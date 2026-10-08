"""Stage configuration dataclasses for simplified parameter passing.

This module provides type-safe configuration objects for each pipeline stage,
replacing the previous approach of passing 10-20 individual parameters.
All validation logic is centralized in the from_yaml_config() class methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .errors import WorkflowConfigError


@dataclass
class SSAMStageConfig:
    """Semantic-SAM stage configuration (validated and ready to use).

    All parameters have been validated and converted to appropriate types.
    Frame range is pre-parsed into start/end/step integers.
    """

    # Essential paths
    data_path: Path
    output_root: Path
    sam_ckpt: Path

    # Processing parameters
    levels: List[int]
    frames_start: int
    frames_end: int
    frames_step: int

    # Filtering thresholds
    min_area: int = 300
    fill_area: int = 300
    stability_threshold: float = 0.9
    cascade_filtering: bool = True  # Enable parent-aware cascade filtering

    # Gap-fill
    add_gaps: bool = False

    # Mask downscaling
    downscale_masks: bool = False
    mask_scale_ratio: float = 1.0

    # Sampling frequency
    ssam_freq: int = 1

    # Progressive refinement control
    max_masks_per_level: int = 2000

    # Output control
    persist_raw: bool = True
    skip_filtering: bool = False
    no_timestamp: bool = False
    tag_in_path: bool = True

    # Optional parameters
    experiment_tag: Optional[str] = None
    sam2_max_propagate: Optional[int] = None
    log_level: Optional[int] = None

    @classmethod
    def from_yaml_config(
        cls,
        stage_cfg: Dict[str, Any],
        experiment_cfg: Dict[str, Any],
        data_path: str,
        output_root: str,
    ) -> SSAMStageConfig:
        """Create configuration from YAML config dict with full validation.

        Args:
            stage_cfg: stages.ssam section from YAML
            experiment_cfg: experiment section from YAML
            data_path: Scene data path
            output_root: Output root directory

        Returns:
            Validated SSAMStageConfig object

        Raises:
            WorkflowConfigError: If configuration is invalid
        """
        from .scenes import resolve_levels, stage_frames_string
        from ..common_utils import parse_range

        # Validate and parse levels
        levels = resolve_levels(stage_cfg, None, experiment_cfg.get('levels'))

        # Validate and parse frames
        frames_str = stage_frames_string(stage_cfg, experiment_cfg)
        start, end, step = parse_range(frames_str)

        # Validate checkpoint
        sam_ckpt_cfg = stage_cfg.get('sam_ckpt') or experiment_cfg.get('sam_ckpt')
        if sam_ckpt_cfg:
            sam_ckpt = Path(sam_ckpt_cfg).expanduser().absolute()
        else:
            # Import default only when needed
            try:
                from ..generate_candidates import DEFAULT_SEMANTIC_SAM_CKPT
                sam_ckpt = Path(DEFAULT_SEMANTIC_SAM_CKPT)
            except ImportError:
                raise WorkflowConfigError(
                    'Cannot import DEFAULT_SEMANTIC_SAM_CKPT. '
                    'Please set stages.ssam.sam_ckpt explicitly.'
                )

        if not sam_ckpt.exists():
            raise WorkflowConfigError(
                f'Semantic-SAM checkpoint not found at {sam_ckpt}. '
                'Set stages.ssam.sam_ckpt or experiment.sam_ckpt to a valid file path.'
            )

        # Validate ssam_freq
        try:
            ssam_freq = int(stage_cfg.get('ssam_freq', 1))
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(
                f"invalid stages.ssam.ssam_freq: {stage_cfg.get('ssam_freq')!r}"
            ) from exc
        ssam_freq = max(1, ssam_freq)

        # Validate max_masks_per_level
        # Special value: 0 or -1 means unlimited (no upper limit)
        max_masks_cfg = stage_cfg.get('max_masks_per_level')
        if max_masks_cfg is None:
            max_masks_per_level = 2000  # Default value
        else:
            try:
                max_masks_per_level = int(max_masks_cfg)
            except (TypeError, ValueError) as exc:
                raise WorkflowConfigError(
                    f"invalid stages.ssam.max_masks_per_level: {max_masks_cfg!r}"
                ) from exc
            # Allow 0 or -1 to represent unlimited (no cap)
            if max_masks_per_level > 0:
                max_masks_per_level = max(1, max_masks_per_level)
            elif max_masks_per_level < 0:
                max_masks_per_level = 0  # Normalize negative values to 0 for "unlimited"

        # Validate filtering parameters
        min_area = int(stage_cfg.get('min_area', 300))

        fill_area_cfg = stage_cfg.get('fill_area')
        if fill_area_cfg is None:
            fill_area = min_area
        else:
            try:
                fill_area = int(fill_area_cfg)
            except (TypeError, ValueError) as exc:
                raise WorkflowConfigError(
                    f'invalid stages.ssam.fill_area: {fill_area_cfg!r}'
                ) from exc
        fill_area = max(0, fill_area)

        # Handle stability_threshold (can be None to disable stability filtering)
        stability_cfg = stage_cfg.get('stability_threshold', 0.9)
        if stability_cfg is None:
            stability = None
        else:
            try:
                stability = float(stability_cfg)
            except (TypeError, ValueError) as exc:
                raise WorkflowConfigError(
                    f'invalid stages.ssam.stability_threshold: {stability_cfg!r}'
                ) from exc

        cascade_filtering = bool(stage_cfg.get('cascade_filtering', True))

        # Validate downscaling
        downscale_masks = bool(stage_cfg.get('downscale_masks', False))
        ssam_downscale_ratio = stage_cfg.get('downscale_ratio',
                                            stage_cfg.get('mask_scale_ratio', 1.0))
        try:
            ssam_downscale_ratio = float(ssam_downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(
                f'invalid stages.ssam.downscale_ratio: {ssam_downscale_ratio!r}'
            ) from exc

        # Output control
        append_timestamp = stage_cfg.get('append_timestamp', True)

        tag_in_path_raw = stage_cfg.get('tag_in_path')
        if tag_in_path_raw is None:
            tag_in_path_raw = experiment_cfg.get('tag_in_path')

        # Resolve bool flag
        if tag_in_path_raw is None:
            tag_in_path = True
        elif isinstance(tag_in_path_raw, bool):
            tag_in_path = tag_in_path_raw
        elif isinstance(tag_in_path_raw, str):
            lowered = tag_in_path_raw.strip().lower()
            tag_in_path = lowered in {'1', 'true', 'yes', 'on'}
        else:
            tag_in_path = bool(tag_in_path_raw)

        return cls(
            data_path=Path(data_path).expanduser().absolute(),
            output_root=Path(output_root).expanduser().absolute(),
            levels=levels,
            frames_start=max(0, start),
            frames_end=end,
            frames_step=step,
            sam_ckpt=sam_ckpt,
            min_area=min_area,
            fill_area=fill_area,
            stability_threshold=stability,
            cascade_filtering=cascade_filtering,
            add_gaps=bool(stage_cfg.get('add_gaps', False)),
            downscale_masks=downscale_masks,
            mask_scale_ratio=ssam_downscale_ratio,
            ssam_freq=ssam_freq,
            max_masks_per_level=max_masks_per_level,
            persist_raw=bool(stage_cfg.get('persist_raw', True)),
            skip_filtering=bool(stage_cfg.get('skip_filtering', False)),
            no_timestamp=not append_timestamp,
            tag_in_path=tag_in_path,
            experiment_tag=stage_cfg.get('experiment_tag') or experiment_cfg.get('tag'),
            sam2_max_propagate=stage_cfg.get('sam2_max_propagate'),
        )

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        """Convert to legacy function call kwargs for backward compatibility.

        This allows gradual migration from the old 19-parameter signature.
        """
        from ..common_utils import list_to_csv

        return {
            'data_path': str(self.data_path),
            'levels': list_to_csv(self.levels),
            'frames': f'{self.frames_start}:{self.frames_end}:{self.frames_step}',
            'sam_ckpt': str(self.sam_ckpt),
            'output': str(self.output_root),
            'min_area': self.min_area,
            'fill_area': self.fill_area,
            'stability_threshold': self.stability_threshold,
            'cascade_filtering': self.cascade_filtering,
            'add_gaps': self.add_gaps,
            'no_timestamp': self.no_timestamp,
            'log_level': self.log_level,
            'ssam_freq': self.ssam_freq,
            'max_masks_per_level': self.max_masks_per_level,
            'sam2_max_propagate': self.sam2_max_propagate,
            'experiment_tag': self.experiment_tag,
            'persist_raw': self.persist_raw,
            'skip_filtering': self.skip_filtering,
            'downscale_masks': self.downscale_masks,
            'mask_scale_ratio': self.mask_scale_ratio,
            'tag_in_path': self.tag_in_path,
        }


@dataclass
class TrackingStageConfig:
    """SAM2 tracking stage configuration (validated and ready to use)."""

    # Essential paths
    data_path: Path
    candidates_root: Path
    output_root: Path

    # SAM2 configuration
    sam2_cfg: Path
    sam2_ckpt: Path

    # Processing parameters
    levels: List[int]
    sam2_max_propagate: Optional[int] = None
    iou_threshold: float = 0.6

    # Mask downscaling
    mask_scale_ratio: float = 1.0

    # Visualization
    render_viz: bool = True
    comparison_sample_stride: Optional[int] = None
    comparison_max_samples: Optional[int] = None

    # Prompt modes
    long_tail_box_prompt: bool = False
    all_box_prompt: bool = False

    # SAM2 tree building (configured via stages.tracker.build_sam2_tree and stages.tracker.sam2_tree in YAML)
    build_sam2_tree: bool = False
    sam2_tree_containment_threshold: float = 0.95
    sam2_tree_temporal_overlap_threshold: float = 0.3

    # SAM2 tree visualization (configured via stages.tracker.visualize_sam2_tree and stages.tracker.tree_viz in YAML)
    visualize_sam2_tree: bool = False
    tree_viz_min_children: int = 1
    tree_viz_max_families: Optional[int] = None

    # Logging
    log_level: Optional[int] = None

    @classmethod
    def from_yaml_config(
        cls,
        stage_cfg: Dict[str, Any],
        experiment_cfg: Dict[str, Any],
        data_path: str,
        candidates_root: str,
        output_root: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> TrackingStageConfig:
        """Create configuration from YAML config dict with full validation.

        Args:
            stage_cfg: stages.tracker section from YAML
            experiment_cfg: experiment section from YAML
            data_path: Scene data path
            candidates_root: Candidates directory from SSAM stage
            output_root: Output root directory
            manifest: Optional manifest from SSAM stage for default values

        Returns:
            Validated TrackingStageConfig object

        Raises:
            WorkflowConfigError: If configuration is invalid
        """
        from .scenes import resolve_levels

        # Validate and parse levels
        levels = resolve_levels(stage_cfg, manifest, experiment_cfg.get('levels'))

        # Validate SAM2 configuration
        sam2_cfg_raw = stage_cfg.get('sam2_cfg') or experiment_cfg.get('sam2_cfg')
        if sam2_cfg_raw:
            sam2_cfg = Path(sam2_cfg_raw).expanduser().absolute()
        else:
            try:
                from ..track_from_candidates import DEFAULT_SAM2_CFG
                sam2_cfg = Path(DEFAULT_SAM2_CFG)
            except ImportError:
                raise WorkflowConfigError(
                    'Cannot import DEFAULT_SAM2_CFG. '
                    'Please set stages.tracker.sam2_cfg explicitly.'
                )

        if not sam2_cfg.exists():
            raise WorkflowConfigError(f'SAM2 config not found at {sam2_cfg}')

        # Validate SAM2 checkpoint
        sam2_ckpt_raw = stage_cfg.get('sam2_ckpt') or experiment_cfg.get('sam2_ckpt')
        if sam2_ckpt_raw:
            sam2_ckpt = Path(sam2_ckpt_raw).expanduser().absolute()
        else:
            try:
                from ..track_from_candidates import DEFAULT_SAM2_CKPT
                sam2_ckpt = Path(DEFAULT_SAM2_CKPT)
            except ImportError:
                raise WorkflowConfigError(
                    'Cannot import DEFAULT_SAM2_CKPT. '
                    'Please set stages.tracker.sam2_ckpt explicitly.'
                )

        if not sam2_ckpt.exists():
            raise WorkflowConfigError(f'SAM2 checkpoint not found at {sam2_ckpt}')

        # Validate max_propagate
        max_propagate = stage_cfg.get('max_propagate') or stage_cfg.get('sam2_max_propagate')
        if max_propagate is not None:
            try:
                max_propagate = int(max_propagate)
            except (TypeError, ValueError):
                max_propagate = None

        # Validate IoU threshold
        iou_threshold = float(stage_cfg.get('iou_threshold', 0.6))

        # Validate mask scaling (use manifest default if available)
        try:
            mask_ratio_default = float(manifest.get('mask_scale_ratio', 1.0)) if manifest else 1.0
        except (TypeError, ValueError):
            mask_ratio_default = 1.0

        downscale_enabled = bool(stage_cfg.get('downscale_masks', mask_ratio_default < 1.0))
        downscale_ratio = stage_cfg.get(
            'downscale_ratio', mask_ratio_default if mask_ratio_default < 1.0 else 0.3
        )
        try:
            downscale_ratio = float(downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f'Invalid tracker.downscale_ratio={downscale_ratio!r}') from exc

        mask_scale_ratio = downscale_ratio if downscale_enabled else 1.0
        if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
            raise WorkflowConfigError('tracker.downscale_ratio must be in (0, 1] when downscale_masks is true')

        # Prompt modes with alias support
        prompt_mode_raw = str(stage_cfg.get('prompt_mode', 'all_mask')).lower()
        prompt_aliases = {
            'none': 'all_mask',
            'all_mask': 'all_mask',
            'long_tail': 'lt_bbox',
            'lt_bbox': 'lt_bbox',
            'all': 'all_bbox',
            'all_bbox': 'all_bbox',
        }
        if prompt_mode_raw not in prompt_aliases:
            raise WorkflowConfigError(f'Unknown tracker.prompt_mode={prompt_mode_raw}')
        prompt_mode = prompt_aliases[prompt_mode_raw]
        long_tail_box_prompt = prompt_mode == 'lt_bbox'
        all_box_prompt = prompt_mode == 'all_bbox'

        # Visualization
        render_viz = bool(stage_cfg.get('render_viz', True))

        # Comparison sampling configuration with validation
        comparison_sample_stride: Optional[int] = None
        comparison_max_samples: Optional[int] = None
        comparison_cfg = stage_cfg.get('comparison_sampling')
        if comparison_cfg is not None:
            if not isinstance(comparison_cfg, dict):
                raise WorkflowConfigError('tracker.comparison_sampling must be a mapping')
            if 'stride' in comparison_cfg:
                try:
                    comparison_sample_stride = int(comparison_cfg['stride'])
                except (TypeError, ValueError):
                    raise WorkflowConfigError(
                        f"Invalid tracker.comparison_sampling.stride={comparison_cfg['stride']!r}"
                    )
                if comparison_sample_stride <= 0:
                    raise WorkflowConfigError('tracker.comparison_sampling.stride must be > 0')
            if 'max_frames' in comparison_cfg:
                try:
                    max_frames_val = int(comparison_cfg['max_frames'])
                except (TypeError, ValueError):
                    raise WorkflowConfigError(
                        f"Invalid tracker.comparison_sampling.max_frames={comparison_cfg['max_frames']!r}"
                    )
                if max_frames_val < 0:
                    raise WorkflowConfigError('tracker.comparison_sampling.max_frames must be >= 0')
                comparison_max_samples = max_frames_val if max_frames_val > 0 else None

        # SAM2 tree building configuration
        build_sam2_tree = bool(stage_cfg.get('build_sam2_tree', False))
        sam2_tree_config = stage_cfg.get('sam2_tree', {})
        sam2_tree_containment_threshold = float(sam2_tree_config.get('containment_threshold', 0.95))
        sam2_tree_temporal_overlap_threshold = float(sam2_tree_config.get('temporal_overlap_threshold', 0.3))

        # Validate thresholds
        if not (0.0 <= sam2_tree_containment_threshold <= 1.0):
            raise WorkflowConfigError('tracker.sam2_tree.containment_threshold must be in [0, 1]')
        if not (0.0 <= sam2_tree_temporal_overlap_threshold <= 1.0):
            raise WorkflowConfigError('tracker.sam2_tree.temporal_overlap_threshold must be in [0, 1]')

        # SAM2 tree visualization configuration
        visualize_sam2_tree = bool(stage_cfg.get('visualize_sam2_tree', False))
        tree_viz_config = stage_cfg.get('tree_viz', {})
        tree_viz_min_children = int(tree_viz_config.get('min_children', 1))
        tree_viz_max_families_raw = tree_viz_config.get('max_families')
        tree_viz_max_families = int(tree_viz_max_families_raw) if tree_viz_max_families_raw is not None else None

        return cls(
            data_path=Path(data_path).expanduser().absolute(),
            candidates_root=Path(candidates_root).expanduser().absolute(),
            output_root=Path(output_root).expanduser().absolute(),
            levels=levels,
            sam2_cfg=sam2_cfg,
            sam2_ckpt=sam2_ckpt,
            sam2_max_propagate=max_propagate,
            iou_threshold=iou_threshold,
            mask_scale_ratio=mask_scale_ratio,
            render_viz=render_viz,
            comparison_sample_stride=comparison_sample_stride,
            comparison_max_samples=comparison_max_samples,
            long_tail_box_prompt=long_tail_box_prompt,
            all_box_prompt=all_box_prompt,
            build_sam2_tree=build_sam2_tree,
            sam2_tree_containment_threshold=sam2_tree_containment_threshold,
            sam2_tree_temporal_overlap_threshold=sam2_tree_temporal_overlap_threshold,
            visualize_sam2_tree=visualize_sam2_tree,
            tree_viz_min_children=tree_viz_min_children,
            tree_viz_max_families=tree_viz_max_families,
        )

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        """Convert to legacy function call kwargs for backward compatibility."""
        from ..common_utils import list_to_csv

        return {
            'data_path': str(self.data_path),
            'candidates_root': str(self.candidates_root),
            'output': str(self.output_root),
            'levels': list_to_csv(self.levels),
            'sam2_cfg': str(self.sam2_cfg),
            'sam2_ckpt': str(self.sam2_ckpt),
            'sam2_max_propagate': self.sam2_max_propagate,
            'log_level': self.log_level,
            'iou_threshold': self.iou_threshold,
            'long_tail_box_prompt': self.long_tail_box_prompt,
            'all_box_prompt': self.all_box_prompt,
            'mask_scale_ratio': self.mask_scale_ratio,
            'comparison_sample_stride': self.comparison_sample_stride,
            'comparison_max_samples': self.comparison_max_samples,
            'render_viz': self.render_viz,
            'build_sam2_tree': self.build_sam2_tree,
            'sam2_tree_containment_threshold': self.sam2_tree_containment_threshold,
            'sam2_tree_temporal_overlap_threshold': self.sam2_tree_temporal_overlap_threshold,
            'visualize_sam2_tree': self.visualize_sam2_tree,
            'tree_viz_min_children': self.tree_viz_min_children,
            'tree_viz_max_families': self.tree_viz_max_families,
        }


@dataclass
class FilterStageConfig:
    """Candidate filtering stage configuration."""

    root: Path
    levels: List[int]
    min_area: int
    stability_threshold: float
    update_manifest: bool = True
    quiet: bool = False

    @classmethod
    def from_yaml_config(
        cls,
        stage_cfg: Dict[str, Any],
        experiment_cfg: Dict[str, Any],
        candidates_root: str,
    ) -> FilterStageConfig:
        """Create configuration from YAML config dict with full validation."""
        from .scenes import resolve_levels

        levels = resolve_levels(stage_cfg, None, experiment_cfg.get('levels'))
        min_area = int(stage_cfg.get('min_area', 300))
        stability = float(stage_cfg.get('stability_threshold', 0.9))

        return cls(
            root=Path(candidates_root).expanduser().absolute(),
            levels=levels,
            min_area=min_area,
            stability_threshold=stability,
            update_manifest=True,
            quiet=False,
        )

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        """Convert to legacy function call kwargs."""
        return {
            'root': str(self.root),
            'levels': self.levels,
            'min_area': self.min_area,
            'stability_threshold': self.stability_threshold,
            'update_manifest': self.update_manifest,
            'quiet': self.quiet,
        }
