"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""
# Ensure src/ is in path for direct execution (inline to avoid circular import)
if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Configure temp directory to use /media partition (avoid root partition space issues)
    tmp_dir = project_root / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    import os
    os.environ['TMPDIR'] = str(tmp_dir)
    os.environ['TEMP'] = str(tmp_dir)
    os.environ['TMP'] = str(tmp_dir)

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

from my3dis.common_utils import ensure_dir, setup_logging
from my3dis.pipeline_defaults import (
    DEFAULT_SAM2_CFG as _DEFAULT_SAM2_CFG_PATH,
    DEFAULT_SAM2_CKPT as _DEFAULT_SAM2_CKPT_PATH,
    DEFAULT_SAM2_ROOT as _DEFAULT_SAM2_ROOT,
    expand_default,
)
from my3dis.tracking import TimingAggregator, format_duration_precise
from my3dis.tracking.level_runner import run_level_tracking
from my3dis.tracking.pipeline_context import (
    LevelRunResult,
    TrackingContext,
    ensure_subset_video,
    prepare_tracking_context,
    resolve_long_tail_area_threshold,
    update_manifest,
)

_SAM2_ROOT_STR = expand_default(_DEFAULT_SAM2_ROOT)
if _SAM2_ROOT_STR not in sys.path:
    sys.path.insert(0, _SAM2_ROOT_STR)

from sam2.build_sam import build_sam2_video_predictor

LOGGER = logging.getLogger("my3dis.track_from_candidates")

DEFAULT_SAM2_ROOT = _SAM2_ROOT_STR
DEFAULT_SAM2_CFG = str(_DEFAULT_SAM2_CFG_PATH)
DEFAULT_SAM2_CKPT = str(_DEFAULT_SAM2_CKPT_PATH)


def resolve_sam2_config_path(config_arg: str) -> str:
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
    return config_arg


def configure_logging(explicit_level: Optional[int] = None) -> int:
    level = setup_logging(explicit_level=explicit_level)
    LOGGER.setLevel(level)
    return level


def run_tracking(
    *,
    data_path: str,
    candidates_root: str,
    output: str,
    levels: Union[str, List[int]] = "2,4,6",
    sam2_cfg: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CFG,
    sam2_ckpt: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CKPT,
    sam2_max_propagate: Optional[int] = None,
    log_level: Optional[int] = None,
    iou_threshold: float = 0.6,
    long_tail_box_prompt: bool = False,
    all_box_prompt: bool = False,
    mask_scale_ratio: float = 1.0,
    comparison_sample_stride: Optional[int] = None,
    comparison_max_samples: Optional[int] = None,
    render_viz: bool = True,
    build_sam2_tree: bool = False,
    sam2_tree_containment_threshold: float = 0.95,
    sam2_tree_temporal_overlap_threshold: float = 0.3,
    visualize_sam2_tree: bool = False,
    tree_viz_min_children: int = 1,
    tree_viz_max_families: Optional[int] = None,
) -> str:
    if not sam2_cfg:
        sam2_cfg = DEFAULT_SAM2_CFG
    if not sam2_ckpt:
        sam2_ckpt = DEFAULT_SAM2_CKPT
    sam2_cfg = os.fspath(sam2_cfg) if isinstance(sam2_cfg, os.PathLike) else sam2_cfg
    sam2_ckpt = os.fspath(sam2_ckpt) if isinstance(sam2_ckpt, os.PathLike) else sam2_ckpt

    # run_tracking() later chdirs into DEFAULT_SAM2_ROOT, so anchor every
    # filesystem path to the caller's working directory first. sam2_cfg may also
    # be a Hydra config name (not a file); only absolutize it if it exists.
    data_path = os.path.abspath(os.path.expanduser(os.fspath(data_path)))
    candidates_root = os.path.abspath(os.path.expanduser(os.fspath(candidates_root)))
    output = os.path.abspath(os.path.expanduser(os.fspath(output)))
    sam2_ckpt = os.path.abspath(os.path.expanduser(sam2_ckpt))
    if os.path.isfile(os.path.expanduser(sam2_cfg)):
        sam2_cfg = os.path.abspath(os.path.expanduser(sam2_cfg))

    # Set PyTorch CUDA allocator configuration to reduce OOM issues
    # Use expandable_segments to avoid memory fragmentation
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        LOGGER.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce OOM risk")

    configure_logging(log_level)

    try:
        mask_scale_ratio = float(mask_scale_ratio)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid mask_scale_ratio={mask_scale_ratio!r}')
    if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
        raise ValueError('mask_scale_ratio must be within (0, 1]')

    overall_start = time.perf_counter()
    if isinstance(levels, str):
        level_list = [int(x) for x in levels.split(',') if x.strip()]
    else:
        level_list = [int(x) for x in levels]

    LOGGER.info("SAM2 tracking started (levels=%s)", ",".join(str(x) for x in level_list))

    context = prepare_tracking_context(
        candidates_root=candidates_root,
        level_list=level_list,
        sam2_max_propagate=sam2_max_propagate,
    )
    manifest = context.manifest
    level_list = context.level_list
    ssam_frames = context.ssam_frames
    ssam_absolute_indices = context.ssam_absolute_indices
    ssam_local_indices = context.ssam_local_indices
    ssam_freq = context.ssam_freq
    sam2_max_propagate = context.sam2_max_propagate

    LOGGER.info(
        "Configuration: ssam_freq=%d, sam2_max_propagate=%s, ssam_frames=%d, iou_threshold=%.2f, mask_scale_ratio=%.3f, preview_stride=%s, preview_max=%s, render_viz=%s",
        ssam_freq,
        sam2_max_propagate,
        len(ssam_frames),
        iou_threshold,
        mask_scale_ratio,
        comparison_sample_stride,
        comparison_max_samples,
        render_viz,
    )

    long_tail_area_threshold = resolve_long_tail_area_threshold(
        manifest=manifest,
        long_tail_box_prompt=long_tail_box_prompt,
        all_box_prompt=all_box_prompt,
    )

    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    sam2_cfg_resolved = resolve_sam2_config_path(sam2_cfg)
    # predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt, vos_optimized=True)
    # print("我有用VOS版本的SAM2了！")
    predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt)

    out_root = ensure_dir(output)
    subset_dir, subset_map = ensure_subset_video(
        context,
        data_path=data_path,
        out_root=out_root,
    )
    LOGGER.info("Selected frames available at %s", subset_dir)

    frame_index_to_name = {
        int(idx): str(name)
        for idx, name in zip(context.selected_indices, context.selected_frames)
    }

    level_results: List[LevelRunResult] = []
    overall_timer = TimingAggregator()

    # Create shared ProvenanceTracker for cross-level parent lookup
    # This tracker accumulates SSAM → SAM2 mappings across all levels,
    # enabling Level 4+ objects to find their Level 2 parents
    from my3dis.tracking.provenance_tracker import ProvenanceTracker
    shared_provenance_tracker = ProvenanceTracker()
    LOGGER.info("Created shared ProvenanceTracker for cross-level parent tracking")

    for level in level_list:
        result = run_level_tracking(
            level=level,
            candidates_root=candidates_root,
            data_path=data_path,
            subset_dir=subset_dir,
            subset_map=subset_map,
            predictor=predictor,
            frame_index_lookup=frame_index_to_name,
            selected_indices=context.selected_indices,
            ssam_local_indices=ssam_local_indices,
            sam2_max_propagate=sam2_max_propagate,
            iou_threshold=iou_threshold,
            long_tail_box_prompt=long_tail_box_prompt,
            all_box_prompt=all_box_prompt,
            long_tail_area_threshold=long_tail_area_threshold,
            mask_scale_ratio=mask_scale_ratio,
            comparison_sample_stride=comparison_sample_stride,
            comparison_max_samples=comparison_max_samples,
            render_viz=render_viz,
            out_root=out_root,
            shared_provenance_tracker=shared_provenance_tracker,
        )
        level_results.append(result)
        overall_timer.merge(result.timer)

    if level_results:
        summary = "; ".join(
            f"L{lvl}: {objs} objects / {frames} frames "
            f"(track={format_duration_precise(track)}, persist={format_duration_precise(persist)}, "
            f"viz={format_duration_precise(viz)}, render={format_duration_precise(render)}, "
            f"total={format_duration_precise(total)})"
            for (
                lvl,
                objs,
                frames,
                track,
                persist,
                viz,
                render,
                total,
            ) in (result.stats for result in level_results)
        )
        LOGGER.info("Tracking summary → %s", summary)

    if overall_timer.items():
        category_summary = []
        for label, prefix in [
            ("track", 'track.'),
            ("persist", 'persist.'),
            ("viz", 'viz.'),
        ]:
            total = overall_timer.total_prefix(prefix)
            if total > 0:
                category_summary.append(f"{label}={format_duration_precise(total)}")
        if category_summary:
            LOGGER.info("Aggregate timing by stage → %s", ", ".join(category_summary))
        LOGGER.debug("Aggregate timing breakdown → %s", overall_timer.format_breakdown())

    # Save unified cross-level provenance tree (BUGFIX 2025-11-05)
    # This ensures cross-level parent relationships are preserved for family tree building
    if shared_provenance_tracker is not None and level_list:
        import json

        relations_dir = ensure_dir(os.path.join(out_root, 'relations'))
        unified_tree_path = os.path.join(relations_dir, 'unified_provenance_tree.json')

        family_result = shared_provenance_tracker.get_family_hierarchy()
        stats = shared_provenance_tracker.get_statistics()

        # Count objects per level for summary
        objects_per_level = {}
        for obj_id in family_result['parent_map'].keys():
            level_id = obj_id // 1000  # L2=2xxx, L4=4xxx, L6=6xxx
            level_key = f'L{level_id}'
            objects_per_level[level_key] = objects_per_level.get(level_key, 0) + 1

        # Count cross-level relationships
        cross_level_children = 0
        for obj_id, parent_id in family_result['parent_map'].items():
            if parent_id is not None:
                obj_level = obj_id // 1000
                parent_level = parent_id // 1000
                if obj_level != parent_level:
                    cross_level_children += 1

        with open(unified_tree_path, 'w', encoding='utf-8') as f:
            json.dump({
                'levels': sorted(level_list),
                'parent_map': family_result['parent_map'],
                'orphan_count': family_result['orphan_count'],
                'orphan_ids': family_result['orphan_ids'],
                'statistics': stats,
                'objects_per_level': objects_per_level,
                'cross_level_children': cross_level_children,
            }, f, indent=2, ensure_ascii=False)

        LOGGER.info(
            "Unified provenance tree saved: %d total objects (%s), %d cross-level relationships, %d orphans (%.1f%% rejection rate)",
            sum(objects_per_level.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(objects_per_level.items())),
            cross_level_children,
            family_result['orphan_count'],
            stats['rejection_rate'] * 100,
        )

    update_manifest(
        context,
        out_root=out_root,
        level_results=level_results,
        mask_scale_ratio=mask_scale_ratio,
        render_viz=render_viz,
    )

    # Build SAM2-based tree if requested
    if build_sam2_tree:
        LOGGER.info("Building SAM2-based containment tree...")
        tree_start = time.perf_counter()

        try:
            from my3dis.sam2_tree_builder import build_sam2_containment_tree

            tree_path = build_sam2_containment_tree(
                run_dir=out_root,
                levels=level_list,
                containment_threshold=sam2_tree_containment_threshold,
                temporal_overlap_threshold=sam2_tree_temporal_overlap_threshold,
                output_filename='sam2_tree.json',
            )

            tree_elapsed = time.perf_counter() - tree_start
            LOGGER.info(
                "✅ SAM2 tree built in %s → %s",
                format_duration_precise(tree_elapsed),
                tree_path
            )

            # Visualize tree if requested
            if visualize_sam2_tree:
                LOGGER.info("Generating SAM2 tree visualizations...")
                viz_start = time.perf_counter()

                try:
                    from pathlib import Path
                    import sys
                    # Add scripts directory to path to import visualization module
                    scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
                    if str(scripts_dir) not in sys.path:
                        sys.path.insert(0, str(scripts_dir))

                    from visualize_family_trees import (
                        load_sam2_tree, find_families, load_npz_manifest,
                        find_best_frame_for_family, visualize_family, get_frame_list
                    )

                    scene_root = Path(out_root)
                    output_dir = scene_root / 'family_viz'
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Load tree
                    tree_data = load_sam2_tree(tree_path)
                    levels_list = tree_data['meta']['levels']

                    # Find families
                    families = find_families(tree_data)
                    families = [f for f in families if len(f['children']) >= tree_viz_min_children]

                    if tree_viz_max_families:
                        families = families[:tree_viz_max_families]

                    if families:
                        LOGGER.info("Found %d families to visualize", len(families))

                        # Load NPZ manifests
                        level_data = {}
                        level_video_npz = {}
                        for level in levels_list:
                            obj_npz_path = scene_root / f"level_{level}" / f"object_segments_L{level:02d}.npz"
                            LOGGER.info("Loading NPZ manifest for L%d: %s", level, obj_npz_path)

                            if obj_npz_path.exists():
                                try:
                                    obj_frames, linked_video = load_npz_manifest(obj_npz_path)
                                    level_data[level] = obj_frames
                                    LOGGER.info("  ✓ Loaded %d objects for L%d", len(obj_frames), level)

                                    video_npz_path = scene_root / f"level_{level}" / f"video_segments_L{level:02d}.npz"
                                    if video_npz_path.exists():
                                        level_video_npz[level] = video_npz_path
                                except Exception as load_exc:
                                    LOGGER.error("  ✗ Failed to load NPZ for L%d: %s", level, load_exc, exc_info=True)
                            else:
                                LOGGER.warning("  ⚠ NPZ not found for L%d: %s", level, obj_npz_path)

                        # Verify all levels are loaded
                        missing_levels = set(levels_list) - set(level_data.keys())
                        if missing_levels:
                            LOGGER.error("Missing level_data for levels: %s", missing_levels)
                            raise RuntimeError(f"Failed to load NPZ manifests for levels: {missing_levels}")

                        # Visualize each family
                        viz_count = 0
                        for i, family in enumerate(families, 1):
                            best_frame = find_best_frame_for_family(family, level_data, level_video_npz)
                            if best_frame is None:
                                continue

                            output_name = f"family_L{family['level']}_obj{family['obj_id']:04d}_frame{best_frame:06d}.jpg"
                            output_path = output_dir / output_name

                            try:
                                visualize_family(
                                    family,
                                    best_frame,
                                    scene_root,
                                    Path(data_path),
                                    output_path,
                                    level_video_npz,
                                    levels=levels_list,
                                )
                                viz_count += 1
                            except Exception as viz_exc:
                                LOGGER.warning("Failed to visualize family %d: %s", family['obj_id'], viz_exc)

                        viz_elapsed = time.perf_counter() - viz_start
                        LOGGER.info(
                            "✅ Generated %d/%d tree visualizations in %s → %s",
                            viz_count, len(families),
                            format_duration_precise(viz_elapsed),
                            output_dir
                        )
                    else:
                        LOGGER.info("No families found matching criteria (min_children=%d)", tree_viz_min_children)

                except ImportError as imp_exc:
                    LOGGER.error("Failed to import visualization modules: %s", imp_exc)
                except Exception as viz_exc:
                    LOGGER.error("Failed to generate tree visualizations: %s", viz_exc, exc_info=True)

        except Exception as exc:
            LOGGER.error("Failed to build SAM2 tree: %s", exc, exc_info=True)
            LOGGER.warning("Continuing without SAM2 tree")

    LOGGER.info("Tracking results saved at %s", out_root)
    LOGGER.info(
        "Tracking completed in %s",
        format_duration_precise(time.perf_counter() - overall_start),
    )

    return out_root


def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking from pre-generated candidates")
    ap.add_argument('--data-path', required=True, help='Original frames dir')
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--sam2-cfg', default=DEFAULT_SAM2_CFG,
                    help='SAM2 config YAML or Hydra path (default: sam2.1_hiera_l)')
    ap.add_argument('--sam2-ckpt', default=DEFAULT_SAM2_CKPT,
                    help='SAM2 checkpoint path (default: sam2.1_hiera_large.pt)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Limit SAM2 propagation to N frames per direction (default: unlimited)')
    ap.add_argument('--iou-threshold', type=float, default=0.6,
                    help='IoU threshold for deduplicating SAM2 prompts (default: 0.6)')
    ap.add_argument('--long-tail-box-prompt', action='store_true',
                    help='Convert long-tail small objects to SAM2 box prompts')
    ap.add_argument('--all-box-prompt', action='store_true',
                    help='Convert all mask prompts to SAM2 box prompts')
    ap.add_argument('--mask-scale-ratio', type=float, default=1.0,
                    help='Downscale masks before persistence (e.g., 0.3 keeps 30% resolution)')
    ap.add_argument('--skip-viz', action='store_true',
                    help='Disable all additional visualization renders to keep outputs minimal')
    ap.add_argument('--build-sam2-tree', action='store_true',
                    help='Build containment-based tree from SAM2 tracked objects')
    ap.add_argument('--sam2-tree-containment-threshold', type=float, default=0.95,
                    help='Minimum avg containment for parent-child relationship (default: 0.95)')
    ap.add_argument('--sam2-tree-temporal-overlap-threshold', type=float, default=0.3,
                    help='Minimum temporal overlap ratio for considering parent-child (default: 0.3)')
    args = ap.parse_args()

    run_tracking(
        data_path=args.data_path,
        candidates_root=args.candidates_root,
        output=args.output,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        sam2_max_propagate=args.sam2_max_propagate,
        iou_threshold=args.iou_threshold,
        long_tail_box_prompt=args.long_tail_box_prompt,
        all_box_prompt=args.all_box_prompt,
        mask_scale_ratio=args.mask_scale_ratio,
        render_viz=not args.skip_viz,
        build_sam2_tree=args.build_sam2_tree,
        sam2_tree_containment_threshold=args.sam2_tree_containment_threshold,
        sam2_tree_temporal_overlap_threshold=args.sam2_tree_temporal_overlap_threshold,
    )


if __name__ == '__main__':
    main()
