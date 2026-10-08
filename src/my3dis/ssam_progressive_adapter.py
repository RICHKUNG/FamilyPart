"""
Adapter that reuses Semantic-SAM's progressive_refinement.py to generate
per-level mask candidates for a list of frames, returning data in the
format expected by our pipeline (bbox XYWH, segmentation, area, level, etc.).

Runs under the Semantic-SAM environment.
"""

import contextlib
import io
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # SciPy may be optional in some environments
    ndimage = None

from my3dis.common_utils import (
    bbox_from_mask_xyxy,
    bbox_xyxy_to_xywh,
    downscale_binary_mask,
    ensure_dir,
    pack_binary_mask,
    unpack_binary_mask,
)
from my3dis.pipeline_defaults import DEFAULT_SEMANTIC_SAM_ROOT as _DEFAULT_SEMANTIC_SAM_ROOT, expand_default

_SEM_ROOT_STR = expand_default(_DEFAULT_SEMANTIC_SAM_ROOT)
DEFAULT_SEMANTIC_SAM_ROOT = _SEM_ROOT_STR

# Prepend Semantic-SAM root to sys.path to ensure it takes precedence over
# any installed semantic_sam package in site-packages
if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SEMANTIC_SAM_ROOT)
elif sys.path[0] != DEFAULT_SEMANTIC_SAM_ROOT:
    # If already in path but not first, move it to the front
    sys.path.remove(DEFAULT_SEMANTIC_SAM_ROOT)
    sys.path.insert(0, DEFAULT_SEMANTIC_SAM_ROOT)

# Import Semantic-SAM modules
from semantic_sam import build_semantic_sam, prepare_image  # noqa: E402
from .semantic_refinement import progressive_refinement_masks, setup_output_directories


LOGGER = logging.getLogger("my3dis.ssam_progressive")


@contextlib.contextmanager
def _semantic_sam_workdir():
    """Temporarily switch to the Semantic-SAM root while preserving the caller's CWD."""

    if not DEFAULT_SEMANTIC_SAM_ROOT:
        yield
        return

    previous_cwd = os.getcwd()
    changed = False

    try:
        os.chdir(DEFAULT_SEMANTIC_SAM_ROOT)
        changed = True
    except FileNotFoundError:
        LOGGER.warning(
            "Semantic-SAM root %s is missing; continuing without changing the working directory",
            DEFAULT_SEMANTIC_SAM_ROOT,
        )
    except OSError as exc:
        LOGGER.warning(
            "Unable to change working directory to Semantic-SAM root %s: %s",
            DEFAULT_SEMANTIC_SAM_ROOT,
            exc,
        )

    try:
        yield
    finally:
        if changed:
            try:
                os.chdir(previous_cwd)
            except OSError as exc:
                LOGGER.warning("Failed to restore working directory to %s: %s", previous_cwd, exc)


def _save_progressive_relations(
    all_results: List[Dict[str, Any]],
    save_root: str,
    levels: List[int],
) -> None:
    """Save parent-child relations from progressive refinement results."""
    from my3dis.relation_index import save_ssam_relations

    # Merge all frame results (they should have consistent tree structure)
    if not all_results:
        return

    # Use the first result as template (tree structure should be identical across frames)
    representative_result = all_results[0]['results']
    save_ssam_relations(representative_result, save_root, levels)


def _extract_gap_components(segs: List[np.ndarray], fill_area: int) -> List[np.ndarray]:
    """Return boolean masks for uncovered regions above the fill_area threshold."""
    valid = [np.asarray(seg, dtype=bool) for seg in segs if seg is not None]
    if not valid:
        return []

    ref_shape = valid[0].shape
    aligned = []
    for seg in valid:
        if seg.shape == ref_shape:
            aligned.append(seg)
        else:
            LOGGER.debug(
                "Skipping gap computation for mask with mismatched shape %s != %s",
                seg.shape,
                ref_shape,
            )
    if not aligned:
        return []

    if len(aligned) == 1:
        coverage = aligned[0].copy()
    else:
        coverage = np.any(np.stack(aligned, axis=0), axis=0)

    gap = np.logical_not(coverage)
    if not gap.any():
        return []

    gap_components: List[np.ndarray] = []

    if ndimage is not None:
        structure = ndimage.generate_binary_structure(gap.ndim, 1)
        labels, num_labels = ndimage.label(gap, structure=structure)
        if num_labels == 0:
            return []

        counts = np.bincount(labels.ravel())
        min_count = max(fill_area, 1)
        for label_id in range(1, num_labels + 1):
            if counts[label_id] < min_count:
                continue
            comp_mask = labels == label_id
            gap_components.append(comp_mask)
        return gap_components

    # Fallback flood-fill when SciPy is unavailable.
    h, w = gap.shape
    visited = np.zeros_like(gap, dtype=bool)

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y + 1 < h:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x + 1 < w:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if not gap[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp_mask = np.zeros_like(gap, dtype=bool)
            comp_mask[y, x] = True
            size = 1

            while stack:
                cy, cx = stack.pop()
                for ny, nx in neighbors(cy, cx):
                    if gap[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        comp_mask[ny, nx] = True
                        size += 1

            if size >= fill_area:
                gap_components.append(comp_mask)

    return gap_components


def generate_with_progressive(
    frames_dir: str,
    selected_frames: List[str],
    sam_ckpt_path: str,
    levels: List[int],
    min_area: int = 300,
    fill_area: Optional[int] = None,
    save_root: str = None,
    persist_outputs: bool = True,
    save_relations: bool = True,
    enable_gap_fill: bool = True,
    mask_scale_ratio: float = 1.0,
    max_masks_per_level: int = 2000,
    cascade_filtering: bool = True,
    stability_threshold: Optional[float] = None,
    verbose: bool = False,
) -> Iterable[Tuple[int, str, Dict[int, List[Dict[str, Any]]]]]:
    """Yield per-level candidates per frame using progressive_refinement.

    Args:
        frames_dir: Directory containing input image frames.
        selected_frames: List of frame filenames to process.
        sam_ckpt_path: Path to Semantic-SAM checkpoint file.
        levels: List of progressive refinement levels (e.g., [2, 4, 6]).
        min_area: Minimum mask area in pixels for filtering.
        fill_area: Minimum area for gap-fill components (or None to disable).
        save_root: Root directory for saving relations and intermediate outputs.
        persist_outputs: If True, save per-frame intermediate outputs (not recommended for large datasets).
        save_relations: If True, save parent-child tree to relations/ directory.
        enable_gap_fill: If True, generate gap-fill masks for uncovered regions.
        mask_scale_ratio: Ratio for downscaling masks (0.0-1.0, where 1.0 = no scaling).
        max_masks_per_level: Maximum number of masks to keep per level during progressive refinement.
        cascade_filtering: Enable parent-aware cascade filtering (default: True).
        stability_threshold: Minimum stability score for cascade filter (default: None).
        verbose: If True, show Semantic-SAM output; if False, suppress stdout/stderr during inference.

    Yields:
        Tuple[int, str, Dict[int, List[Dict[str, Any]]]]: (frame_index, frame_name, {level: [candidate_dicts]})

    Each candidate dict contains: {'frame_idx', 'frame_name', 'bbox'(XYWH), 'area', 'stability_score'(1.0),
    'level', 'segmentation', 'mask_scale_ratio'}.
    """
    # _semantic_sam_workdir() chdirs into the Semantic-SAM repo; anchor paths first.
    sam_ckpt_path = os.path.abspath(os.path.expanduser(os.fspath(sam_ckpt_path)))
    frames_dir = os.path.abspath(os.path.expanduser(os.fspath(frames_dir)))
    with _semantic_sam_workdir():
        semantic_sam = build_semantic_sam(model_type="L", ckpt=sam_ckpt_path)

    try:
        mask_scale_ratio = float(mask_scale_ratio)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid mask_scale_ratio=%r; defaulting to 1.0", mask_scale_ratio)
        mask_scale_ratio = 1.0
    if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
        raise ValueError("mask_scale_ratio must be within (0, 1]")

    if fill_area is None:
        fill_area = min_area
    fill_area = int(max(0, fill_area))

    # When persist_outputs is False we rely on temporary directories so no artifacts remain on disk.
    if save_root is not None and persist_outputs:
        save_root = ensure_dir(save_root)

    verbose = os.environ.get("MY3DIS_VERBOSE_PROGRESSIVE") == "1"

    base_level = min(levels) if levels else None

    # Storage for collecting all results (for relation saving)
    all_results: List[Dict[str, Any]] = []

    for f_idx, fname in enumerate(selected_frames):
        image_path = os.path.join(frames_dir, fname)

        def run_progressive(output_dirs: Dict[str, str]):
            def _invoke_progressive():
                with _semantic_sam_workdir():
                    return progressive_refinement_masks(
                        semantic_sam,
                        image_path,
                        level_sequence=levels,
                        output_dirs=output_dirs,
                        ssam_frame_idx=f_idx,  # Pass current frame index
                        min_area=min_area,
                        max_masks_per_level=max_masks_per_level,
                        cascade_filtering=cascade_filtering,
                        stability_threshold=stability_threshold,
                        save_viz=False,
                    )

            if verbose:
                return _invoke_progressive()

            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    return _invoke_progressive()
            except Exception:
                # Preserve captured logs to aid debugging before re-raising
                print(buf_out.getvalue(), file=sys.stderr, end="")
                print(buf_err.getvalue(), file=sys.stderr, end="")
                raise

        if persist_outputs and save_root is not None:
            image_out = ensure_dir(os.path.join(save_root, f"pr_{os.path.splitext(fname)[0]}"))
            out_dirs = setup_output_directories(image_out)
            results = run_progressive(out_dirs)
        else:
            tmp_kwargs = {}
            if save_root is not None:
                ensure_dir(save_root)
                tmp_kwargs["dir"] = save_root
            with tempfile.TemporaryDirectory(**tmp_kwargs) as tmp_root:
                out_dirs = setup_output_directories(tmp_root)
                results = run_progressive(out_dirs)

        additional_gap_masks: List[Dict[str, Any]] = []
        if enable_gap_fill and base_level is not None:
            base_masks = results['levels'].get(base_level, {}).get('masks', [])
            base_segs = [m.get('segmentation') for m in base_masks if m.get('segmentation') is not None]
            gap_components = _extract_gap_components(base_segs, fill_area)
            if gap_components:
                gap_seq = 1  # Gap-fill sequence counter for this frame
                for comp in gap_components:
                    # Generate globally unique ID for gap-fill masks (format: {frame:04d}_{level}_gap{seq:04d})
                    gap_unique_id = f"{f_idx:04d}_{base_level}_gap{gap_seq:04d}"
                    gap_seq += 1

                    additional_gap_masks.append({
                        'segmentation': pack_binary_mask(comp, full_resolution_shape=comp.shape),
                        'stability_score': 1.0,
                        'area': int(comp.sum()),
                        'level': base_level,
                        'source': 'gap_fill',
                        # BUGFIX (2025-11-13): Add provenance fields for gap-fill masks
                        # This enables provenance tracking in SAM2 stage
                        'unique_id': gap_unique_id,
                        'parent_unique_id': None,  # Gap-fill masks have no parent
                        'ssam_frame_idx': f_idx,
                        'lineage': [],  # Empty lineage for gap-fill masks
                    })
                LOGGER.info(
                    "Frame %s: added %d gap-fill masks at level %s",
                    fname,
                    len(additional_gap_masks),
                    base_level,
                )

        frame_payload: Dict[int, List[Dict[str, Any]]] = {}
        for L in levels:
            masks = list(results['levels'].get(L, {}).get('masks', []))
            if L == base_level and additional_gap_masks:
                masks.extend(additional_gap_masks)
            # Build candidate list for this frame
            frame_cands: List[Dict[str, Any]] = []
            for m in masks:
                seg = m.get('segmentation')
                if seg is None:
                    continue
                seg_bool = unpack_binary_mask(seg)
                orig_shape = seg_bool.shape
                area = int(m.get('area', int(np.sum(seg_bool))))
                if area == 0:
                    continue
                seg_scaled = seg_bool
                applied_ratio = mask_scale_ratio
                if mask_scale_ratio < 1.0:
                    seg_scaled = downscale_binary_mask(seg_bool, mask_scale_ratio)
                    if not seg_scaled.any() and seg_bool.any():
                        LOGGER.debug(
                            "Mask collapsed after downscale (ratio %.3f); keeping full resolution",
                            mask_scale_ratio,
                        )
                        seg_scaled = seg_bool
                        applied_ratio = 1.0
                x1, y1, x2, y2 = bbox_from_mask_xyxy(seg_scaled)
                bbox = bbox_xyxy_to_xywh((x1, y1, x2, y2))
                stability = float(m.get('stability_score', 1.0))
                # Preserve all metadata from progressive_refinement (including provenance)
                cand = dict(m)  # Copy all fields
                # Override/add computed fields
                cand.update({
                    'frame_idx': f_idx,
                    'frame_name': fname,
                    'bbox': bbox,
                    'area': area,
                    'stability_score': stability,
                    'level': L,
                    'mask_scale_ratio': applied_ratio,
                    'segmentation': pack_binary_mask(
                        seg_scaled,
                        full_resolution_shape=orig_shape,
                    ),
                })
                frame_cands.append(cand)
            frame_payload[L] = frame_cands

        # Store results for relation saving
        if save_relations and save_root is not None:
            all_results.append({'results': results, 'frame_idx': f_idx, 'frame_name': fname})

        yield f_idx, fname, frame_payload

    # After all frames processed, save relations if requested
    if save_relations and save_root is not None and all_results:
        _save_progressive_relations(all_results, save_root, levels)
