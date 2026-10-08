#!/usr/bin/env python3
"""
Analyze GT proposal 3D-to-2D projection quality (no visualization, stats only).

Investigates why some GT proposals have low SigLIP similarity.
"""

import argparse
import logging
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GTInstance:
    """Metadata for a single GT instance mask."""
    raw_label: int
    label_id: int
    instance_id: int
    point_count: int
    text: str


def load_joint_label_map() -> Dict[int, str]:
    """Load Search3D joint labels (object + part) to text."""
    constants_path = (
        REPO_ROOT / "eval" / "search3d" / "data" / "multiscan_search3d_constants.py"
    )
    if not constants_path.exists():
        # Fallback to old location
        constants_path = (
            REPO_ROOT / "data" / "search3d_gt" / "multiscan_data_search3d" / "multiscan_search3d_constants.py"
        )

    if not constants_path.exists():
        raise FileNotFoundError(f"Cannot find constants file at {constants_path}")

    constants = runpy.run_path(str(constants_path))
    names = constants["VALID_JOINT_TUPLE_NAMES"]
    label_ids = constants["VALID_JOINT_SEMANTIC_IDS"]

    label_map = {}
    for (obj_text, part_text), label_id in zip(names, label_ids):
        text = f"{obj_text} {part_text}".strip()
        label_map[int(label_id)] = text

    return label_map


def load_gt_annotations(
    annotation_file: Path,
    scene_path: Path,
    limit: Optional[int] = None,
    min_points: int = 150,
) -> Tuple[List[GTInstance], List[np.ndarray], Dict[int, str]]:
    """
    Load GT annotations and convert to proposal format.

    Returns:
        instances: List of GT instances
        proposal_masks: List of boolean masks (one per instance)
        label_text_map: Mapping from label_id to text
    """
    # Load label map
    label_text_map = load_joint_label_map()

    # Load annotations
    labels = np.loadtxt(annotation_file, dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)

    instances: List[GTInstance] = []
    masks: List[np.ndarray] = []

    # Sort by point count descending
    sorted_indices = np.argsort(-counts)

    for idx in sorted_indices:
        raw_label = int(unique_labels[idx])
        if raw_label <= 0:
            continue

        point_count = int(counts[idx])
        if point_count < min_points:
            continue

        composite_label = raw_label // 1000
        instance_id = raw_label % 1000

        text = label_text_map.get(composite_label, "UNKNOWN")
        if text == "UNKNOWN":
            continue

        mask = (labels == raw_label)

        instances.append(
            GTInstance(
                raw_label=raw_label,
                label_id=int(composite_label),
                instance_id=int(instance_id),
                point_count=point_count,
                text=text,
            )
        )
        masks.append(mask)

        if limit is not None and len(instances) >= limit:
            break

    if not masks:
        raise ValueError("No GT instances matched the filtering criteria.")

    return instances, masks, label_text_map


def prepare_world2cam(
    scene_path: Path,
    experiment_root: Path,
) -> AggregationPipeline:
    """Initialize WORLD_2_CAM through AggregationPipeline."""
    experiment_root.mkdir(parents=True, exist_ok=True)

    config = {
        "uni3dis": {"vis_depth_threshold": 0.03, "frequency": 1},
        "network2d": {"masklet": {"resize_ratio": 0.3}},
    }

    pipeline = AggregationPipeline(
        exp_dir=str(experiment_root),
        scene_path=str(scene_path),
        device="cuda",
        config=config,
        iou_threshold=0.8,
        area_fraction=1 / 500,
        pooling_mode="average",
    )
    pipeline._init_world2cam()  # pylint: disable=protected-access
    return pipeline


def analyze_projection_quality(
    pipeline: AggregationPipeline,
    proposal_mask: np.ndarray,
    instance: GTInstance,
) -> Dict:
    """
    Analyze projection quality for a single GT proposal.

    Returns statistics about visibility across frames.
    """
    mask_indices = np.where(proposal_mask)[0]
    num_points = len(mask_indices)

    # Check visibility in each frame
    frame_visibility = {}
    visible_points_set = set()

    # Also track projection coordinates for bbox analysis
    frame_bboxes = {}

    for frame_idx in range(len(pipeline.world2cam.color_paths)):
        # Get points visible in this frame
        frame_mask = pipeline.inside_mask[frame_idx]  # [num_points]

        # Count proposal points visible in this frame
        visible_in_frame_mask = proposal_mask & frame_mask
        visible_count = np.sum(visible_in_frame_mask)

        if visible_count > 0:
            frame_visibility[frame_idx] = int(visible_count)
            visible_points_set.update(mask_indices[frame_mask[mask_indices]])

            # Get 2D projected coordinates for bbox
            projected_2d = pipeline.projected_points[:, :, frame_idx]  # [2, num_points]
            visible_points_2d = projected_2d[:, visible_in_frame_mask]

            if visible_points_2d.shape[1] > 0:
                x_coords = visible_points_2d[0, :]
                y_coords = visible_points_2d[1, :]

                bbox_area = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
                frame_bboxes[frame_idx] = {
                    'area': float(bbox_area),
                    'width': float(x_coords.max() - x_coords.min()),
                    'height': float(y_coords.max() - y_coords.min()),
                }

    num_visible_frames = len(frame_visibility)
    visibility_counts = list(frame_visibility.values())
    avg_points = np.mean(visibility_counts) if visibility_counts else 0
    median_points = np.median(visibility_counts) if visibility_counts else 0
    max_points = max(visibility_counts) if visibility_counts else 0
    min_points = min(visibility_counts) if visibility_counts else 0
    coverage_ratio = len(visible_points_set) / num_points if num_points > 0 else 0

    bbox_areas = [b['area'] for b in frame_bboxes.values()]
    avg_bbox_area = np.mean(bbox_areas) if bbox_areas else 0
    median_bbox_area = np.median(bbox_areas) if bbox_areas else 0

    return {
        'instance_id': instance.instance_id,
        'label_id': instance.label_id,
        'text': instance.text,
        'num_points': num_points,
        'num_visible_frames': num_visible_frames,
        'frame_visibility': frame_visibility,
        'avg_points_per_frame': avg_points,
        'median_points_per_frame': median_points,
        'max_points_per_frame': max_points,
        'min_points_per_frame': min_points,
        'coverage_ratio': coverage_ratio,
        'visible_points_count': len(visible_points_set),
        'avg_bbox_area': avg_bbox_area,
        'median_bbox_area': median_bbox_area,
        'top_5_frames': sorted(frame_visibility.items(), key=lambda x: x[1], reverse=True)[:5],
    }


def print_detailed_analysis(analyses: List[Dict], output_path: Path):
    """Generate detailed analysis report."""

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("GT PROPOSAL 3D-TO-2D PROJECTION ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        # Summary table
        f.write(f"{'Text':<25} {'ID':<8} {'Points':<8} {'Frames':<8} {'Avg':<10} {'Med':<10} {'Max':<10} {'Coverage':<10} {'BBox Area':<12}\n")
        f.write("-" * 115 + "\n")

        analyses_sorted = sorted(analyses, key=lambda x: x['avg_points_per_frame'], reverse=True)

        for a in analyses_sorted:
            f.write(f"{a['text']:<25} "
                   f"{a['instance_id']:<8} "
                   f"{a['num_points']:<8} "
                   f"{a['num_visible_frames']:<8} "
                   f"{a['avg_points_per_frame']:<10.1f} "
                   f"{a['median_points_per_frame']:<10.1f} "
                   f"{a['max_points_per_frame']:<10} "
                   f"{a['coverage_ratio']:<10.2%} "
                   f"{a['avg_bbox_area']:<12.0f}\n")

        # Detailed per-instance analysis
        f.write("\n" + "=" * 100 + "\n")
        f.write("DETAILED PER-INSTANCE ANALYSIS\n")
        f.write("=" * 100 + "\n")

        for a in analyses_sorted:
            f.write(f"\n{a['text']} (ID: {a['instance_id']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Points:          {a['num_points']}\n")
            f.write(f"  Visible Frames:        {a['num_visible_frames']}\n")
            f.write(f"  Avg Points/Frame:      {a['avg_points_per_frame']:.1f}\n")
            f.write(f"  Median Points/Frame:   {a['median_points_per_frame']:.1f}\n")
            f.write(f"  Max Points/Frame:      {a['max_points_per_frame']}\n")
            f.write(f"  Min Points/Frame:      {a['min_points_per_frame']}\n")
            f.write(f"  Coverage Ratio:        {a['coverage_ratio']:.2%}\n")
            f.write(f"  Visible Points:        {a['visible_points_count']} / {a['num_points']}\n")
            f.write(f"  Avg BBox Area:         {a['avg_bbox_area']:.0f} px²\n")
            f.write(f"  Median BBox Area:      {a['median_bbox_area']:.0f} px²\n")

            f.write(f"\n  Top 5 Frames by Visibility:\n")
            for rank, (frame_idx, count) in enumerate(a['top_5_frames'], 1):
                f.write(f"    {rank}. Frame {frame_idx:04d}: {count} points\n")
            f.write("\n")

        # Key findings
        f.write("=" * 100 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 100 + "\n\n")

        # Compare high vs low performers
        high_perf = [a for a in analyses if a['avg_points_per_frame'] > 100]
        low_perf = [a for a in analyses if a['avg_points_per_frame'] < 50]

        f.write(f"HIGH PERFORMERS (>100 avg points/frame): {len(high_perf)}\n")
        for a in high_perf:
            f.write(f"  - {a['text']}: {a['avg_points_per_frame']:.1f} avg, "
                   f"{a['num_visible_frames']} frames, {a['coverage_ratio']:.1%} coverage\n")

        f.write(f"\nLOW PERFORMERS (<50 avg points/frame): {len(low_perf)}\n")
        for a in low_perf:
            f.write(f"  - {a['text']}: {a['avg_points_per_frame']:.1f} avg, "
                   f"{a['num_visible_frames']} frames, {a['coverage_ratio']:.1%} coverage\n")

        # Statistical comparison
        if high_perf and low_perf:
            f.write(f"\nSTATISTICAL COMPARISON:\n")
            f.write(f"  High performers - Avg BBox Area: {np.mean([a['avg_bbox_area'] for a in high_perf]):.0f} px²\n")
            f.write(f"  Low performers  - Avg BBox Area: {np.mean([a['avg_bbox_area'] for a in low_perf]):.0f} px²\n")
            f.write(f"  High performers - Avg Frames:    {np.mean([a['num_visible_frames'] for a in high_perf]):.0f}\n")
            f.write(f"  Low performers  - Avg Frames:    {np.mean([a['num_visible_frames'] for a in low_perf]):.0f}\n")

    logger.info(f"Saved detailed analysis to {output_path}")

    # Also print summary to console
    print("\n" + "=" * 100)
    print("PROJECTION QUALITY SUMMARY")
    print("=" * 100)
    print(f"\n{'Text':<25} {'Avg Pts/Frame':<15} {'Frames':<10} {'Coverage':<12} {'BBox Area':<12}")
    print("-" * 80)
    for a in analyses_sorted:
        print(f"{a['text']:<25} {a['avg_points_per_frame']:<15.1f} "
              f"{a['num_visible_frames']:<10} {a['coverage_ratio']:<12.2%} "
              f"{a['avg_bbox_area']:<12.0f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GT proposal 3D-to-2D projection quality")
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=Path("/tmp/gt_projection_analysis.txt"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-points", type=int, default=150)
    args = parser.parse_args()

    # Load GT annotations
    logger.info(f"Loading GT annotations from {args.annotation_file}")
    instances, proposal_masks, label_text_map = load_gt_annotations(
        annotation_file=args.annotation_file,
        scene_path=args.scene_path,
        limit=args.limit,
        min_points=args.min_points,
    )
    logger.info(f"Loaded {len(instances)} GT instances")

    # Prepare WORLD_2_CAM
    logger.info("Preparing WORLD_2_CAM projection...")
    pipeline = prepare_world2cam(
        scene_path=args.scene_path,
        experiment_root=Path("./outputs/gt_projection_analysis"),
    )
    logger.info(f"Loaded projection for {len(pipeline.world2cam.color_paths)} frames")

    # Analyze all instances
    logger.info(f"\nAnalyzing {len(instances)} instances...")
    analyses = []
    for instance, proposal_mask in zip(instances, proposal_masks):
        analysis = analyze_projection_quality(pipeline, proposal_mask, instance)
        analyses.append(analysis)

        logger.info(f"  {instance.text}: {analysis['avg_points_per_frame']:.1f} avg pts/frame, "
                   f"{analysis['num_visible_frames']} frames")

    # Generate report
    print_detailed_analysis(analyses, args.output_file)

    logger.info(f"\n✓ Analysis complete! Report saved to {args.output_file}")


if __name__ == "__main__":
    main()
