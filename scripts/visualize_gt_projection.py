#!/usr/bin/env python3
"""
Visualize GT proposal 3D-to-2D projections to debug feature extraction.

This script investigates why some GT proposals (e.g., cabinet door) have very low
SigLIP similarity while others (e.g., door frame) succeed.

Key questions:
1. Which RGB frames does each GT proposal project to?
2. Does the projected mask cover meaningful visual content?
3. Are there differences in projection quality between successful and failed cases?
4. Is visibility filtering working correctly?
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline

# Import from existing sanity check script
sys.path.insert(0, str(Path(__file__).parent))
from siglip_gt_sanity_check_with_scaling import (
    GTInstance,
    load_gt_annotations,
    prepare_world2cam,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_projection_quality(
    pipeline: AggregationPipeline,
    proposal_mask: np.ndarray,
    instance: GTInstance,
) -> Dict:
    """
    Analyze projection quality for a single GT proposal.

    Returns:
        Dictionary with projection statistics:
        - num_points: Total points in proposal
        - num_visible_frames: Number of frames where proposal is visible
        - frame_visibility: Dict[frame_idx, num_visible_points]
        - avg_points_per_frame: Average visible points per frame
        - max_points_per_frame: Maximum visible points in any frame
        - coverage_ratio: Ratio of points visible in at least one frame
    """
    mask_indices = np.where(proposal_mask)[0]
    num_points = len(mask_indices)

    # Check visibility in each frame
    frame_visibility = {}
    visible_points = set()

    for frame_idx, frame_path in enumerate(pipeline.world2cam.color_paths):
        # Get points visible in this frame
        frame_mask = pipeline.inside_mask[frame_idx]  # [num_points]

        # Count proposal points visible in this frame
        visible_in_frame = np.sum(proposal_mask & frame_mask)

        if visible_in_frame > 0:
            frame_visibility[frame_idx] = int(visible_in_frame)
            visible_points.update(mask_indices[frame_mask[mask_indices]])

    num_visible_frames = len(frame_visibility)
    avg_points = np.mean(list(frame_visibility.values())) if frame_visibility else 0
    max_points = max(frame_visibility.values()) if frame_visibility else 0
    coverage_ratio = len(visible_points) / num_points if num_points > 0 else 0

    return {
        'instance_id': instance.instance_id,
        'label_id': instance.label_id,
        'text': instance.text,
        'num_points': num_points,
        'num_visible_frames': num_visible_frames,
        'frame_visibility': frame_visibility,
        'avg_points_per_frame': avg_points,
        'max_points_per_frame': max_points,
        'coverage_ratio': coverage_ratio,
        'visible_points_count': len(visible_points),
    }


def visualize_projection_on_frame(
    frame_path: str,
    projected_points: np.ndarray,  # [2, num_points]
    proposal_mask: np.ndarray,  # [num_points]
    inside_mask: np.ndarray,  # [num_points]
    output_path: Path,
    instance: GTInstance,
) -> None:
    """
    Visualize projected mask on RGB frame.

    Args:
        frame_path: Path to RGB image
        projected_points: Projected 2D coordinates [2, num_points]
        proposal_mask: Boolean mask for this proposal
        inside_mask: Boolean mask for points visible in this frame
        output_path: Where to save visualization
        instance: GT instance info
    """
    # Load image
    img = cv2.imread(frame_path)
    if img is None:
        logger.error(f"Failed to load image: {frame_path}")
        return

    h, w = img.shape[:2]

    # Create overlay
    overlay = img.copy()

    # Get points that belong to this proposal AND are visible in this frame
    valid_points = proposal_mask & inside_mask

    if not np.any(valid_points):
        logger.warning(f"No valid points for {instance.text} in frame {frame_path}")
        return

    # Get 2D coordinates
    x_coords = projected_points[0, valid_points].astype(int)
    y_coords = projected_points[1, valid_points].astype(int)

    # Filter out-of-bounds coordinates
    valid_coords = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    x_coords = x_coords[valid_coords]
    y_coords = y_coords[valid_coords]

    if len(x_coords) == 0:
        logger.warning(f"All points out of bounds for {instance.text}")
        return

    # Draw points
    for x, y in zip(x_coords, y_coords):
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)

    # Draw bounding box
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add text
    label_text = f"{instance.text} (ID: {instance.instance_id})"
    cv2.putText(overlay, label_text, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Blend
    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Add statistics text
    stats_text = f"Points: {len(x_coords)} | Area: {(x_max-x_min)*(y_max-y_min)} px"
    cv2.putText(result, stats_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    cv2.imwrite(str(output_path), result)
    logger.info(f"Saved visualization to {output_path}")


def compare_instances(
    pipeline: AggregationPipeline,
    instances: List[GTInstance],
    proposal_masks: List[np.ndarray],
    output_dir: Path,
    max_frames_per_instance: int = 5,
) -> None:
    """
    Compare projection quality across multiple instances.

    Generates:
    1. Projection statistics for each instance
    2. Visualizations of top-K frames for each instance
    3. Side-by-side comparison of successful vs failed cases
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all instances
    analyses = []
    for instance, proposal_mask in zip(instances, proposal_masks):
        analysis = analyze_projection_quality(pipeline, proposal_mask, instance)
        analyses.append(analysis)

        logger.info(f"\n{instance.text} (ID: {instance.instance_id}):")
        logger.info(f"  Total points: {analysis['num_points']}")
        logger.info(f"  Visible in {analysis['num_visible_frames']} frames")
        logger.info(f"  Avg points/frame: {analysis['avg_points_per_frame']:.1f}")
        logger.info(f"  Max points/frame: {analysis['max_points_per_frame']}")
        logger.info(f"  Coverage ratio: {analysis['coverage_ratio']:.2%}")

    # Generate visualizations for each instance
    for analysis, proposal_mask in zip(analyses, proposal_masks):
        instance_id = analysis['instance_id']
        text = analysis['text']

        # Sort frames by number of visible points
        sorted_frames = sorted(
            analysis['frame_visibility'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_frames_per_instance]

        logger.info(f"\nVisualizing top {len(sorted_frames)} frames for {text}:")

        for rank, (frame_idx, num_points) in enumerate(sorted_frames, 1):
            frame_path = pipeline.world2cam.color_paths[frame_idx]
            projected_points = pipeline.projected_points[:, :, frame_idx]
            inside_mask = pipeline.inside_mask[frame_idx]

            output_path = output_dir / f"{text.replace(' ', '_')}_{instance_id}_frame{frame_idx:04d}_rank{rank}.jpg"

            visualize_projection_on_frame(
                frame_path=frame_path,
                projected_points=projected_points,
                proposal_mask=proposal_mask,
                inside_mask=inside_mask,
                output_path=output_path,
                instance=GTInstance(
                    instance_id=instance_id,
                    label_id=analysis['label_id'],
                    text=text,
                    num_points=num_points,
                ),
            )

    # Generate comparison report
    report_path = output_dir / "projection_analysis.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GT PROPOSAL 3D-TO-2D PROJECTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Sort by coverage ratio
        analyses_sorted = sorted(analyses, key=lambda x: x['coverage_ratio'], reverse=True)

        f.write(f"{'Text':<25} {'ID':<8} {'Points':<8} {'Frames':<8} {'Avg/Frame':<12} {'Max/Frame':<12} {'Coverage':<10}\n")
        f.write("-" * 100 + "\n")

        for analysis in analyses_sorted:
            f.write(f"{analysis['text']:<25} "
                   f"{analysis['instance_id']:<8} "
                   f"{analysis['num_points']:<8} "
                   f"{analysis['num_visible_frames']:<8} "
                   f"{analysis['avg_points_per_frame']:<12.1f} "
                   f"{analysis['max_points_per_frame']:<12} "
                   f"{analysis['coverage_ratio']:<10.2%}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY OBSERVATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Identify problematic cases
        low_coverage = [a for a in analyses if a['coverage_ratio'] < 0.5]
        few_frames = [a for a in analyses if a['num_visible_frames'] < 10]
        low_points = [a for a in analyses if a['avg_points_per_frame'] < 50]

        if low_coverage:
            f.write("Instances with low coverage ratio (<50%):\n")
            for a in low_coverage:
                f.write(f"  - {a['text']} (ID {a['instance_id']}): {a['coverage_ratio']:.2%}\n")
            f.write("\n")

        if few_frames:
            f.write("Instances visible in few frames (<10):\n")
            for a in few_frames:
                f.write(f"  - {a['text']} (ID {a['instance_id']}): {a['num_visible_frames']} frames\n")
            f.write("\n")

        if low_points:
            f.write("Instances with few points per frame (<50):\n")
            for a in low_points:
                f.write(f"  - {a['text']} (ID {a['instance_id']}): {a['avg_points_per_frame']:.1f} points/frame\n")
            f.write("\n")

    logger.info(f"\nSaved projection analysis to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize GT proposal 3D-to-2D projections")
    parser.add_argument("--scene-path", type=Path, required=True,
                       help="Path to scene (e.g., /path/to/scene_00005_00)")
    parser.add_argument("--annotation-file", type=Path, required=True,
                       help="Path to GT annotation file")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/gt_projection_viz"),
                       help="Output directory for visualizations")
    parser.add_argument("--limit", type=int, default=20,
                       help="Limit number of instances to analyze")
    parser.add_argument("--min-points", type=int, default=150,
                       help="Minimum points per instance")
    parser.add_argument("--max-frames-per-instance", type=int, default=5,
                       help="Max frames to visualize per instance")
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

    # Compare instances
    logger.info(f"\nAnalyzing {len(instances)} instances...")
    compare_instances(
        pipeline=pipeline,
        instances=instances,
        proposal_masks=proposal_masks,
        output_dir=args.output_dir,
        max_frames_per_instance=args.max_frames_per_instance,
    )

    logger.info(f"\n✓ Analysis complete! Check {args.output_dir} for results")


if __name__ == "__main__":
    main()
