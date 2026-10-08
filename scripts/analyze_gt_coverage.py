#!/usr/bin/env python3
"""
Analyze GT coverage statistics across all Search3D scenes.
Helps identify scenes with dense, complete annotations for testing.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

def analyze_scene_gt(gt_file, ply_file):
    """Analyze GT coverage for a single scene."""
    # Read GT labels
    gt_labels = np.loadtxt(gt_file, dtype=np.int32)

    # Read PLY to get total points (read as binary, decode header only)
    num_points = None
    with open(ply_file, 'rb') as f:
        for line in f:
            try:
                line_str = line.decode('ascii').strip()
                if line_str.startswith('element vertex'):
                    num_points = int(line_str.split()[-1])
                if line_str == 'end_header':
                    break
            except UnicodeDecodeError:
                continue

    if num_points is None:
        return None

    # Calculate statistics
    labeled_points = (gt_labels > 0).sum()
    coverage_pct = 100.0 * labeled_points / num_points

    # Get unique labels
    unique_labels = set()
    for label in gt_labels[gt_labels > 0]:
        semantic_id = label // 1000
        unique_labels.add(semantic_id)

    num_labels = len(unique_labels)

    # Count label distribution
    label_counts = defaultdict(int)
    for label in gt_labels[gt_labels > 0]:
        semantic_id = label // 1000
        label_counts[semantic_id] += 1

    # Find largest label
    if label_counts:
        largest_label = max(label_counts.items(), key=lambda x: x[1])
        largest_label_pct = 100.0 * largest_label[1] / labeled_points
    else:
        largest_label = (None, 0)
        largest_label_pct = 0.0

    return {
        'total_points': num_points,
        'labeled_points': labeled_points,
        'coverage_pct': coverage_pct,
        'num_labels': num_labels,
        'largest_label': largest_label[0],
        'largest_label_points': largest_label[1],
        'largest_label_pct': largest_label_pct
    }

def main():
    gt_dir = Path('./data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations')
    ply_dir = Path('./data/search3d_gt/multiscan_data_search3d/multiscan_test_plys_only')

    scenes = []

    # Analyze all scenes
    for gt_file in sorted(gt_dir.glob('*.txt')):
        scene_name = gt_file.stem.replace('_obj_part_inst', '')
        ply_file = ply_dir / f"{scene_name}.ply"

        if not ply_file.exists():
            print(f"Warning: PLY file not found for {scene_name}", file=sys.stderr)
            continue

        stats = analyze_scene_gt(gt_file, ply_file)
        if stats is None:
            continue

        scenes.append({
            'scene': scene_name,
            **stats
        })

    # Sort by coverage (descending)
    scenes.sort(key=lambda x: x['coverage_pct'], reverse=True)

    # Print results
    print(f"{'Scene':<20} {'Total Pts':>10} {'Labeled':>10} {'Coverage':>10} {'Labels':>8} {'Largest':>10} {'Largest%':>10}")
    print("=" * 100)

    for scene in scenes:
        print(f"{scene['scene']:<20} "
              f"{scene['total_points']:>10,} "
              f"{scene['labeled_points']:>10,} "
              f"{scene['coverage_pct']:>9.1f}% "
              f"{scene['num_labels']:>8} "
              f"{scene['largest_label'] or 'N/A':>10} "
              f"{scene['largest_label_pct']:>9.1f}%")

    print("\n" + "=" * 100)
    print("Recommendations for testing:")
    print("-" * 100)

    # Find good candidates (>30% coverage, >10 labels)
    good_scenes = [s for s in scenes if s['coverage_pct'] > 30 and s['num_labels'] > 10]

    if good_scenes:
        print("\nScenes with >30% coverage and >10 labels (BEST for testing):")
        for scene in good_scenes[:5]:
            print(f"  - {scene['scene']}: {scene['coverage_pct']:.1f}% coverage, {scene['num_labels']} labels")
    else:
        print("\nNo scenes with >30% coverage and >10 labels found.")
        print("Showing top 5 scenes by coverage:")
        for scene in scenes[:5]:
            print(f"  - {scene['scene']}: {scene['coverage_pct']:.1f}% coverage, {scene['num_labels']} labels")

    # Show current scene stats
    current_scene = [s for s in scenes if s['scene'] == 'scene_00005_01']
    if current_scene:
        print(f"\nCurrent scene (scene_00005_01):")
        s = current_scene[0]
        print(f"  Coverage: {s['coverage_pct']:.1f}%")
        print(f"  Labels: {s['num_labels']}")
        print(f"  Rank: {scenes.index(s) + 1}/{len(scenes)}")

if __name__ == '__main__':
    main()
