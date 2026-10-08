#!/usr/bin/env python3
"""
2D→3D Projection Pipeline Script

這個腳本將 pipeline.ipynb 的邏輯轉換為可批量執行的腳本，
自動處理從 My3DIS tracking 輸出到 3D proposals + SigLIP features 的完整流程。

Usage:
    python scripts/run_3d_projection_pipeline.py \
        --exp-dir outputs/experiments/my_exp/scene_00005_00 \
        --scene-path data/multiscan/scene_00005_00 \
        --output-dir /tmp/3d_proposals \
        --levels 2 4 6

Output:
    - {output_dir}/proposal_data_level2.npz
    - {output_dir}/proposal_data_level4.npz
    - {output_dir}/proposal_data_level6.npz
    - {output_dir}/Proposal_relation.json
"""

import sys
from pathlib import Path

# Add 3DprojToSiglip to path
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root / "3DprojToSiglip"))
sys.path.insert(0, str(proj_root / "src"))

import argparse
import json
import logging
import numpy as np
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_inputs(exp_dir: Path, scene_path: Path, levels: list):
    """驗證必要的輸入文件存在"""
    errors = []

    # Check experiment outputs
    for level in levels:
        video_file = exp_dir / f"level_{level}" / f"video_segments_L{level:02d}.npz"
        if not video_file.exists():
            errors.append(f"Missing: {video_file}")

    family_tree = exp_dir / "relations" / "family_tree.json"
    if not family_tree.exists():
        errors.append(f"Missing: {family_tree}")

    # Check scene data
    ply_file = scene_path / f"{scene_path.name}.ply"
    if not ply_file.exists():
        ply_file = scene_path / f"{scene_path.name}_converted.ply"
    if not ply_file.exists():
        errors.append(f"Missing PLY: {scene_path.name}.ply or *_converted.ply")

    depth_dir = scene_path / "outputs" / "depth"
    if not depth_dir.exists():
        errors.append(f"Missing depth directory: {depth_dir}")

    color_dir = scene_path / "outputs" / "color"
    if not color_dir.exists():
        errors.append(f"Missing color directory: {color_dir}")

    intrinsics_file = scene_path / "outputs" / "intrinsics.txt"
    if not intrinsics_file.exists():
        errors.append(f"Missing intrinsics: {intrinsics_file}")

    if errors:
        for err in errors:
            logger.error(err)
        raise FileNotFoundError(f"Missing {len(errors)} required file(s)")

    return ply_file


def main():
    parser = argparse.ArgumentParser(
        description="Run 2D→3D projection pipeline (from pipeline.ipynb)"
    )

    parser.add_argument('--exp-dir', required=True,
                        help='My3DIS experiment directory (scene-specific)')
    parser.add_argument('--scene-path', required=True,
                        help='MultiScan scene directory')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 4, 6],
                        help='Levels to process')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                        help='IoU threshold for merging proposals')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip SigLIP feature extraction (faster, for testing)')

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    scene_path = Path(args.scene_path)
    output_dir = Path(args.output_dir)

    logger.info("="*60)
    logger.info("2D→3D PROJECTION PIPELINE")
    logger.info("="*60)
    logger.info(f"Experiment: {exp_dir}")
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Levels: {args.levels}")

    # Validate inputs
    logger.info("\nValidating inputs...")
    ply_file = validate_inputs(exp_dir, scene_path, args.levels)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import required modules from 3DprojToSiglip
    logger.info("\nImporting 3DprojToSiglip modules...")
    try:
        from utils_load.common_utils import unpack_binary_mask
        from utils_load.compat import load_legacy_video_segments
        from utils_new import WORLD_2_CAM
        from utils_new.utils_2d import get_visible_points

        logger.info("✓ Successfully imported 3DprojToSiglip modules")
    except ImportError as e:
        logger.error(f"Failed to import: {e}")
        logger.error("Make sure 3DprojToSiglip directory is present")
        sys.exit(1)

    # Configuration (from pipeline.ipynb)
    config = {
        "uni3dis": {"vis_depth_threshold": 0.1, "frequency": 50},
        "network2d": {"masklet": {"resize_ratio": 0.3}}
    }

    scaling_params = [1/config["network2d"]["masklet"]["resize_ratio"]] * 2
    vis_depth_threshold = config["uni3dis"]["vis_depth_threshold"]
    frequency = config["uni3dis"]["frequency"]
    depth_scale = 1000.0

    # Initialize WORLD_2_CAM
    logger.info("\nInitializing WORLD_2_CAM for 3D projection...")
    device_str = args.device if torch.cuda.is_available() else "cpu"

    world2cam = WORLD_2_CAM(
        dataset_type="multiscan",
        path_2_scene=str(scene_path),
        path_2_ply=str(ply_file),
        path_2_sp="",  # Not used
        path_2_data=str(output_dir / "temp"),
        depth_scale=depth_scale,
        uni3dis_config=config,
        device=device_str
    )

    # Get mesh projections
    logger.info("Computing mesh projections to 2D...")
    projected_points, inside_mask, point_depth, depth_maps = world2cam.get_mesh_projections()

    F, P = inside_mask.shape
    logger.info(f"  Frames: {F}, 3D Points: {P}")

    color_paths = world2cam.color_paths

    # Get visible points for all frames
    logger.info("Computing visible points per frame...")
    point_masks_all_true = torch.ones(P, dtype=torch.bool)

    pts_coords_per_frame, points_inside_mask_num, pts_idx_per_frame = get_visible_points(
        color_paths=color_paths,
        inside_mask=inside_mask,
        projected_points=projected_points,
        point_depth=point_depth,
        point_masks=point_masks_all_true,
        depth_maps=depth_maps,
        device=torch.device('cpu'),
        scaling_params=scaling_params,
        vis_depth_threshold=vis_depth_threshold
    )

    del point_masks_all_true

    # Load family tree for relation tracking
    logger.info("\nLoading family tree...")
    family_tree_path = exp_dir / "relations" / "family_tree.json"
    with open(family_tree_path, 'r') as f:
        relations = json.load(f)

    logger.info(f"✓ Loaded family tree from {family_tree_path}")

    # Process each level
    logger.info("\n" + "="*60)
    logger.info("PROCESSING LEVELS")
    logger.info("="*60)

    # Import helper functions (from notebook cells)
    # TODO: 這裡需要將 notebook 的 helper functions 複製過來
    # 包括：
    # - build_proposal_masks_from_masklets
    # - merge_drop_level*_mask functions
    # - generate_grounding_features_siglip (如果需要特徵提取)

    logger.warning("\n⚠️  WARNING: Core processing functions not yet implemented!")
    logger.warning("This script is a template. You need to:")
    logger.warning("1. Copy helper functions from pipeline.ipynb cells")
    logger.warning("2. Implement build_proposal_masks_from_masklets")
    logger.warning("3. Implement merge_drop_level* functions")
    logger.warning("4. Optionally implement SigLIP feature extraction")
    logger.warning("\nRefer to pipeline.ipynb cells for the full implementation.")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. 將 pipeline.ipynb 的以下 cells 的函數複製到此腳本：")
    print("   - build_proposal_masks_from_masklets (cell id: 82ce062b)")
    print("   - merge_drop_level2_mask (cell id: cfe3a16a)")
    print("   - merge_drop_level4_within_siblings (cell id: 4bc6682c)")
    print("   - merge_drop_level6_within_siblings (cell id: ee66a822)")
    print("   - pool_point_to_proposal_features (cell id: 7c698705)")
    print("   - generate_grounding_features_siglip (cell id: e7f5c56e)")
    print("\n2. 或者，直接使用現有的 .npz 文件配合 eval_from_npz.py")
    print("="*60)


if __name__ == '__main__':
    main()
