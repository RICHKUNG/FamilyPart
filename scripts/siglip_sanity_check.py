"""
SigLIP Feature Sanity Check

This script performs a sanity check on SigLIP features by:
1. Loading GT annotations and converting them to 3D proposals
2. Extracting SigLIP visual features from GT proposals (via RGB frames)
3. Extracting SigLIP text features from GT labels (with scale=300.0 and softmax)
4. Comparing visual and text feature similarity

The expectation is that GT visual features should be highly similar to their
corresponding text features.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my3dis.inference.feature_extraction import SigLIPFeatureExtractor
from my3dis.inference.retrieval import compute_cosine_similarity

# Import Search3D constants
try:
    from eval.search3d.data.multiscan_search3d_constants import (
        VALID_JOINT_SEMANTIC_IDS,
        VALID_JOINT_LABEL_LIST,
    )
except ImportError:
    # Try alternative path
    try:
        import runpy
        constants_path = Path(__file__).parent.parent / "eval" / "search3d" / "data" / "multiscan_search3d_constants.py"
        if not constants_path.exists():
            constants_path = Path(__file__).parent.parent / "data" / "search3d_gt" / "multiscan_data_search3d" / "multiscan_search3d_constants.py"
        constants = runpy.run_path(str(constants_path))
        VALID_JOINT_SEMANTIC_IDS = constants["VALID_JOINT_SEMANTIC_IDS"]
        VALID_JOINT_LABEL_LIST = constants["VALID_JOINT_LABEL_LIST"]
    except Exception as e:
        print(f"Warning: Could not import Search3D constants: {e}")
        VALID_JOINT_SEMANTIC_IDS = []
        VALID_JOINT_LABEL_LIST = []

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTProposal:
    """Represents a GT proposal (instance)."""
    def __init__(
        self,
        instance_id: int,
        semantic_id: int,
        label: str,
        mask_3d: np.ndarray,
        point_indices: np.ndarray
    ):
        self.instance_id = instance_id
        self.semantic_id = semantic_id
        self.label = label
        self.mask_3d = mask_3d  # Binary mask [N_points]
        self.point_indices = point_indices  # Indices of points in this instance


def load_gt_annotations(
    annotation_path: Path,
    ply_path: Path
) -> Tuple[np.ndarray, Dict[int, GTProposal]]:
    """
    Load GT annotations and convert to proposals.

    Args:
        annotation_path: Path to annotation .txt file
        ply_path: Path to point cloud .ply file

    Returns:
        Tuple of (point_cloud, proposals_dict)
        - point_cloud: [N_points, 3] array of xyz coordinates
        - proposals_dict: {semantic_id: GTProposal}
    """
    logger.info(f"Loading GT annotations from {annotation_path}")

    # Load point cloud
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        logger.info(f"Loaded point cloud with {len(points)} points")
    except ImportError:
        logger.error("plyfile library required. Install with: pip install plyfile")
        raise

    # Load annotations (one label per point)
    labels = []
    with open(annotation_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                continue
            labels.append(int(stripped))

    labels = np.array(labels, dtype=np.int64)
    logger.info(f"Loaded {len(labels)} labels")

    if len(labels) != len(points):
        logger.error(f"Mismatch: {len(points)} points but {len(labels)} labels")
        raise ValueError("Point cloud and labels size mismatch")

    # Extract unique instances
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]  # Skip background (0)

    logger.info(f"Found {len(unique_ids)} unique instances")

    # Create semantic_id to label mapping
    id_to_label = {}
    for sem_id, label in zip(VALID_JOINT_SEMANTIC_IDS, VALID_JOINT_LABEL_LIST):
        id_to_label[sem_id] = label

    # Convert to proposals
    proposals = {}
    for unique_id in unique_ids:
        # Decode: semantic_id * 1000 + instance_number
        semantic_id = unique_id // 1000
        instance_num = unique_id % 1000

        # Skip if not in valid set
        if semantic_id not in id_to_label:
            logger.warning(f"Skipping unknown semantic_id: {semantic_id}")
            continue

        # Get mask
        mask_3d = (labels == unique_id)
        point_indices = np.where(mask_3d)[0]

        label = id_to_label[semantic_id]

        proposal = GTProposal(
            instance_id=unique_id,
            semantic_id=semantic_id,
            label=label,
            mask_3d=mask_3d,
            point_indices=point_indices
        )

        proposals[unique_id] = proposal
        logger.debug(f"Instance {unique_id}: {label} ({len(point_indices)} points)")

    logger.info(f"Created {len(proposals)} GT proposals")

    return points, proposals


def extract_visual_features_from_gt(
    proposals: Dict[int, GTProposal],
    points: np.ndarray,
    scene_path: Path,
    extractor: SigLIPFeatureExtractor,
    max_proposals: int = None,
    max_frames_per_proposal: int = 5,
    use_projection: bool = True
) -> Dict[int, torch.Tensor]:
    """
    Extract SigLIP visual features for GT proposals.

    Args:
        proposals: Dictionary of GT proposals
        points: Point cloud [N_points, 3]
        scene_path: Path to scene directory
        extractor: SigLIP feature extractor
        max_proposals: Maximum number of proposals to process (for testing)
        max_frames_per_proposal: Maximum frames to use per proposal
        use_projection: Whether to use 3D-to-2D projection (if False, use random features)

    Returns:
        Dictionary mapping instance_id to feature tensor
    """
    if not use_projection:
        logger.warning("Using PLACEHOLDER random features (use_projection=False)")
        visual_features = {}
        proposal_list = list(proposals.items())
        if max_proposals:
            proposal_list = proposal_list[:max_proposals]

        for instance_id, proposal in tqdm(proposal_list, desc="Extracting visual features"):
            feature = torch.randn(extractor.feature_dim)
            feature = feature / feature.norm()
            visual_features[instance_id] = feature

        logger.info(f"Extracted random features for {len(visual_features)} proposals")
        return visual_features

    # Use 3D-to-2D projection for real visual features
    from my3dis.aggregation.mask_projection_3d_to_2d import Mask3Dto2DProjector

    # Check if scene has required files
    outputs_dir = scene_path / "outputs"
    intrinsics_path = outputs_dir / "intrinsics.txt"
    pose_dir = outputs_dir / "pose"
    depth_dir = outputs_dir / "depth"
    color_dir = outputs_dir / "color"

    if not all([intrinsics_path.exists(), pose_dir.exists(), color_dir.exists()]):
        logger.error(f"Missing required files in {outputs_dir}")
        logger.error("Falling back to random features")
        return extract_visual_features_from_gt(
            proposals, points, scene_path, extractor, max_proposals,
            max_frames_per_proposal, use_projection=False
        )

    # Initialize projector
    logger.info("Initializing 3D-to-2D projector...")
    projector = Mask3Dto2DProjector(
        intrinsics_path=str(intrinsics_path),
        pose_dir=str(pose_dir),
        depth_dir=str(depth_dir) if depth_dir.exists() else None
    )

    # Get available frame indices
    color_files = sorted(color_dir.glob("*.png"))
    if len(color_files) == 0:
        color_files = sorted(color_dir.glob("*.jpg"))

    frame_indices = [int(f.stem) for f in color_files]

    if len(frame_indices) == 0:
        logger.error(f"No RGB frames found in {color_dir}")
        logger.error("Falling back to random features")
        return extract_visual_features_from_gt(
            proposals, points, scene_path, extractor, max_proposals,
            max_frames_per_proposal, use_projection=False
        )

    logger.info(f"Found {len(frame_indices)} RGB frames")

    # Determine image size from first frame
    sample_img = Image.open(color_files[0])
    image_size = (sample_img.height, sample_img.width)
    logger.info(f"Image size: {image_size}")

    visual_features = {}

    proposal_list = list(proposals.items())
    if max_proposals:
        proposal_list = proposal_list[:max_proposals]

    for instance_id, proposal in tqdm(proposal_list, desc="Extracting visual features"):
        # Find frames where this proposal is visible
        visible_frames = projector.get_visible_frames(
            points, proposal.mask_3d, frame_indices, image_size,
            min_visible_points=50
        )

        if len(visible_frames) == 0:
            logger.warning(f"No visible frames for instance {instance_id}, using random feature")
            feature = torch.randn(extractor.feature_dim)
            feature = feature / feature.norm()
            visual_features[instance_id] = feature
            continue

        # Sample frames
        if len(visible_frames) > max_frames_per_proposal:
            step = len(visible_frames) // max_frames_per_proposal
            visible_frames = visible_frames[::step][:max_frames_per_proposal]

        logger.debug(f"Instance {instance_id}: {len(visible_frames)} visible frames")

        # Extract features from each frame
        frame_features = []

        for frame_idx in visible_frames:
            # Project to 2D
            mask_2d = projector.project_3d_mask_to_2d(
                points, proposal.mask_3d, frame_idx, image_size, check_visibility=True
            )

            if mask_2d is None or mask_2d.sum() < 10:
                continue

            # Load RGB image
            rgb_path = color_dir / f"{frame_idx}.png"
            if not rgb_path.exists():
                rgb_path = color_dir / f"{frame_idx}.jpg"
            if not rgb_path.exists():
                continue

            try:
                image = Image.open(rgb_path).convert("RGB")

                # Get bounding box from 2D mask
                y_coords, x_coords = np.where(mask_2d)
                if len(y_coords) == 0:
                    continue

                x1, y1 = x_coords.min(), y_coords.min()
                x2, y2 = x_coords.max(), y_coords.max()

                # Expand bbox slightly
                margin = 20
                H, W = image.height, image.width
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(W, x2 + margin)
                y2 = min(H, y2 + margin)

                # Crop and extract features
                crop = image.crop((x1, y1, x2, y2))
                inputs = extractor.processor(images=crop, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(extractor.device)

                with torch.no_grad():
                    features = extractor.model.get_image_features(pixel_values=pixel_values)
                    features = features / features.norm(dim=-1, keepdim=True)
                    frame_features.append(features.cpu())

            except Exception as e:
                logger.debug(f"Error processing frame {frame_idx} for instance {instance_id}: {e}")
                continue

        if len(frame_features) == 0:
            logger.warning(f"No valid frames for instance {instance_id}, using random feature")
            feature = torch.randn(extractor.feature_dim)
            feature = feature / feature.norm()
        else:
            # Aggregate features across frames (mean pooling)
            aggregated = torch.stack(frame_features).mean(dim=0)
            feature = aggregated / aggregated.norm(dim=-1, keepdim=True)
            feature = feature.squeeze(0)

        visual_features[instance_id] = feature

    logger.info(f"Extracted visual features for {len(visual_features)} proposals")

    return visual_features


def extract_text_features(
    proposals: Dict[int, GTProposal],
    extractor: SigLIPFeatureExtractor,
    scale_semantic_score: float = 300.0,
    apply_softmax: bool = True
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Extract SigLIP text features from GT labels.

    Args:
        proposals: Dictionary of GT proposals
        extractor: SigLIP feature extractor
        scale_semantic_score: Scaling constant (default 300.0)
        apply_softmax: Whether to apply softmax (default True)

    Returns:
        Tuple of (raw_features, processed_features)
        - raw_features: {instance_id: raw_text_feature}
        - processed_features: {instance_id: scaled_and_softmaxed_feature}
    """
    logger.info("Extracting text features from GT labels")

    # Get unique labels
    unique_labels = {}
    for instance_id, proposal in proposals.items():
        if proposal.semantic_id not in unique_labels:
            unique_labels[proposal.semantic_id] = proposal.label

    labels_list = list(unique_labels.values())
    logger.info(f"Extracting features for {len(labels_list)} unique labels")

    # Extract raw text features
    text_features_raw = extractor.extract_text_features(labels_list)  # [N_labels, D]

    # Create mapping from semantic_id to feature
    semantic_to_feature = {}
    for sem_id, label in unique_labels.items():
        idx = labels_list.index(label)
        semantic_to_feature[sem_id] = text_features_raw[idx]

    # Assign to each proposal
    raw_features = {}
    for instance_id, proposal in proposals.items():
        raw_features[instance_id] = semantic_to_feature[proposal.semantic_id]

    # Process features (scale + softmax)
    if apply_softmax:
        logger.info(f"Applying scale_semantic_score={scale_semantic_score} and softmax")

        # Stack all features
        all_features = torch.stack(list(raw_features.values()))  # [N, D]

        # Scale
        scaled_features = scale_semantic_score * all_features  # [N, D]

        # Apply softmax across instances (NOTE: This is per-dimension softmax)
        # Following the implementation in assignment_utils.py
        softmax_features = torch.softmax(scaled_features, dim=0)  # [N, D]

        # Map back to instances
        processed_features = {}
        for i, instance_id in enumerate(raw_features.keys()):
            processed_features[instance_id] = softmax_features[i]
    else:
        processed_features = raw_features

    logger.info(f"Extracted text features for {len(raw_features)} proposals")

    return raw_features, processed_features


def compute_similarity_matrix(
    visual_features: Dict[int, torch.Tensor],
    text_features: Dict[int, torch.Tensor]
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute cosine similarity matrix between visual and text features.

    Args:
        visual_features: {instance_id: visual_feature}
        text_features: {instance_id: text_feature}

    Returns:
        Tuple of (similarity_matrix, instance_ids)
        - similarity_matrix: [N, N] array of cosine similarities
        - instance_ids: List of instance IDs (row/column order)
    """
    # Ensure same instances
    common_ids = sorted(set(visual_features.keys()) & set(text_features.keys()))

    if len(common_ids) == 0:
        logger.error("No common instances between visual and text features")
        return np.array([]), []

    # Stack features
    vis_feats = torch.stack([visual_features[iid] for iid in common_ids]).numpy()
    txt_feats = torch.stack([text_features[iid] for iid in common_ids]).numpy()

    # Normalize
    vis_feats = vis_feats / (np.linalg.norm(vis_feats, axis=1, keepdims=True) + 1e-8)
    txt_feats = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity matrix [N, N]
    similarity_matrix = vis_feats @ txt_feats.T

    return similarity_matrix, common_ids


def analyze_similarity_results(
    similarity_matrix: np.ndarray,
    instance_ids: List[int],
    proposals: Dict[int, GTProposal]
) -> Dict:
    """
    Analyze similarity results and generate statistics.

    Args:
        similarity_matrix: [N, N] similarity matrix
        instance_ids: List of instance IDs
        proposals: Dictionary of GT proposals

    Returns:
        Dictionary of statistics
    """
    N = len(instance_ids)

    # Diagonal values (correct matches)
    diagonal_sims = np.diag(similarity_matrix)

    # Off-diagonal values (incorrect matches)
    off_diagonal_mask = ~np.eye(N, dtype=bool)
    off_diagonal_sims = similarity_matrix[off_diagonal_mask]

    # Statistics
    stats = {
        'num_instances': N,
        'diagonal_mean': float(np.mean(diagonal_sims)),
        'diagonal_std': float(np.std(diagonal_sims)),
        'diagonal_min': float(np.min(diagonal_sims)),
        'diagonal_max': float(np.max(diagonal_sims)),
        'off_diagonal_mean': float(np.mean(off_diagonal_sims)) if len(off_diagonal_sims) > 0 else 0.0,
        'off_diagonal_std': float(np.std(off_diagonal_sims)) if len(off_diagonal_sims) > 0 else 0.0,
        'off_diagonal_max': float(np.max(off_diagonal_sims)) if len(off_diagonal_sims) > 0 else 0.0,
    }

    # Per-instance analysis
    instance_stats = []
    for i, instance_id in enumerate(instance_ids):
        proposal = proposals[instance_id]

        correct_sim = similarity_matrix[i, i]
        max_incorrect_sim = np.max(np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]]))

        instance_stats.append({
            'instance_id': int(instance_id),
            'label': proposal.label,
            'correct_similarity': float(correct_sim),
            'max_incorrect_similarity': float(max_incorrect_sim),
            'margin': float(correct_sim - max_incorrect_sim)
        })

    stats['per_instance'] = instance_stats

    # Sort by margin (ascending = problematic cases)
    instance_stats.sort(key=lambda x: x['margin'])

    return stats


def print_summary(stats: Dict):
    """Print summary of sanity check results."""
    print("\n" + "="*80)
    print("SigLIP Feature Sanity Check Results")
    print("="*80)

    print(f"\nNumber of instances: {stats['num_instances']}")

    print("\nDiagonal Similarities (Correct Matches):")
    print(f"  Mean: {stats['diagonal_mean']:.4f}")
    print(f"  Std:  {stats['diagonal_std']:.4f}")
    print(f"  Min:  {stats['diagonal_min']:.4f}")
    print(f"  Max:  {stats['diagonal_max']:.4f}")

    print("\nOff-Diagonal Similarities (Incorrect Matches):")
    print(f"  Mean: {stats['off_diagonal_mean']:.4f}")
    print(f"  Std:  {stats['off_diagonal_std']:.4f}")
    print(f"  Max:  {stats['off_diagonal_max']:.4f}")

    print("\nTop 5 Most Problematic Instances (Lowest Margin):")
    for i, inst_stat in enumerate(stats['per_instance'][:5]):
        print(f"  {i+1}. ID {inst_stat['instance_id']}: {inst_stat['label']}")
        print(f"     Correct: {inst_stat['correct_similarity']:.4f}, "
              f"Max Incorrect: {inst_stat['max_incorrect_similarity']:.4f}, "
              f"Margin: {inst_stat['margin']:.4f}")

    print("\nTop 5 Best Instances (Highest Margin):")
    for i, inst_stat in enumerate(stats['per_instance'][-5:][::-1]):
        print(f"  {i+1}. ID {inst_stat['instance_id']}: {inst_stat['label']}")
        print(f"     Correct: {inst_stat['correct_similarity']:.4f}, "
              f"Max Incorrect: {inst_stat['max_incorrect_similarity']:.4f}, "
              f"Margin: {inst_stat['margin']:.4f}")

    # Sanity check verdict
    print("\n" + "-"*80)
    if stats['diagonal_mean'] > 0.7 and stats['diagonal_mean'] > stats['off_diagonal_mean'] + 0.2:
        print("✓ PASS: Visual and text features show high similarity (expected for GT)")
    else:
        print("✗ FAIL: Visual and text features do not show expected similarity")
        print("  This may indicate issues with feature extraction or alignment")
    print("-"*80)
    print()


def main():
    parser = argparse.ArgumentParser(description='SigLIP Feature Sanity Check')
    parser.add_argument('--scene', required=True,
                        help='Scene name (e.g., scene_00005_00)')
    parser.add_argument('--gt-root', default='./data/search3d_gt/multiscan_data_search3d',
                        help='Root directory for Search3D GT data')
    parser.add_argument('--scene-root', default='data/multiscan',
                        help='Root directory for MultiScan scenes (for RGB frames)')
    parser.add_argument('--model-name', default='google/siglip-so400m-patch14-384',
                        help='SigLIP model name')
    parser.add_argument('--device', default='cuda',
                        help='Device to run on')
    parser.add_argument('--max-proposals', type=int, default=None,
                        help='Maximum number of proposals to test (for quick testing)')
    parser.add_argument('--scale-semantic-score', type=float, default=300.0,
                        help='Semantic score scaling factor')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON file for detailed results')

    args = parser.parse_args()

    # Paths
    gt_root = Path(args.gt_root)
    scene_root = Path(args.scene_root)

    annotation_path = gt_root / "multiscan_annotations_search3d" / "ov_part_annotations" / f"{args.scene}_obj_part_inst.txt"
    ply_path = gt_root / "multiscan_test_plys_only" / f"{args.scene}.ply"
    scene_path = scene_root / args.scene

    # Verify paths
    if not annotation_path.exists():
        logger.error(f"Annotation file not found: {annotation_path}")
        sys.exit(1)
    if not ply_path.exists():
        logger.error(f"PLY file not found: {ply_path}")
        sys.exit(1)
    if not scene_path.exists():
        logger.warning(f"Scene path not found: {scene_path}")
        logger.warning("Visual feature extraction will use placeholders")

    # Load GT annotations
    logger.info("Loading GT annotations...")
    points, proposals = load_gt_annotations(annotation_path, ply_path)

    if args.max_proposals:
        logger.info(f"Limiting to {args.max_proposals} proposals for testing")
        proposal_ids = list(proposals.keys())[:args.max_proposals]
        proposals = {k: v for k, v in proposals.items() if k in proposal_ids}

    # Initialize SigLIP extractor
    logger.info("Initializing SigLIP feature extractor...")
    extractor = SigLIPFeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )

    # Extract visual features
    logger.info("Extracting visual features from GT proposals...")
    visual_features = extract_visual_features_from_gt(
        proposals, points, scene_path, extractor, max_proposals=args.max_proposals
    )

    # Extract text features
    logger.info("Extracting text features from GT labels...")
    text_features_raw, text_features_processed = extract_text_features(
        proposals, extractor,
        scale_semantic_score=args.scale_semantic_score,
        apply_softmax=True
    )

    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity_matrix, instance_ids = compute_similarity_matrix(
        visual_features, text_features_processed
    )

    # Analyze results
    logger.info("Analyzing results...")
    stats = analyze_similarity_results(similarity_matrix, instance_ids, proposals)

    # Print summary
    print_summary(stats)

    # Save detailed results
    if args.output:
        logger.info(f"Saving detailed results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
