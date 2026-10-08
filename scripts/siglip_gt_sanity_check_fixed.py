#!/usr/bin/env python3
"""
SigLIP GT Sanity Check with FIXED Feature Extraction (Bug Fixes Applied)

This script uses the FIXED feature extraction from notebook_pipeline_fixed.py
with the following critical bug fixes:
- FIX 1 (CRITICAL): Track accumulation counts and average instead of accumulate
- FIX 2 (HIGH): Use only 1 crop (tight bbox) instead of 3 multi-scale crops
- FIX 3 (HIGH): Dynamic threshold based on proposal size instead of hard-coded 20

Expected improvement: 0.03-0.06 → >0.4-0.6 similarity, 60% → >85% rank-1 accuracy

Usage example:
    PYTHONPATH=src python scripts/siglip_gt_sanity_check_fixed.py \
        --scene-path data/multiscan/scene_00005_00 \
        --annotation-file data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations/scene_00005_00_obj_part_inst.txt

Date: 2025-11-16
"""

import argparse
import json
import logging
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SIGLIP_ROOT = REPO_ROOT / "3DprojToSiglip"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline
from notebook_pipeline_fixed import extract_proposal_features_siglip_fixed

logger = logging.getLogger("gt_siglip_sanity_check_fixed")


@dataclass
class GTInstance:
    """Metadata for a single GT instance mask."""

    raw_label: int
    label_id: int
    instance_id: int
    point_count: int


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


def load_gt_masks(
    annotation_file: Path,
    min_points: int,
    limit: Optional[int] = None,
) -> Tuple[torch.Tensor, List[GTInstance]]:
    """Convert GT annotation into proposal masks."""
    labels = np.loadtxt(annotation_file, dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)

    instances: List[GTInstance] = []
    masks: List[torch.Tensor] = []

    # Sort by point count descending for reproducibility
    sorted_indices = np.argsort(-counts)

    for idx in sorted_indices:
        raw_label = int(unique_labels[idx])
        if raw_label <= 0:
            continue

        point_count = int(counts[idx])
        if point_count < min_points:
            continue

        composite_label = raw_label // 1000  # Remove per-instance suffix
        instance_id = raw_label % 1000

        mask = torch.from_numpy((labels == raw_label))

        instances.append(
            GTInstance(
                raw_label=raw_label,
                label_id=int(composite_label),
                instance_id=int(instance_id),
                point_count=point_count,
            )
        )
        masks.append(mask)

        if limit is not None and len(instances) >= limit:
            break

    if not masks:
        raise ValueError("No GT instances matched the filtering criteria.")

    proposal_masks = torch.stack(masks, dim=1).to(torch.bool)
    return proposal_masks, instances


def prepare_world2cam(
    scene_path: Path,
    exp_dir: Path,
    device: str,
    vis_depth_threshold: float,
    frame_frequency: int,
    resize_ratio: float,
) -> AggregationPipeline:
    """Initialize WORLD_2_CAM through AggregationPipeline."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "uni3dis": {"vis_depth_threshold": vis_depth_threshold, "frequency": frame_frequency},
        "network2d": {"masklet": {"resize_ratio": resize_ratio}},
    }

    pipeline = AggregationPipeline(
        exp_dir=str(exp_dir),
        scene_path=str(scene_path),
        device=device,
        config=config,
        iou_threshold=0.8,
        area_fraction=1 / 500,
        pooling_mode="average",
    )
    pipeline._init_world2cam()  # pylint: disable=protected-access
    return pipeline


def extract_text_features_raw(
    tokenizer,
    model,
    label_text_map: Dict[int, str],
    label_ids: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Compute SigLIP text embeddings (L2-normalized only).

    Args:
        tokenizer: SigLIP tokenizer
        model: SigLIP model
        label_text_map: Mapping from label_id to text
        label_ids: List of label IDs to process
        device: Torch device

    Returns:
        Dictionary mapping label_id to L2-normalized text feature
    """
    unique_ids = sorted(set(label_ids))
    texts = [label_text_map[label_id] for label_id in unique_ids]

    # Extract raw text features
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalization

    return {label_id: feats[idx].cpu() for idx, label_id in enumerate(unique_ids)}


def compute_similarity_matrix(
    proposal_features: torch.Tensor,
    text_features_dict: Dict[int, torch.Tensor],
    instances: List[GTInstance],
    scale_semantic_score: float = 300.0,
    apply_softmax: bool = True,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute full similarity matrix between proposals and text features.

    This matches the inference pipeline behavior:
    1. Compute cosine similarity: features @ text_features.T
    2. Scale by scale_semantic_score (default 300.0)
    3. Apply softmax across labels (dim=-1)

    Args:
        proposal_features: [N_proposals, D] tensor of proposal features
        text_features_dict: Dictionary mapping label_id to text feature
        instances: List of GT instances
        scale_semantic_score: Scaling factor (default 300.0)
        apply_softmax: Whether to apply softmax (default True)

    Returns:
        Tuple of (similarity_matrix, proposal_indices, text_label_ids)
        - similarity_matrix: [N_proposals, N_unique_labels] similarity matrix (after scaling & softmax)
        - proposal_indices: Indices of proposals
        - text_label_ids: Corresponding label IDs for text features
    """
    # Get unique label IDs that have text features
    unique_label_ids = sorted(set(inst.label_id for inst in instances if inst.label_id in text_features_dict))

    if len(unique_label_ids) == 0:
        logger.warning("No text features found for any instances")
        return np.array([]), [], []

    # Stack text features
    text_feats = torch.stack([text_features_dict[lid] for lid in unique_label_ids])  # [M, D]

    # Ensure both tensors are on the same device (CPU)
    text_feats = text_feats.cpu()
    proposal_features = proposal_features.cpu()

    # Compute cosine similarity matrix
    # proposal_features: [N, D]
    # text_feats: [M, D]
    # similarity: [N, M]
    similarity_matrix = proposal_features @ text_feats.T  # Cosine similarity (already normalized)

    if apply_softmax:
        logger.info(f"Applying scale_semantic_score={scale_semantic_score} and softmax to similarity matrix")

        # Scale similarity matrix
        # Following assignment_utils.py line 76-77:
        # similarity = scale_factor * (features @ text_features.T)
        # similarity = F.softmax(similarity, dim=-1)
        scaled_similarity = scale_semantic_score * similarity_matrix  # [N, M]

        # Apply softmax across labels (dim=-1)
        softmax_similarity = torch.softmax(scaled_similarity, dim=-1)  # [N, M]

        logger.info(f"Similarity matrix shape: {softmax_similarity.shape}")
        logger.info(f"Softmax sum per proposal (should be ~1.0): {softmax_similarity.sum(dim=-1)[:5]}")

        similarity_matrix = softmax_similarity

    proposal_indices = list(range(len(instances)))

    return similarity_matrix.cpu().numpy(), proposal_indices, unique_label_ids


def analyze_similarity_matrix(
    similarity_matrix: np.ndarray,
    instances: List[GTInstance],
    text_label_ids: List[int],
    label_text_map: Dict[int, str],
) -> Dict:
    """
    Analyze similarity matrix to check if GT visual features match their text features.

    Args:
        similarity_matrix: [N_proposals, M_labels] similarity matrix
        instances: List of GT instances
        text_label_ids: List of label IDs for columns in similarity matrix
        label_text_map: Mapping from label_id to text

    Returns:
        Dictionary with analysis results
    """
    N = len(instances)
    M = len(text_label_ids)

    logger.info(f"Similarity matrix shape: {similarity_matrix.shape} ({N} proposals x {M} labels)")

    # Create label_id to column index mapping
    label_to_col = {lid: idx for idx, lid in enumerate(text_label_ids)}

    # For each proposal, check similarity with its GT label
    results = []

    for i, inst in enumerate(instances):
        if inst.label_id not in label_to_col:
            # No text feature for this label
            results.append({
                'proposal_idx': i,
                'raw_label': inst.raw_label,
                'label_id': inst.label_id,
                'instance_id': inst.instance_id,
                'points': inst.point_count,
                'text': label_text_map.get(inst.label_id, "UNKNOWN"),
                'correct_similarity': None,
                'max_incorrect_similarity': None,
                'margin': None,
                'rank': None,
                'status': 'missing_text'
            })
            continue

        # Get correct column
        correct_col = label_to_col[inst.label_id]
        correct_sim = float(similarity_matrix[i, correct_col])

        # Get all similarities for this proposal
        all_sims = similarity_matrix[i, :]

        # Remove correct similarity to get incorrect ones
        incorrect_sims = np.concatenate([all_sims[:correct_col], all_sims[correct_col+1:]])

        max_incorrect_sim = float(np.max(incorrect_sims)) if len(incorrect_sims) > 0 else 0.0

        # Compute rank (1 = highest similarity)
        rank = int(np.sum(all_sims > correct_sim) + 1)

        # Status
        margin = correct_sim - max_incorrect_sim
        if rank == 1:
            status = "rank_1"
        elif margin > 0.1:
            status = "ok"
        else:
            status = "low"

        results.append({
            'proposal_idx': i,
            'raw_label': inst.raw_label,
            'label_id': inst.label_id,
            'instance_id': inst.instance_id,
            'points': inst.point_count,
            'text': label_text_map.get(inst.label_id, "UNKNOWN"),
            'correct_similarity': round(correct_sim, 4),
            'max_incorrect_similarity': round(max_incorrect_sim, 4),
            'margin': round(margin, 4),
            'rank': rank,
            'status': status
        })

    # Sort by margin (ascending = problematic)
    results_sorted = sorted(results, key=lambda r: (r['margin'] is None, -(r['margin'] or -1)))

    # Statistics
    correct_sims = [r['correct_similarity'] for r in results if r['correct_similarity'] is not None]
    rank_1_count = sum(1 for r in results if r['rank'] == 1)

    summary = {
        'num_proposals': N,
        'num_labels': M,
        'num_with_text': len(correct_sims),
        'num_rank_1': rank_1_count,
        'rank_1_percentage': 100.0 * rank_1_count / len(correct_sims) if correct_sims else 0.0,
        'mean_correct_similarity': float(np.mean(correct_sims)) if correct_sims else None,
        'mean_margin': float(np.mean([r['margin'] for r in results if r['margin'] is not None])) if correct_sims else None,
        'min_correct_similarity': float(np.min(correct_sims)) if correct_sims else None,
        'max_correct_similarity': float(np.max(correct_sims)) if correct_sims else None,
        'details': results_sorted
    }

    return summary


def print_summary(summary: Dict):
    """Print summary of sanity check results."""
    print("\n" + "="*80)
    print("SigLIP GT Sanity Check Results - FIXED Feature Extraction")
    print("(Bug Fixes: Average instead of accumulate, 1 crop, dynamic threshold)")
    print("="*80)

    print(f"\nDataset Statistics:")
    print(f"  Total proposals: {summary['num_proposals']}")
    print(f"  Unique labels: {summary['num_labels']}")
    print(f"  Proposals with text: {summary['num_with_text']}")

    print(f"\nRanking Performance:")
    print(f"  Rank-1 matches: {summary['num_rank_1']} / {summary['num_with_text']} ({summary['rank_1_percentage']:.1f}%)")

    print(f"\nSimilarity Statistics:")
    if summary['mean_correct_similarity'] is not None:
        print(f"  Mean correct similarity: {summary['mean_correct_similarity']:.4f}")
        print(f"  Mean margin: {summary['mean_margin']:.4f}")
        print(f"  Min correct similarity: {summary['min_correct_similarity']:.4f}")
        print(f"  Max correct similarity: {summary['max_correct_similarity']:.4f}")

    print(f"\nTop 10 Most Problematic Cases (Lowest Margin):")
    for i, r in enumerate(summary['details'][:10]):
        if r['margin'] is None:
            print(f"  {i+1}. [{r['status']}] ID {r['raw_label']}: {r['text']} - NO TEXT FEATURE")
        else:
            print(f"  {i+1}. [{r['status']}] ID {r['raw_label']}: {r['text']}")
            print(f"      Correct: {r['correct_similarity']:.4f}, Max Incorrect: {r['max_incorrect_similarity']:.4f}, "
                  f"Margin: {r['margin']:.4f}, Rank: {r['rank']}")

    print(f"\nTop 5 Best Cases (Highest Margin):")
    valid_results = [r for r in summary['details'] if r['margin'] is not None]
    for i, r in enumerate(valid_results[-5:][::-1]):
        print(f"  {i+1}. [{r['status']}] ID {r['raw_label']}: {r['text']}")
        print(f"      Correct: {r['correct_similarity']:.4f}, Max Incorrect: {r['max_incorrect_similarity']:.4f}, "
              f"Margin: {r['margin']:.4f}, Rank: {r['rank']}")

    # Sanity check verdict
    print("\n" + "-"*80)
    if summary['rank_1_percentage'] >= 80.0 and summary['mean_margin'] > 0.1:
        print("✓ PASS: Visual and text features show high similarity (expected for GT)")
        print(f"  - {summary['rank_1_percentage']:.1f}% rank-1 accuracy")
        print(f"  - Mean margin: {summary['mean_margin']:.4f}")
    elif summary['rank_1_percentage'] >= 60.0:
        print("⚠ PARTIAL PASS: Most GT visual features match their text features")
        print(f"  - {summary['rank_1_percentage']:.1f}% rank-1 accuracy (expected >= 80%)")
        print(f"  - Mean margin: {summary['mean_margin']:.4f}")
    else:
        print("✗ FAIL: Visual and text features do not show expected similarity")
        print(f"  - {summary['rank_1_percentage']:.1f}% rank-1 accuracy (expected >= 80%)")
        print("  This may indicate issues with feature extraction, alignment, or scaling")
    print("-"*80)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT SigLIP feature sanity check with scaling & softmax")
    parser.add_argument("--scene-path", required=True, type=Path, help="Path to MultiScan scene directory")
    parser.add_argument("--annotation-file", required=True, type=Path, help="Search3D GT annotation file")
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--local-files-only", action="store_true", help="Force transformers to use local files only")
    parser.add_argument("--device", default="cuda:0", help="Torch device for SigLIP")
    parser.add_argument("--temp-exp-dir", type=Path, help="Temporary experiment directory for WORLD_2_CAM caches")
    parser.add_argument("--min-points", type=int, default=150, help="Minimum points per GT instance")
    parser.add_argument("--limit", type=int, help="Process at most N GT instances")
    parser.add_argument("--topk-views", type=int, default=1, help="Number of best views per proposal")
    parser.add_argument("--batch-size", type=int, default=32, help="SigLIP proposal batch size")
    parser.add_argument("--frame-frequency", type=int, default=50, help="Frame sampling frequency for WORLD_2_CAM")
    parser.add_argument("--masklet-resize", type=float, default=0.3, help="Masklet resize ratio")
    parser.add_argument("--vis-depth-threshold", type=float, default=0.1, help="Visibility depth threshold")
    parser.add_argument("--scale-semantic-score", type=float, default=300.0, help="Temperature scaling factor")
    parser.add_argument("--no-softmax", dest="apply_softmax", action="store_false", default=True,
                        help="Disable softmax normalization (default: enabled)")
    parser.add_argument("--output-json", type=Path, help="Optional path to dump JSON summary")
    parser.add_argument("--no-optimized", dest="use_optimized", action="store_false", default=True,
                        help="Disable batched optimization")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    if not args.scene_path.exists():
        raise FileNotFoundError(f"Scene path not found: {args.scene_path}")
    if not args.annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")

    scene_name = args.scene_path.name
    temp_exp_dir = args.temp_exp_dir or (REPO_ROOT / "outputs" / "gt_siglip_sanity_check" / scene_name)

    label_text_map = load_joint_label_map()

    logger.info("Loading GT annotations from %s", args.annotation_file)
    proposal_masks, instances = load_gt_masks(
        annotation_file=args.annotation_file,
        min_points=args.min_points,
        limit=args.limit,
    )
    logger.info("Loaded %d GT instances (mask tensor shape=%s)", len(instances), tuple(proposal_masks.shape))

    logger.info("Initializing WORLD_2_CAM for scene %s", args.scene_path)
    pipeline = prepare_world2cam(
        scene_path=args.scene_path,
        exp_dir=temp_exp_dir,
        device=args.device,
        vis_depth_threshold=args.vis_depth_threshold,
        frame_frequency=args.frame_frequency,
        resize_ratio=args.masklet_resize,
    )

    if proposal_masks.shape[0] != pipeline.P:
        raise ValueError(
            f"Mismatch between GT mask points ({proposal_masks.shape[0]}) and scene points ({pipeline.P})."
        )

    logger.info("Loading SigLIP model: %s", args.siglip_model)
    try:
        torch_device = torch.device(args.device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            torch_device = torch.device("cpu")
    except (RuntimeError, ValueError):
        logger.warning("Invalid device '%s', falling back to CPU", args.device)
        torch_device = torch.device("cpu")

    local_only = args.local_files_only or Path(args.siglip_model).expanduser().exists()

    try:
        model = AutoModel.from_pretrained(args.siglip_model, local_files_only=local_only).to(torch_device)
        processor = AutoProcessor.from_pretrained(args.siglip_model, local_files_only=local_only)
        tokenizer = AutoTokenizer.from_pretrained(args.siglip_model, local_files_only=local_only)
        model.eval()
    except OSError as exc:
        raise RuntimeError(f"Failed to load SigLIP model '{args.siglip_model}': {exc}") from exc

    logger.info("Extracting proposal features with FIXED feature extraction (topk=%d, batch_size=%d)...",
                args.topk_views, args.batch_size)
    proposal_features = extract_proposal_features_siglip_fixed(
        pipeline=pipeline,
        proposal_masks=proposal_masks,
        model=model,
        processor=processor,
        device=torch_device,
        topk=args.topk_views,
        batch_size=args.batch_size,
    )

    logger.info("Extracting text features for %d unique labels",
                len({inst.label_id for inst in instances}))
    text_features = extract_text_features_raw(
        tokenizer=tokenizer,
        model=model,
        label_text_map=label_text_map,
        label_ids=[inst.label_id for inst in instances if inst.label_id in label_text_map],
        device=torch_device,
    )

    logger.info("Computing similarity matrix (with scaling & softmax)...")
    similarity_matrix, proposal_indices, text_label_ids = compute_similarity_matrix(
        proposal_features=proposal_features,
        text_features_dict=text_features,
        instances=instances,
        scale_semantic_score=args.scale_semantic_score,
        apply_softmax=args.apply_softmax,
    )

    logger.info("Analyzing results...")
    summary = analyze_similarity_matrix(
        similarity_matrix=similarity_matrix,
        instances=instances,
        text_label_ids=text_label_ids,
        label_text_map=label_text_map,
    )

    print_summary(summary)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Saved summary to %s", args.output_json)


if __name__ == "__main__":
    main()
