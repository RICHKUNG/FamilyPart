#!/usr/bin/env python3
"""
Ground-truth SigLIP feature similarity checker.

This script converts Search3D GT annotations into 3D proposal masks, extracts
SigLIP features for the proposals using actual MultiScan RGB frames, and
compares them with the SigLIP text embeddings produced from the GT query text.

Usage example:
    PYTHONPATH=src python scripts/check_siglip_gt_similarity.py \
        --scene-path data/multiscan/scene_00005_00 \
        --annotation-file data/search3d_gt/.../ov_part_annotations/scene_00005_00_obj_part_inst.txt
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
from transformers import AutoModel, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SIGLIP_ROOT = REPO_ROOT / "3DprojToSiglip"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline

from notebook_pipeline import extract_proposal_features_siglip

logger = logging.getLogger("gt_siglip_check")


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


def extract_text_features(
    tokenizer,
    model,
    label_text_map: Dict[int, str],
    label_ids: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Compute SigLIP text embeddings for selected label IDs."""
    unique_ids = sorted(set(label_ids))
    texts = [label_text_map[label_id] for label_id in unique_ids]

    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return {label_id: feats[idx].cpu() for idx, label_id in enumerate(unique_ids)}


def summarize_results(
    instances: List[GTInstance],
    similarities: List[Optional[float]],
    label_text_map: Dict[int, str],
    threshold: float,
) -> Dict:
    """Prepare summary statistics and logging output."""
    rows = []
    pass_count = 0
    valid_sims = []

    for inst, sim in zip(instances, similarities):
        text_label = label_text_map.get(inst.label_id, "UNKNOWN")
        status = "missing_text" if sim is None else ("ok" if sim >= threshold else "low")
        if sim is not None:
            valid_sims.append(sim)
            if sim >= threshold:
                pass_count += 1

        rows.append(
            {
                "raw_label": inst.raw_label,
                "label_id": inst.label_id,
                "instance_id": inst.instance_id,
                "points": inst.point_count,
                "similarity": None if sim is None else round(float(sim), 4),
                "text": text_label,
                "status": status,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: (r["similarity"] is None, -(r["similarity"] or -1.0)))

    logger.info("=== Similarity Results ===")
    for row in rows_sorted:
        logger.info(
            "label=%d (raw=%d inst=%03d) pts=%d sim=%s status=%s text=\"%s\"",
            row["label_id"],
            row["raw_label"],
            row["instance_id"],
            row["points"],
            "None" if row["similarity"] is None else f"{row['similarity']:.4f}",
            row["status"],
            row["text"],
        )

    summary = {
        "num_instances": len(instances),
        "num_with_text": len([sim for sim in similarities if sim is not None]),
        "num_above_threshold": pass_count,
        "threshold": threshold,
        "mean_similarity": float(np.mean(valid_sims)) if valid_sims else None,
        "min_similarity": float(np.min(valid_sims)) if valid_sims else None,
        "max_similarity": float(np.max(valid_sims)) if valid_sims else None,
        "details": rows_sorted,
    }

    logger.info(
        "Summary: %d/%d instances matched text labels, %d above threshold %.2f",
        summary["num_with_text"],
        summary["num_instances"],
        summary["num_above_threshold"],
        threshold,
    )
    if valid_sims:
        logger.info(
            "Similarity stats -> mean: %.4f | min: %.4f | max: %.4f",
            summary["mean_similarity"],
            summary["min_similarity"],
            summary["max_similarity"],
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT SigLIP feature similarity checker")
    parser.add_argument("--scene-path", required=True, type=Path, help="Path to MultiScan scene directory")
    parser.add_argument("--annotation-file", required=True, type=Path, help="Search3D GT annotation file")
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--local-files-only", action="store_true", help="Force transformers to use local files only")
    parser.add_argument("--device", default="cuda:0", help="Torch device for SigLIP")
    parser.add_argument("--temp-exp-dir", type=Path, help="Temporary experiment directory for WORLD_2_CAM caches")
    parser.add_argument("--min-points", type=int, default=150, help="Minimum points per GT instance")
    parser.add_argument("--limit", type=int, help="Process at most N GT instances")
    parser.add_argument("--topk-views", type=int, default=1, help="Number of best views per proposal")
    parser.add_argument("--batch-size", type=int, default=32, help="SigLIP proposal batch size (default: 32)")
    parser.add_argument("--frame-frequency", type=int, default=50, help="Frame sampling frequency for WORLD_2_CAM")
    parser.add_argument("--masklet-resize", type=float, default=0.3, help="Masklet resize ratio")
    parser.add_argument("--vis-depth-threshold", type=float, default=0.1, help="Visibility depth threshold")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Threshold to flag high similarity")
    parser.add_argument("--output-json", type=Path, help="Optional path to dump JSON summary")
    parser.add_argument("--no-optimized", dest="use_optimized", action="store_false", default=True, help="Disable batched optimization (default: enabled)")
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
    temp_exp_dir = args.temp_exp_dir or (REPO_ROOT / "outputs" / "gt_siglip_check" / scene_name)

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

    def load_siglip_components(model_name: str, local_only: bool):
        try:
            model = AutoModel.from_pretrained(model_name, local_files_only=local_only).to(torch_device)
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only)
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
            model.eval()
            return model, processor, tokenizer
        except OSError as exc:
            extra = (
                "Provide a local path via --siglip-model or pre-download the weights "
                "and re-run with --local-files-only."
            )
            raise RuntimeError(f"Failed to load SigLIP model '{model_name}': {exc}\n{extra}") from exc

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
    model, processor, tokenizer = load_siglip_components(args.siglip_model, local_only)

    logger.info("Extracting proposal features (optimized=%s, batch_size=%d, sparse=True)...",
                args.use_optimized, args.batch_size)
    proposal_features = extract_proposal_features_siglip(
        pipeline=pipeline,
        proposal_masks=proposal_masks,
        model=model,
        processor=processor,
        device=torch_device,
        topk=args.topk_views,
        batch_size=args.batch_size,
        use_optimized=args.use_optimized,
    )

    logger.info("Extracting text features for %d unique labels", len({inst.label_id for inst in instances}))
    text_features = extract_text_features(
        tokenizer=tokenizer,
        model=model,
        label_text_map=label_text_map,
        label_ids=[inst.label_id for inst in instances if inst.label_id in label_text_map],
        device=torch_device,
    )

    similarities: List[Optional[float]] = []
    for idx, inst in enumerate(instances):
        text_feat = text_features.get(inst.label_id)
        if text_feat is None:
            similarities.append(None)
        else:
            sim = torch.dot(proposal_features[idx], text_feat)
            similarities.append(float(sim.item()))

    summary = summarize_results(
        instances=instances,
        similarities=similarities,
        label_text_map=label_text_map,
        threshold=args.similarity_threshold,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Saved summary to %s", args.output_json)


if __name__ == "__main__":
    main()
