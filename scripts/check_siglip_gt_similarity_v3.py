#!/usr/bin/env python3
"""
GT SigLIP feature similarity checker (v3) - with improved text prompts.

Key improvement: Use "part of object" format for better text-image alignment.
Example: "door frame" -> "frame of door"

Usage:
    PYTHONPATH=src python scripts/check_siglip_gt_similarity_v3.py \
        --scene-path data/multiscan/scene_00005_00 \
        --annotation-file data/search3d_gt/.../scene_00005_00_obj_part_inst.txt
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
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline

logger = logging.getLogger("gt_siglip_check_v3")


@dataclass
class GTInstance:
    """Metadata for a single GT instance mask."""

    raw_label: int
    label_id: int
    instance_id: int
    point_count: int


def load_joint_label_map_with_of() -> Dict[int, str]:
    """
    Load Search3D joint labels and convert to 'part of object' format.

    Example:
        (door, frame) -> "frame of door"
        (toilet, lid) -> "lid of toilet"
    """
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
        # Convert to "part of object" format
        if part_text:
            # Has part: "part of object"
            text = f"{part_text} of {obj_text}"
        else:
            # No part: just object
            text = obj_text

        label_map[int(label_id)] = text.strip()

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
    pipeline._init_world2cam()
    return pipeline


def extract_proposal_features_v2(
    pipeline: AggregationPipeline,
    proposal_masks: torch.Tensor,
    model,
    processor,
    device: torch.device,
    topk: int,
    multi_scale: bool = True,
) -> torch.Tensor:
    """Extract SigLIP features per-proposal."""
    N = proposal_masks.shape[1]
    D = 1152  # SigLIP dimension

    # Load images once
    images = [np.array(Image.open(p)) for p in tqdm(pipeline.world2cam.color_paths, desc="Loading images")]

    proposal_features = torch.zeros((N, D), dtype=torch.float32, device=device)

    for prop_idx in tqdm(range(N), desc="Extracting proposal features"):
        prop_mask = proposal_masks[:, prop_idx]
        prop_points = torch.where(prop_mask)[0]

        if len(prop_points) == 0:
            continue

        # Find visible frames for this proposal
        frame_scores = compute_frame_visibility_scores(
            prop_points=prop_points,
            inside_mask=pipeline.inside_mask,
            projected_points=pipeline.projected_points,
            point_depth=pipeline.point_depth,
            depth_maps=pipeline.depth_maps,
            vis_threshold=pipeline.vis_depth_threshold,
        )

        # Select top-k frames
        valid_frames = torch.nonzero(frame_scores > 0).view(-1)
        if len(valid_frames) == 0:
            continue

        k = min(topk, len(valid_frames))
        topk_frames = torch.topk(frame_scores[valid_frames], k=k).indices
        selected_frames = valid_frames[topk_frames]

        # Extract features from selected frames
        frame_features = []
        for frame_idx in selected_frames.tolist():
            feat = extract_proposal_feature_from_frame(
                frame_idx=frame_idx,
                prop_points=prop_points,
                inside_mask=pipeline.inside_mask,
                projected_points=pipeline.projected_points,
                image=images[frame_idx],
                model=model,
                processor=processor,
                device=device,
                scaling_params=pipeline.scaling_params,
                multi_scale=multi_scale,
            )

            if feat is not None:
                frame_features.append(feat)

        if len(frame_features) > 0:
            # Average across frames and normalize
            avg_feat = torch.stack(frame_features).mean(dim=0)
            proposal_features[prop_idx] = F.normalize(avg_feat, dim=0)

    return proposal_features


def compute_frame_visibility_scores(
    prop_points: torch.Tensor,
    inside_mask: torch.Tensor,
    projected_points: torch.Tensor,
    point_depth: torch.Tensor,
    depth_maps: List[torch.Tensor],
    vis_threshold: float,
) -> torch.Tensor:
    """Compute visibility score for each frame."""
    F = inside_mask.shape[0]
    scores = torch.zeros(F, dtype=torch.float32)

    for frame_idx in range(F):
        # Check which proposal points are inside this frame
        inside = inside_mask[frame_idx, prop_points]

        if inside.sum() == 0:
            continue

        # Get projected coordinates and depths
        proj_coords = projected_points[frame_idx, prop_points[inside]]
        depths = point_depth[frame_idx, prop_points[inside]]

        # Check depth visibility
        H, W = depth_maps[frame_idx].shape
        x_coords = proj_coords[:, 0].long().clamp(0, H - 1)
        y_coords = proj_coords[:, 1].long().clamp(0, W - 1)

        depth_at_pixels = depth_maps[frame_idx][x_coords, y_coords]
        depth_diff = torch.abs(depths - depth_at_pixels)
        visible = depth_diff < vis_threshold

        scores[frame_idx] = visible.sum().item()

    return scores


def extract_proposal_feature_from_frame(
    frame_idx: int,
    prop_points: torch.Tensor,
    inside_mask: torch.Tensor,
    projected_points: torch.Tensor,
    image: np.ndarray,
    model,
    processor,
    device: torch.device,
    scaling_params: List[float],
    multi_scale: bool = True,
) -> Optional[torch.Tensor]:
    """Extract SigLIP feature for a proposal from a single frame."""
    # Get points visible in this frame
    inside = inside_mask[frame_idx, prop_points]

    if inside.sum() == 0:
        return None

    # Get 2D coordinates
    coords = projected_points[frame_idx, prop_points[inside]]

    # Apply scaling
    scale_x, scale_y = scaling_params
    coords_scaled = coords.clone().float()
    coords_scaled[:, 0] *= scale_x
    coords_scaled[:, 1] *= scale_y

    # Compute bounding box
    mi = torch.min(coords_scaled, dim=0).values
    ma = torch.max(coords_scaled, dim=0).values

    x1, y1 = int(mi[0].item()), int(mi[1].item())
    x2, y2 = int(ma[0].item()), int(ma[1].item())

    if x2 <= x1 or y2 <= y1:
        return None

    # Extract multi-scale crops
    H, W = image.shape[:2]
    crops = []
    kexp = 0.2

    num_scales = 3 if multi_scale else 1

    for r in range(num_scales):
        # Clamp to image bounds
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(H, x2)
        y2_clip = min(W, y2)

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            break

        crop = image[x1_clip:x2_clip, y1_clip:y2_clip, :]

        if crop.size != 0:
            crop_tensor = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
            crops.append(crop_tensor)

        # Grow bounding box for next scale
        dx = int((x2 - x1) * kexp)
        dy = int((y2 - y1) * kexp)
        x1 -= dx
        y1 -= dy
        x2 += dx
        y2 += dy

    if len(crops) == 0:
        return None

    # Encode crops
    crops_tensor = torch.cat(crops, dim=0)

    with torch.no_grad():
        feats = model.get_image_features(pixel_values=crops_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Average across scales
        frame_feature = feats.mean(dim=0)

    return frame_feature


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

    logger.info("Text prompts used:")
    for label_id, text in zip(unique_ids, texts):
        logger.info(f"  {label_id}: \"{text}\"")

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
    """Prepare summary statistics."""
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

    logger.info("=== Similarity Results (v3 - with 'of' prompts) ===")
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
        "version": "v3",
        "text_format": "part of object",
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
        "Summary: %d/%d instances matched text, %d above threshold %.2f",
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
    parser = argparse.ArgumentParser(description="GT SigLIP feature similarity checker (v3)")
    parser.add_argument("--scene-path", required=True, type=Path)
    parser.add_argument("--annotation-file", required=True, type=Path)
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temp-exp-dir", type=Path)
    parser.add_argument("--min-points", type=int, default=150)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topk-views", type=int, default=5)
    parser.add_argument("--frame-frequency", type=int, default=50)
    parser.add_argument("--masklet-resize", type=float, default=0.3)
    parser.add_argument("--vis-depth-threshold", type=float, default=0.1)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--no-multi-scale", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    scene_name = args.scene_path.name
    temp_exp_dir = args.temp_exp_dir or (REPO_ROOT / "outputs" / "gt_siglip_check_v3" / scene_name)

    # Load label map with 'of' format
    label_text_map = load_joint_label_map_with_of()

    logger.info("Loading GT annotations from %s", args.annotation_file)
    proposal_masks, instances = load_gt_masks(
        annotation_file=args.annotation_file,
        min_points=args.min_points,
        limit=args.limit,
    )
    logger.info("Loaded %d GT instances", len(instances))

    logger.info("Initializing WORLD_2_CAM")
    pipeline = prepare_world2cam(
        scene_path=args.scene_path,
        exp_dir=temp_exp_dir,
        device=args.device,
        vis_depth_threshold=args.vis_depth_threshold,
        frame_frequency=args.frame_frequency,
        resize_ratio=args.masklet_resize,
    )

    if proposal_masks.shape[0] != pipeline.P:
        raise ValueError(f"Mismatch: GT has {proposal_masks.shape[0]} points, scene has {pipeline.P}")

    logger.info("Loading SigLIP model: %s", args.siglip_model)
    torch_device = torch.device(args.device)
    model = AutoModel.from_pretrained(args.siglip_model).to(torch_device)
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.siglip_model)
    model.eval()

    logger.info("Extracting proposal features (v2 method)...")
    proposal_features = extract_proposal_features_v2(
        pipeline=pipeline,
        proposal_masks=proposal_masks,
        model=model,
        processor=processor,
        device=torch_device,
        topk=args.topk_views,
        multi_scale=not args.no_multi_scale,
    )

    logger.info("Extracting text features with 'of' format")
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
            sim = torch.dot(proposal_features[idx].cpu(), text_feat)
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
        logger.info("Saved to %s", args.output_json)


if __name__ == "__main__":
    main()
