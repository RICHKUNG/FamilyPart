#!/usr/bin/env python3
"""
Verify SigLIP model is working correctly with known image-text pairs.

This script tests:
1. Load SigLIP model correctly
2. Extract image and text features
3. Compute similarity scores
4. Verify similarity values are in expected range

Expected behavior:
- Matched pairs should have similarity > 0.2 (ideally > 0.3)
- Mismatched pairs should have similarity < 0.15
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_siglip_model(model_name: str = "google/siglip-so400m-patch14-384", device: str = "cuda"):
    """Load SigLIP model, processor, and tokenizer."""
    logger.info(f"Loading SigLIP model: {model_name}")
    
    try:
        # Try local first
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        logger.warning("Local model not found, downloading from HuggingFace...")
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.eval()
    logger.info("✓ Model loaded successfully")
    return model, processor, tokenizer


def extract_image_features(model, processor, image_path: Path, device: str = "cuda") -> torch.Tensor:
    """Extract L2-normalized image features."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    
    return feats.squeeze(0).cpu()


def extract_text_features_batch(model, tokenizer, texts: List[str], device: str = "cuda") -> torch.Tensor:
    """Extract L2-normalized text features for a batch of texts."""
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    
    return feats.cpu()


def test_with_sample_images(scene_path: Path, model, processor, tokenizer, device: str = "cuda"):
    """Test SigLIP with real scene images."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Real Scene Images from MultiScan")
    logger.info("="*80)
    
    # Get some sample frames
    color_dir = scene_path / "color"
    if not color_dir.exists():
        logger.warning(f"Color directory not found: {color_dir}")
        return
    
    image_files = sorted(color_dir.glob("*.jpg"))[:3]  # Take first 3 frames
    if not image_files:
        logger.warning(f"No images found in {color_dir}")
        return
    
    logger.info(f"Testing with {len(image_files)} sample frames")
    
    # Test descriptions (generic indoor scene descriptions)
    test_texts = [
        "a bathroom",
        "a kitchen",
        "a bedroom",
        "an office",
        "indoor furniture",
        "a door",
        "a toilet",
        "a cabinet",
    ]
    
    # Extract features
    logger.info("Extracting image features...")
    image_features = []
    for img_path in image_files:
        feat = extract_image_features(model, processor, img_path, device)
        image_features.append(feat)
    image_features = torch.stack(image_features)  # [N_images, D]
    
    logger.info("Extracting text features...")
    text_features = extract_text_features_batch(model, tokenizer, test_texts, device)  # [N_texts, D]
    
    # Compute similarity matrix
    similarity = image_features @ text_features.T  # [N_images, N_texts]
    similarity_np = similarity.numpy()
    
    logger.info(f"\nSimilarity Matrix Shape: {similarity_np.shape}")
    logger.info(f"Similarity Range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    logger.info(f"Mean Similarity: {similarity_np.mean():.4f}")
    logger.info(f"Std Similarity: {similarity_np.std():.4f}")
    
    # Print top matches for each image
    for img_idx, img_path in enumerate(image_files):
        sims = similarity_np[img_idx]
        top_k = 3
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        logger.info(f"\n  Image: {img_path.name}")
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"    {rank}. '{test_texts[idx]}': {sims[idx]:.4f}")
    
    # Check if similarity values are in reasonable range
    if similarity_np.max() < 0.15:
        logger.error("❌ FAIL: Maximum similarity is very low (< 0.15)")
        logger.error("  This suggests features are not properly aligned!")
    elif similarity_np.max() < 0.25:
        logger.warning("⚠️  WARNING: Maximum similarity is low (< 0.25)")
        logger.warning("  Expected values should be > 0.3 for good matches")
    else:
        logger.info(f"✓ PASS: Similarity values in reasonable range (max={similarity_np.max():.4f})")


def test_with_synthetic_examples(model, processor, tokenizer, device: str = "cuda"):
    """Test with synthetic text-text similarity (sanity check)."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Text-Text Similarity (Sanity Check)")
    logger.info("="*80)
    
    # Test with highly similar and dissimilar text pairs
    test_cases = [
        ("door frame", ["door frame", "door", "window frame", "toilet", "cabinet"]),
        ("cabinet door", ["cabinet door", "door", "cabinet", "drawer", "window"]),
        ("toilet lid", ["toilet lid", "toilet", "lid", "sink", "bathtub"]),
    ]
    
    for query_text, candidate_texts in test_cases:
        # Extract features
        query_feat = extract_text_features_batch(model, tokenizer, [query_text], device)[0]
        candidate_feats = extract_text_features_batch(model, tokenizer, candidate_texts, device)
        
        # Compute similarity
        similarities = (query_feat @ candidate_feats.T).numpy()
        
        logger.info(f"\nQuery: '{query_text}'")
        for cand_text, sim in zip(candidate_texts, similarities):
            match_indicator = "✓" if cand_text == query_text else " "
            logger.info(f"  {match_indicator} '{cand_text}': {sim:.4f}")
        
        # Check if self-similarity is highest
        if similarities[0] < max(similarities[1:]):
            logger.warning(f"  ⚠️  Self-match is not highest! ({similarities[0]:.4f} < {max(similarities[1:]):.4f})")
        else:
            logger.info(f"  ✓ Self-match is highest ({similarities[0]:.4f})")


def test_prompt_templates(model, tokenizer, device: str = "cuda"):
    """Test different text prompt templates."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Text Prompt Template Comparison")
    logger.info("="*80)
    
    base_labels = ["door frame", "cabinet door", "toilet lid"]
    templates = [
        "{}",  # No template
        "a {}",
        "a photo of a {}",
        "{} in an indoor scene",
        "indoor {}",
        "a {} in a room",
    ]
    
    for label in base_labels:
        logger.info(f"\nBase label: '{label}'")
        
        # Generate all prompt variants
        prompts = [template.format(label) for template in templates]
        
        # Extract features
        feats = extract_text_features_batch(model, tokenizer, prompts, device)
        
        # Compute pairwise similarities
        similarity_matrix = feats @ feats.T
        
        # Compare with base (no template)
        base_feat = feats[0]
        sims_to_base = (base_feat @ feats.T).numpy()
        
        for idx, (template, sim) in enumerate(zip(templates, sims_to_base)):
            prompt = prompts[idx]
            logger.info(f"  '{prompt}': {sim:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify SigLIP model baseline functionality")
    parser.add_argument("--scene-path", type=Path, 
                        default=Path("data/multiscan/scene_00005_00"),
                        help="Path to scene directory")
    parser.add_argument("--model-name", default="google/siglip-so400m-patch14-384",
                        help="SigLIP model name")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    model, processor, tokenizer = load_siglip_model(args.model_name, args.device)
    
    # Run tests
    test_with_sample_images(args.scene_path, model, processor, tokenizer, args.device)
    test_with_synthetic_examples(model, processor, tokenizer, args.device)
    test_prompt_templates(model, tokenizer, args.device)
    
    logger.info("\n" + "="*80)
    logger.info("Baseline verification complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
