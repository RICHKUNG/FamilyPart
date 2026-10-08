#!/usr/bin/env python3
"""
Performance benchmark for feature extraction optimization.

This script compares the original vs optimized feature extraction implementations
on a single scene to demonstrate speedup.

Usage:
    # Test on a single scene
    python scripts/benchmark_feature_extraction.py \
        --exp-dir outputs/experiments/SUCCESS/v7_246_ssam4_filter100_fill100_prop30_max3k_m_1/scene_00005_00 \
        --dataset-root data/multiscan

    # Compare different batch sizes
    python scripts/benchmark_feature_extraction.py \
        --exp-dir outputs/experiments/SUCCESS/v7_246_ssam4_filter100_fill100_prop30_max3k_m_1/scene_00005_00 \
        --dataset-root data/multiscan \
        --batch-sizes 8,16,32,64

    # Test with cache disabled
    python scripts/benchmark_feature_extraction.py \
        --exp-dir outputs/experiments/SUCCESS/v7_246_ssam4_filter100_fill100_prop30_max3k_m_1/scene_00005_00 \
        --dataset-root data/multiscan \
        --no-cache
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import logging
from typing import Dict, List

import torch
import numpy as np

from my3dis.inference import SigLIPFeatureExtractor, OptimizedSigLIPFeatureExtractor
from my3dis.aggregation import ProposalLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_proposals(exp_dir: Path, levels: List[str]) -> Dict[str, List[Dict]]:
    """Load proposals from aggregation output."""
    aggregation_dir = exp_dir / "aggregation_output_siglip_so400m_average"
    if not aggregation_dir.exists():
        raise RuntimeError(f"Aggregation output not found: {aggregation_dir}")

    loader = ProposalLoader(aggregation_dir)
    proposals_by_level = loader.load_proposals(levels=levels)

    # Count total proposals
    total = sum(len(props) for props in proposals_by_level.values())
    logger.info(f"Loaded {total} proposals across {len(proposals_by_level)} levels")

    return proposals_by_level


def benchmark_original(
    proposals_by_level: Dict[str, List[Dict]],
    exp_dir: Path,
    scene_path: Path,
    model_name: str,
    device: str,
    max_frames: int = 5
):
    """Benchmark original feature extraction (no batching)."""
    logger.info("="*80)
    logger.info("ORIGINAL IMPLEMENTATION (No Batching)")
    logger.info("="*80)

    extractor = SigLIPFeatureExtractor(model_name=model_name, device=device)

    start_time = time.time()

    features = extractor.extract_proposal_features(
        proposals_by_level=proposals_by_level,
        exp_dir=exp_dir,
        scene_path=scene_path,
        max_frames_per_object=max_frames,
        batch_size=8  # Ignored by original implementation
    )

    elapsed = time.time() - start_time

    total_objects = sum(len(f) for f in features.values())

    logger.info(f"✓ Original completed in {elapsed:.1f}s")
    logger.info(f"  Objects processed: {total_objects}")
    logger.info(f"  Speed: {total_objects/elapsed:.2f} objects/sec")

    return elapsed, total_objects


def benchmark_optimized(
    proposals_by_level: Dict[str, List[Dict]],
    exp_dir: Path,
    scene_path: Path,
    model_name: str,
    device: str,
    batch_size: int = 32,
    cache_size: int = 100,
    max_frames: int = 5
):
    """Benchmark optimized feature extraction (with batching and caching)."""
    logger.info("="*80)
    logger.info(f"OPTIMIZED IMPLEMENTATION (Batch={batch_size}, Cache={cache_size})")
    logger.info("="*80)

    extractor = OptimizedSigLIPFeatureExtractor(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        cache_size=cache_size
    )

    start_time = time.time()

    features = extractor.extract_proposal_features(
        proposals_by_level=proposals_by_level,
        exp_dir=exp_dir,
        scene_path=scene_path,
        max_frames_per_object=max_frames,
        batch_size=batch_size
    )

    elapsed = time.time() - start_time

    total_objects = sum(len(f) for f in features.values())

    # Get cache statistics
    cache_stats = extractor.image_cache.get_stats()

    logger.info(f"✓ Optimized completed in {elapsed:.1f}s")
    logger.info(f"  Objects processed: {total_objects}")
    logger.info(f"  Speed: {total_objects/elapsed:.2f} objects/sec")
    logger.info(f"  Cache hit rate: {cache_stats['hit_rate']:.1f}%")
    logger.info(f"  Cache hits/misses: {cache_stats['hits']}/{cache_stats['misses']}")

    return elapsed, total_objects, cache_stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark feature extraction performance")
    parser.add_argument('--exp-dir', type=str, required=True,
                       help='Experiment directory (single scene)')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Dataset root directory')
    parser.add_argument('--model', type=str, default='google/siglip-so400m-patch14-384',
                       help='SigLIP model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--levels', type=str, default='L2,L4,L6',
                       help='Levels to process (comma-separated)')
    parser.add_argument('--batch-sizes', type=str, default='32',
                       help='Batch sizes to test (comma-separated)')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Image cache size')
    parser.add_argument('--max-frames', type=int, default=5,
                       help='Maximum frames per object')
    parser.add_argument('--skip-original', action='store_true',
                       help='Skip original implementation (faster testing)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable image cache for testing')

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        return

    # Get scene name from exp_dir
    scene_name = exp_dir.name
    scene_path = Path(args.dataset_root) / scene_name

    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        return

    levels = [lvl.strip() for lvl in args.levels.split(',')]
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(',')]

    logger.info(f"Experiment: {exp_dir}")
    logger.info(f"Scene: {scene_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Levels: {levels}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info("")

    # Load proposals
    proposals_by_level = load_proposals(exp_dir, levels)

    results = []

    # Benchmark original (optional)
    if not args.skip_original:
        try:
            original_time, original_objects = benchmark_original(
                proposals_by_level=proposals_by_level,
                exp_dir=exp_dir,
                scene_path=scene_path,
                model_name=args.model,
                device=args.device,
                max_frames=args.max_frames
            )
            results.append({
                'method': 'Original',
                'batch_size': 1,
                'time': original_time,
                'objects': original_objects,
                'speed': original_objects / original_time
            })
        except Exception as e:
            logger.error(f"Original implementation failed: {e}")
            original_time = None
    else:
        logger.info("Skipping original implementation (--skip-original)")
        original_time = None

    # Benchmark optimized with different batch sizes
    for batch_size in batch_sizes:
        cache_size = 0 if args.no_cache else args.cache_size

        try:
            opt_time, opt_objects, cache_stats = benchmark_optimized(
                proposals_by_level=proposals_by_level,
                exp_dir=exp_dir,
                scene_path=scene_path,
                model_name=args.model,
                device=args.device,
                batch_size=batch_size,
                cache_size=cache_size,
                max_frames=args.max_frames
            )

            results.append({
                'method': 'Optimized',
                'batch_size': batch_size,
                'cache_size': cache_size,
                'time': opt_time,
                'objects': opt_objects,
                'speed': opt_objects / opt_time,
                'cache_hit_rate': cache_stats['hit_rate']
            })
        except Exception as e:
            logger.error(f"Optimized implementation (batch={batch_size}) failed: {e}")

    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)

    print("\n{:<15} {:<10} {:<10} {:<15} {:<15} {:<15}".format(
        "Method", "Batch", "Cache", "Time (s)", "Speed (obj/s)", "Speedup"
    ))
    print("-" * 80)

    baseline_time = original_time if original_time else results[0]['time']

    for r in results:
        speedup = baseline_time / r['time'] if r['time'] > 0 else 0
        cache_info = f"{r.get('cache_size', 0)}" if 'cache_size' in r else "N/A"

        print("{:<15} {:<10} {:<10} {:<15.1f} {:<15.2f} {:<15.1f}x".format(
            r['method'],
            r['batch_size'],
            cache_info,
            r['time'],
            r['speed'],
            speedup
        ))

    # Print cache statistics if available
    if not args.no_cache:
        print("\nCache Statistics:")
        for r in results:
            if 'cache_hit_rate' in r:
                print(f"  Batch {r['batch_size']}: {r['cache_hit_rate']:.1f}% hit rate")

    logger.info("="*80)


if __name__ == "__main__":
    main()
