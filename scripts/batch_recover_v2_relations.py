#!/usr/bin/env python3
"""
Batch recovery script for v2_* experiments.

Automatically finds all v2 experiment runs and recovers their parent-child relations.

Usage:
    python scripts/batch_recover_v2_relations.py \
        --experiments-root ./outputs/experiments

Author: Rich Kung
Created: 2025-10-21
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from my3dis.recover_relations import recover_legacy_relations

LOGGER = logging.getLogger(__name__)


def find_v2_experiment_runs(experiments_root: Path) -> List[Path]:
    """
    Find all v2_* experiment run directories.

    Returns:
        List of paths like: experiments_root/v2_*/scene_*/
        (v2 experiments have scene dirs as run dirs directly)
    """
    candidates: List[Path] = []

    for experiment_dir in experiments_root.glob('v2_*'):
        if not experiment_dir.is_dir():
            continue

        # Find scene directories (which are run directories in v2 layout)
        for scene_dir in experiment_dir.glob('scene_*'):
            if not scene_dir.is_dir():
                continue

            # Check if scene_dir itself has level_* directories
            level_dirs = list(scene_dir.glob('level_*'))
            if level_dirs:
                candidates.append(scene_dir)
                continue

            # Fallback: check subdirectories (for mixed layouts)
            for run_dir in scene_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                level_dirs = list(run_dir.glob('level_*'))
                if level_dirs:
                    candidates.append(run_dir)

    return sorted(candidates)


def check_needs_recovery(run_dir: Path) -> bool:
    """Check if run directory needs relation recovery."""
    # If relations.json already exists and has data, skip
    relations_json = run_dir / 'relations.json'
    if relations_json.exists():
        try:
            import json
            with relations_json.open('r') as f:
                data = json.load(f)

            # Check both old and new formats
            if 'objects' in data and len(data['objects']) > 0:
                return False  # Old format with data
            elif 'hierarchy' in data and any(len(v) > 0 for v in data['hierarchy'].values()):
                return False  # New format with data
            # If relations.json exists but is empty, needs recovery
        except Exception:
            pass  # If can't read, assume needs recovery

    # Check if any level has tree.json
    for level_dir in run_dir.glob('level_*'):
        tree_json = level_dir / 'relations' / 'tree.json'
        if tree_json.exists():
            return False  # Already has relations

    return True


def infer_levels(run_dir: Path) -> List[int]:
    """Infer levels from level_* directories."""
    levels: List[int] = []
    for level_dir in run_dir.glob('level_*'):
        if level_dir.is_dir():
            try:
                level = int(level_dir.name.split('_')[1])
                levels.append(level)
            except (ValueError, IndexError):
                pass
    return sorted(levels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch recover relations for all v2 experiments"
    )
    parser.add_argument(
        '--experiments-root',
        type=str,
        default='./outputs/experiments',
        help='Root directory containing v2_* experiments',
    )
    parser.add_argument(
        '--containment-threshold',
        type=float,
        default=0.95,
        help='Minimum containment ratio for parent-child (default: 0.95)',
    )
    parser.add_argument(
        '--mask-scale-ratio',
        type=float,
        default=0.3,
        help='Mask downscale ratio (default: 0.3)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only list experiments, do not process',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of experiments to process (for testing)',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    experiments_root = Path(args.experiments_root)
    if not experiments_root.exists():
        LOGGER.error("Experiments root not found: %s", experiments_root)
        return

    LOGGER.info("Scanning for v2 experiments in: %s", experiments_root)
    all_runs = find_v2_experiment_runs(experiments_root)
    LOGGER.info("Found %d total run directories", len(all_runs))

    # Filter to only those needing recovery
    runs_to_process = [run for run in all_runs if check_needs_recovery(run)]
    LOGGER.info("Need recovery: %d runs", len(runs_to_process))

    if args.limit:
        runs_to_process = runs_to_process[:args.limit]
        LOGGER.info("Limited to first %d runs", args.limit)

    if args.dry_run:
        print("\n=== Dry run: would process ===")
        for run_dir in runs_to_process:
            levels = infer_levels(run_dir)
            print(f"  {run_dir} (levels: {levels})")
        return

    # Process each run
    success_count = 0
    fail_count = 0

    for idx, run_dir in enumerate(runs_to_process, 1):
        levels = infer_levels(run_dir)
        if not levels:
            LOGGER.warning("[%d/%d] Skipping %s (no levels found)", idx, len(runs_to_process), run_dir)
            continue

        LOGGER.info(
            "[%d/%d] Processing %s (levels: %s)",
            idx,
            len(runs_to_process),
            run_dir.relative_to(experiments_root),
            levels,
        )

        try:
            relations_path = recover_legacy_relations(
                experiment_dir=run_dir,
                levels=levels,
                containment_threshold=args.containment_threshold,
                mask_scale_ratio=args.mask_scale_ratio,
            )
            LOGGER.info("✅ Success → %s", relations_path)
            success_count += 1
        except Exception as exc:
            LOGGER.error("❌ Failed: %s", exc, exc_info=args.verbose)
            fail_count += 1

    LOGGER.info(
        "\n=== Summary ===\nTotal: %d | Success: %d | Failed: %d",
        len(runs_to_process),
        success_count,
        fail_count,
    )


if __name__ == '__main__':
    main()
