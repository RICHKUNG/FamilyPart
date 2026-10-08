#!/usr/bin/env python3
"""
Verify that gap-fill masks in existing run have unique_id.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from my3dis.raw_archive import RawCandidateArchiveReader


def verify_run(run_dir):
    """Verify gap-fill masks in a run directory."""
    print("=" * 60)
    print(f"Verifying: {run_dir}")
    print("=" * 60)

    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"✗ Directory not found: {run_dir}")
        return False

    # Check all levels
    levels = [2, 4, 6]
    all_good = True

    for level in levels:
        level_raw = run_path / f"level_{level}" / "raw"
        if not level_raw.exists():
            print(f"\n⚠ Level {level} raw directory not found, skipping")
            continue

        print(f"\nChecking Level {level}:")
        print(f"  Raw directory: {level_raw}")

        try:
            reader = RawCandidateArchiveReader(str(level_raw))
            frame_indices = reader.frame_indices()
            print(f"  Found {len(frame_indices)} frames")

            total_candidates = 0
            total_gap_fills = 0
            gap_fills_with_uid = 0
            missing_uid_items = []

            for frame_idx in frame_indices:
                frame_data = reader.load_frame(frame_idx)
                if not frame_data or 'meta' not in frame_data:
                    continue

                candidates = frame_data['meta'].get('candidates', [])
                total_candidates += len(candidates)

                for cand_idx, cand in enumerate(candidates):
                    source = cand.get('source')
                    frame_name = cand.get('frame_name', '')

                    # Check if this is a gap-fill mask
                    is_gap_fill = (source == 'gap_fill' or frame_name.startswith('gap_'))

                    if is_gap_fill:
                        total_gap_fills += 1
                        unique_id = cand.get('unique_id')

                        if unique_id:
                            gap_fills_with_uid += 1
                        else:
                            missing_uid_items.append({
                                'frame': frame_idx,
                                'index': cand_idx,
                                'frame_name': frame_name,
                                'source': source,
                            })

            print(f"  Total candidates: {total_candidates}")
            print(f"  Gap-fill masks: {total_gap_fills}")
            print(f"  With unique_id: {gap_fills_with_uid}")
            print(f"  Missing unique_id: {len(missing_uid_items)}")

            if missing_uid_items:
                print(f"\n  ✗ Gap-fills without unique_id:")
                for item in missing_uid_items[:5]:  # Show first 5
                    print(f"     - Frame {item['frame']} [{item['index']}]: "
                          f"frame_name={item['frame_name']}, source={item['source']}")
                if len(missing_uid_items) > 5:
                    print(f"     ... and {len(missing_uid_items) - 5} more")
                all_good = False
            elif total_gap_fills > 0:
                print(f"  ✓ All {total_gap_fills} gap-fill masks have unique_id")

        except Exception as e:
            print(f"  ✗ Error reading level {level}: {e}")
            import traceback
            traceback.print_exc()
            all_good = False

    return all_good


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Use the most recent test run
        run_dir = "./outputs/experiments/test_v7_246_ssam4_filter10_fill50_propinf_1114/scene_00005_00"

    success = verify_run(run_dir)

    print("\n" + "=" * 60)
    if success:
        print("✓ VERIFICATION PASSED")
    else:
        print("✗ VERIFICATION FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)
