#!/usr/bin/env python3
"""
Verify Search3D Ground Truth setup for My3DIS evaluation.

This script checks:
1. GT directory structure
2. Required files present
3. File formats correct
4. Evaluation code availability
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple


def check_gt_structure(gt_root: Path) -> Tuple[bool, List[str]]:
    """Check if GT directory has correct structure."""
    issues = []

    # Check main directory
    if not gt_root.exists():
        issues.append(f"❌ GT root not found: {gt_root}")
        return False, issues

    print(f"✓ Found GT root: {gt_root}")

    # Check for multiscan_data_search3d
    data_dir = gt_root / "multiscan_data_search3d"
    if not data_dir.exists():
        issues.append(f"❌ Missing: {data_dir}")
        return False, issues

    print(f"✓ Found data directory: {data_dir}")

    # Check annotations
    annotations_dir = data_dir / "multiscan_annotations_search3d"
    if not annotations_dir.exists():
        issues.append(f"❌ Missing annotations: {annotations_dir}")
        return False, issues

    print(f"✓ Found annotations directory")

    # Check ov_part_annotations (most important)
    ov_part_dir = annotations_dir / "ov_part_annotations"
    if not ov_part_dir.exists():
        issues.append(f"❌ Missing OV part annotations: {ov_part_dir}")
        return False, issues

    # Count annotation files
    annotation_files = list(ov_part_dir.glob("*_obj_part_inst.txt"))
    print(f"✓ Found {len(annotation_files)} OV part annotation files")

    if len(annotation_files) == 0:
        issues.append(f"❌ No annotation files found in {ov_part_dir}")
        return False, issues

    # Check PLY files
    ply_dir = data_dir / "multiscan_test_plys_only"
    if ply_dir.exists():
        ply_files = list(ply_dir.glob("*.ply"))
        print(f"✓ Found {len(ply_files)} PLY files")
    else:
        print(f"⚠ PLY directory not found (optional): {ply_dir}")

    # Check constants file
    constants_file = data_dir / "multiscan_search3d_constants.py"
    if constants_file.exists():
        print(f"✓ Found constants file")
    else:
        print(f"⚠ Constants file not found (optional): {constants_file}")

    # Sample annotation file
    if annotation_files:
        sample_file = annotation_files[0]
        with open(sample_file, 'r') as f:
            lines = f.readlines()
        print(f"\n📄 Sample annotation: {sample_file.name}")
        print(f"   - Total lines: {len(lines)}")
        print(f"   - First few labels: {' '.join(lines[:5]).strip()}")

    return True, issues


def check_eval_code(project_root: Path) -> Tuple[bool, List[str]]:
    """Check if evaluation code is available."""
    issues = []

    eval_dir = project_root / "eval" / "search3d"
    if not eval_dir.exists():
        issues.append(f"❌ Evaluation code not found: {eval_dir}")
        return False, issues

    print(f"\n✓ Found evaluation directory: {eval_dir}")

    # Check required files
    required_files = [
        "eval_semantic_instance_parts_OV.py",
        "util.py",
        "util_3d.py"
    ]

    for filename in required_files:
        filepath = eval_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            issues.append(f"❌ Missing: {filename}")

    # Check data directory
    data_dir = eval_dir / "data"
    if data_dir.exists():
        constants_file = data_dir / "multiscan_search3d_constants.py"
        test_scenes_file = data_dir / "test_scenes_multiscan.txt"

        if constants_file.exists():
            print(f"  ✓ data/multiscan_search3d_constants.py")
        if test_scenes_file.exists():
            with open(test_scenes_file, 'r') as f:
                scenes = [line.strip() for line in f if line.strip()]
            print(f"  ✓ data/test_scenes_multiscan.txt ({len(scenes)} scenes)")

    return len(issues) == 0, issues


def check_test_scene(gt_root: Path, scene_name: str = "scene_00005_00") -> Tuple[bool, List[str]]:
    """Check if test scene exists in GT."""
    issues = []

    annotation_file = (
        gt_root / "multiscan_data_search3d" / "multiscan_annotations_search3d" /
        "ov_part_annotations" / f"{scene_name}_obj_part_inst.txt"
    )

    if annotation_file.exists():
        print(f"\n✓ Test scene annotation found: {scene_name}")

        # Read first few lines
        with open(annotation_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()[:100]]

        # Count non-zero labels
        non_zero = [l for l in labels if l != '0']
        print(f"  - Sample size: {len(labels)} points")
        print(f"  - Annotated: {len(non_zero)} points ({len(non_zero)/len(labels)*100:.1f}%)")

        if non_zero:
            print(f"  - Example labels: {', '.join(non_zero[:5])}")
    else:
        issues.append(f"❌ Test scene not found: {scene_name}")
        return False, issues

    return True, issues


def main():
    print("="*60)
    print("Search3D Ground Truth Setup Verification")
    print("="*60)

    project_root = Path(__file__).parent.parent
    gt_root = project_root / "data" / "search3d_gt"

    all_passed = True
    all_issues = []

    # Check 1: GT structure
    print("\n[Check 1/3] Verifying GT directory structure...")
    passed, issues = check_gt_structure(gt_root)
    if not passed:
        all_passed = False
        all_issues.extend(issues)
        print(f"\n❌ GT structure check failed:")
        for issue in issues:
            print(f"  {issue}")

    # Check 2: Evaluation code
    print("\n[Check 2/3] Verifying evaluation code...")
    passed, issues = check_eval_code(project_root)
    if not passed:
        all_passed = False
        all_issues.extend(issues)
        print(f"\n❌ Evaluation code check failed:")
        for issue in issues:
            print(f"  {issue}")

    # Check 3: Test scene
    print("\n[Check 3/3] Verifying test scene...")
    passed, issues = check_test_scene(gt_root)
    if not passed:
        all_passed = False
        all_issues.extend(issues)
        print(f"\n❌ Test scene check failed:")
        for issue in issues:
            print(f"  {issue}")

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! GT setup is complete.")
        print("\nYou can now run evaluation with:")
        gt_path = gt_root / "multiscan_data_search3d" / "multiscan_annotations_search3d" / "ov_part_annotations"
        print(f"\n  python scripts/quick_eval_experiment.py \\")
        print(f"    --exp-dir outputs/experiments/YOUR_EXP/scene_XXXXX_XX \\")
        print(f"    --dataset-root data/multiscan \\")
        print(f"    --gt-path {gt_path} \\")
        print(f"    --output-dir eval_results")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nIssues found:")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nSee: scripts/DOWNLOAD_GT_INSTRUCTIONS.md for setup guide")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
