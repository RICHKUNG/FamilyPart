#!/bin/bash
# Test script for Orphan Fix Phase 1-2 validation
# Tests: Dedup timing fix + Cascade filtering

set -e

echo "============================================"
echo "Orphan Fix Test - Phase 1-2 Validation"
echo "============================================"
echo ""
echo "This test will:"
echo "1. Run SSAM with cascade_filtering=true"
echo "2. Run SAM2 tracking with provenance tracking"
echo "3. Generate family tree and visualizations"
echo "4. Report orphan statistics"
echo ""

# Configuration
CONFIG_FILE="configs/multiscan/test_orphan_fix.yaml"
OUTPUT_BASE="./outputs/experiments/test_orphan_fix_1104"

echo "Step 1: Running workflow..."
echo "-------------------------------------------"
PYTHONPATH=src python3 -m my3dis.run_workflow --config "$CONFIG_FILE"

# Wait for completion
if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Analyzing results..."
    echo "-------------------------------------------"

    # Find the output directory
    SCENE_DIR=$(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "scene_*" | head -1)

    if [ -z "$SCENE_DIR" ]; then
        echo "ERROR: No scene directory found in $OUTPUT_BASE"
        exit 1
    fi

    echo "Scene directory: $SCENE_DIR"
    echo ""

    # Check if provenance trees exist
    echo "Checking provenance trees..."
    RELATIONS_DIR="$SCENE_DIR/relations"
    if [ -d "$RELATIONS_DIR" ]; then
        echo "  ✓ Relations directory found"
        ls -lh "$RELATIONS_DIR"/provenance_tree_*.json 2>/dev/null | awk '{print "   ", $NF}'
    else
        echo "  ✗ Relations directory not found"
    fi
    echo ""

    # Check family tree
    echo "Checking family tree..."
    FAMILY_TREE="$RELATIONS_DIR/family_tree.json"
    if [ -f "$FAMILY_TREE" ]; then
        echo "  ✓ Family tree found: $FAMILY_TREE"

        # Extract statistics using Python
        python3 << 'EOF'
import json
import sys

family_tree_path = "$FAMILY_TREE"
# Replace with actual path
family_tree_path = sys.argv[1] if len(sys.argv) > 1 else None

if family_tree_path:
    with open(family_tree_path) as f:
        tree = json.load(f)

    stats = tree.get('statistics', {})
    print(f"  Total objects: {stats.get('total_objects', 'N/A')}")
    print(f"  Total families: {stats.get('total_families', 'N/A')}")
    print(f"  Orphan count: {stats.get('orphan_count', 'N/A')}")
    print(f"  Orphan rate: {stats.get('orphan_rate', 'N/A'):.2%}" if isinstance(stats.get('orphan_rate'), (int, float)) else f"  Orphan rate: N/A")

    # Level breakdown
    print("")
    print("  Level breakdown:")
    for level_key in ['level_2', 'level_4', 'level_6']:
        if level_key in stats:
            level_stats = stats[level_key]
            print(f"    {level_key}: {level_stats.get('objects', 'N/A')} objects, {level_stats.get('orphans', 'N/A')} orphans")
EOF
python3 -c "
import json
with open('$FAMILY_TREE') as f:
    tree = json.load(f)
stats = tree.get('statistics', {})
print(f'  Total objects: {stats.get(\"total_objects\", \"N/A\")}')
print(f'  Total families: {stats.get(\"total_families\", \"N/A\")}')
print(f'  Orphan count: {stats.get(\"orphan_count\", \"N/A\")}')
orphan_rate = stats.get('orphan_rate', 0)
if isinstance(orphan_rate, (int, float)):
    print(f'  Orphan rate: {orphan_rate:.2%}')
print('')
print('  Level breakdown:')
for level_key in ['level_2', 'level_4', 'level_6']:
    if level_key in stats:
        level_stats = stats[level_key]
        print(f'    {level_key}: {level_stats.get(\"objects\", \"N/A\")} objects, {level_stats.get(\"orphans\", \"N/A\")} orphans')
"
    else
        echo "  ✗ Family tree not found"
    fi
    echo ""

    # Check for unresolved rejections in provenance trees
    echo "Checking unresolved rejections..."
    for prov_tree in "$RELATIONS_DIR"/provenance_tree_L*.json; do
        if [ -f "$prov_tree" ]; then
            level=$(basename "$prov_tree" | sed 's/provenance_tree_L\([0-9]*\)\.json/\1/')
            unresolved=$(python3 -c "import json; t=json.load(open('$prov_tree')); print(len(t.get('unresolved_rejections', [])))")
            echo "  Level $level: $unresolved unresolved rejections"
        fi
    done
    echo ""

    # Check visualizations
    echo "Checking family visualizations..."
    VIZ_DIR="$SCENE_DIR/visualizations"
    if [ -d "$VIZ_DIR" ]; then
        viz_count=$(find "$VIZ_DIR" -name "family_*.png" | wc -l)
        echo "  ✓ Visualizations directory found: $viz_count images"
        find "$VIZ_DIR" -name "family_*.png" | head -5 | awk '{print "   ", $0}'
    else
        echo "  ✗ Visualizations directory not found"
    fi
    echo ""

    echo "============================================"
    echo "Test completed successfully!"
    echo "============================================"
    echo ""
    echo "Output directory: $SCENE_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Review family tree statistics above"
    echo "2. Check visualizations in: $VIZ_DIR"
    echo "3. Compare orphan rate with baseline (expected <0.5%)"
    echo ""
else
    echo "ERROR: Workflow failed"
    exit 1
fi
