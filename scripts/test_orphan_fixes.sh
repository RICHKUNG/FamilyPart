#!/bin/bash
# Test script for orphan fixes (Phase 1-2)
# Usage: ./scripts/test_orphan_fixes.sh <scene_name>

set -e

SCENE=${1:-scene_00006_00}
TEST_OUTPUT_DIR="outputs/experiments/test_orphan_fix_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Testing Orphan Fixes (Phase 1-2)"
echo "Scene: $SCENE"
echo "Output: $TEST_OUTPUT_DIR"
echo "=========================================="
echo ""

# Set PYTHONPATH
export PYTHONPATH=$(pwd)/src

# Create test config
TEST_CONFIG="/tmp/test_orphan_fix.yaml"

cat > "$TEST_CONFIG" <<EOF
experiment:
  name: test_orphan_fix
  dataset_root: data/multiscan
  scenes:
    - $SCENE

  levels: [2, 4, 6]
  frames: "0:15000:50"  # Quick test

  output_root: $TEST_OUTPUT_DIR
  parallel_scenes: 1

stages:
  ssam:
    enabled: true
    ssam_frequency: 4
    min_area: 1000
    stability_threshold: 0.9
    cascade_filtering: true  # NEW: Enable cascade filtering
    max_masks_per_level: 0

  filter:
    enabled: false  # Skip filter stage (cascade filtering in SSAM)

  tracker:
    enabled: true
    max_propagate_frames: 30
    iou_threshold: 0.6
    downscale_ratio: 0.3

  report:
    enabled: false

logging:
  level: INFO
EOF

echo "📝 Test config created: $TEST_CONFIG"
echo ""
cat "$TEST_CONFIG"
echo ""

# Run pipeline
echo "=========================================="
echo "Running pipeline..."
echo "=========================================="

python -m my3dis.run_workflow --config "$TEST_CONFIG" || {
    echo "❌ Pipeline failed!"
    exit 1
}

# Analyze results
echo ""
echo "=========================================="
echo "Analyzing results..."
echo "=========================================="

SCENE_DIR="$TEST_OUTPUT_DIR/$SCENE"

if [ ! -d "$SCENE_DIR" ]; then
    echo "❌ Scene directory not found: $SCENE_DIR"
    exit 1
fi

# Check provenance trees
echo ""
echo "📊 Provenance Tree Statistics:"
echo "----------------------------------------"

for LEVEL in 2 4 6; do
    PROV_TREE="$SCENE_DIR/relations/provenance_tree_L${LEVEL}.json"

    if [ -f "$PROV_TREE" ]; then
        STATS=$(python3 -c "
import json
with open('$PROV_TREE') as f:
    tree = json.load(f)
    parent_map = tree.get('parent_map', {})
    orphan_ids = tree.get('orphan_ids', [])
    metadata = tree.get('metadata', {})

    print(f\"Level {metadata.get('level', $LEVEL)}:\")
    print(f\"  Objects: {len(parent_map)}\")
    print(f\"  Orphans: {len(orphan_ids)}\")

    # Check for unresolved rejections (NEW)
    unresolved = metadata.get('unresolved_rejections', [])
    if unresolved:
        print(f\"  Unresolved rejections: {len(unresolved)}\")
" 2>/dev/null)
        echo "$STATS"
    else
        echo "Level $LEVEL: No provenance tree found"
    fi
done

# Check family tree
echo ""
echo "📊 Family Tree Statistics:"
echo "----------------------------------------"

FAMILY_TREE="$SCENE_DIR/relations/family_tree.json"

if [ -f "$FAMILY_TREE" ]; then
    python3 -c "
import json
with open('$FAMILY_TREE') as f:
    tree = json.load(f)
    stats = tree['statistics']

    print(f\"Total objects: {stats['total_objects']}\")
    print(f\"Total families: {stats['total_families']}\")
    print(f\"Orphan count: {stats['orphan_count']}\")
    print(f\"Objects per level: {stats['objects_per_level']}\")
    print(f\"\")

    # Analyze family roots
    families = tree['families']
    root_levels = {}
    for family in families:
        root_id = family['root_id']
        root_obj = tree['objects'][str(root_id)]
        level = root_obj['level']
        root_levels[f'L{level}'] = root_levels.get(f'L{level}', 0) + 1

    print(f\"Family roots by level: {root_levels}\")

    # Expected: Only L2 roots after fixes
    if root_levels.get('L4', 0) > 0 or root_levels.get('L6', 0) > 0:
        print(f\"⚠️  WARNING: Found L4/L6 family roots (orphans still exist)\")
    else:
        print(f\"✓ All families rooted at L2 (no L4/L6 orphans!)\")
"
else
    echo "❌ Family tree not found"
fi

# Check cascade filtering statistics (if available in manifest)
echo ""
echo "📊 Cascade Filtering Statistics:"
echo "----------------------------------------"

# Look for cascade filtering stats in level manifests
for LEVEL in 2 4 6; do
    MANIFEST="$SCENE_DIR/level_${LEVEL}/raw/manifest.json"

    if [ -f "$MANIFEST" ]; then
        python3 -c "
import json
try:
    with open('$MANIFEST') as f:
        manifest = json.load(f)
        cascade = manifest.get('cascade_filtering', {})

        if cascade.get('enabled'):
            stats = cascade.get('statistics', {})
            print(f\"Level $LEVEL:\")
            print(f\"  Total input: {stats.get('total_input', 'N/A')}\")
            print(f\"  Passed: {stats.get('passed', 'N/A')}\")
            print(f\"  Rejected: {stats.get('rejected', 'N/A')}\")
            print(f\"  Cascade rejected: {stats.get('breakdown', {}).get('cascade_rejected', 'N/A')}\")
            print(f\"\")
except:
    pass
" 2>/dev/null || echo "Level $LEVEL: No cascade filtering data"
    fi
done

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Scene directory: $SCENE_DIR"
echo ""
echo "Key files:"
echo "  - Relations: $SCENE_DIR/relations/"
echo "  - Family tree: $FAMILY_TREE"
echo "  - Workflow summary: $SCENE_DIR/workflow_summary.json"
echo ""

# Compare with baseline (if exists)
BASELINE_DIR="outputs/experiments/test_1030_1_ssam4_filter1000/$SCENE"

if [ -d "$BASELINE_DIR" ] && [ -f "$BASELINE_DIR/relations/family_tree.json" ]; then
    echo "📊 Comparison with Baseline:"
    echo "----------------------------------------"

    python3 -c "
import json

# Load baseline
with open('$BASELINE_DIR/relations/family_tree.json') as f:
    baseline = json.load(f)
    baseline_stats = baseline['statistics']

# Load new results
with open('$FAMILY_TREE') as f:
    new = json.load(f)
    new_stats = new['statistics']

print(f\"Metric                  | Baseline | New     | Change\")
print(f\"------------------------|----------|---------|--------\")
print(f\"Total objects           | {baseline_stats['total_objects']:8} | {new_stats['total_objects']:7} | {new_stats['total_objects'] - baseline_stats['total_objects']:+7}\")
print(f\"Total families          | {baseline_stats['total_families']:8} | {new_stats['total_families']:7} | {new_stats['total_families'] - baseline_stats['total_families']:+7}\")
print(f\"Orphan count            | {baseline_stats['orphan_count']:8} | {new_stats['orphan_count']:7} | {new_stats['orphan_count'] - baseline_stats['orphan_count']:+7}\")
print(f\"\")

# Orphan rate
baseline_rate = baseline_stats['orphan_count'] / baseline_stats['total_objects'] * 100
new_rate = new_stats['orphan_count'] / new_stats['total_objects'] * 100
print(f\"Orphan rate: {baseline_rate:.2f}% → {new_rate:.2f}% (Δ {new_rate - baseline_rate:+.2f}%)\")

# Check if L4/L6 roots eliminated
baseline_families = baseline['families']
new_families = new['families']

baseline_l4_roots = sum(1 for f in baseline_families if baseline['objects'][str(f['root_id'])]['level'] == 4)
baseline_l6_roots = sum(1 for f in baseline_families if baseline['objects'][str(f['root_id'])]['level'] == 6)

new_l4_roots = sum(1 for f in new_families if new['objects'][str(f['root_id'])]['level'] == 4)
new_l6_roots = sum(1 for f in new_families if new['objects'][str(f['root_id'])]['level'] == 6)

print(f\"\")
print(f\"L4/L6 family roots:\")
print(f\"  Baseline: L4={baseline_l4_roots}, L6={baseline_l6_roots}\")
print(f\"  New:      L4={new_l4_roots}, L6={new_l6_roots}\")

if new_l4_roots == 0 and new_l6_roots == 0:
    print(f\"  ✓ SUCCESS: All L4/L6 orphan roots eliminated!\")
else:
    print(f\"  ⚠️  Some L4/L6 roots remain\")
"
fi

echo ""
echo "=========================================="
echo "✓ Test completed!"
echo "=========================================="
