#!/bin/bash
# Example usage script for quick evaluation

# Set your paths here
EXP_DIR="./outputs/experiments/test_1105_ssam4_filter300_fill500_fix"
DATASET_ROOT="data/multiscan"
GT_PATH="/path/to/multiscan_annotations_search3d/ov_part_annotations"
OUTPUT_DIR="/tmp/eval_results"

echo "==================================================================="
echo "Quick Evaluation Example Usage"
echo "==================================================================="
echo ""

# Step 0: Test setup (optional)
echo "Step 0: Testing setup (optional)..."
echo "Command:"
echo "  python scripts/test_quick_eval.py \\"
echo "    --exp-dir $EXP_DIR \\"
echo "    --dataset-root $DATASET_ROOT"
echo ""
read -p "Run test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/test_quick_eval.py \
        --exp-dir "$EXP_DIR" \
        --dataset-root "$DATASET_ROOT"
    echo ""
fi

# Step 1: Single experiment evaluation
echo "==================================================================="
echo "Step 1: Evaluate Single Experiment"
echo "==================================================================="
echo ""
echo "Command:"
echo "  python scripts/quick_eval_experiment.py \\"
echo "    --exp-dir $EXP_DIR \\"
echo "    --dataset-root $DATASET_ROOT \\"
echo "    --gt-path $GT_PATH \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --config configs/inference/quick_eval.yaml"
echo ""
read -p "Run single evaluation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/quick_eval_experiment.py \
        --exp-dir "$EXP_DIR" \
        --dataset-root "$DATASET_ROOT" \
        --gt-path "$GT_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --config configs/inference/quick_eval.yaml
    echo ""
fi

# Step 2: Batch evaluation (if applicable)
echo "==================================================================="
echo "Step 2: Batch Evaluation (Multiple Experiments)"
echo "==================================================================="
echo ""
echo "Command:"
echo "  python scripts/batch_eval_experiments.py \\"
echo "    --exp-root $(dirname $EXP_DIR) \\"
echo "    --dataset-root $DATASET_ROOT \\"
echo "    --gt-path $GT_PATH \\"
echo "    --output-dir ${OUTPUT_DIR}_batch \\"
echo "    --config configs/inference/quick_eval.yaml \\"
echo "    --num-workers 4"
echo ""
read -p "Run batch evaluation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/batch_eval_experiments.py \
        --exp-root "$(dirname "$EXP_DIR")" \
        --dataset-root "$DATASET_ROOT" \
        --gt-path "$GT_PATH" \
        --output-dir "${OUTPUT_DIR}_batch" \
        --config configs/inference/quick_eval.yaml \
        --num-workers 4
    echo ""
fi

echo "==================================================================="
echo "Done! Check results in:"
echo "  Single: $OUTPUT_DIR"
echo "  Batch:  ${OUTPUT_DIR}_batch"
echo "==================================================================="
