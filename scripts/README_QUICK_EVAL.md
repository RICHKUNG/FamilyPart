# Quick Evaluation Scripts

Fast evaluation pipeline for My3DIS experiments on MultiScan dataset with Search3D benchmark.

## Overview

These scripts provide a streamlined way to:
1. **Aggregate** multi-level proposals from My3DIS tracking outputs
2. **Extract** SigLIP features for proposals and query embeddings
3. **Infer** object-part pairs with multi-instance handling
4. **Evaluate** using Search3D metrics (mAP, mAP@50, mAP@25)

## Quick Start

### 1. Test Your Setup

```bash
python scripts/test_quick_eval.py \
  --exp-dir outputs/experiments/your_experiment \
  --dataset-root /path/to/multiscan
```

This checks:
- Required dependencies (transformers, torch, etc.)
- Experiment directory structure
- Dataset accessibility
- SigLIP model loading

### 2. Evaluate Single Experiment

```bash
python scripts/quick_eval_experiment.py \
  --exp-dir outputs/experiments/test_1105_ssam4_filter300_fill500_fix \
  --dataset-root data/multiscan \
  --gt-path /path/to/multiscan_annotations_search3d/ov_part_annotations \
  --output-dir /tmp/eval_results \
  --config configs/inference/quick_eval.yaml
```

**Output**: JSON file with mAP scores and detailed metrics

### 3. Batch Evaluate Multiple Experiments

```bash
python scripts/batch_eval_experiments.py \
  --exp-root outputs/experiments \
  --dataset-root data/multiscan \
  --gt-path /path/to/gt \
  --output-dir /tmp/batch_results \
  --num-workers 4
```

**Output**: Report with rankings, CSV table, and full JSON results

## Scripts

| Script | Purpose |
|--------|---------|
| `test_quick_eval.py` | Test setup and dependencies |
| `quick_eval_experiment.py` | Evaluate single experiment |
| `batch_eval_experiments.py` | Batch evaluate multiple experiments |
| `example_eval_usage.sh` | Interactive example usage |

## Configuration

Edit `configs/inference/quick_eval.yaml` to customize:

```yaml
# Key parameters for tuning
nms:
  iou_threshold: 0.35          # Lower = more instances preserved
  centroid_distance_ratio: 0.2 # Spatial separation threshold

retrieval:
  top_k_per_level: 50          # Proposals to retrieve per level
  min_similarity: 0.2          # Minimum retrieval score

scoring:
  weights:
    object: 0.4
    part: 0.4
    geometric: 0.2
```

## Expected Input Structure

Your experiment directory should contain:

```
exp_dir/
├── level_L2/
│   └── tracking/
│       └── object_segments_scale0.3x.npz
├── level_L4/
│   └── tracking/
│       └── object_segments_scale0.3x.npz
├── level_L6/
│   └── tracking/
│       └── object_segments_scale0.3x.npz
└── relations/
    └── family_tree.json (optional)
```

## Output Format

### Single Experiment
```json
{
  "scene_name": "scene_00005_00",
  "num_proposals": {"L2": 45, "L4": 123, "L6": 267},
  "num_predictions": 89,
  "metrics": {
    "mAP": 0.2345,
    "mAP@50": 0.3456,
    "mAP@25": 0.4567
  }
}
```

### Batch Evaluation
- `evaluation_report.txt`: Human-readable rankings
- `evaluation_results.csv`: Spreadsheet with metrics
- `evaluation_results.json`: Full structured data

## Implementation Status

### ✅ Complete
- Multi-level proposal aggregation
- SigLIP model integration
- Multi-instance NMS
- Search3D format output
- Parallel batch processing

### ⚠️ Placeholder
- **SigLIP feature extraction from RGB frames** (currently uses random features)
  - Needs: 2D projection, visibility checking, multi-view aggregation
  - For production use, implement based on `3DprojToSiglip/utils_new/feature_extraction.py`

## Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers pillow pandas tqdm

# For real evaluation (optional)
pip install git+https://github.com/aycatakmaz/search3d.git
```

## Tips for Better Results

**Higher Recall** (find more instances):
- Lower `nms.iou_threshold` to 0.25-0.30
- Increase `retrieval.top_k_per_level` to 100-200
- Lower `retrieval.min_similarity` to 0.15-0.20

**Higher Precision** (fewer false positives):
- Increase `nms.iou_threshold` to 0.40-0.45
- Increase `retrieval.min_similarity` to 0.25-0.30
- Stricter `pairing.containment_threshold`

**Multi-Instance Handling**:
- Lower `nms.iou_threshold`
- Increase `nms.centroid_distance_ratio`
- Enable `nms.use_soft_nms`

## Documentation

- **Full Guide**: [`docs/inference/QUICK_EVAL_GUIDE.md`](../docs/inference/QUICK_EVAL_GUIDE.md)
- **Inference System**: [`docs/inference/README.md`](../docs/inference/README.md)
- **Search3D**: https://github.com/aycatakmaz/search3d

## Troubleshooting

### "No proposals loaded"
- Check experiment directory structure
- Ensure `object_segments_*.npz` files exist in tracking directories

### "Search3D not available"
- Install with: `pip install git+https://github.com/aycatakmaz/search3d.git`
- Or continue with dummy metrics (0.0) for testing

### Low mAP scores
- Implement real feature extraction (currently using random features)
- Tune NMS parameters (see Tips section)
- Increase retrieval top-k and lower thresholds

### Out of memory
- Use `--device cpu` for CPU-only processing
- Reduce batch sizes in feature extraction
- Process experiments sequentially (`--num-workers 1`)

## Example Workflow

```bash
# 1. Test setup
python scripts/test_quick_eval.py \
  --exp-dir outputs/experiments/your_exp \
  --dataset-root /path/to/multiscan

# 2. Quick eval (single scene)
python scripts/quick_eval_experiment.py \
  --exp-dir outputs/experiments/your_exp \
  --dataset-root /path/to/multiscan \
  --gt-path /path/to/gt \
  --output-dir results/

# 3. Batch eval (all experiments)
python scripts/batch_eval_experiments.py \
  --exp-root outputs/experiments \
  --dataset-root /path/to/multiscan \
  --gt-path /path/to/gt \
  --output-dir results_batch/ \
  --num-workers 4

# 4. Analyze results
cat results_batch/evaluation_report.txt
# or open results_batch/evaluation_results.csv in spreadsheet
```

## Contact

For issues or questions about the evaluation pipeline, see:
- Project documentation: `docs/inference/`
- CLAUDE.md for general guidance
- GitHub issues (if applicable)
