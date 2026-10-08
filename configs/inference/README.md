# Inference Pipeline Configuration

This directory contains YAML configuration files for the My3DIS inference pipeline (Aggregation + OV-3DIS).

## Quick Start

### 1. Create Your Configuration

Copy the template and customize for your experiment:

```bash
cp configs/inference/pipeline_template.yaml configs/inference/my_experiment.yaml
```

Edit the two required paths in `my_experiment.yaml`:

```yaml
experiment:
  # Path to your My3DIS experiment directory
  exp_dir: ./outputs/experiments/test_1108_ssam4_filter500_fill500_propinf

  # Path to the corresponding MultiScan scene
  scene_path: data/multiscan/scene_00005_00
```

### 2. Run the Pipeline

```bash
python scripts/run_full_pipeline.py --config configs/inference/my_experiment.yaml
```

That's it! The pipeline will:
1. **Aggregation**: Convert 2D masks to 3D proposals with SigLIP features
2. **Inference**: Run open-vocabulary instance segmentation
3. Outputs will be saved to:
   - `{exp_dir}/aggregation_output/` - NPZ files with proposals and features
   - `{exp_dir}/inference_output/` - Predictions JSON

## Configuration Files

### `pipeline_template.yaml`
Clean template for creating your own configs. Copy this and modify the paths.

### `full_pipeline.yaml`
Fully documented reference config with all available parameters and descriptions.

### `base.yaml`, `hierarchical.yaml`, etc.
Legacy inference-only configs (kept for backward compatibility).

### `quick_eval.yaml`
Optimized config for fast evaluation on MultiScan dataset.

## Command-Line Overrides

You can override config values from the command line:

```bash
# Use different experiment directory
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --exp-dir /path/to/another/experiment

# Skip aggregation if already done
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --skip-aggregation

# Only run aggregation (skip inference)
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --skip-inference

# Override pooling mode
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --pooling-mode voting

# Override ground truth path for evaluation
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --gt-path /path/to/ground_truth

# Override GPU device
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --device cuda:1

# Combine multiple overrides
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --exp-dir /path/to/experiment \
    --pooling-mode voting \
    --device cuda:0 \
    --gt-path /path/to/gt
```

## Configuration Structure

The YAML config has four main sections:

### 1. Experiment Settings

```yaml
experiment:
  exp_dir: <path>              # My3DIS experiment directory
  scene_path: <path>           # MultiScan scene directory
  aggregation_output_dir: null # null = auto-generate
  inference_output_dir: null   # null = auto-generate
  levels: [2, 4, 6]
  device: cuda:0               # cuda, cuda:0, cuda:1, cpu
```

**Device Options:**
- `cuda` - Use default GPU (equivalent to `cuda:0`)
- `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3` - Use specific GPU
- `cpu` - Use CPU (slower but no GPU required)

### 2. Aggregation Stage

Controls 2D→3D aggregation and feature extraction:

```yaml
aggregation:
  enabled: true
  iou_threshold: 0.8           # Merging threshold
  area_fraction: 0.002         # Min area (1/500 of total points)
  extract_features: true       # Extract SigLIP features
  siglip_model: google/siglip-so400m-patch14-384
  pooling_mode: average        # average or voting
```

**Key Parameters:**
- `iou_threshold`: Higher = more aggressive merging (0.7-0.9 recommended)
- `area_fraction`: Minimum proposal size (default 1/500 ≈ 0.002)
- `pooling_mode`:
  - `voting`: Raw averaged features (preserves magnitude) - **Recommended for OV-3DIS**
    - Matches the behavior in `utils_ov_inference.py`
    - Preserves similarity scale information
  - `average`: L2-normalized features (standard for retrieval tasks)
    - Normalizes all features to unit length
    - Better for pure similarity ranking without magnitude

**Why `voting` is recommended:**
The OV-3DIS inference pipeline expects features that preserve their original magnitude, which helps with score calibration and multi-instance detection. Using `average` mode would normalize all features to the same length, losing important scale information.

### 3. Inference Stage

Controls open-vocabulary instance segmentation:

```yaml
inference:
  enabled: true
  strategy: independent        # independent or hierarchical

  retrieval:
    temperature: 0.2           # Lower = sharper similarity scores
    min_similarity: 0.2        # Minimum similarity threshold
    top_k_per_level: 50        # Max proposals per level

  pairing:
    containment_threshold: 0.5 # Min part-in-object ratio
    use_tree_pruning: true     # Use family tree for pruning

  nms:
    iou_threshold: 0.35        # Lower = preserve more instances
    keep_top_k: 200            # Max predictions per scene

  scoring:
    method: weighted_sum
    weights:
      object: 0.4
      part: 0.4
      geometric: 0.2
```

**Strategies:**
- `independent`: Retrieve objects and parts independently, then pair
  - Faster, better for multi-instance queries
  - Recommended for most use cases

- `hierarchical`: Coarse-to-fine retrieval using hierarchy
  - More computationally expensive
  - Better for exploiting hierarchical structure

### 4. Evaluation Stage (Optional)

```yaml
evaluation:
  enabled: false               # Enable for Search3D evaluation
  gt_path: <path>              # Path to ground truth annotations
  query_file: null             # null = use default queries
  num_workers: 4
```

## Input Requirements

Your My3DIS experiment directory should contain:

```
exp_dir/
├── level_2/
│   └── video_segments_L02.npz
├── level_4/
│   └── video_segments_L04.npz
├── level_6/
│   └── video_segments_L06.npz
└── relations/
    └── family_tree.json
```

The MultiScan scene directory should contain:

```
scene_path/
├── scene_XXXXX_XX.ply (or scene_XXXXX_XX_converted.ply)
├── outputs/
│   ├── color/
│   │   └── frame_*.jpg
│   └── depth/
│       └── frame_*.png
└── intrinsic/
    └── intrinsic_*.txt
```

## Output Structure

After running the pipeline:

```
exp_dir/
├── aggregation_output/
│   ├── proposal_data_level2.npz
│   ├── proposal_data_level4.npz
│   ├── proposal_data_level6.npz
│   ├── Proposal_relation.json
│   └── aggregation_metadata.json
├── inference_output/
│   ├── predictions.json
│   └── (evaluation results if enabled)
└── pipeline.log
```

### NPZ File Format

Each `proposal_data_levelX.npz` contains:

```python
{
  'proposal_masks': (P, N),    # Binary masks (P points, N proposals)
  'proposal_features': (N, D), # SigLIP features (N proposals, D dims)
  'proposal_ids': (N,)         # Object IDs from tracking
}
```

### Predictions Format

`predictions.json`:

```json
[
  {
    "query": "chair leg",
    "query_type": "part",
    "num_instances": 1,
    "score": 0.85,
    "object_level": "L2",
    "part_level": "L4",
    "object_id": 2050,
    "part_id": 4120,
    "mask_points": 1523
  },
  ...
]
```

## Troubleshooting

### "Video segments not found"

Make sure your experiment directory contains the correct level directories:
- `level_2/video_segments_L02.npz`
- `level_4/video_segments_L04.npz`
- `level_6/video_segments_L06.npz`

### "PLY file not found"

The scene directory should contain either:
- `{scene_name}.ply`, or
- `{scene_name}_converted.ply`

### "Family tree not found"

The pipeline will continue with a warning, but object-part pairing may be less effective. Generate family tree using:

```bash
PYTHONPATH=src python -m my3dis.family_tree_builder \
    --exp-dir <exp_dir> \
    --output <exp_dir>/relations/family_tree.json
```

### Out of memory

Reduce memory usage by:
1. Lowering `area_fraction` to filter more proposals
2. Reducing `top_k_per_level` in retrieval config
3. Setting `keep_top_k` to limit final predictions
4. Using CPU instead of GPU: `device: cpu`

### Slow feature extraction

Speed up SigLIP feature extraction by:
1. Using a smaller model: `siglip-base-patch16-384`
2. Reducing number of proposals (adjust `area_fraction`)
3. Skipping feature extraction and loading pre-computed features

## Advanced Usage

### Using Pre-computed Features

If you already ran aggregation, skip it and use existing outputs:

```bash
python scripts/run_full_pipeline.py \
    --config configs/inference/my_experiment.yaml \
    --skip-aggregation
```

### Custom Inference Queries

Modify `scripts/run_full_pipeline.py` to load queries from a file:

```python
# Load queries from file
with open('my_queries.txt', 'r') as f:
    queries = [line.strip().split(',') for line in f]
    # Format: "query_text,type,num_instances"
```

### Batch Processing Multiple Scenes

Create a script to loop over multiple configs:

```bash
for config in configs/inference/scene_*.yaml; do
    python scripts/run_full_pipeline.py --config $config
done
```

## Performance Tips

### For Speed:
- Use `independent` strategy (faster than `hierarchical`)
- Lower `top_k_per_level` (e.g., 30 instead of 50)
- Increase `min_similarity` threshold (e.g., 0.3)
- Set `keep_top_k` to limit NMS candidates

### For Quality:
- Use `hierarchical` strategy for better hierarchy exploitation
- Increase `top_k_per_level` (e.g., 100)
- Lower `min_similarity` threshold (e.g., 0.1)
- Use `use_tree_pruning: true` for better pairing
- Tune `scoring.weights` based on your task

### For Multi-Instance Queries:
- Lower `nms.iou_threshold` (e.g., 0.25)
- Increase `keep_top_k` (e.g., 300)
- Use `centroid_distance_ratio` to separate spatial duplicates

## See Also

- **Full documentation**: `./CLAUDE.md`
- **Aggregation guide**: `./scripts/AGGREGATION_INTEGRATION_SUMMARY.md`
- **Inference strategies**: `./src/my3dis/inference/`
- **Example scripts**: `./scripts/quick_eval_experiment.py`
