# FamilyPart: Training-Free Open-Vocabulary 3D Part Segmentation

**Hsiang-Yu (Rich) Kung** · Advisor: Prof. Min Sun · Vision Science Lab, National Tsing Hua University · 2025

[**Report (PDF)**](report_B611.pdf) · [**Method**](#method-overview) · [**Results**](#experimental-results) · [**Failure Analysis**](#key-failure-modes)

## TL;DR

- **Goal:** find object *parts* in 3D scenes from free-text queries (e.g. "the handle of the drawer") without training any model.
- **Method:** build parent → child → grandchild masks with Semantic-SAM, track them with SAM2, then lift them to 3D and retrieve parts with SigLIP / CLIP.
- **Implementation:** an RGB-D-to-3D pipeline covering multilevel mask generation, provenance tracking, proposal aggregation and text-query retrieval.
- **Finding:** temporal granularity drift can break object–part relationships, and the reported AP remains below the cited Search3D reference. See [Results](#experimental-results) and [Failure Analysis](#key-failure-modes).

![FamilyPart pipeline](assets/fig1_pipeline.jpg)

<p align="center">
  <img src="assets/fig2_family_example.jpg" alt="One frame's family tree: original frame, one L2 instance, its seven L4 children and five L6 grandchildren" width="100%">
  <br><sub><b>What it builds:</b> one frame's family tree. Left to right: the frame, one L2 instance, its seven L4 children and five L6 grandchildren.</sub>
</p>

<p align="center">
  <img src="assets/fig6_granularity_drift.jpg" alt="Granularity drift: the chair's tracked mask no longer matches the granularity of the Semantic-SAM prompt" width="100%">
  <br><sub><b>Where it breaks:</b> after SAM2 tracking, the chair's mask drifts away from the granularity of its Semantic-SAM prompt. See <a href="#key-failure-modes">Failure Analysis</a>.</sub>
</p>

---

## Research Motivation

Traditional 3D instance segmentation approaches operate at a single granularity level, missing the hierarchical nature of real-world scenes. In more complex scenarios, it is crucial to identify specific parts of objects. However, conventional 3D part-level instance segmentation pipelines are often constrained to a closed set of objects or parts, causing them to lose functionality outside of their training datasets.

**Key Challenge**:
1. Maintain consistent identity across frames at multiple granularity levels
2. Track which fine-level masks belong to which coarse-level objects
3. Handle deduplication when the same object is proposed at different levels
4. Preserve complete provenance from 2D masks → tracked objects → 3D proposals

---

## Method Overview

### Stage 1: Multi-level Class-agnostic Family Tree Construction with Semantic-SAM

**Solo Photo Method**:
1. Prompt Semantic-SAM at three levels (e.g., L2=coarse, L4=medium, L6=fine)
2. For each L2 instance, conduct logical AND operation between original image and mask, padding black pixels as background → "solo photo"
3. Use solo photos as input and prompt Semantic-SAM at L4 to segment children
4. Output binary masks of children are ANDed with parents, ensuring children masks are always inside parents
5. Discard masks identical to parents or whose area is below threshold
6. Apply same procedure for L6 instances
7. Construct entire family tree and assign unique mask IDs via lookup table

![A family of instances in a single frame](assets/fig2_family_example.png)
*A family of instances in one frame. Left to right: original frame, one L2 instance, its seven L4 children, and five L6 grandchildren.*

**Gap Filling**:
- Extract connected pixel regions not covered by any highest-level mask
- Record as independent mask if area exceeds threshold
- Only applied at highest level to avoid missing objects

![Gap filling before and after](assets/fig3_gap_filling.jpg)
*Before and after gap filling. The rightmost image has gap filling applied.*

### Stage 2: Provenance-aware SAM2 Tracking and Deduplication

**Window-Based Method**:
- Sampling strategy with propagation stride of 50 (retain one frame every 50 frames)
- For each Semantic-SAM prompt frame, construct subsequence of length 2F+1 (F=30)
- Run SAM2 propagation within each overlapping window
- SAM2 maintains internal memory across windows

![Window-based tracking](assets/fig5_window_tracking.jpg)
*Window-based tracking at a single level. Each Semantic-SAM prompt frame is deduplicated against existing tracks, then propagated F frames in both directions.*

![Semantic-SAM vs SAM2 tracking](assets/fig4_ssam_vs_sam2.jpg)
*One frame: Semantic-SAM segments (middle) and the result after SAM2 tracking (right) at a single level.*

**Deduplication and Virtual Children**:
- Use Semantic-SAM masks as prompts for SAM2 video propagation
- Compare SAM2-tracked masks with Semantic-SAM masks using IoU
- If IoU > threshold, consider object already tracked
- **Virtual children mechanism**: Rejected Semantic-SAM candidates preserved at provenance level
  - System uses mask ID to determine which existing SAM2 object to merge into
  - Updates lookup table with virtual child relationships
  - O(1) time traceability to parent object
  - Records provenance for rejected candidates; this does not prevent the tracking granularity failures discussed below

### Stage 3: 3D Proposal Aggregation and Feature Extraction

**Family-aware Proposal Merging**:
- Back-project 2D masklets into 3D space using camera intrinsic/extrinsic matrices
- Aggregate per-frame 3D masks by taking spatial union
- Apply minimum volume threshold to filter small proposals
- Merge overlapping proposals using IoU threshold of 0.5:
  - Top-level objects: merge and reassign child nodes to same parent
  - Lower-level nodes: merge restricted to siblings within same family tree

**3D Proposal Feature Extraction**:
- Use LAION-CLIP (768-dim) or SigLIP (1152-dim) encoders
- Multi-scale cropping (3 scales) by expanding bounding box outward
- L₂ normalization followed by averaging
- Temporal Gaussian weighting + bounding-box area weighting:
  - `wₖ = exp(-((k-c)/σ)²) × √Aₖ`
  - `c = (K-1)/2`, `σ = K/4`
  - The report normalizes these weights with `αₖ = exp(wₖ) / Σⱼ exp(wⱼ)`, uses them to pool the features, and applies final L₂ normalization (Section 3.3, printed p. 9)
- Template-based text encoding: "a {part} of {object} in a room"

### Stage 4: Open-Vocabulary Query Inference

Three retrieval strategies are designed:

1. **Hierarchical Strategy** (Top-down family-tree guided):
   - Retrieve top-K₁ L2 candidates with highest similarity to object query
   - Collect L4 children if similarity exceeds threshold
   - Search descendants (L4/L6) for top-K₂ part candidates
   - Apply Cross-path NMS (only part proposals as NMS masks)
   - Sensitive to quality of highest-level proposals

2. **Exhaustive Pairing Strategy** (Enumerate all combinations):
   - Consider six (object, part) level combinations: (L2,L2), (L2,L4), (L2,L6), (L4,L4), (L4,L6), (L6,L6)
   - Retain top-K pairs with highest similarity scores
   - Validate pairings in 3D space, discard non-overlapping
   - Apply Cross-path NMS
   - Higher pairing complexity but bypasses family tree limitations

3. **Independent Strategy** (Decouple object and part retrieval):
   - Match highest level (L2) against object query
   - Match lower levels (L4/L6) against part query
   - Compute geometric score based on IoU and size ratio
   - Combine with retrieval scores via weighted average
   - Highest mean mAP, AP50 and AP25 among the three strategies in the report's seven-experiment comparison (Table 3)

---

## Experimental Results

### Dataset and Metrics
- **Dataset**: MultiScan with Search3D benchmark
  - 155 object + 15 part categories
  - 47 (object, part) pairs
  - 41 scenes (4 without ground-truth annotations)
- **Metric**: Mean Average Precision (mAP) under Search3D protocol. All AP values below are percentages, as labeled in the report (e.g. 1.0 means 1.0%).

### Reported Summary

The following values are transcribed from Section 4.2 of the [report](report_B611.pdf) (printed p. 12 / PDF p. 16).

| Configuration / source | mAP (%) | AP50 (%) | AP25 (%) |
|------------------------|---------|----------|----------|
| FamilyPart: L1/3/5, min. area=100, SigLIP, Independent (Section 4.2 prose) | 1.0 | 3.4 | 14.7 |
| Search3D [1] (baseline cited in the report) | 7.9 | 14.5 | 31.5 |

**Unresolved discrepancy in the report:** Table 2 lists **1.1 / 3.1 / 11.8** for SigLIP with the same stated L1/3/5, min. area=100 and Independent labels. It also lists **1.2 / 3.7 / 11.4** for LAION-CLIP under those level, area and strategy labels. The report does not explain why its prose summary differs from Table 2. These values are retained as reported; the 1.0 result is not treated here as a verified best-mAP configuration, and the configuration labels alone do not establish that the underlying runs are otherwise comparable. The Search3D figures are a cited reference, not a reproduction in this repository.

The three comparisons below reproduce Tables 3, 1 and 2 respectively. The seven-experiment averages are distinct from the individual configurations above.

### Strategy Comparison (Averaged over 7 Experiments)

| Strategy | mAP | AP50 | AP25 |
|----------|-----|------|------|
| Independent | 0.84 | 2.90 | 8.59 |
| Exhaustive | 0.51 | 1.54 | 8.39 |
| Hierarchical | 0.25 | 0.75 | 7.01 |

### Level Comparison (LAION-CLIP, min. area=500)

| Level | Independent<br/>mAP/AP50/AP25 | Hierarchical<br/>mAP/AP50/AP25 | Exhaustive<br/>mAP/AP50/AP25 |
|-------|------|------------|------------|
| L1/L3/L5 | 1.2 / 3.7 / 8.0 | 0.0 / 0.1 / 4.3 | 1.1 / 3.2 / 11.0 |
| L2/L4/L6 | 0.3 / 1.6 / 6.8 | 0.9 / 2.3 / 10.7 | 0.0 / 0.2 / 7.8 |

### Feature Extractor Comparison (L1/3/5, min. area=100)

| Model | Independent<br/>mAP/AP50/AP25 | Hierarchical<br/>mAP/AP50/AP25 | Exhaustive<br/>mAP/AP50/AP25 |
|-------|------|------------|------------|
| LAION-CLIP | 1.2 / 3.7 / 11.4 | 0.0 / 0.1 / 3.8 | 0.5 / 1.4 / 8.3 |
| SigLIP | 1.1 / 3.1 / 11.8 | 0.1 / 0.4 / 6.6 | 0.9 / 2.7 / 9.7 |

**Key Findings**:
- Fewer than half of 41 scenes achieve non-zero mAP
- Hierarchical strategy performs particularly poorly (only 1 scene with non-zero mAP)
- Independent has the highest mean mAP, AP50 and AP25 in the report's seven-experiment strategy comparison; this is not a claim of general superiority across datasets or settings.
- In Table 2, SigLIP scores higher on all three metrics for Hierarchical and Exhaustive, while LAION-CLIP scores higher on mAP and AP50 for Independent; SigLIP has higher Independent AP25.
- All listed FamilyPart results remain below the Search3D reference cited in the report.

---

## Key Failure Modes

### 1. Deduplication Issue (Primary Problem)

**Problem**: IoU-only criterion destroys hierarchical structure
- When SAM2 tracks across viewpoints, same object may be represented by masks of different granularities
- Masks at different scales suppress each other using IoU-only comparison
- Intended hierarchical structure is effectively destroyed
- Part masks occasionally larger than corresponding object masks

![Granularity drift after SAM2 tracking](assets/fig6_granularity_drift.jpg)
*Granularity drift after SAM2 tracking (see the chair): the tracked mask no longer matches the granularity of the Semantic-SAM prompt.*

**Impact**:
- Hierarchical strategy fails almost completely
- Assumption that children are always contained inside parents no longer holds
- L2/L4/L6 combination performs worse than L1/L3/L5

### 2. Aggressive Filtering

Due to hardware constraints:
- Increased minimum area threshold to save memory
- Many fine-grained part-level masks discarded
- Further degrades part-level performance

### 3. Feature Alignment

- SigLIP and LAION-CLIP alone insufficient for tight alignment
- Even with ground-truth 3D masks, not every label attains similarity > 0.8
- Considerable variation across labels

---

## Future Directions

Based on current findings, we identify promising directions for improvement:

1. **Size-aware Deduplication Module**
   - Distinguish whether two masks stand in containment relationship or are truly identical
   - Explicitly account for relative mask size
   - Preserve children when appropriate, rather than relying solely on IoU

2. **Scale-adaptive Filtering**
   - Different granularity levels should not share single, global minimum-area threshold
   - Prevent fine-grained part-level masks from being discarded

3. **Enhanced Scoring Mechanisms**
   - More sophisticated weighting or feature fusion strategies
   - Improve feature alignment between 3D proposals and text queries

---

## Getting Started

### Prerequisites
- Two conda environments:
  - **Semantic-SAM**: Detectron2 0.6 + PyTorch 1.13
  - **SAM2**: PyTorch 2.x
- MultiScan dataset (or custom RGB-D sequences with camera poses)
- **Semantic-SAM checkpoint**: `swinl_only_sam_many2many`
- **SAM2 checkpoint**: `sam2.1_hiera_large`

### Running the Archived Pipeline

This repository preserves research code and experiment configurations. The commands below identify the current entry points; they have not been validated end-to-end in a fresh environment and are not a recipe for reproducing the report's figures. The configurations contain machine-specific paths and GPU selections that must be adapted before use.

**Stage 1-2: Segmentation + Tracking**

Start from [configs/multiscan/base.yaml](configs/multiscan/base.yaml). Set `experiment.dataset_root`, `experiment.output_root`, the scene selection (`scenes` or the configured scene range), `stages.gpu`, and the checkpoint/config paths under `stages.ssam` and `stages.tracker` for your installation. Check the enabled filtering, gap-filling and tracking settings as well; the checked-in configuration is one experiment, not a canonical reproduction setting.

```bash
PYTHONPATH=src python -m my3dis.run_workflow \
  --config configs/multiscan/base.yaml
```

The legacy single-scene wrapper `run_experiment.sh` still looks for `generate_candidates.py` and `track_from_candidates.py` beside the wrapper, while those files now live under `src/my3dis/`. It requires repair before use, so it is not presented as a working quick-start command.

**Stage 3-4: Aggregation + Inference (After Tracking)**

The checked-in runner configuration is [configs/inference/experiment_pipeline.yaml](configs/inference/experiment_pipeline.yaml). It defaults to multi-scene mode, where `--exp-root` selects the directory containing per-scene tracking outputs; `--exp-dir` is for single-scene mode. Review the model, levels, pooling and strategy settings, along with device settings in the inference subsections.

```bash
python scripts/run_full_pipeline.py \
  --config configs/inference/experiment_pipeline.yaml \
  --exp-root "/path/to/tracking-output-root" \
  --dataset-root "/path/to/multiscan" \
  --device cuda:0 \
  --skip-evaluation
```

Replace the placeholder paths with your local data. Evaluation is skipped in this example because it requires separately prepared Search3D annotations and paths. To enable it, configure the `evaluation` section, including annotation/point-cloud paths and the evaluation script, then omit `--skip-evaluation`.

These two entry points use different configuration schemas. Use their linked YAML files as the schema references rather than combining stage settings into one generic configuration.

---

## Project Structure

```
FamilyPart/
├── src/my3dis/           # Core package
│   ├── workflow/         # Multi-scene orchestration
│   ├── tracking/         # SAM2 tracking + provenance
│   ├── aggregation/      # 3D proposal generation
│   ├── inference/        # Feature extraction + retrieval
│   ├── mask/             # Mask encoding/geometry utilities
│   └── utils/            # Shared utilities
│
├── configs/              # YAML configuration templates
│   ├── multiscan/        # Dataset-specific configs
│   └── inference/        # Retrieval strategy configs
│
├── scripts/              # Automation and analysis tools
│   ├── run_full_pipeline.py
│   └── benchmark_*.py
│
├── run_experiment.sh     # Legacy single-scene wrapper (requires path repair)
└── README.md             # This file
```

---

## Output Format

### Per-Scene Outputs
```
outputs/<experiment_name>/<scene_id>/
├── level_2/ (or level_1/)
│   ├── raw/              # Raw Semantic-SAM candidates
│   ├── filtered/         # Filtered candidates
│   └── tracking/         # SAM2 tracked objects
├── level_4/ (or level_3/)
├── level_6/ (or level_5/)
├── relations/
│   ├── provenance_tree_L2.json  # Per-level lineage
│   ├── provenance_tree_L4.json
│   ├── provenance_tree_L6.json
│   └── family_tree.json         # Unified hierarchy
├── proposals_L*.npz      # 3D aggregated proposals
├── features_L*.npz       # Feature embeddings
└── workflow_summary.json # Timing/resource statistics
```

### Inference Outputs
```
outputs/inference/<experiment_name>/
├── predictions.json      # Search3D format
├── object_counts.json    # Per-scene statistics
└── inference_summary.json
```

---

## Implementation Details

- **Semantic-SAM**: `swinl_only_sam_many2many` checkpoint
- **SAM2**: `sam2.1_hiera_large` checkpoint
- **Feature Extractors**:
  - LAION-CLIP ViT-L/14: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`
  - SigLIP: `google/siglip-so400m-patch14-384`
- **Sampling**:
  - Semantic-SAM: Every 4th frame (~T/200 frames)
  - SAM2: Every 50th frame (~T/50 frames)
- **Thresholds**:
  - IoU for deduplication: 0.6
  - IoU for merging: 0.5
  - Downscale ratio: 0.3
  - Minimum area: 100-500 (depends on memory)
  - Minimum visible area fraction: 0.0005

---

## Design Philosophy

### Training-Free Approach
This work demonstrates the potential of developing 3D part segmentation pipelines using only:
- Posed RGB-D image sequences
- 3D point clouds
- Pre-trained foundation models (Semantic-SAM, SAM2, SigLIP/CLIP)

No training or fine-tuning required for novel environments.

### Provenance Tracking
Rather than post-hoc spatial containment, FamilyPart tracks provenance from mask generation:
- Explicit parent-child relationships across levels
- Virtual children intended to retain lineage for deduplicated candidates
- O(1) parent lookup through unified mapping

---

## Python Package Name

The repository is published as **FamilyPart**, but the Python package remains `my3dis` for backward compatibility:

```python
# All imports use my3dis
from my3dis.workflow import SceneWorkflow
from my3dis.tracking import ProvenanceTracker
```

When running scripts:
```bash
export PYTHONPATH=src
python -m my3dis.run_workflow --config config.yaml
```

---

## Acknowledgments

FamilyPart builds upon:
- **Semantic-SAM** [5] for multi-level candidate generation
- **SAM2** [6] for temporal mask tracking
- **SigLIP** [7] and **LAION-CLIP** [8] for open-vocabulary feature extraction
- **MultiScan** [3] dataset and **Search3D** [1] benchmark

**Mentors at Vision Science Lab, NTHU**:
- Yen Hong-Xuan, Ou Yeh, Chen Chia-Min and Wang Yan-Qing 

---

## References

[1] A. Takmaz et al., "Search3D: Hierarchical Open-Vocabulary 3D Segmentation," IEEE RA-L, vol. 10, no. 3, pp. 2558-2565, 2025.

[2] A. Takmaz et al., "OpenMask3D: Open-Vocabulary 3D Instance Segmentation," NeurIPS, vol. 36, 2023.

[3] Y. Mao et al., "MultiScan: Scalable RGBD Scanning for 3D Environments with Articulated Objects," NeurIPS, vol. 35, 2022.

[4] H. Yin et al., "Semantic Consistent Language Gaussian Splatting for Point-Level Open-vocabulary Querying," arXiv:2503.21767, 2025.

[5] F. Li et al., "Semantic-SAM: Segment and Recognize Anything at Any Granularity," ECCV, pp. 467-484, 2024.

[6] N. Ravi et al., "SAM 2: Segment Anything in Images and Videos," arXiv:2408.00714, 2024.

[7] X. Zhai et al., "Sigmoid Loss for Language Image Pre-Training," ICCV, pp. 11975-11986, 2023.

[8] LAION-AI, "LAION-CLIP: OpenCLIP Models Trained on the LAION-2B Dataset," Hugging Face / OpenCLIP, 2023-2024.

