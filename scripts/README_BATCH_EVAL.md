# Batch Evaluation with Integrated AggregationPipeline

## 概述

`batch_eval_experiment.py` 使用整合的 `AggregationPipeline`，實現完整的 2D→3D 投影、SigLIP 特徵提取、推論與評估工作流。

### 主要特色

- ✅ **預設使用 SigLIP features** - 準確的視覺特徵表示
- ✅ **結構化輸出** - 結果保存在實驗目錄結構中
  - `{scene_dir}/aggregation_result/` - 3D proposals & features
  - `{scene_dir}/inference_result/` - 推論結果 & 評估指標
- ✅ **完整的 pipeline.ipynb 整合** - 產生 3D point cloud masks (79,614 points)
- ✅ **可配置的 inference** - 支援 YAML 配置檔案
- ✅ **批量處理** - 自動發現並評估多個場景

## 工作流程

### 完整流程（針對每個場景）

```
1. Aggregation (Step 1-2/4)
   ├─ 初始化 WORLD_2_CAM
   ├─ 計算 mesh projections
   ├─ 獲取每幀可見點
   ├─ 從 video_segments 建立 3D masks
   ├─ 合併與過濾 proposals (hierarchical merging)
   ├─ 提取 SigLIP features (預設啟用)
   └─ 導出到 {scene_dir}/aggregation_result/
       ├─ proposal_data_level2.npz
       ├─ proposal_data_level4.npz
       ├─ proposal_data_level6.npz
       ├─ Proposal_relation.json
       └─ aggregation_metadata.json

2. Inference (Step 3/4)
   ├─ 載入 proposals 從 NPZ files
   ├─ 載入 SigLIP 進行文字編碼
   ├─ 對每個 query 執行推論
   ├─ 格式化預測結果
   └─ 保存到 {scene_dir}/inference_result/
       ├─ inference_results.json        # 詳細推論結果
       ├─ formatted_predictions.npz     # Search3D 格式
       └─ evaluation_results.json       # 評估指標

3. Evaluation (Step 4/4)
   └─ Search3D 評估協議 (mAP, mAP@50, mAP@25)

4. Summary
   └─ 聚合多場景結果到 {exp_root}/batch_eval_summary.json
```

## 使用方式

### 基本用法（預設使用 SigLIP features）

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root /path/to/experiment/root \
    --dataset-root data/multiscan \
    --gt-path /path/to/search3d_gt/ov_part_annotations \
    --scenes scene_00005_00
```

**預期時間**: 每個場景 ~40-60 分鐘（含 SigLIP 特徵提取）

**輸出位置**:
```
experiment/root/scene_00005_00/
├── aggregation_result/          # 3D proposals
│   ├── proposal_data_level*.npz
│   └── aggregation_metadata.json
└── inference_result/            # 推論與評估
    ├── inference_results.json
    ├── formatted_predictions.npz
    └── evaluation_results.json

experiment/root/
└── batch_eval_summary.json      # 聚合結果
```

### 快速模式（跳過 SigLIP 特徵提取）

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root /path/to/experiment/root \
    --dataset-root data/multiscan \
    --gt-path /path/to/search3d_gt/ov_part_annotations \
    --no-extract-features
```

**預期時間**: 每個場景 ~5-10 分鐘（使用 dummy features）

⚠️ **注意**: 使用 `--no-extract-features` 會產生隨機特徵，評估結果僅供測試參考。

### 評估多個場景

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root /path/to/experiment/root \
    --dataset-root data/multiscan \
    --gt-path /path/to/search3d_gt/ov_part_annotations \
    --scenes scene_00005_00 scene_00005_01 scene_00006_00
```

### 使用自訂 inference 配置

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root /path/to/experiment/root \
    --dataset-root data/multiscan \
    --gt-path /path/to/search3d_gt/ov_part_annotations \
    --inference-config configs/inference/hierarchical.yaml
```

**可用配置**:
- `configs/inference/base.yaml` - 基礎配置
- `configs/inference/quick_eval.yaml` - 快速評估（預設）
- `configs/inference/hierarchical.yaml` - 階層式檢索
- `configs/inference/ablation_low_nms.yaml` - 消融實驗

### 完整參數範例

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03 \
    --dataset-root data/multiscan \
    --gt-path data/search3d_gt/multiscan_annotations_search3d/ov_part_annotations \
    --inference-config configs/inference/quick_eval.yaml \
    --levels 2 4 6 \
    --iou-threshold 0.8 \
    --area-fraction 0.002 \
    --strategy independent \
    --device cuda \
    --summary-output results/my_experiment_summary.json
```

## 參數說明

### 必要參數

| 參數 | 說明 | 範例 |
|-----|------|-----|
| `--exp-root` | 實驗根目錄（包含多個 scene_XX 子目錄） | `/path/to/outputs/experiments/exp_v2` |
| `--dataset-root` | MultiScan 數據集根目錄 | `data/multiscan` |
| `--gt-path` | Search3D ground truth 標註路徑 | `/path/to/ov_part_annotations` |

### 可選參數

#### Inference 配置

| 參數 | 預設值 | 說明 |
|-----|--------|-----|
| `--inference-config` | `configs/inference/quick_eval.yaml` | Inference config YAML 路徑 |
| `--strategy` | `independent` | 推論策略 (`independent`/`hierarchical`) |

#### Aggregation 配置

| 參數 | 預設值 | 說明 |
|-----|--------|-----|
| `--levels` | `2 4 6` | 要處理的 levels |
| `--iou-threshold` | `0.8` | Proposal 合併的 IoU 閾值 |
| `--area-fraction` | `0.002` (1/500) | 最小面積佔總點數的比例 |

#### Feature Extraction

| 參數 | 預設值 | 說明 |
|-----|--------|-----|
| `--no-extract-features` | `False` | 加上此參數則跳過 SigLIP 特徵提取 |
| `--siglip-model-dir` | (local path) | SigLIP 模型目錄 |

#### 其他

| 參數 | 預設值 | 說明 |
|-----|--------|-----|
| `--device` | `cuda` | 使用的設備 |
| `--scenes` | `None` | 特定場景列表（預設：全部） |
| `--summary-output` | `{exp_root}/batch_eval_summary.json` | 聚合摘要輸出路徑 |

## 輸出結構

### 場景級輸出

```
{exp_root}/scene_XXXXX_XX/
├── aggregation_result/
│   ├── proposal_data_level2.npz         # 3D proposals (79614, N2) + features
│   ├── proposal_data_level4.npz         # 3D proposals (79614, N4) + features
│   ├── proposal_data_level6.npz         # 3D proposals (79614, N6) + features
│   ├── Proposal_relation.json           # 更新後的階層關係
│   └── aggregation_metadata.json        # Pipeline 配置與統計
│
└── inference_result/
    ├── inference_results.json           # 詳細推論結果（含每個 prediction）
    ├── formatted_predictions.npz        # Search3D 評估格式
    └── evaluation_results.json          # 評估指標（mAP 等）
```

### 實驗級輸出

```
{exp_root}/
└── batch_eval_summary.json              # 聚合所有場景的結果
```

### 輸出文件內容

#### 1. aggregation_metadata.json

```json
{
  "exp_dir": "...",
  "scene_path": "...",
  "ply_file": "...",
  "iou_threshold": 0.8,
  "area_fraction": 0.002,
  "area_threshold": 159,
  "num_points": 79614,
  "outputs": {
    "2": "aggregation_result/proposal_data_level2.npz",
    "4": "aggregation_result/proposal_data_level4.npz",
    "6": "aggregation_result/proposal_data_level6.npz"
  },
  "feature_extraction": true
}
```

#### 2. inference_results.json

```json
{
  "scene_name": "scene_00005_00",
  "timestamp": "2025-11-08T14:30:00",
  "config": {
    "strategy": "independent",
    "retrieval": {...},
    "pairing": {...},
    "nms": {...}
  },
  "num_queries": 8,
  "num_predictions": 150,
  "predictions": [
    {
      "label_id": 121003,
      "score": 0.85,
      "object_level": "L2",
      "part_level": "L4",
      "num_points": 1234
    },
    ...
  ]
}
```

#### 3. evaluation_results.json

```json
{
  "scene_name": "scene_00005_00",
  "timestamp": "2025-11-08T14:30:00",
  "metrics": {
    "mAP": 0.1234,
    "mAP@50": 0.2345,
    "mAP@25": 0.3456,
    "details": {...}
  }
}
```

#### 4. batch_eval_summary.json

```json
{
  "experiment": "/path/to/experiment/root",
  "timestamp": "2025-11-08T14:30:00",
  "config": {
    "strategy": "independent",
    "levels": [2, 4, 6],
    "iou_threshold": 0.8,
    "area_fraction": 0.002,
    "extract_features": true,
    "inference_config": {...}
  },
  "aggregated": {
    "num_scenes": 10,
    "mean_mAP": 0.1234,
    "mean_mAP@50": 0.2345,
    "mean_mAP@25": 0.3456,
    "total_proposals": {
      "L2": 2270,
      "L4": 3850,
      "L6": 1330
    },
    "total_predictions": 1500,
    "per_scene": {
      "mAP": {
        "scene_00005_00": 0.1234,
        ...
      }
    }
  },
  "scene_results": {
    "scene_00005_00": {
      "num_proposals": {...},
      "num_predictions": 150,
      "metrics": {...},
      "aggregation_output_dir": "...",
      "inference_output_dir": "..."
    }
  },
  "failed_scenes": []
}
```

## 與舊版本的差異

### 主要變更

| 特性 | 舊版 (v1) | 新版 (v2) |
|------|----------|-----------|
| **輸出位置** | `/tmp/batch_eval_results/` ❌ | `{scene_dir}/aggregation_result/`<br>`{scene_dir}/inference_result/` ✅ |
| **SigLIP features** | 需手動指定 `--extract-features` | 預設啟用（可用 `--no-extract-features` 關閉） ✅ |
| **Inference 配置** | 僅支援 CLI 參數 | 支援 YAML 配置檔案 ✅ |
| **輸出結構** | 扁平結構 | 階層式結構（aggregation + inference） ✅ |
| **結果保存** | 僅摘要 | 完整詳細結果 + 摘要 ✅ |

### 遷移指南

**舊命令**:
```bash
python scripts/batch_eval_experiment.py \
    --exp-root /path/to/exp \
    --dataset-root /path/to/multiscan \
    --gt-path /path/to/gt \
    --output-dir /tmp/results \          # ❌ 不再使用
    --extract-features                   # ❌ 現在是預設
```

**新命令**:
```bash
python scripts/batch_eval_experiment.py \
    --exp-root /path/to/exp \
    --dataset-root /path/to/multiscan \
    --gt-path /path/to/gt
    # 輸出自動保存到實驗目錄
    # SigLIP features 預設啟用
```

## 性能優化

### 1. 快速測試（跳過 SigLIP）

適用於快速驗證 pipeline 正確性：

```bash
--no-extract-features
```

- **時間**: ~5-10 分鐘/場景
- **用途**: 測試、除錯
- **限制**: 使用隨機特徵，評估結果不準確

### 2. 完整評估（預設）

準確的特徵表示與評估：

```bash
# 不需要額外參數，預設就會提取 SigLIP features
```

- **時間**: ~40-60 分鐘/場景
- **用途**: 正式評估、論文結果
- **優勢**: 準確的視覺特徵

### 3. 批量並行處理

腳本本身是序列處理，但可以使用 GNU parallel 並行處理多場景：

```bash
#!/bin/bash
# parallel_batch_eval.sh

SCENES=(scene_00005_00 scene_00005_01 scene_00006_00 scene_00006_01)
EXP_ROOT="/path/to/experiment/root"
DATASET_ROOT="data/multiscan"
GT_PATH="/path/to/ov_part_annotations"

# 並行處理（每次2個場景）
parallel -j 2 \
    "PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
        --exp-root ${EXP_ROOT} \
        --dataset-root ${DATASET_ROOT} \
        --gt-path ${GT_PATH} \
        --scenes {}" \
    ::: "${SCENES[@]}"

# 合併結果
python scripts/merge_batch_results.py --exp-root ${EXP_ROOT}
```

### 4. 記憶體優化

如果遇到 CUDA OOM:

```bash
--device cpu  # 使用 CPU（較慢但不會 OOM）
```

或調整 batch size（需修改 AggregationPipeline 內部參數）

## 常見問題

### Q1: 為什麼預設啟用 SigLIP features？

**A**: SigLIP features 對於準確的開放詞彙檢索至關重要。雖然提取時間較長（~30-40分鐘），但能提供準確的評估結果。如需快速測試，可使用 `--no-extract-features`。

### Q2: 輸出目錄在哪裡？

**A**: 新版本將結果保存在實驗目錄結構中：
- Aggregation: `{scene_dir}/aggregation_result/`
- Inference: `{scene_dir}/inference_result/`
- Summary: `{exp_root}/batch_eval_summary.json`

這樣更容易管理和追蹤不同實驗的結果。

### Q3: 如何使用不同的 inference 策略？

**A**: 有兩種方式：

**方法 1**: 使用 YAML 配置檔案（推薦）
```bash
--inference-config configs/inference/hierarchical.yaml
```

**方法 2**: 使用 CLI 參數
```bash
--strategy hierarchical
```

### Q4: Video segments 找不到

**錯誤**: `Video segments not found: .../level_2/video_segments_L02.npz`

**原因**: 實驗使用不同的 level 命名（如 level_1 而非 level_2）

**解決**: 檢查實驗目錄結構並調整 `--levels` 參數

```bash
ls /path/to/experiment/scene_XXX/
# 如果看到 level_1, level_3, level_5
--levels 1 3 5
```

### Q5: 如何查看詳細的 inference 結果？

**A**: 檢查 `{scene_dir}/inference_result/inference_results.json`，包含：
- 每個 prediction 的詳細資訊
- Score, level, num_points
- Inference 配置參數

### Q6: 評估結果保存在哪裡？

**A**:
- **單場景**: `{scene_dir}/inference_result/evaluation_results.json`
- **聚合**: `{exp_root}/batch_eval_summary.json`

## 完整範例

### 範例 1: 快速測試單個場景（無特徵提取）

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03 \
    --dataset-root data/multiscan \
    --gt-path data/search3d_gt/multiscan_annotations_search3d/ov_part_annotations \
    --scenes scene_00005_00 \
    --no-extract-features
```

**預期時間**: ~5-10 分鐘

**輸出**:
```
scene_00005_00/
├── aggregation_result/
│   └── ... (3D proposals with dummy features)
└── inference_result/
    └── ... (evaluation results)
```

### 範例 2: 完整評估單個場景（含 SigLIP features）

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03 \
    --dataset-root data/multiscan \
    --gt-path data/search3d_gt/multiscan_annotations_search3d/ov_part_annotations \
    --scenes scene_00005_00
```

**預期時間**: ~40-60 分鐘

**預期輸出**:
```
================================================================================
Scene 1/1: scene_00005_00
================================================================================
Processing scene_00005_00...
  Step 1/4: Generating 3D proposals...
    Frames: 196, 3D Points: 79614
    ✓ WORLD_2_CAM initialization complete
  Step 2/4: Loading proposals...
    Total 3D points: 79614
    Proposals: L2=227, L4=385, L6=133
  Step 3/4: Running inference...
    Generated 150 predictions
  Step 4/4: Running Search3D evaluation...

scene_00005_00 Results:
  Proposals: {'L2': 227, 'L4': 385, 'L6': 133}
  Predictions: 150
  mAP:    0.1234
  mAP@50: 0.2345
  mAP@25: 0.3456
  Aggregation: .../scene_00005_00/aggregation_result
  Inference:   .../scene_00005_00/inference_result
```

### 範例 3: 批量評估多場景

```bash
PYTHONPATH=src conda run -n SAM2 python scripts/batch_eval_experiment.py \
    --exp-root outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03 \
    --dataset-root data/multiscan \
    --gt-path data/search3d_gt/multiscan_annotations_search3d/ov_part_annotations \
    --scenes scene_00005_00 scene_00005_01 scene_00006_00
```

**預期時間**: ~2-3 小時（3個場景）

**預期輸出**:
```
================================================================================
OVERALL RESULTS
================================================================================
Scenes evaluated: 3
Total proposals: {'L2': 681, 'L4': 1155, 'L6': 399}
Total predictions: 450

Mean Metrics:
  mAP:    0.1234
  mAP@50: 0.2345
  mAP@25: 0.3456
================================================================================

Aggregated summary saved to .../batch_eval_summary.json
```

## 相關文件

- **整合說明**: `scripts/AGGREGATION_INTEGRATION_SUMMARY.md`
- **評估工作流**: `scripts/EVALUATION_WORKFLOW.md`
- **AggregationPipeline**: `src/my3dis/aggregation/aggregation_pipeline.py`
- **InferencePipeline**: `src/my3dis/inference/inference_pipeline.py`
- **Inference configs**: `configs/inference/*.yaml`

## 參考

- **Original workflow**: `3DprojToSiglip/pipeline.ipynb`
- **Search3D benchmark**: https://github.com/3dlg-hcvc/search3d
- **SigLIP model**: https://huggingface.co/google/siglip-so400m-patch14-384
