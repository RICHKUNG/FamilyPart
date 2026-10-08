# My3DIS Evaluation Workflow - 完整指南

## 問題總結

**原始問題**：`quick_eval_experiment.py` 使用 2D masks，但 Search3D 評估需要 3D point cloud masks。

**解決方案**：使用 `pipeline.ipynb` 的 2D→3D 投影邏輯。

## 當前工作流程（可用）

### 方法 1：使用 Jupyter Notebook（推薦）

這是目前**最快且經過驗證**的方法：

```bash
# Step 1: 打開 Jupyter Notebook
cd ./3DprojToSiglip
jupyter notebook pipeline.ipynb

# Step 2: 修改路徑配置（cell 2）
path_2_masklets_l2 = "/path/to/your/experiment/scene_XXX/level_2/video_segments_L02.npz"
path_2_masklets_l4 = "/path/to/your/experiment/scene_XXX/level_4/video_segments_L04.npz"
path_2_masklets_l6 = "/path/to/your/experiment/scene_XXX/level_6/video_segments_L06.npz"
path_2_relations = "/path/to/your/experiment/scene_XXX/relations/family_tree.json"
path_2_scene = "data/multiscan/scene_XXX"
path_2_ply = "data/multiscan/scene_XXX/scene_XXX_converted.ply"

# Step 3: 執行所有 cells（Kernel → Restart & Run All）

# Step 4: 檢查輸出
ls -lh ./3DprojToSiglip/proposal_data_level*.npz

# Step 5: 運行評估
python3 scripts/eval_from_npz.py \
    --npz-l2 ./3DprojToSiglip/proposal_data_level2.npz \
    --npz-l4 ./3DprojToSiglip/proposal_data_level4.npz \
    --npz-l6 ./3DprojToSiglip/proposal_data_level6.npz \
    --family-tree ./3DprojToSiglip/Proposal_relation.json \
    --gt-path ./data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations \
    --scene-name scene_00005_00 \
    --output-dir /tmp/eval_results
```

**優點**：
- ✅ 已驗證可行
- ✅ 完整的可視化和debug能力
- ✅ 無需修改代碼

**缺點**：
- ❌ 手動操作，難以批量處理
- ❌ 需要逐場景修改路徑

### 方法 2：批量評估（腳本化）

當前 `eval_from_npz.py` 可以直接使用已生成的 `.npz` 文件。

創建批量腳本：

```bash
#!/bin/bash
# batch_eval_with_npz.sh

SCENES=(
    "scene_00005_00"
    "scene_00005_01"
    "scene_00006_00"
    "scene_00006_01"
)

for scene in "${SCENES[@]}"; do
    echo "Evaluating $scene..."

    # 假設 .npz 文件已經生成在標準位置
    NPZ_DIR="/path/to/npz_outputs/${scene}"

    python3 scripts/eval_from_npz.py \
        --npz-l2 "${NPZ_DIR}/proposal_data_level2.npz" \
        --npz-l4 "${NPZ_DIR}/proposal_data_level4.npz" \
        --npz-l6 "${NPZ_DIR}/proposal_data_level6.npz" \
        --family-tree "${NPZ_DIR}/Proposal_relation.json" \
        --gt-path ./data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations \
        --scene-name "${scene}" \
        --output-dir "/tmp/eval_results"
done
```

## pipeline.ipynb 做了什麼？

### 核心步驟

1. **載入 2D masks**（從 My3DIS tracking 輸出）
   ```python
   frames, _ = load_legacy_video_segments("video_segments_L02.npz", unpack_masks=False)
   ```

2. **初始化 3D 投影器**
   ```python
   world2cam = WORLD_2_CAM(
       dataset_type="multiscan",
       path_2_scene=scene_path,
       path_2_ply=ply_path,
       ...
   )
   projected_points, inside_mask, point_depth, depth_maps = world2cam.get_mesh_projections()
   ```

3. **計算可見點**（哪些 3D 點在每個視角可見）
   ```python
   pts_coords_per_frame, _, pts_idx_per_frame = get_visible_points(...)
   ```

4. **從 2D masks 建立 3D masks**
   ```python
   proposal_masks_l2, obj_ids_l2 = build_proposal_masks_from_masklets(
       masklet_path=path_2_masklets_l2,
       pts_coords_per_frame=pts_coords_per_frame,
       pts_idx_per_frame=pts_idx_per_frame,
       P=P,  # 3D 點數量
       frequency=frequency,
   )
   ```

5. **合併/過濾 proposals**
   ```python
   # 基於 IoU 合併相似 proposals，過濾小物體
   new_mask_l2, col_ids, old2rep, dropped_ids, groups = merge_drop_level2_mask(
       proposal_masks_l2, iou_thresh=0.8, area_thresh=P/500
   )
   ```

6. **提取 SigLIP 特徵**
   ```python
   pc_features_l2 = generate_grounding_features_siglip(
       model=model,
       processor=processor,
       ...
       proposal_masks=proposal_masks_l2,
   )
   ```

7. **Pooling 點特徵到 proposal 特徵**
   ```python
   proposal_features_l2 = pool_point_to_proposal_features(pc_features_l2, new_mask_l2)
   ```

8. **保存為 NPZ**
   ```python
   np.savez(
       "proposal_data_level2.npz",
       proposal_masks=new_mask_l2.cpu().numpy(),
       proposal_features=proposal_features_l2.cpu().numpy(),
       proposal_ids=col_ids
   )
   ```

### 關鍵依賴

- `WORLD_2_CAM` 類（3DprojToSiglip/utils_new/__init__.py）
- `get_visible_points` 函數（3DprojToSiglip/utils_new/utils_2d.py）
- SigLIP model（transformers）
- Open3D, torch, PIL 等

## 未來整合計劃

### 短期（現在可用）

✅ **使用現有工具**：
- `pipeline.ipynb` 生成 `.npz`
- `eval_from_npz.py` 進行評估

### 中期（1-2週）

🔄 **腳本化 pipeline.ipynb**：

創建 `scripts/generate_3d_proposals.py`，包含：

```python
class Pipeline3D:
    """
    整合 pipeline.ipynb 的所有邏輯
    """
    def __init__(self, exp_dir, scene_path, config):
        self.exp_dir = Path(exp_dir)
        self.scene_path = Path(scene_path)
        self.config = config

        # 初始化 WORLD_2_CAM
        self.world2cam = WORLD_2_CAM(...)

    def project_2d_to_3d(self, level):
        """將 2D masks 投影到 3D"""
        # 實現 build_proposal_masks_from_masklets 邏輯
        pass

    def merge_and_filter(self, proposal_masks, level):
        """合併和過濾 proposals"""
        # 實現 merge_drop_level* 邏輯
        pass

    def extract_features(self, proposal_masks):
        """提取 SigLIP 特徵"""
        # 實現 generate_grounding_features_siglip 邏輯
        pass

    def run(self, output_dir, levels=[2, 4, 6]):
        """運行完整 pipeline"""
        for level in levels:
            masks = self.project_2d_to_3d(level)
            masks = self.merge_and_filter(masks, level)
            features = self.extract_features(masks)
            self.save_npz(masks, features, output_dir, level)
```

### 長期（1-2月）

🎯 **整合到 my3dis.aggregation**：

將整個流程整合為 `my3dis.aggregation.AggregationPipeline` 的一部分：

```python
from my3dis.aggregation import AggregationPipeline

pipeline = AggregationPipeline(
    family_tree_path="...",
    pointcloud_path="...",
    depth_maps_dir="...",
    color_frames_dir="...",
    camera_params_path="..."
)

# 運行 2D→3D 投影
results = pipeline.run(output_dir="...", levels=[2, 4, 6])

# 直接評估
from my3dis.inference import InferencePipeline
inference = InferencePipeline(
    proposals_by_level=pipeline.get_proposals(),
    ...
)
```

## 實際操作建議

### 當前（立即可用）

**單個場景評估**：
1. 運行 `pipeline.ipynb`（10-20分鐘）
2. 使用 `eval_from_npz.py`（5-10分鐘）

**多個場景評估**：
1. 為每個場景運行 `pipeline.ipynb`
2. 創建批量評估腳本調用 `eval_from_npz.py`

### 下一步開發優先級

1. **[高優先級]** 將 `pipeline.ipynb` 轉換為 `.py` 腳本
   - 複製所有 helper functions
   - 添加 CLI 參數
   - 實現批量處理

2. **[中優先級]** 整合到 `quick_eval_experiment.py`
   - 修改 `quick_eval` 使用 3D projection
   - 統一 API

3. **[低優先級]** 重構到 `my3dis.aggregation`
   - 創建統一的模組結構
   - 完整文檔和測試

## 文件路徑參考

```
./
├── 3DprojToSiglip/
│   ├── pipeline.ipynb              # 核心邏輯（已驗證可行）
│   ├── utils_new/
│   │   ├── __init__.py            # WORLD_2_CAM 類
│   │   ├── utils_2d.py            # get_visible_points
│   │   └── feature_extraction.py  # SigLIP 批量提取
│   └── utils_load/
│       ├── compat.py              # load_legacy_video_segments
│       └── common_utils.py        # unpack_binary_mask
├── src/my3dis/
│   ├── aggregation/
│   │   ├── aggregation_pipeline.py    # 當前實現（需要擴展）
│   │   └── proposal_loader.py         # 2D mask loader
│   └── inference/
│       ├── feature_extraction.py      # SigLIP extractor（2D版本）
│       └── inference_pipeline.py      # 評估 pipeline
├── scripts/
│   ├── eval_from_npz.py                # ✅ 使用 .npz 評估（可用）
│   ├── quick_eval_experiment.py        # ❌ 當前不兼容 Search3D
│   └── run_3d_projection_pipeline.py   # 🔄 模板（需完成）
└── data/
    └── search3d_gt/
        └── multiscan_annotations_search3d/
            └── ov_part_annotations/    # Ground truth

常用場景數據：
data/multiscan/
├── scene_00005_00/
│   ├── scene_00005_00_converted.ply    # 3D 點雲
│   └── outputs/
│       ├── color/                      # RGB 圖像
│       ├── depth/                      # 深度圖
│       ├── pose/                       # 相機位姿
│       └── intrinsics.txt              # 相機內參
```

## 常見問題

### Q: 為什麼不直接修改 quick_eval_experiment.py？

A: 2D→3D 投影需要：
- 深度圖（depth maps）
- 相機參數（camera intrinsics/extrinsics）
- 複雜的可見性計算
- 多視角聚合邏輯

這些邏輯在 `pipeline.ipynb` 已經實現並驗證，直接整合比重寫更可靠。

### Q: .npz 文件可以復用嗎？

A: 可以！只要：
- 場景相同（同一個 3D 點雲）
- Tracking 輸出相同（同樣的 2D masks）
- 配置相同（IoU 閾值、面積閾值等）

則 `.npz` 文件可以在多個評估中復用。

### Q: 如何加速 SigLIP 特徵提取？

A: `pipeline.ipynb` 已經包含優化版本：

```python
pc_features = generate_grounding_features_siglip(
    ...
    use_optimized=True,   # 使用批量處理
    batch_size=32,        # 調整以適應 GPU 記憶體
)
```

預期加速：2-3倍（~40分鐘 → ~15分鐘）

## 總結

**當前最佳實踐**：
1. ✅ 使用 `pipeline.ipynb` 生成 3D proposals
2. ✅ 使用 `eval_from_npz.py` 進行評估
3. 🔄 逐步將邏輯腳本化以實現批量處理

**不要**：
- ❌ 嘗試修復 `quick_eval_experiment.py` 的 2D mask 問題
- ❌ 從頭重寫 2D→3D 投影邏輯

**優先做**：
- ✅ 驗證 `pipeline.ipynb` 對你的實驗輸出可行
- ✅ 記錄不同配置的 `.npz` 輸出位置
- ✅ 創建批量評估腳本
