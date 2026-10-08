# Search3D Ground Truth 下載指引

## 下載連結

🔗 **Google Drive**: https://drive.google.com/drive/folders/1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx?usp=sharing

## 手動下載步驟

### 1. 開啟 Google Drive 連結
打開上方連結，你會看到資料夾內容。

### 2. 下載所需文件
需要下載以下內容：
- **multiscan_data_search3d/** 資料夾（整個資料夾）
  - 包含 annotations 和 PLY 文件

### 3. 保存位置
將下載的資料夾解壓到：
```
./data/search3d_gt/
```

### 4. 預期目錄結構
```
./data/search3d_gt/
└── multiscan_data_search3d/
    ├── multiscan_annotations_search3d/
    │   ├── obj_annotations/
    │   ├── part_annotations/
    │   └── ov_part_annotations/          ← 主要評估數據
    │       ├── scene_00005_00_obj_part_inst.txt
    │       ├── scene_00005_01_obj_part_inst.txt
    │       └── ... (41 scenes total)
    ├── multiscan_test_plys_only/
    │   ├── scene_00005_00.ply
    │   ├── scene_00005_01.ply
    │   └── ... (41 scenes)
    └── multiscan_search3d_constants.py
```

## 驗證下載

下載完成後，執行：
```bash
python scripts/verify_gt_setup.py
```

## GT 路徑設置

評估時使用的路徑：
```bash
GT_PATH="./data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations"
```

## 使用示例

```bash
conda run -n 3Dsiglip python scripts/quick_eval_experiment.py \
    --exp-dir outputs/experiments/test_1105_ssam4_filter300_fill500_fix/scene_00005_00 \
    --dataset-root data/multiscan \
    --gt-path ${GT_PATH} \
    --output-dir eval_results
```

## 數據統計

- **測試場景**: 20 個獨特場景
- **總場景文件**: 41 個（含變體 _00, _01, _02）
- **標註類型**: OV Part Annotations（物體-部件對）
- **物體-部件對**: 47 個有效關係
  - 例如: cabinet-door, table-drawer, toilet-seat
- **標籤格式**: 6 位數字 `AAAIII`
  - `AAA`: 語義 ID（物體-部件對）
  - `III`: 實例編號

## 快速測試（無需完整 GT）

如果只想測試 pipeline 而不需要真實評估指標：
```bash
mkdir -p /tmp/dummy_gt
python scripts/quick_eval_experiment.py \
    --exp-dir ... \
    --gt-path /tmp/dummy_gt \  # 使用 dummy GT
    --output-dir eval_results
```

這會運行完整 pipeline 但跳過 mAP 計算。
