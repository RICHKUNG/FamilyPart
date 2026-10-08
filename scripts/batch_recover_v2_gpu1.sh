#!/bin/bash
# 批次救援 v2 實驗關係 - 使用所有可用 GPU
# Author: Rich Kung
# Created: 2025-10-22

set -e  # 遇到錯誤立即停止

# ===== 配置 =====
EXPERIMENTS_ROOT="./outputs/experiments"
CONTAINMENT_THRESHOLD=0.95
MASK_SCALE_RATIO=1.0
LOG_DIR="logs/recovery_$(date +%Y%m%d_%H%M%S)"
PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"

# ===== 初始化 =====
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"
ERROR_LOG="$LOG_DIR/errors.log"

echo "======================================" | tee "$SUMMARY_LOG"
echo "v2 實驗關係救援 - 使用所有可用 GPU" | tee -a "$SUMMARY_LOG"
echo "開始時間: $(date)" | tee -a "$SUMMARY_LOG"
echo "======================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# ===== GPU 資訊 =====
if command -v nvidia-smi &> /dev/null; then
    echo "可用 GPU:" | tee -a "$SUMMARY_LOG"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$SUMMARY_LOG"
    echo "" | tee -a "$SUMMARY_LOG"
fi

# ===== 掃描實驗 =====
echo "[1/3] 掃描 v2 實驗目錄..." | tee -a "$SUMMARY_LOG"
python3 scripts/batch_recover_v2_relations.py --dry-run > "$LOG_DIR/scan_result.txt" 2>&1

TOTAL_SCENES=$(grep "Found.*total run directories" "$LOG_DIR/scan_result.txt" | awk '{print $2}')
NEED_RECOVERY=$(grep "Need recovery:" "$LOG_DIR/scan_result.txt" | awk '{print $3}')

echo "找到 $TOTAL_SCENES 個實驗，需要救援 $NEED_RECOVERY 個" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

if [ "$NEED_RECOVERY" == "0" ]; then
    echo "沒有需要救援的實驗，退出。" | tee -a "$SUMMARY_LOG"
    exit 0
fi

# ===== 開始批次處理 =====
echo "[2/3] 開始批次救援..." | tee -a "$SUMMARY_LOG"
echo "預計時間: $(echo "$NEED_RECOVERY * 45 / 3600" | bc -l | xargs printf "%.1f") 小時" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 執行批次救援
PYTHONPATH="$PYTHONPATH" python3 scripts/batch_recover_v2_relations.py \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --containment-threshold "$CONTAINMENT_THRESHOLD" \
    --mask-scale-ratio "$MASK_SCALE_RATIO" \
    2>&1 | tee "$LOG_DIR/batch_output.log"

# ===== 統計結果 =====
echo "" | tee -a "$SUMMARY_LOG"
echo "[3/3] 統計結果..." | tee -a "$SUMMARY_LOG"

SUCCESS_COUNT=$(grep "✅ Success" "$LOG_DIR/batch_output.log" | wc -l)
FAIL_COUNT=$(grep "❌ Failed" "$LOG_DIR/batch_output.log" | wc -l)
COMPLETED=$(find "$EXPERIMENTS_ROOT"/v2_* -name "relations.json" | wc -l)

echo "成功: $SUCCESS_COUNT" | tee -a "$SUMMARY_LOG"
echo "失敗: $FAIL_COUNT" | tee -a "$SUMMARY_LOG"
echo "總計完成: $COMPLETED / $TOTAL_SCENES" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 提取錯誤訊息
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "錯誤訊息:" | tee -a "$SUMMARY_LOG"
    grep "❌ Failed" "$LOG_DIR/batch_output.log" | head -20 | tee -a "$ERROR_LOG"
    echo "" | tee -a "$SUMMARY_LOG"
    echo "完整錯誤日誌: $ERROR_LOG" | tee -a "$SUMMARY_LOG"
fi

# ===== 完成 =====
echo "======================================" | tee -a "$SUMMARY_LOG"
echo "完成時間: $(date)" | tee -a "$SUMMARY_LOG"
echo "日誌目錄: $LOG_DIR" | tee -a "$SUMMARY_LOG"
echo "======================================" | tee -a "$SUMMARY_LOG"

# 發送通知（如果有 ntfy）
if command -v ntfy &> /dev/null; then
    ntfy send "v2 關係救援完成: $SUCCESS_COUNT 成功, $FAIL_COUNT 失敗"
fi
