#!/bin/bash
#
# Setup Search3D Ground Truth Data for MultiScan Evaluation
#
# This script:
# 1. Downloads Search3D GT data from Google Drive
# 2. Sets up the correct directory structure
# 3. Copies evaluation code from search3d repo
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
GT_ROOT="${PROJECT_ROOT}/data/search3d_gt"
DOWNLOAD_URL="https://drive.google.com/drive/folders/1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx?usp=sharing"

echo "========================================="
echo "Search3D Ground Truth Setup"
echo "========================================="
echo

# Step 1: Check for gdown
echo "[Step 1/4] Checking dependencies..."
if ! command -v gdown &> /dev/null; then
    echo "  ⚠ gdown not found. Installing..."
    pip install gdown
fi

# Step 2: Create directory structure
echo
echo "[Step 2/4] Creating directory structure..."
mkdir -p "${GT_ROOT}"
echo "  ✓ Created: ${GT_ROOT}"

# Step 3: Download instructions
echo
echo "[Step 3/4] Downloading Ground Truth data..."
echo
echo "  📥 Download Link: ${DOWNLOAD_URL}"
echo
echo "  Please download the following files from Google Drive:"
echo "    1. multiscan_annotations_search3d.zip (or folder)"
echo "    2. multiscan_test_plys_only.zip (or folder)"
echo "    3. multiscan_search3d_constants.py"
echo
echo "  Save them to: ${GT_ROOT}/"
echo

# Try automatic download using gdown
echo "  Attempting automatic download..."
echo "  (If this fails, please download manually from the link above)"
echo

# Note: Google Drive folder download with gdown requires folder ID
# The folder ID from the URL is: 1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx
FOLDER_ID="1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx"

if gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${GT_ROOT}/" --remaining-ok 2>/dev/null; then
    echo "  ✓ Download completed successfully"
else
    echo "  ⚠ Automatic download failed"
    echo
    echo "  Please download manually:"
    echo "    1. Open: ${DOWNLOAD_URL}"
    echo "    2. Download all files/folders"
    echo "    3. Extract to: ${GT_ROOT}/"
    echo
    echo "  Expected structure:"
    echo "    ${GT_ROOT}/"
    echo "      ├── multiscan_annotations_search3d/"
    echo "      │   ├── obj_annotations/"
    echo "      │   ├── part_annotations/"
    echo "      │   └── ov_part_annotations/    ← Main evaluation data"
    echo "      ├── multiscan_test_plys_only/"
    echo "      └── multiscan_search3d_constants.py"
    echo
    read -p "  Press Enter after manual download is complete..."
fi

# Step 4: Verify structure
echo
echo "[Step 4/4] Verifying data structure..."

REQUIRED_DIRS=(
    "${GT_ROOT}/multiscan_annotations_search3d/ov_part_annotations"
    "${GT_ROOT}/multiscan_test_plys_only"
)

all_found=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "  ✓ Found: $dir ($count files)"
    else
        echo "  ✗ Missing: $dir"
        all_found=false
    fi
done

if [ -f "${GT_ROOT}/multiscan_search3d_constants.py" ]; then
    echo "  ✓ Found: multiscan_search3d_constants.py"
else
    echo "  ✗ Missing: multiscan_search3d_constants.py"
    all_found=false
fi

echo
if [ "$all_found" = true ]; then
    echo "✅ Ground Truth setup complete!"
    echo
    echo "GT path for evaluation: ${GT_ROOT}/multiscan_annotations_search3d/ov_part_annotations"
    echo
    echo "Usage:"
    echo "  python scripts/quick_eval_experiment.py \\"
    echo "    --exp-dir outputs/experiments/YOUR_EXP/scene_XXXXX_XX \\"
    echo "    --dataset-root data/multiscan \\"
    echo "    --gt-path ${GT_ROOT}/multiscan_annotations_search3d/ov_part_annotations \\"
    echo "    --output-dir eval_results"
else
    echo "❌ Setup incomplete. Please check missing files."
    exit 1
fi
