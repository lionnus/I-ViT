#!/bin/bash
# Run evaluation N times and calculate accuracy statistics

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Configuration
NUM_RUNS=${1:-5}
WEIGHTS=${2:-"results/checkpoint_a04a495ac414499c86b097ded7fc1b7a.pth.tar"}
DATA_PATH=${3:-"/scratch/ml_datasets/ILSVRC2012"}
DEVICE=${4:-"cuda:0"}
GELU_TYPE=${5:-"ppoly_deg_2_seg_16_scale-bits_26_backend_ibert_optim-bounds_false"}
SOFTMAX_TYPE=${6:-"ppoly_deg_2_seg_16_scale-bits_26_exp-bits_20_backend_ibert_optim-bounds_false"}
LAYERNORM_TYPE=${7:-"ibert"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/eval_stats_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Running $NUM_RUNS evaluations..."
echo "Weights: $WEIGHTS | Device: $DEVICE"
echo "GELU: $GELU_TYPE"
echo "Softmax: $SOFTMAX_TYPE"
echo "LayerNorm: $LAYERNORM_TYPE"
echo "Logs: $LOG_DIR/"
echo ""

# Run evaluations
declare -a TOP1 TOP5
for i in $(seq 1 $NUM_RUNS); do
    echo "=== Run $i/$NUM_RUNS ==="
    RUN_LOG="${LOG_DIR}/run_${i}.log"
    
    # Run evaluation with PYTHONPATH set to project root
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/inference.py" \
        --weights "$WEIGHTS" \
        --data-path "$DATA_PATH" \
        --device "$DEVICE" \
        --batch-size 128 \
        --gelu-type "$GELU_TYPE" \
        --softmax-type "$SOFTMAX_TYPE" \
        --layernorm-type "$LAYERNORM_TYPE" \
        --no-io-stats 2>&1 | tee "$RUN_LOG"
    
    # Extract metrics from log file
    T1=$(grep "Top-1 =" "$RUN_LOG" | grep -oP "Top-1 = \K[0-9]+\.[0-9]+")
    T5=$(grep "Top-5 =" "$RUN_LOG" | grep -oP "Top-5 = \K[0-9]+\.[0-9]+")
    
    if [ -n "$T1" ]; then
        TOP1+=("$T1")
        TOP5+=("$T5")
        echo "✓ Extracted: Top-1=$T1%, Top-5=$T5%"
    else
        echo "✗ ERROR: Failed to extract metrics from run $i"
    fi
    echo ""
done

# Calculate statistics
if [ ${#TOP1[@]} -eq 0 ]; then
    echo "ERROR: No successful runs to analyze"
    exit 1
fi

# Convert arrays to comma-separated strings for Python
TOP1_STR=$(IFS=,; echo "${TOP1[*]}")
TOP5_STR=$(IFS=,; echo "${TOP5[*]}")

python3 << EOF
import numpy as np
top1 = np.array([$TOP1_STR])
top5 = np.array([$TOP5_STR])
print("=" * 50)
print(f"Statistics from {len(top1)} successful runs:")
print("=" * 50)
print(f"Top-1: {top1.mean():.2f}% ± {top1.std():.3f}% (range: {top1.min():.2f}-{top1.max():.2f}%)")
print(f"Top-5: {top5.mean():.2f}% ± {top5.std():.3f}% (range: {top5.min():.2f}-{top5.max():.2f}%)")
print("=" * 50)
EOF

echo "Logs saved in: $LOG_DIR/"
