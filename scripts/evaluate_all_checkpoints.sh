#!/bin/bash
# Evaluate all checkpoints with their respective configurations
# This script reads from checkpoints.json and runs evaluation for each checkpoint

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RESULTS_DIR="$PROJECT_ROOT/results"
DATA_PATH="/scratch2/ml_datasets/ILSVRC2012"
DEVICE="${1:-cuda:2}"

echo "================================================="
echo "Checkpoint Evaluation Suite"
echo "================================================="
echo "Data path: $DATA_PATH"
echo "Device: $DEVICE"
echo ""

# Function to run evaluation for a checkpoint
run_evaluation() {
    local checkpoint_file=$1
    local gelu_type=$2
    local softmax_type=$3
    local layernorm_type=$4
    local bitwidth=$5
    
    local checkpoint_path="$RESULTS_DIR/$checkpoint_file"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "Error: Checkpoint not found: $checkpoint_file"
        return 1
    fi
    
    echo "================================================="
    echo "Evaluating: $checkpoint_file"
    echo "GELU: $gelu_type"
    echo "Softmax: $softmax_type"
    echo "LayerNorm: $layernorm_type"
    echo "Bitwidth: $bitwidth"
    echo "================================================="
    
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/inference.py" \
        --weights "$checkpoint_path" \
        --data-path "$DATA_PATH" \
        --device "$DEVICE" \
        --batch-size 128 \
        --gelu-type "$gelu_type" \
        --softmax-type "$softmax_type" \
        --layernorm-type "$layernorm_type" \
        --bitwidth "$bitwidth" \
        --num-calibration-batches 100 \
        --no-io-stats \
        --use-random-warmup \
    
    echo ""
}

# Evaluate each checkpoint

echo "1/6: Evaluating checkpoint_ivit_16b (I-ViT, INT16)"
run_evaluation "checkpoint_ivit_16b.pth.tar" \
    "ivit" "ivit" "ivit" 16

echo "2/6: Evaluating checkpoint_0965b4e299764beb9005f39bc24ac5b2 (I-ViT, INT8)"
run_evaluation "checkpoint_0965b4e299764beb9005f39bc24ac5b2.pth.tar" \
    "ivit" "ivit" "ivit" 8

echo "3/6: Evaluating checkpoint_9f32c5ad39174ff88e0107293039530c (I-BERT, INT16)"
run_evaluation "checkpoint_9f32c5ad39174ff88e0107293039530c.pth.tar" \
    "ibert" "ibert" "ibert" 16

echo "4/6: Evaluating checkpoint_a04a495ac414499c86b097ded7fc1b7a (I-BERT, INT8)"
run_evaluation "checkpoint_a04a495ac414499c86b097ded7fc1b7a.pth.tar" \
    "ibert" "ibert" "ibert" 8

echo "5/6: Evaluating checkpoint_3b9f5d806f1047019d2b7d498a118f0e (Float, INT16)"
run_evaluation "checkpoint_3b9f5d806f1047019d2b7d498a118f0e.pth.tar" \
    "float" \
    "float" \
    "ibert" 16

echo "6/6: Evaluating checkpoint_13e53befe81e4088b9894b4e04bc1b37 (Float, INT8)"
run_evaluation "checkpoint_13e53befe81e4088b9894b4e04bc1b37.pth.tar" \
    "float" \
    "float" \
    "ibert" 8

echo "================================================="
echo "All evaluations completed!"
echo "================================================="
