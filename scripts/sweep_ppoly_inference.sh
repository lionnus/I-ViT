#!/bin/bash

# =============================================================================
# Inference Parameter Sweep Script
# =============================================================================
# This script sweeps over different GELU and Softmax configurations
# for model inference evaluation on ImageNet validation set.
#
# Parameters swept:
# - GELU: degree 1 and 2, segments 8, 16, 32
# - Softmax: degree 1 and 2, segments 8, 16, 32
# - LayerNorm: fixed to 'ibert'
#
# Author: Lionnus Kesting (lkesting@ethz.ch)
# =============================================================================

# --- Configuration Parameters ---
WEIGHTS_PATH="results/checkpoint_9f32c5ad39174ff88e0107293039530c.pth.tar"
DATA_PATH="/scratch2/ml_datasets/ILSVRC2012"
DEVICE="cuda:0"
BATCH_SIZE=128
PYTHON_ENV="/scratch/lkesting/I-ViT/venv/bin/python"
WORKSPACE_DIR="/scratch/lkesting/I-ViT"

# --- Sweep Parameters (degrees & segments ONLY) ---
GELU_DEGREES=(1 2)
GELU_SEGMENTS=(8 16 32)

SOFTMAX_DEGREES=(1 2)
SOFTMAX_SEGMENTS=(8 16 32)

LAYERNORM_TYPE="ibert"

# --- Extra params (NOT swept, separate for GELU / Softmax) ---
GELU_SCALE_BITS=24
SOFTMAX_SCALE_BITS=24
SOFTMAX_EXP_BITS=16

GELU_OPTIM_BOUNDS=true      # true / false
SOFTMAX_OPTIM_BOUNDS=true    # true / false

# backend as param (NOT swept, separate for GELU / Softmax)
GELU_BACKEND="ibert"
SOFTMAX_BACKEND="ibert"

# --- Function to generate type string ---
generate_type_string() {
    local degree=$1
    local segment=$2
    local scale_bits=$3
    local backend=$4
    local optim_bounds=$5
    local exp_bits=${6:-}
    
    echo "ppoly_deg_${degree}_seg_${segment}_scale-bits_${scale_bits}${exp_bits:+_exp-bits_${exp_bits}}_backend_${backend}_optim-bounds_${optim_bounds}"
}

# --- Main sweep function ---
run_sweep() {
    echo "=========================================="
    echo "Starting Inference Parameter Sweep"
    echo "=========================================="
    echo "Weights: $WEIGHTS_PATH"
    echo "Data: $DATA_PATH"
    echo "Device: $DEVICE"
    echo "Batch size: $BATCH_SIZE"
    echo "Python env: $PYTHON_ENV"
    echo "Working directory: $WORKSPACE_DIR"
    echo "=========================================="
    
    cd "$WORKSPACE_DIR" || exit 1
    
    local total_configs=0
    local current_config=0
    
    # Count total configurations
    for gelu_deg in "${GELU_DEGREES[@]}"; do
        for gelu_seg in "${GELU_SEGMENTS[@]}"; do
            for softmax_deg in "${SOFTMAX_DEGREES[@]}"; do
                for softmax_seg in "${SOFTMAX_SEGMENTS[@]}"; do
                    ((total_configs++))
                done
            done
        done
    done
    
    echo "Total configurations to test: $total_configs"
    echo "=========================================="
    
    # Run sweep
    for gelu_deg in "${GELU_DEGREES[@]}"; do
        for gelu_seg in "${GELU_SEGMENTS[@]}"; do
            for softmax_deg in "${SOFTMAX_DEGREES[@]}"; do
                for softmax_seg in "${SOFTMAX_SEGMENTS[@]}"; do
                    ((current_config++))
                    
                    gelu_type=$(generate_type_string \
                        "$gelu_deg" "$gelu_seg" "$GELU_SCALE_BITS" "$GELU_BACKEND" "$GELU_OPTIM_BOUNDS")
                    
                    softmax_type=$(generate_type_string \
                        "$softmax_deg" "$softmax_seg" "$SOFTMAX_SCALE_BITS" "$SOFTMAX_BACKEND" "$SOFTMAX_OPTIM_BOUNDS" "$SOFTMAX_EXP_BITS")
                    
                    echo ""
                    echo "[$current_config/$total_configs] Configuration:"
                    echo "  GELU:      $gelu_type"
                    echo "  Softmax:   $softmax_type"
                    echo "  LayerNorm: $LAYERNORM_TYPE"
                    echo "----------------------------------------"
                    
                    start_time=$(date +%s)
                    
                    "$PYTHON_ENV" -m scripts.inference \
                        --weights "$WEIGHTS_PATH" \
                        --data-path "$DATA_PATH" \
                        --device "$DEVICE" \
                        --batch-size "$BATCH_SIZE" \
                        --gelu-type "$gelu_type" \
                        --softmax-type "$softmax_type" \
                        --layernorm-type "$LAYERNORM_TYPE" \
                        --use-random-warmup \
                        --no-io-stats 
                    
                    exit_code=$?
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    
                    if [ $exit_code -eq 0 ]; then
                        echo "? Configuration completed successfully in ${duration}s"
                    else
                        echo "? Configuration failed with exit code $exit_code after ${duration}s"
                    fi
                    
                    echo "========================================"
                done
            done
        done
    done
    
    echo ""
    echo "Sweep completed!"
    echo "Total configurations tested: $total_configs"
}

# --- Help function ---
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --weights PATH Override weights path (default: $WEIGHTS_PATH)"
    echo "  --data PATH    Override data path (default: $DATA_PATH)"
    echo "  --device DEV   Override device (default: $DEVICE)"
    echo "  --batch-size N Override batch size (default: $BATCH_SIZE)"
    echo "  --python PATH  Override python environment path (default: $PYTHON_ENV)"
    echo ""
    echo "To redirect output to a file:"
    echo "  $0 > sweep_results.txt 2>&1"
}

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --weights)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --python)
            PYTHON_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# --- Validate required paths exist ---
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: Weights file not found: $WEIGHTS_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data directory not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_ENV" ]; then
    echo "Error: Python environment not found: $PYTHON_ENV"
    exit 1
fi

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Error: Workspace directory not found: $WORKSPACE_DIR"
    exit 1
fi

# --- Run the sweep ---
run_sweep
