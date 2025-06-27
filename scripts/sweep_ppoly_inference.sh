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
WEIGHTS_PATH="results/checkpoint_a04a495ac414499c86b097ded7fc1b7a.pth.tar"
DATA_PATH="/scratch/ml_datasets/ILSVRC2012"
DEVICE="cuda:3"
BATCH_SIZE=128
PYTHON_ENV="/scratch/msc25f5/I-ViT/venv/bin/python"
WORKSPACE_DIR="/scratch/msc25f5/I-ViT"

# --- Sweep Parameters ---
DEGREES=(1 2)
SEGMENTS=(8 16 32)
LAYERNORM_TYPE="ibert"

# --- Function to generate type string ---
generate_type_string() {
    local degree=$1
    local segment=$2
    echo "ppoly_deg_${degree}_seg_${segment}_scale-bits_30_backend_ibert"
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
    
    # Change to workspace directory
    cd "$WORKSPACE_DIR" || exit 1
    
    local total_configs=0
    local current_config=0
    
    # Count total configurations
    for gelu_deg in "${DEGREES[@]}"; do
        for gelu_seg in "${SEGMENTS[@]}"; do
            for softmax_deg in "${DEGREES[@]}"; do
                for softmax_seg in "${SEGMENTS[@]}"; do
                    ((total_configs++))
                done
            done
        done
    done
    
    echo "Total configurations to test: $total_configs"
    echo "=========================================="
    
    # Run sweep
    for gelu_deg in "${DEGREES[@]}"; do
        for gelu_seg in "${SEGMENTS[@]}"; do
            for softmax_deg in "${DEGREES[@]}"; do
                for softmax_seg in "${SEGMENTS[@]}"; do
                    ((current_config++))
                    
                    # Generate type strings
                    gelu_type=$(generate_type_string $gelu_deg $gelu_seg)
                    softmax_type=$(generate_type_string $softmax_deg $softmax_seg)
                    
                    echo ""
                    echo "[$current_config/$total_configs] Configuration:"
                    echo "  GELU:      $gelu_type"
                    echo "  Softmax:   $softmax_type"
                    echo "  LayerNorm: $LAYERNORM_TYPE"
                    echo "----------------------------------------"
                    
                    # Run inference
                    start_time=$(date +%s)
                    
                    "$PYTHON_ENV" -m inference \
                        --weights "$WEIGHTS_PATH" \
                        --data-path "$DATA_PATH" \
                        --device "$DEVICE" \
                        --batch-size "$BATCH_SIZE" \
                        --gelu-type "$gelu_type" \
                        --softmax-type "$softmax_type" \
                        --layernorm-type "$LAYERNORM_TYPE"
                    
                    exit_code=$?
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    
                    if [ $exit_code -eq 0 ]; then
                        echo "✓ Configuration completed successfully in ${duration}s"
                    else
                        echo "✗ Configuration failed with exit code $exit_code after ${duration}s"
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
