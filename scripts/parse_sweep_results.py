#!/usr/bin/env python3
"""
Parse sweep results and output to CSV
"""
import re
import csv
import sys
from pathlib import Path

def parse_ppoly_params(ppoly_string):
    """Parse ppoly configuration string and extract parameters."""
    params = {}
    
    # Extract degree
    deg_match = re.search(r'deg_(\d+)', ppoly_string)
    if deg_match:
        params['deg'] = deg_match.group(1)
    
    # Extract segments
    seg_match = re.search(r'seg_(\d+)', ppoly_string)
    if seg_match:
        params['seg'] = seg_match.group(1)
    
    # Extract scale-bits
    scale_bits_match = re.search(r'scale-bits_(\d+)', ppoly_string)
    if scale_bits_match:
        params['scale_bits'] = scale_bits_match.group(1)
    
    # Extract exp-bits (for softmax)
    exp_bits_match = re.search(r'exp-bits_(\d+)', ppoly_string)
    if exp_bits_match:
        params['exp_bits'] = exp_bits_match.group(1)
    
    # Extract backend
    backend_match = re.search(r'backend_(\w+)', ppoly_string)
    if backend_match:
        params['backend'] = backend_match.group(1)
    
    # Extract optim-bounds
    if 'optim-bounds_true' in ppoly_string:
        params['optim_bounds'] = 'true'
    elif 'optim-bounds_false' in ppoly_string:
        params['optim_bounds'] = 'false'
    
    return params

def parse_sweep_results(input_file):
    """Parse sweep results file and extract configuration and accuracy data."""
    
    results = []
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split by configuration blocks
    config_blocks = re.split(r'\[\d+/\d+\] Configuration:', content)
    
    # Skip the header (first split)
    for block in config_blocks[1:]:
        result = {}
        
        # Extract GELU type
        gelu_match = re.search(r'GELU:\s+(\S+)', block)
        if gelu_match:
            gelu_full = gelu_match.group(1)
            result['gelu'] = gelu_full
            # Parse GELU parameters
            gelu_params = parse_ppoly_params(gelu_full)
            for key, val in gelu_params.items():
                result[f'gelu_{key}'] = val
        
        # Extract Softmax type
        softmax_match = re.search(r'Softmax:\s+(\S+)', block)
        if softmax_match:
            softmax_full = softmax_match.group(1)
            result['softmax'] = softmax_full
            # Parse Softmax parameters
            softmax_params = parse_ppoly_params(softmax_full)
            for key, val in softmax_params.items():
                result[f'softmax_{key}'] = val
        
        # Extract LayerNorm type
        layernorm_match = re.search(r'LayerNorm:\s+(\S+)', block)
        if layernorm_match:
            result['layernorm'] = layernorm_match.group(1)
        
        # Extract checkpoint path
        checkpoint_match = re.search(r'Loading checkpoint from:\s+(.+)', block)
        if checkpoint_match:
            checkpoint_path = checkpoint_match.group(1).strip()
            # Extract checkpoint ID from filename
            checkpoint_id_match = re.search(r'checkpoint_([a-f0-9]+)\.pth\.tar', checkpoint_path)
            if checkpoint_id_match:
                result['checkpoint_id'] = checkpoint_id_match.group(1)
            else:
                # Handle other checkpoint naming schemes (e.g., ivit_16b)
                checkpoint_name = Path(checkpoint_path).stem.replace('checkpoint_', '').replace('.pth', '')
                result['checkpoint_id'] = checkpoint_name
        
        # Extract model info
        model_match = re.search(r'Model:\s+(\S+)', block)
        if model_match:
            result['model'] = model_match.group(1)
        
        # Extract quantization bitwidths
        bitwidth_match = re.search(r'Quantization bitwidths:\s+\[([^\]]+)\]', block)
        if bitwidth_match:
            bitwidths = bitwidth_match.group(1)
            # Take the first value (they should all be the same)
            first_bitwidth = bitwidths.split(',')[0].strip()
            result['bitwidth'] = first_bitwidth
        
        # Extract epoch
        epoch_match = re.search(r'Checkpoint from epoch:\s+(\d+)', block)
        if epoch_match:
            result['epoch'] = epoch_match.group(1)
        
        # Extract best validation accuracy from checkpoint
        best_acc_match = re.search(r'Best validation accuracy:\s+([\d.]+)%', block)
        if best_acc_match:
            result['checkpoint_best_acc'] = best_acc_match.group(1)
        
        # Extract Top-1 accuracy (ignore lines with WARNING)
        top1_matches = re.findall(r'^(?!\[WARNING\]).*Top-1\s*=\s*([\d.]+)%', block, re.MULTILINE)
        if top1_matches:
            result['top1'] = top1_matches[-1]  # Take the last match (final result)
        
        # Extract Top-5 accuracy (ignore lines with WARNING)
        top5_matches = re.findall(r'^(?!\[WARNING\]).*Top-5\s*=\s*([\d.]+)%', block, re.MULTILINE)
        if top5_matches:
            result['top5'] = top5_matches[-1]  # Take the last match (final result)
        
        # Extract timing info
        time_match = re.search(r'avg/batch=([\d.]+)\s*ms.*avg/img=([\d.]+)\s*ms', block)
        if time_match:
            result['avg_batch_ms'] = time_match.group(1)
            result['avg_img_ms'] = time_match.group(2)
        
        # Extract total time
        total_time_match = re.search(r'completed successfully in (\d+)s', block)
        if total_time_match:
            result['total_time_s'] = total_time_match.group(1)
        
        # Only add if we have the essential data
        if 'gelu' in result and 'top1' in result:
            results.append(result)
    
    return results

def write_csv(results, output_file):
    """Write results to CSV file."""
    
    if not results:
        print("No results to write!")
        return
    
    # Define column order
    fieldnames = [
        'checkpoint_id',
        'model',
        'bitwidth',
        'epoch',
        'checkpoint_best_acc',
        'gelu',
        'gelu_deg',
        'gelu_seg',
        'gelu_scale_bits',
        'gelu_backend',
        'gelu_optim_bounds',
        'softmax',
        'softmax_deg',
        'softmax_seg',
        'softmax_scale_bits',
        'softmax_exp_bits',
        'softmax_backend',
        'softmax_optim_bounds',
        'layernorm',
        'top1',
        'top5',
        'avg_batch_ms',
        'avg_img_ms',
        'total_time_s'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Wrote {len(results)} results to {output_file}")

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'results/sweep_results_2025-12-10.txt'
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = input_file.replace('.txt', '.csv')
    
    print(f"Parsing {input_file}...")
    results = parse_sweep_results(input_file)
    print(f"Found {len(results)} configurations")
    
    write_csv(results, output_file)
    print(f"Results saved to {output_file}")
    
    # Print summary
    if results:
        print("\nSummary:")
        print(f"  Top-1 range: {min(float(r['top1']) for r in results):.2f}% - {max(float(r['top1']) for r in results):.2f}%")
        print(f"  Top-5 range: {min(float(r['top5']) for r in results):.2f}% - {max(float(r['top5']) for r in results):.2f}%")

if __name__ == "__main__":
    main()
