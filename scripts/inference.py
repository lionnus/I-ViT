#!/usr/bin/env python
# Author: Lionnus Kesting (lkesting@ethz.ch)
"""
evaluate_deit_tiny.py - inference / evaluation utility for DeiT-tiny
====================================================================
Evaluate given weights on ImageNet validation set and report Top-1/Top-5 accuracy.

In both modes statistics about the model are collected using `attach_io_stat_hooks`.
After forward-pass completion it writes them via `save_io_stats_df`, naming the 
file automatically based on the mode.
"""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from models import *
from utils import *
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def load_model(checkpoint_path, device='cuda', num_classes=1000, gelu_type=None, softmax_type=None, layernorm_type=None):
    """
    Load the model with the given weights checkpoint, checks for configuration.
    Optional parameters override the saved configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model_config exists in checkpoint
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"\nLoading model with saved configuration:")
        print(f"  Model: {config.get('model_name', 'deit_tiny')}")
        print(f"  Quantization bitwidths: {config.get('quant_bitwidths', [8,8,8,8,8,8,8,8])}")
        
        # Use override parameters if provided, otherwise use saved config
        final_gelu_type = gelu_type if gelu_type is not None else config.get('gelu_type', 'ibert')
        final_softmax_type = softmax_type if softmax_type is not None else config.get('softmax_type', 'ibert')
        final_layernorm_type = layernorm_type if layernorm_type is not None else config.get('layernorm_type', 'ibert')
        
        print(f"  GELU type: {final_gelu_type} {'(overridden)' if gelu_type is not None else ''}")
        print(f"  Softmax type: {final_softmax_type} {'(overridden)' if softmax_type is not None else ''}")
        print(f"  LayerNorm type: {final_layernorm_type} {'(overridden)' if layernorm_type is not None else ''}")
        
        # Create model with configuration
        model = deit_tiny_patch16_224(
            pretrained=False,
            num_classes=config.get('num_classes', num_classes),
            drop_rate=config.get('drop_rate', 0.0),
            drop_path_rate=config.get('drop_path_rate', 0.1),
            patch_embed_bw=config.get('patch_embed_bw', 8),
            pos_encoding_bw=config.get('pos_encoding_bw', 8),
            block_input_bw=config.get('block_input_bw', 8),
            attention_out_bw=config.get('attention_out_bw', 8),
            softmax_bw=config.get('softmax_bw', 8),
            mlp_out_bw=config.get('mlp_out_bw', 8),
            norm2_in_bw=config.get('norm2_in_bw', 8),
            att_block_out_bw=config.get('att_block_out_bw', 8),
            gelu_type=final_gelu_type,
            softmax_type=final_softmax_type,
            layernorm_type=final_layernorm_type
        )
        
        # Print additional info if available
        if 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_acc1' in checkpoint:
            print(f"  Best validation accuracy: {checkpoint['best_acc1']:.2f}%")
    else:
        # Old checkpoint format or missing config - use defaults or overrides
        print("\nNo model configuration found in checkpoint, using default/override configuration")
        
        # Use override parameters if provided, otherwise use defaults
        final_gelu_type = gelu_type if gelu_type is not None else 'ppoly_deg_1_seg_32_scale-bits_30_backend_ibert'
        final_softmax_type = softmax_type if softmax_type is not None else 'ppoly_deg_1_seg_32_scale-bits_30_backend_ibert'
        final_layernorm_type = layernorm_type if layernorm_type is not None else 'ibert'
        
        print(f"  GELU type: {final_gelu_type}")
        print(f"  Softmax type: {final_softmax_type}")
        print(f"  LayerNorm type: {final_layernorm_type}")
        
        model = deit_tiny_patch16_224(
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            gelu_type=final_gelu_type,
            softmax_type=final_softmax_type,
            layernorm_type=final_layernorm_type
        )
    
    # Load model weights
    if 'model' in checkpoint:
        # New format with model state dict under 'model' key
        checkpoint_weights = checkpoint['model']
    else:
        # Old format where checkpoint is the state dict
        checkpoint_weights = checkpoint
    
    model_state_dict = model.state_dict()
    for name, param in checkpoint_weights.items():
        if name in model_state_dict:
            # If checkpoint param is a scalar [] but the model expects [1], unsqueeze it.
            if param.shape == torch.Size([]) and model_state_dict[name].shape == torch.Size([1]):
                checkpoint_weights[name] = param.unsqueeze(0)
    
    model.load_state_dict(checkpoint_weights, strict=False)
    model.to(device)
    model.eval()
    
    # One dummy forward pass to warm up the model
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Freeze the model to prevent further training
    freeze_model(model)
    return model

def preprocess_image(path: Path | str) -> torch.Tensor:
    """Load and normalise a single image for DeiT-tiny (224by224)."""
    tf = T.Compose([
        T.Resize(int(224 * 1.14)),  # 256 if you prefer standard
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img)


# -----------------------------------------------------------------------------
#  Evaluation / inference helpers
# -----------------------------------------------------------------------------

def evaluate_dataset(model, data_loader, device, *, print_batch_stats: bool = True):
    """Run full evaluation.

    Returns
    -------
    tuple[float, float]
        Top-1 and Top-5 accuracy in percent.
    """
    correct1 = correct5 = tot = 0
    batch_times: list[float] = []
    start_total = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for imgs, targets in tqdm(data_loader, desc="Val", unit="batch", dynamic_ncols=True):
            t0 = time.perf_counter()
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            _, pred5 = logits.topk(5, dim=1)
            pred1 = pred5[:, 0]
            correct1 += (pred1 == targets).sum().item()
            correct5 += sum(t in p for t, p in zip(targets, pred5))
            tot += imgs.size(0)
            t_batch = time.perf_counter() - t0
            batch_times.append(t_batch)

    # ───── summary timing ─────────────────────────────────────────────
    total_time = time.perf_counter() - start_total
    avg_batch  = sum(batch_times) / len(batch_times)
    avg_img    = total_time / tot if tot else 0.0
    print(
        f"Finished evaluation: total={total_time:.1f}s | "
        f"avg/batch={avg_batch*1000:.1f} ms | "
        f"avg/img={avg_img*1000:.2f} ms")

    return 100 * correct1 / tot, 100 * correct5 / tot


def predict_single(model, img: torch.Tensor, device, labels_map: dict[int, str] | None):
    """Run forward pass on *one* image tensor and print Top-5."""
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]
        top5_prob, top5_idx = probs.topk(5)
    latency = (time.perf_counter() - start) * 1000

    print("Top-5 predictions (latency = {:.1f} ms):".format(latency))
    for rank, (p, idx) in enumerate(zip(top5_prob.tolist(), top5_idx.tolist()), 1):
        label = labels_map.get(idx, str(idx)) if labels_map else str(idx)
        print(f"  #{rank}: class {idx:4} - {label:<25}  |  p={p:.4f}")


# -----------------------------------------------------------------------------
#  CLI / entry-point
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate DeiT-tiny or run single-image inference")
    ap.add_argument("--weights", default="results/checkpoint.pth.tar",
                    help="Path to model checkpoint (*.pth.tar)")
    ap.add_argument("--device", default="cuda:1",
                    help="Compute device (e.g. 'cuda:0' or 'cpu')")

    # Dataset config
    ap.add_argument("--data-path", default=None,
                    help="Path to ImageNet root containing 'val'. If omitted and --single-image is not set, script aborts.")
    ap.add_argument("--batch-size", type=int, default=128, help="Validation batch size")
    
    # Model configuration overrides
    ap.add_argument("--gelu-type", default=None,
                    help="Override GELU type (e.g., 'ppoly_deg_1_seg_32_scale-bits_30_backend_ibert')")
    ap.add_argument("--softmax-type", default=None,
                    help="Override Softmax type (e.g., 'ppoly_deg_1_seg_32_scale-bits_30_backend_ibert')")
    ap.add_argument("--layernorm-type", default=None,
                    help="Override LayerNorm type (e.g., 'ibert')")

    # Export
    ap.add_argument("--export-onnx", default=None,
                    help="Export the model to ONNX and exit (path to *.onnx)")
    ap.add_argument("--ort", action="store_true",
                    help="Run ORT extended graph fusions on the exported ONNX model")
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # Load model from checkpoint.
    print(f"Loading checkpoint from: {args.weights}")
    model = load_model(args.weights, device=device, num_classes=1000, 
                      gelu_type=args.gelu_type, 
                      softmax_type=args.softmax_type, 
                      layernorm_type=args.layernorm_type)
    
    # If --export-onnx flag is provided, export to ONNX and exit.
    if args.export_onnx is not None:
        # Create a dummy input tensor for the model
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        # Export the model to ONNX format
        onnx_output = args.export_onnx
        # 1) Export FP32 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            do_constant_folding=True
        )

        # 2) Optionally run ORT extended graph fusions
        if args.ort:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            so.optimized_model_filepath = onnx_output
            # create session to trigger optimization and dump optimized model
            _ = ort.InferenceSession(onnx_output, sess_options=so)
            print("Optimized (ORT_ENABLE_EXTENDED) model saved ->", onnx_output)
        else:
            print("ONNX model exported ->", onnx_output)
        return
    
    # Attach IO stat hooks to the model
    attach_io_stat_hooks(model)
    

    # Dataset eval
    if args.data_path is None:
        sys.exit("ERR: --data-path is required")

    val_dir = Path(args.data_path) / "val"
    if not val_dir.exists():
        sys.exit(f"ERR: validation directory '{val_dir}' not found")

    print(f"Evaluating on validation set: {val_dir}")
    
    tf = T.Compose([
        T.Resize(int(224 * 1.14)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_data = torchvision.datasets.ImageFolder(val_dir, transform=tf)
    loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=16, pin_memory=True)

    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Starting evaluation ...")
    top1, top5 = evaluate_dataset(model, loader, device, print_batch_stats=True)
    print(f"Validation set: Top-1 = {top1:.2f}%  |  Top-5 = {top5:.2f}%")
    save_io_stats_df("io_stats_val.pkl", to_csv=True)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Evaluation finished.")

if __name__ == "__main__":
    main()