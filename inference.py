#!/usr/bin/env python
"""
evaluate_deit_tiny.py – inference / evaluation utility for DeiT‑tiny
====================================================================
Two modes of operation
---------------------
1. **Dataset evaluation** (default)
   Evaluate on an ImageNet‑style validation set and report Top‑1/Top‑5 accuracy.

2. **Single‑image inference** via `--single-image [<path>]`
   Run the model on exactly one image (default: `sample.png`).
   Prints the Top‑5 predicted class indices (or the human‑readable labels if
   `--labels` is given), saves the per‑layer IO‑stats, and exits.

The script keeps `attach_io_stat_hooks` in both modes so the same statistics
are collected.  After forward‑pass completion it writes them via
`save_io_stats_df`, naming the file automatically based on the mode.
"""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
import torchvision

# --- project‑specific imports -------------------------------------------------
from models import deit_tiny_patch16_224
from models.quantization_utils.quant_modules import attach_io_stat_hooks, save_io_stats_df

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def load_model(checkpoint_path, device='cuda', num_classes=1000):
    """
    Load the model with the given weights checkpoint.
    """
    model = deit_tiny_patch16_224(pretrained=False, num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name in model_state_dict:
            # If checkpoint param is a scalar [] but the model expects [1], unsqueeze it.
            if param.shape == torch.Size([]) and model_state_dict[name].shape == torch.Size([1]):
                checkpoint[name] = param.unsqueeze(0)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    attach_io_stat_hooks(model)
    return model


def preprocess_image(path: Path | str) -> torch.Tensor:
    """Load and normalise a single image for DeiT‑tiny (224×224)."""
    tf = T.Compose([
        T.Resize(int(224 * 1.14)),  # 256 if you prefer standard
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img)


def evaluate_dataset(model, data_loader, device):
    correct1 = correct5 = tot = 0
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            _, pred5 = logits.topk(5, dim=1)
            pred1 = pred5[:, 0]
            correct1 += (pred1 == targets).sum().item()
            correct5 += sum(t in p for t, p in zip(targets, pred5))
            tot += imgs.size(0)
    return 100 * correct1 / tot, 100 * correct5 / tot


def predict_single(model, img: torch.Tensor, device, labels_map: dict[int, str] | None):
    """Run forward pass on *one* image tensor and print Top‑5."""
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)[0]
        top5_prob, top5_idx = probs.topk(5)
    print("Top‑5 predictions:")
    for rank, (p, idx) in enumerate(zip(top5_prob.tolist(), top5_idx.tolist()), 1):
        label = labels_map.get(idx, str(idx)) if labels_map else str(idx)
        print(f"  #{rank}: class {idx:4} – {label:<25}  |  p={p:.4f}")


# -----------------------------------------------------------------------------
#  CLI / entry‑point
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate DeiT‑tiny or run single‑image inference")
    ap.add_argument("--weights", default="results/checkpoint.pth.tar",
                    help="Path to model checkpoint (*.pth.tar)")
    ap.add_argument("--device", default="cuda:0",
                    help="Compute device (e.g. 'cuda:0' or 'cpu')")

    # Dataset mode
    ap.add_argument("--data-path", default=None,
                    help="Path to ImageNet root containing 'val'. If omitted and --single-image is not set, script aborts.")
    ap.add_argument("--batch-size", type=int, default=128, help="Validation batch size")

    # Single‑image mode
    ap.add_argument("--single-image", nargs="?", const="sample.png", default=None,
                    help="Run on exactly one image. If the flag is given without a value, uses 'sample.png'.")
    ap.add_argument("--labels", default=None,
                    help="Optional JSON mapping class‑id → label for pretty output")

    # Export
    ap.add_argument("--export-onnx", default=None,
                    help="Export the model to ONNX and exit (path to *.onnx)")
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # Load model from checkpoint.
    model = load_model(args.weights, device=device, num_classes=1000)
    
    # If --export-onnx flag is provided, export to ONNX and exit.
    if args.export_onnx is not None:
        # Create a dummy input with batch size 1 (change shape as needed)
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        traced = torch.jit.trace(model, dummy_input)
        # Save the traced model
        onnx_output = args.export_onnx
        # Export the model to ONNX format
        torch.onnx.export(
            traced,
            dummy_input,
            onnx_output,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            export_params=True,
            do_constant_folding=True,
        )
        print("ONNX model exported →", args.export_onnx)
        return

    # ───────── single‑image inference ────────────────────────────────────
    if args.single_image is not None:
        img_path = Path(args.single_image)
        if not img_path.exists():
            sys.exit(f"ERR: image '{img_path}' not found")
        img_tensor = preprocess_image(img_path)
        # optional labels map
        labels_map = None
        if args.labels:
            with open(args.labels) as fp:
                labels_map = {int(k): v for k, v in json.load(fp).items()}
        predict_single(model, img_tensor, device, labels_map)
        save_io_stats_df("io_stats_single.pkl", to_csv=True)
        return

    # ───────── dataset evaluation ────────────────────────────────────────
    if args.data_path is None:
        sys.exit("ERR: --data-path is required unless --single-image is used")

    val_dir = Path(args.data_path) / "val"
    if not val_dir.exists():
        sys.exit(f"ERR: validation directory '{val_dir}' not found")

    tf = T.Compose([
        T.Resize(int(224 * 1.14)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_data = torchvision.datasets.ImageFolder(val_dir, transform=tf)
    loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=16, pin_memory=True)

    top1, top5 = evaluate_dataset(model, loader, device)
    print(f"Validation set: Top‑1 = {top1:.2f}%  |  Top‑5 = {top5:.2f}%")
    save_io_stats_df("io_stats_val.pkl", to_csv=True)


if __name__ == "__main__":
    main()
