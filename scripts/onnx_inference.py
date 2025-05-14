#!/usr/bin/env python
"""
onnx_inference.py - run inference or dataset evaluation with a
DeiT-tiny ONNX model produced by `inference.py --export-onnx …`.

------------------------------------------------------------------------
Usage examples
------------------------------------------------------------------------
# 1) Export the model once (from the original script)
python inference.py --weights results/checkpoint.pth.tar \
       --export-onnx deit_tiny.onnx --ort             # optional --ort

# 2) Single-image inference (Top-5)
python onnx_inference.py --onnx deit_tiny.onnx \
       --single-image sample.png --labels imagenet_labels.json

# 3) Full ImageNet validation set
python evaluate_deit_tiny_onnx.py --onnx deit_tiny.onnx \
       --data-path /path/to/imagenet --batch-size 64 --device cuda
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

# We keep torchvision / torch ONLY for transforms and the ImageNet loader.
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_session(onnx_path: str | Path, device: str = "cpu") -> ort.InferenceSession:
    """Create an ONNX Runtime session on CPU or CUDA."""
    device = device.lower()
    if device.startswith("cuda") or device.startswith("gpu"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    # NB: the model is already optimised if --ort during export
    return ort.InferenceSession(str(onnx_path), providers=providers, sess_options=sess_opts)


def preprocess_image(path: str | Path) -> np.ndarray:
    """Load + normalise one image (224x224, NCHW) as expected by DeiT-tiny."""
    tf = T.Compose([
        T.Resize(int(224 * 1.14)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(Image.open(path).convert("RGB"))     # (C,H,W), torch.float32
    return tensor.unsqueeze(0).numpy()               # -> (1,C,H,W), numpy


def evaluate_dataset(session: ort.InferenceSession,
                     loader: DataLoader,
                     input_name: str) -> tuple[float, float]:
    """Compute Top-1 / Top-5 accuracy on the dataloader."""
    correct1 = correct5 = total = 0
    for imgs, targets in loader:
        logits = session.run(None, {input_name: imgs.numpy()})[0]  # (B,1000)
        top5 = np.argsort(-logits, axis=1)[:, :5]                  # (B,5)
        pred1 = top5[:, 0]
        targets_np = targets.numpy()
        correct1 += np.sum(pred1 == targets_np)
        # membership test for each row
        correct5 += sum(t in p for t, p in zip(targets_np, top5))
        total += imgs.shape[0]
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    return top1, top5


def predict_single(session: ort.InferenceSession,
                   img: np.ndarray,
                   input_name: str,
                   labels_map: dict[int, str] | None = None) -> None:
    logits = session.run(None, {input_name: img})[0][0]            # (1000,)
    probs = np.exp(logits) / np.exp(logits).sum()                  # softmax w/o torch
    top5_idx = probs.argsort()[-5:][::-1]
    print("Top-5 predictions:")
    for rank, idx in enumerate(top5_idx, 1):
        label = labels_map.get(idx, str(idx)) if labels_map else str(idx)
        print(f"  #{rank}: class {idx:4} - {label:<25} | p={probs[idx]:.4f}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Inference / evaluation with a DeiT-tiny ONNX model")
    ap.add_argument("--onnx", required=True, help="Path to *.onnx file produced by evaluate_deit_tiny.py")
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (selects ORT ExecutionProvider)")
    ap.add_argument("--batch-size", type=int, default=128, help="Validation batch size (dataset mode)")

    # dataset mode
    ap.add_argument("--data-path", default=None,
                    help="ImageNet root containing 'val'. If omitted and --single-image is not set, the script aborts.")

    # single-image mode
    ap.add_argument("--single-image", nargs="?", const="sample.png", default=None,
                    help="Run on exactly one image (default: sample.png).")
    ap.add_argument("--labels", default=None,
                    help="Optional JSON map class-id to human-readable label")
    args = ap.parse_args()

    # ─────────── session ────────────────────────────────────────────────
    if not Path(args.onnx).exists():
        sys.exit(f"ERR: ONNX model '{args.onnx}' not found")
    ort_sess = make_session(args.onnx, device=args.device)
    input_name = ort_sess.get_inputs()[0].name

    # ─────────── single image mode ──────────────────────────────────────
    if args.single_image is not None:
        img_path = Path(args.single_image)
        if not img_path.exists():
            sys.exit(f"ERR: image '{img_path}' not found")
        img = preprocess_image(img_path)                     # (1,3,224,224)
        labels_map = None
        if args.labels:
            with open(args.labels) as fp:
                labels_map = {int(k): v for k, v in json.load(fp).items()}
        predict_single(ort_sess, img, input_name, labels_map)
        return

    # ─────────── dataset mode ───────────────────────────────────────────
    if args.data_path is None:
        sys.exit("ERR: --data-path is required unless --single-image is used")

    val_dir = Path(args.data_path) / "val"
    if not val_dir.exists():
        sys.exit(f"ERR: validation directory '{val_dir}' not found")

    tf_val = T.Compose([
        T.Resize(int(224 * 1.14)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=tf_val)
    loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True)

    top1, top5 = evaluate_dataset(ort_sess, loader, input_name)
    print(f"Validation set: Top-1 = {top1:.2f}%  |  Top-5 = {top5:.2f}%")


if __name__ == "__main__":
    main()
