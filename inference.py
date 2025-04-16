import argparse
import json
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import deit_tiny_patch16_224  # Make sure this imports your DeiT-tiny model definition

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
    return model

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the provided DataLoader.
    Returns Top-1 and Top-5 accuracies.
    """
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            # Compute Top-1 and Top-5 predictions
            _, pred_top5 = outputs.topk(5, dim=1)
            
            # Top-1 accuracy
            pred_top1 = pred_top5[:, 0]
            correct_top1 += pred_top1.eq(targets).sum().item()
            
            # Top-5 accuracy
            # For each sample in the batch, check if the target is in the top-5 predictions
            for i in range(images.size(0)):
                if targets[i] in pred_top5[i]:
                    correct_top5 += 1

            total_samples += images.size(0)

    top1_acc = 100.0 * correct_top1 / total_samples
    top5_acc = 100.0 * correct_top5 / total_samples
    return top1_acc, top5_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeiT-tiny on ILSVRC2012 validation set or export to ONNX")
    parser.add_argument('--data-path', type=str,
                        help="Path to the ILSVRC2012/Imagenet root folder (should contain 'val' subfolder).",
                        default='/scratch2/ml_datasets/ILSVRC2012')
    parser.add_argument('--weights', type=str, default='results/checkpoint.pth.tar',
                        help="Path to model checkpoint")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument('--num-classes', type=int, default=1000,
                        help="Number of classes (default: 1000 for ImageNet)")
    parser.add_argument('--device', default='cuda:0',
                        help="Device to use (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument('--export-onnx', action='store_true',
                        help="Flag to export the loaded model to ONNX format and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load model from checkpoint.
    model = load_model(args.weights, device=device, num_classes=args.num_classes)
    
    # If --export-onnx flag is provided, export to ONNX and exit.
    if args.export_onnx:
        # Create a dummy input with batch size 1 (change shape as needed)
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        onnx_output = "model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )
        print(f"ONNX model exported to {onnx_output}")
        return

    # Else, continue with evaluation
    # Define the transformation (for evaluation)
    val_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.14)),  # or transforms.Resize(256) if you prefer standard
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset from the 'val' subdirectory
    val_dir = os.path.join(args.data_path, 'val')
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

    # Create DataLoader for evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # Evaluate the model and report Top-1 and Top-5 accuracies.
    top1_acc, top5_acc = evaluate(model, val_loader, device)
    print(f"Validation set: Top-1 accuracy = {top1_acc:.2f}%, Top-5 accuracy = {top5_acc:.2f}%")

if __name__ == '__main__':
    main()
