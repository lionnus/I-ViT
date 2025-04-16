import argparse
import torch
from PIL import Image
from torchvision import transforms
from models import deit_tiny_patch16_224
import json


def load_model(checkpoint_path, device='cuda', num_classes=1000):
    model = deit_tiny_patch16_224(pretrained=False, num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=device) # Adjust the shape of parameters in the checkpoint if needed
    model_state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name in model_state_dict:
            # If checkpoint param is a scalar [] but the model expects [1], unsqueeze it
            if param.shape == torch.Size([]) and model_state_dict[name].shape == torch.Size([1]):
                checkpoint[name] = param.unsqueeze(0)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size=224):
    preprocess = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--image', help="Path to the image", default='bike.jpg')
    parser.add_argument('--weights', help="Path to checkpoint", default='results/checkpoint.pth.tar')
    parser.add_argument('--num-classes', default=1000, type=int, help="Number of classes")
    parser.add_argument('--device', default='cuda:2', help="Device (cuda or cpu)")

    args = parser.parse_args()

    device = torch.device(args.device)

    model = load_model(args.weights, device=device, num_classes=args.num_classes)
    image = preprocess_image(args.image).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    class_idx = json.load(open("imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    print("Top 5 predictions:")
    for i in range(top5_prob.size(0)):
        print(f"Class {top5_catid[i].item()} - {idx2label[top5_catid[i].item()]} - Probability {top5_prob[i].item():.4f}")


if __name__ == '__main__':
    main()
