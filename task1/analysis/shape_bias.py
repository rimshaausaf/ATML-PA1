import os
import torch
from torchvision import transforms
from PIL import Image
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP
import torch.nn as nn
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(STL10_CLASSES)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'task1/data/cue_conflicts/output'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    backbones = {
        'ResNet': FrozenResNet().to(device),
        'ViT': FrozenViT().to(device),
        'CLIP': FrozenCLIP().to(device)
    }
    heads = {
        'ResNet': nn.Linear(2048, 10).to(device),
        'ViT': nn.Linear(768, 10).to(device),
        'CLIP': nn.Linear(512, 10).to(device)
    }
    for name in backbones:
        backbones[name].eval()
        heads[name].load_state_dict(torch.load(f"task1/models/{name.lower()}_head.pth", map_location=device))
        heads[name].eval()
    valid_images = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    if len(valid_images) < 200:
        print(f"Warning: Only {len(valid_images)} images found. The manual requires at least 200.")
    results = {name: {'shape': 0, 'texture': 0, 'other': 0} for name in backbones}
    with torch.no_grad():
        for filename in valid_images:
            parts = filename.split('_')
            shape_cls = parts[0]
            texture_cls = parts[2]
            shape_idx = CLASS_TO_IDX[shape_cls]
            texture_idx = CLASS_TO_IDX[texture_cls]
            img_path = os.path.join(output_dir, filename)
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            for name in backbones:
                features = backbones[name](tensor)
                logits = heads[name](features)
                pred_idx = torch.argmax(logits, dim=-1).item()
                if pred_idx == shape_idx:
                    results[name]['shape'] += 1
                elif pred_idx == texture_idx:
                    results[name]['texture'] += 1
                else:
                    results[name]['other'] += 1
    print(f"--- Shape Bias & Coverage Evaluation ({len(valid_images)} Images) ---")
    for name in backbones:
        s = results[name]['shape']
        t = results[name]['texture']
        o = results[name]['other']
        total = s + t + o
        valid_decisions = s + t
        shape_bias = (s / valid_decisions * 100) if valid_decisions > 0 else 0
        coverage = (valid_decisions / total * 100) if total > 0 else 0
        print(f"\n[{name}]")
        print(f"Decisions -> Shape: {s} | Texture: {t} | Other: {o}")
        print(f"Shape Bias: {shape_bias:.2f}%")
        print(f"Coverage: {coverage:.2f}%")

if __name__ == "__main__":
    main()