import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP

def get_translated_images(images, shift):
    translated = []
    for img in images:
        t_img = F.affine(img, angle=0.0, translate=(shift, shift), scale=1.0, shear=0.0)
        translated.append(t_img)
    return torch.stack(translated)

def evaluate_translation(backbone, head, dataloader, device, shift=0):
    head.eval()
    predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            if shift > 0:
                images = get_translated_images(images, shift)
            images = images.to(device)
            features = backbone(images)
            logits = head(features)
            _, preds = torch.max(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
    return np.array(predictions)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('task1/analysis/figures', exist_ok=True)
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    base_dataset = datasets.STL10(root='./data', split='test', download=False, transform=common_transform)
    with open('./task1/data/stl10_test_subset_indices.json', 'r') as f:
        indices = json.load(f)
    eval_dataset = Subset(base_dataset, indices)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
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

    ground_truth = np.array([base_dataset[idx][1] for idx in indices])
    shifts = [0, 8, 16, 32]
    model_names = list(backbones.keys())
    shift_results = {name: [] for name in model_names}
    for name in model_names:
        backbones[name].eval()
        heads[name].load_state_dict(torch.load(f"task1/models/{name.lower()}_head.pth", map_location=device))
        for shift in shifts:
            preds = evaluate_translation(backbones[name], heads[name], dataloader, device, shift=shift)
            acc = np.mean(preds == ground_truth) * 100
            shift_results[name].append(acc)
            print(f"[{name}] Shift {shift}px -> Accuracy: {acc:.2f}%")

    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^']
    colors = ['#4C72B0', '#DD8452', '#55A868']
    
    for i, name in enumerate(model_names):
        plt.plot(shifts, shift_results[name], marker=markers[i], color=colors[i], label=name, linewidth=2, markersize=8)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Spatial Translation Shift (Pixels)')
    plt.title('Translation Invariance across Architectures')
    plt.xticks(shifts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    fig_path = 'task1/analysis/figures/spatial_translation.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure dynamically saved to {fig_path}")

if __name__ == "__main__":
    main()
