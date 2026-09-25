import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP

def get_shuffled_images(images, patch_size=16):
    B, C, H, W = images.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    x = images.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(B, num_patches, C, patch_size, patch_size)
    shuffled = torch.zeros_like(x)
    for i in range(B):
        idx = torch.randperm(num_patches, device=images.device)
        shuffled[i] = x[i, idx]
    shuffled = shuffled.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled = shuffled.view(B, C, H, W)
    return shuffled

def evaluate_shuffling(backbone, head, dataloader, device, shuffle_patches=False):
    head.eval()
    predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if shuffle_patches:
                images = get_shuffled_images(images, patch_size=16)
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
    model_names = list(backbones.keys())
    clean_accs = []
    shuffled_accs = []
    for name in model_names:
        backbones[name].eval()
        heads[name].load_state_dict(torch.load(f"task1/models/{name.lower()}_head.pth", map_location=device))
        clean_preds = evaluate_shuffling(backbones[name], heads[name], dataloader, device, shuffle_patches=False)
        clean_acc = np.mean(clean_preds == ground_truth) * 100
        clean_accs.append(clean_acc)
        shuffled_preds = evaluate_shuffling(backbones[name], heads[name], dataloader, device, shuffle_patches=True)
        shuffled_acc = np.mean(shuffled_preds == ground_truth) * 100
        shuffled_accs.append(shuffled_acc)
        cons = np.mean(shuffled_preds == clean_preds) * 100
        print(f"[{name}] Clean: {clean_acc:.2f}% | Shuffled: {shuffled_acc:.2f}% | Drop: {clean_acc - shuffled_acc:.2f}% | Consistency: {cons:.2f}%")

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', color='#4C72B0')
    ax.bar(x + width/2, shuffled_accs, width, label='Shuffled Accuracy', color='#C44E52')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Impact of 16x16 Patch Shuffling on Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    fig_path = 'task1/analysis/figures/patch_shuffling.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure dynamically saved to {fig_path}")

if __name__ == "__main__":
    main()
