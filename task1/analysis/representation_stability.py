import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP

def apply_transforms(images):
    gray = TF.rgb_to_grayscale(images, num_output_channels=3)
    shifted = torch.stack([TF.affine(img, angle=0.0, translate=(16, 16), scale=1.0, shear=0.0) for img in images])
    return gray, shifted

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
    subset_indices = indices[:100]
    eval_dataset = Subset(base_dataset, subset_indices)
    dataloader = DataLoader(eval_dataset, batch_size=100, shuffle=False)
    backbones = {
        'ResNet': FrozenResNet().to(device),
        'ViT': FrozenViT().to(device),
        'CLIP': FrozenCLIP().to(device)
    }    
    images, labels = next(iter(dataloader))
    images = images.to(device)
    gray_images, shifted_images = apply_transforms(images)
    features_dict = {}
    with torch.no_grad():
        for name, backbone in backbones.items():
            backbone.eval()
            clean_feat = backbone(images)
            gray_feat = backbone(gray_images)
            shift_feat = backbone(shifted_images)
            sim_gray = F.cosine_similarity(clean_feat, gray_feat, dim=-1).mean().item()
            sim_shift = F.cosine_similarity(clean_feat, shift_feat, dim=-1).mean().item()
            print(f"[{name}] Clean vs Grayscale Sim: {sim_gray:.4f}")
            print(f"[{name}] Clean vs Shifted Sim: {sim_shift:.4f}")            
            combined_feats = torch.cat([clean_feat, gray_feat, shift_feat], dim=0).cpu().numpy()
            features_dict[name] = combined_feats

    print("\nGenerating t-SNE plot for CLIP...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    clip_tsne = tsne.fit_transform(features_dict['CLIP'])
    
    N = 100
    plt.figure(figsize=(8, 6))
    plt.scatter(clip_tsne[:N, 0], clip_tsne[:N, 1], c='#4C72B0', label='Clean', alpha=0.7)
    plt.scatter(clip_tsne[N:2*N, 0], clip_tsne[N:2*N, 1], c='#DD8452', label='Grayscale', alpha=0.7)
    plt.scatter(clip_tsne[2*N:, 0], clip_tsne[2*N:, 1], c='#55A868', label='Shifted', alpha=0.7)
    plt.title("t-SNE of CLIP Representations under Perturbations")
    plt.legend()
    plt.savefig('task1/analysis/figures/clip_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("t-SNE saved to 'task1/analysis/figures/clip_tsne.png'")

if __name__ == "__main__":
    main()
