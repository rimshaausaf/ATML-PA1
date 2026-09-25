import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP
from task1.data.transforms import get_grayscale, get_hue_rotation

def evaluate_intervention(backbone, head, dataloader, device, transform_func=None):
    head.eval()
    predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)            
            if transform_func:
                images = transform_func(images)
            features = backbone(images)
            logits = head(features)
            _, preds = torch.max(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
    return np.array(predictions)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for name in backbones.keys():
        backbones[name].eval()
        heads[name].load_state_dict(torch.load(f"task1/models/{name.lower()}_head.pth", map_location=device))
        clean_preds = evaluate_intervention(backbones[name], heads[name], dataloader, device, transform_func=None)
        gray_preds = evaluate_intervention(backbones[name], heads[name], dataloader, device, transform_func=get_grayscale)
        hue_preds = evaluate_intervention(backbones[name], heads[name], dataloader, device, transform_func=get_hue_rotation)
        clean_acc = np.mean(clean_preds == ground_truth) * 100
        gray_acc = np.mean(gray_preds == ground_truth) * 100
        hue_acc = np.mean(hue_preds == ground_truth) * 100
        gray_cons = np.mean(gray_preds == clean_preds) * 100
        hue_cons = np.mean(hue_preds == clean_preds) * 100
        
        print(f"\n[{name}]")
        print(f"Clean Accuracy: {clean_acc:.2f}%")
        print(f"Grayscale Accuracy: {gray_acc:.2f}% (Drop: {clean_acc - gray_acc:.2f}%) | Consistency: {gray_cons:.2f}%")
        print(f"Hue Rotation Accuracy: {hue_acc:.2f}% (Drop: {clean_acc - hue_acc:.2f}%) | Consistency: {hue_cons:.2f}%")

if __name__ == "__main__":
    main()