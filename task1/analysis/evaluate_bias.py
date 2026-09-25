import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import open_clip
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def evaluate_classifier(backbone, head, dataloader, device, model_name):
    head.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = backbone(images)
            logits = head(features)
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_conf = np.mean(all_confidences)
    
    print(f"--- {model_name} (Trained Head) ---")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Macro-F1 Score: {f1:.4f}")
    print(f"Mean Max Confidence: {mean_conf:.4f}\n")

def evaluate_zeroshot_clip(clip_backbone, dataloader, device):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text_prompts = [f"a photo of a {cls}." for cls in STL10_CLASSES]
    text_tokens = tokenizer(text_prompts).to(device)    
    with torch.no_grad():
        text_features = clip_backbone.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            image_features = clip_backbone(images)
            logit_scale = clip_backbone.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_conf = np.mean(all_confidences)
    print(f"--- Zero-Shot CLIP ---")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Macro-F1 Score: {f1:.4f}")
    print(f"Mean Max Confidence: {mean_conf:.4f}\n")

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
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
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

    for name in backbones.keys():
        backbones[name].eval()
        weight_path = f"task1/models/{name.lower()}_head.pth"
        heads[name].load_state_dict(torch.load(weight_path, map_location=device))
        evaluate_classifier(backbones[name], heads[name], eval_loader, device, name)
    evaluate_zeroshot_clip(backbones['CLIP'], eval_loader, device)

if __name__ == "__main__":
    main()