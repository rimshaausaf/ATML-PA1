import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    base_dataset = datasets.STL10(
        root='./data', 
        split='test', 
        download=False, 
        transform=common_transform
    )

    with open('./task1/data/stl10_test_subset_indices.json', 'r') as f:
        indices = json.load(f)
        
    dataset = Subset(base_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    models = {
        'resnet': FrozenResNet().to(device),
        'vit': FrozenViT().to(device),
        'clip': FrozenCLIP().to(device)
    }
    for model in models.values():
        model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            resnet_features = models['resnet'](images)
            vit_features = models['vit'](images)
            clip_features = models['clip'](images)
            
            print(f"ResNet feature shape: {resnet_features.shape}")
            print(f"ViT feature shape: {vit_features.shape}")
            print(f"CLIP feature shape: {clip_features.shape}")
            break

if __name__ == "__main__":
    main()