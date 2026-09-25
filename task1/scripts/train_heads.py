import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import copy
import os
from task1.models.backbones import FrozenResNet, FrozenViT, FrozenCLIP

def train_linear_head(backbone, head, train_loader, val_loader, device, model_name):
    print(f"\n--- Training {model_name} Head ---")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_head_weights = None
    patience = 5
    epochs_without_improvement = 0
    max_epochs = 30
    
    for epoch in range(max_epochs):
        head.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)            
            with torch.no_grad():
                features = backbone(images)            
            optimizer.zero_grad()
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                features = backbone(images)
                outputs = head(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_weights = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered for {model_name} at epoch {epoch+1}.")
            break
            
    head.load_state_dict(best_head_weights)
    print(f"Best Validation Accuracy for {model_name}: {best_val_acc:.2f}%")
    return head

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    full_train_dataset = datasets.STL10(root='./data', split='train', download=False, transform=common_transform)
    targets = full_train_dataset.labels
    train_idx, val_idx = train_test_split(
        range(len(targets)), 
        test_size=0.20, 
        random_state=6304, 
        stratify=targets
    )
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    backbones = {
        'ResNet': FrozenResNet().to(device),
        'ViT': FrozenViT().to(device),
        'CLIP': FrozenCLIP().to(device)
    }
    for b in backbones.values():
        b.eval()

    heads = {
        'ResNet': nn.Linear(2048, 10).to(device),
        'ViT': nn.Linear(768, 10).to(device),
        'CLIP': nn.Linear(512, 10).to(device)
    }
    os.makedirs('task1/models', exist_ok=True)
    for name in backbones.keys():
        trained_head = train_linear_head(
            backbones[name], heads[name], train_loader, val_loader, device, name
        )
        
        save_path = f"task1/models/{name.lower()}_head.pth"
        torch.save(trained_head.state_dict(), save_path)
        print(f"Saved best {name} head weights to {save_path}\n")

if __name__ == "__main__":
    main()