import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from task4.data.make_splits import get_cifar_dataloaders
from task4.models.resnet_cifar import get_cifar_resnet18
def train_baseline(method="vanilla", seed=6304):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_randaug = (method == "gcsc")
    train_loader, val_loader, _, _, _, _ = get_cifar_dataloaders(batch_size=128, seed=seed, use_randaug=use_randaug)
    model = get_cifar_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    best_val_acc = 0.0
    print(f"Starting {method.upper()} training...")
    for epoch in range(100):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("task4/results", exist_ok=True)
            torch.save(model.state_dict(), f"task4/results/{method}_best.pth")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["vanilla", "gcsc"], required=True)
    args = parser.parse_args()
    train_baseline(args.method)
