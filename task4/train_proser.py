import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from task4.data.make_splits import get_cifar_dataloaders
from task4.models.resnet_cifar import get_cifar_resnet18
from task4.methods.proser import PROSERResNet
def train_proser(seed=6304):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _, _, _ = get_cifar_dataloaders(batch_size=128, seed=seed)
    base_model = get_cifar_resnet18(num_classes=10).to(device)
    base_model.load_state_dict(torch.load("task4/results/vanilla_best.pth", map_location=device, weights_only=True))
    model = PROSERResNet(base_model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    best_val_csa = 0.0
    print("Starting PROSER Fine-Tuning (50 epochs)...")
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            half = x.size(0) // 2
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:], y[half:]
            logits1, _ = model.forward_post_mixup(model.forward_pre_mixup(x1))
            ce_loss = F.cross_entropy(logits1[:, :10], y1)
            mask = torch.ones_like(logits1[:, :10], dtype=torch.bool)
            mask.scatter_(1, y1.unsqueeze(1), False)
            non_target_max, _ = logits1[:, :10][mask].view(half, 9).max(dim=1)
            dummy_max, _ = logits1[:, 10:].max(dim=1)
            cp_loss = F.relu(non_target_max - dummy_max + 1.0).mean()
            perm = torch.randperm(half)
            diff_mask = y2 != y2[perm]
            if diff_mask.sum() > 0:
                h2 = model.forward_pre_mixup(x2)
                h_mixed = np.random.beta(2.0, 2.0) * h2[diff_mask] + (1.0 - np.random.beta(2.0, 2.0)) * h2[perm][diff_mask]
                logits_mix, _ = model.forward_post_mixup(h_mixed)
                data_placeholder_loss = F.cross_entropy(logits_mix, torch.randint(10, 15, (h_mixed.size(0),), device=device))
            else:
                data_placeholder_loss = torch.tensor(0.0, device=device)
            loss = ce_loss + cp_loss + 0.1 * data_placeholder_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, predicted = model(inputs)[0][:, :10].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_csa = 100. * correct / total
        if val_csa > best_val_csa:
            best_val_csa = val_csa
            torch.save(model.state_dict(), "task4/results/proser_best.pth")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Val CSA: {val_csa:.2f}% (Best: {best_val_csa:.2f}%)")
if __name__ == "__main__":
    train_proser()
