import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

NEAR_CLASSES = [10, 61, 48, 89, 97, 34, 42, 15]
FAR_CLASSES = [9, 10, 20, 22, 39, 51, 82, 94]

def get_cifar_dataloaders(batch_size=128, seed=6304, use_randaug=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    base_train_tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if use_randaug:
        base_train_tf.append(transforms.RandAugment(num_ops=2, magnitude=9))
    base_train_tf.extend([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_transform = transforms.Compose(base_train_tf)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    c10_train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    c10_train_unaug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)
    c10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
    targets = c10_train_full.targets
    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.10, random_state=seed, stratify=targets)
    train_loader = DataLoader(Subset(c10_train_full, train_idx), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(c10_train_unaug, val_idx), batch_size=batch_size, shuffle=False, num_workers=2)
    train_unaug_loader = DataLoader(Subset(c10_train_unaug, train_idx), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(c10_test, batch_size=batch_size, shuffle=False, num_workers=2)
    c100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=eval_transform)
    near_names = ['bus', 'pickup_truck', 'motorcycle', 'tractor', 'wolf', 'fox', 'leopard', 'camel']
    far_names = ['bottle', 'bowl', 'chair', 'clock', 'keyboard', 'mushroom', 'sunflower', 'wardrobe']
    class_to_idx = c100_test.class_to_idx
    near_idx = [class_to_idx[n] for n in near_names]
    far_idx = [class_to_idx[n] for n in far_names]
    near_samples = [i for i, t in enumerate(c100_test.targets) if t in near_idx]
    far_samples = [i for i, t in enumerate(c100_test.targets) if t in far_idx]
    near_loader = DataLoader(Subset(c100_test, near_samples), batch_size=batch_size, shuffle=False, num_workers=2)
    far_loader = DataLoader(Subset(c100_test, far_samples), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_unaug_loader, test_loader, near_loader, far_loader