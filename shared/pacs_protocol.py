import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from shared.pacs import (
    get_train_transform,
    get_eval_transform,
    build_or_load_splits,
    InfiniteDomainIterator
)

def get_pacs_dataloaders(data_root="data/pacs/PACS", seed=6304):
    splits = build_or_load_splits(data_root=data_root, seed=seed)
    train_tx = get_train_transform()
    eval_tx = get_eval_transform()
    photo_ds = datasets.ImageFolder(os.path.join(data_root, "photo"), transform=train_tx)
    art_ds = datasets.ImageFolder(os.path.join(data_root, "art_painting"), transform=train_tx)
    cartoon_ds = datasets.ImageFolder(os.path.join(data_root, "cartoon"), transform=train_tx)
    sketch_ds = datasets.ImageFolder(os.path.join(data_root, "sketch"), transform=train_tx)
    source_train_loaders = {
        "photo": DataLoader(Subset(photo_ds, splits["photo"]["train"]), batch_size=8, shuffle=True, drop_last=True),
        "art_painting": DataLoader(Subset(art_ds, splits["art_painting"]["train"]), batch_size=8, shuffle=True, drop_last=True),
        "cartoon": DataLoader(Subset(cartoon_ds, splits["cartoon"]["train"]), batch_size=8, shuffle=True, drop_last=True)
    }
    target_train_loader = DataLoader(
        Subset(sketch_ds, splits["sketch"]["all"]),
        batch_size=24,
        shuffle=True,
        drop_last=True
    )
    photo_val_ds = datasets.ImageFolder(os.path.join(data_root, "photo"), transform=eval_tx)
    art_val_ds = datasets.ImageFolder(os.path.join(data_root, "art_painting"), transform=eval_tx)
    cartoon_val_ds = datasets.ImageFolder(os.path.join(data_root, "cartoon"), transform=eval_tx)
    source_val_loaders = {
        "photo": DataLoader(Subset(photo_val_ds, splits["photo"]["val"]), batch_size=32, shuffle=False),
        "art_painting": DataLoader(Subset(art_val_ds, splits["art_painting"]["val"]), batch_size=32, shuffle=False),
        "cartoon": DataLoader(Subset(cartoon_val_ds, splits["cartoon"]["val"]), batch_size=32, shuffle=False)
    }
    sketch_eval_ds = datasets.ImageFolder(os.path.join(data_root, "sketch"), transform=eval_tx)
    target_eval_loader = DataLoader(sketch_eval_ds, batch_size=32, shuffle=False)
    return source_train_loaders, target_train_loader, source_val_loaders, target_eval_loader
