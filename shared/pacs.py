import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def build_or_load_splits(data_root="data/pacs/PACS", split_file="shared/splits/pacs_sketch_seed6304.json", seed=6304):
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)

    splits = {}
    sources = ["photo", "art_painting", "cartoon"]
    for domain in sources:
        domain_path = os.path.join(data_root, domain)
        dataset = datasets.ImageFolder(domain_path)
        targets = dataset.targets
        train_idx, val_idx = train_test_split(
            list(range(len(targets))),
            test_size=0.20,
            stratify=targets,
            random_state=seed
        )
        splits[domain] = {"train": train_idx, "val": val_idx}

    sketch_path = os.path.join(data_root, "sketch")
    sketch_ds = datasets.ImageFolder(sketch_path)
    splits["sketch"] = {"all": list(range(len(sketch_ds)))}

    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    return splits

class InfiniteDomainIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
