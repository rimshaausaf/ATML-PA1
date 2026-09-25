import json
import sys
from pathlib import Path
import numpy as np
from torchvision import datasets

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.seed import set_seed

def generate_balanced_subset(data_dir: str):
    set_seed() 
    total_samples = 500
    test_data = datasets.STL10(root=data_dir, split='test', download=True)
    labels = np.array(test_data.labels)
    unique_classes = np.unique(labels)
    samples_per_class = total_samples // len(unique_classes)
    selected_indices = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < samples_per_class:
            chosen = cls_indices.tolist()
        else:
            chosen = np.random.choice(cls_indices, samples_per_class, replace=False).tolist()
        selected_indices.extend(chosen)
            
    selected_indices.sort()
    save_file = Path("./task1/data/stl10_test_subset_indices.json")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_file, 'w') as f:
        json.dump(selected_indices, f, indent=4)
        
    print(f"Saved {len(selected_indices)} balanced indices to {save_file}")

if __name__ == "__main__":
    generate_balanced_subset(data_dir="./data")