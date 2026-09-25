import json
import os
import random
from PIL import Image
from torchvision import datasets
CLASS_PAIRS = [
    ('cat', 'airplane'),
    ('bird', 'horse'),
    ('truck', 'dog'),
    ('ship', 'monkey'),
    ('deer', 'car')
]
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def main():
    base_dataset = datasets.STL10(root='./data', split='test', download=False)
    with open('./task1/data/stl10_test_subset_indices.json', 'r') as f:
        indices = json.load(f)
    class_to_imgs = {cls: [] for cls in STL10_CLASSES}
    for idx in indices:
        img, label_idx = base_dataset[idx]
        class_name = STL10_CLASSES[label_idx]
        class_to_imgs[class_name].append(img)
        
    os.makedirs('task1/data/cue_conflicts/content', exist_ok=True)
    os.makedirs('task1/data/cue_conflicts/style', exist_ok=True)
    pair_count = 0
    for shape_cls, texture_cls in CLASS_PAIRS:
        directions = [(shape_cls, texture_cls), (texture_cls, shape_cls)]
        for content_cls, style_cls in directions:
            content_imgs = class_to_imgs[content_cls]
            style_imgs = class_to_imgs[style_cls]
            num_pairs = min(len(content_imgs), len(style_imgs), 25)
            for i in range(num_pairs):
                content_img = content_imgs[i]
                style_img = random.choice(style_imgs)
                filename = f"{content_cls}_shape_{style_cls}_texture_{i:02d}.jpg"
                content_img.save(f"task1/data/cue_conflicts/content/{filename}")
                style_img.save(f"task1/data/cue_conflicts/style/{filename}")
                pair_count += 1
    print(f"Saved {pair_count} image pairs to task1/data/cue_conflicts/.")

if __name__ == "__main__":
    main()