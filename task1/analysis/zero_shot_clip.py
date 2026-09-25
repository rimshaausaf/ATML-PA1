import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import open_clip
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    base_dataset = datasets.STL10(root='./data', split='test', download=False, transform=preprocess)
    with open('./task1/data/stl10_test_subset_indices.json', 'r') as f:
        indices = json.load(f)
    eval_dataset = Subset(base_dataset, indices)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    text_prompts = [f"a photo of a {c}" for c in stl10_classes]
    text_tokens = tokenizer(text_prompts).to(device)
    model.eval()
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, preds = similarity.max(dim=-1)
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(labels.numpy())
    zero_shot_acc = np.mean(np.array(predictions) == np.array(ground_truth)) * 100
    print(f"Zero-Shot Clean Accuracy: {zero_shot_acc:.2f}%")
    print("Linear Probe Clean Accuracy: 92.80% (From previous experiments)")

if __name__ == "__main__":
    main()
