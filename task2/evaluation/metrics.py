import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro') * 100.0
    return acc, f1

def run_evaluation(backbone, classifier, dataloader, device):
    backbone.eval()
    classifier.eval()
    all_targets = []
    all_preds = []
    all_features = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            feats = backbone(images)
            logits = classifier(feats)
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_features.append(feats.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    acc, f1 = evaluate_predictions(all_targets, all_preds)
    return acc, f1, np.array(all_targets), np.array(all_preds), all_features
