import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from task4.scores.extract_and_score import extract_features, compute_mahalanobis_params, get_scores
from task4.data.make_splits import get_cifar_dataloaders
from task4.models.resnet_cifar import get_cifar_resnet18
from task4.methods.proser import PROSERResNet
import argparse
def evaluate_model(method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, train_unaug_loader, test_loader, near_loader, far_loader = get_cifar_dataloaders()
    if method == "proser":
        base_model = get_cifar_resnet18(num_classes=10).to(device)
        model = PROSERResNet(base_model).to(device)
    else:
        model = get_cifar_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(f"task4/results/{method}_best.pth", map_location=device, weights_only=True))
    print(f"--- Extracting Features for {method.upper()} ---")
    train_f, _, train_y = extract_features(model, train_unaug_loader, device)
    val_f, val_l, _ = extract_features(model, val_loader, device)
    test_f, test_l, test_y = extract_features(model, test_loader, device)
    near_f, near_l, _ = extract_features(model, near_loader, device)
    far_f, far_l, _ = extract_features(model, far_loader, device)
    if method == "proser":
        csa = (test_l[:, :10].max(dim=1)[1] == test_y).float().mean().item() * 100
    else:
        csa = (test_l.max(dim=1)[1] == test_y).float().mean().item() * 100
    class_means, shared_var = compute_mahalanobis_params(train_f, train_y)
    def process_group(feats, logits):
        if method == "proser":
            logits = logits[:, :10]
        return get_scores(logits, feats, class_means, shared_var)
    val_scores = process_group(val_f, val_l)
    test_scores = process_group(test_f, test_l)
    near_scores = process_group(near_f, near_l)
    far_scores = process_group(far_f, far_l)
    print(f"\nClosed-Set Accuracy (CSA): {csa:.2f}%")
    print(f"{'Score':<15} | {'Near AUROC':<12} | {'Far AUROC':<12} | {'Near FPR@95':<12} | {'Far FPR@95':<12}")
    print("-" * 70)
    score_names = ["MSP", "MLS", "Energy", "Mahalanobis"]
    for i, name in enumerate(score_names):
        if (method == "gcsc" or method == "proser") and name != "MLS":
            continue
        s_val, s_test, s_near, s_far = val_scores[i].numpy(), test_scores[i].numpy(), near_scores[i].numpy(), far_scores[i].numpy()
        tau = np.percentile(s_val, 95)
        y_true_near = np.concatenate([np.zeros(len(s_test)), np.ones(len(s_near))])
        auroc_near = roc_auc_score(y_true_near, np.concatenate([s_test, s_near])) * 100
        fpr95_near = (s_near <= tau).mean() * 100
        y_true_far = np.concatenate([np.zeros(len(s_test)), np.ones(len(s_far))])
        auroc_far = roc_auc_score(y_true_far, np.concatenate([s_test, s_far])) * 100
        fpr95_far = (s_far <= tau).mean() * 100
        print(f"{name:<15} | {auroc_near:<12.2f} | {auroc_far:<12.2f} | {fpr95_near:<12.2f} | {fpr95_far:<12.2f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["vanilla", "gcsc", "proser"], required=True)
    args = parser.parse_args()
    evaluate_model(args.method)
