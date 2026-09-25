import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from task4.scores.extract_and_score import extract_features, compute_mahalanobis_params, get_scores
from task4.data.make_splits import get_cifar_dataloaders
from task4.models.resnet_cifar import get_cifar_resnet18
def generate_evidence_and_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, train_unaug_loader, test_loader, near_loader, far_loader = get_cifar_dataloaders()
    model = get_cifar_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("task4/results/vanilla_best.pth", map_location=device, weights_only=True))
    train_f, _, train_y = extract_features(model, train_unaug_loader, device)
    val_f, val_l, _ = extract_features(model, val_loader, device)
    test_f, test_l, _ = extract_features(model, test_loader, device)
    near_f, near_l, _ = extract_features(model, near_loader, device)
    far_f, far_l, _ = extract_features(model, far_loader, device)
    class_means, shared_var = compute_mahalanobis_params(train_f, train_y)
    val_msp, val_mls, _, val_mah = get_scores(val_l, val_f, class_means, shared_var)
    test_msp, test_mls, _, test_mah = get_scores(test_l, test_f, class_means, shared_var)
    all_unk_f = torch.cat([near_f, far_f], dim=0)
    all_unk_l = torch.cat([near_l, far_l], dim=0)
    unk_msp, unk_mls, _, unk_mah = get_scores(all_unk_l, all_unk_f, class_means, shared_var)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scores = [("MSP", test_msp, unk_msp), ("MLS", test_mls, unk_mls), ("Mahalanobis", test_mah, unk_mah)]
    y_true = np.concatenate([np.zeros(len(test_msp)), np.ones(len(unk_msp))])
    for ax, (name, s_known, s_unk) in zip(axes, scores):
        y_scores = np.concatenate([s_known.numpy(), s_unk.numpy()])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores) * 100
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUROC = {auc:.1f}%)')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title(f'{name} ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("task4_roc_curves.png", dpi=300)
    plt.close()
    print("Saved task4_roc_curves.png")
    c10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    near_names = ['bus', 'pickup_truck', 'motorcycle', 'tractor', 'wolf', 'fox', 'leopard', 'camel']
    far_names = ['bottle', 'bowl', 'chair', 'clock', 'keyboard', 'mushroom', 'sunflower', 'wardrobe']
    tau_mls = np.percentile(val_mls.numpy(), 95)
    print("\n=== FAILURE ANALYSIS: INCORRECTLY ACCEPTED UNKNOWNS (MLS) ===")
    print(f"Calibrated Threshold (tau): {tau_mls:.4f}\n")
    near_mls = get_scores(near_l, near_f)[1].numpy()
    near_preds = near_l.max(dim=1)[1].numpy()
    accepted_near = np.where(near_mls <= tau_mls)[0]
    print("--- 3 Informative Near-Unknown Failures ---")
    for idx in accepted_near[:3]:
        print(f"True: {near_names[idx % len(near_names)]:<12} | Predicted: {c10_classes[near_preds[idx]]:<10} | MLS Score: {near_mls[idx]:.4f}")
    far_mls = get_scores(far_l, far_f)[1].numpy()
    far_preds = far_l.max(dim=1)[1].numpy()
    accepted_far = np.where(far_mls <= tau_mls)[0]
    print("\n--- 3 Informative Far-Unknown Failures ---")
    for idx in accepted_far[:3]:
        print(f"True: {far_names[idx % len(far_names)]:<12} | Predicted: {c10_classes[far_preds[idx]]:<10} | MLS Score: {far_mls[idx]:.4f}")
if __name__ == "__main__":
    generate_evidence_and_plots()
