import torch
import numpy as np
from shared.pacs_protocol import get_pacs_dataloaders
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.metrics import run_evaluation
from task2.evaluation.domain_separability import compute_domain_separability
from task2.evaluation.class_analysis import analyze_class_transfers

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, source_val, target_eval = get_pacs_dataloaders(seed=6304)
    methods = ["source_only", "dan", "dann", "cdan"]
    results = {}    
    baseline_true = None
    baseline_pred = None
    for method in methods:
        print(f"\nEvaluating {method.upper()}...")
        backbone = ResNet18Backbone().to(device)
        classifier = ClassifierHead().to(device)
        checkpoint_path = f"task2/results/{method}_alpha1.0_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        
        t_acc, t_f1, t_true, t_pred, t_feats = run_evaluation(backbone, classifier, target_eval, device)
        s_feats_list = []
        s_accs = []
        s_f1s = []
        for domain, loader in source_val.items():
            s_acc, s_f1, _, _, s_feats = run_evaluation(backbone, classifier, loader, device)
            s_feats_list.append(s_feats)
            s_accs.append(s_acc)
            s_f1s.append(s_f1)
            
        mean_s_acc = np.mean(s_accs)
        mean_s_f1 = np.mean(s_f1s)
        all_s_feats = np.concatenate(s_feats_list, axis=0)
        separability = compute_domain_separability(all_s_feats, t_feats, seed=6304)
        results[method] = {
            "mean_s_acc": mean_s_acc,
            "mean_s_f1": mean_s_f1,
            "t_acc": t_acc,
            "t_f1": t_f1,
            "separability": separability
        }
        
        if method == "source_only":
            baseline_true = t_true
            baseline_pred = t_pred
            print(f"Target Accuracy: {t_acc:.2f}%, Target F1: {t_f1:.2f}%, Separability: {separability:.2f}%")
        else:
            analyze_class_transfers(baseline_true, baseline_pred, t_true, t_pred, method_name=method.upper())
            print(f"Target Accuracy: {t_acc:.2f}%, Target F1: {t_f1:.2f}%, Separability: {separability:.2f}%")

    print("\n=== Final LaTeX Table Data ===")
    print("Method | Mean Source Acc | Target Acc | Target Delta | Target F1 | Separability")
    for method in methods:
        r = results[method]
        delta = r['t_acc'] - results['source_only']['t_acc'] if method != "source_only" else 0.0
        print(f"{method.upper():<12} | {r['mean_s_acc']:.2f} | {r['t_acc']:.2f} | {delta:+.2f} | {r['t_f1']:.2f} | {r['separability']:.2f}")

if __name__ == "__main__":
    evaluate_all()
