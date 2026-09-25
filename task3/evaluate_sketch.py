import torch
import numpy as np
from shared.pacs_protocol import get_pacs_dataloaders
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from task2.evaluation.metrics import run_evaluation
from task2.evaluation.class_analysis import analyze_class_transfers
from task3.evaluation.source_domain_separability import compute_source_domain_separability
from task3.evaluation.sharpness import compute_sharpness_proxy

def evaluate_task3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, source_val, target_eval = get_pacs_dataloaders(seed=6304)
    models_to_eval = [
        ("ERM", "task2/results/source_only_alpha1.0_best.pth"),
        ("DAN-DG", "task3/results/dan_dg_lambda1.0_best.pth"),
        ("SAM", "task3/results/sam_rho0.05_best.pth")
    ]
    
    results = {}
    baseline_true = None
    baseline_pred = None    
    for name, path in models_to_eval:
        print(f"\nEvaluating {name} from {path}...")
        backbone = ResNet18Backbone().to(device)
        classifier = ClassifierHead().to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        s_accs = {}
        s_f1s = {}
        s_feats_dict = {}
        for d, loader in source_val.items():
            acc, f1, _, _, feats = run_evaluation(backbone, classifier, loader, device)
            s_accs[d] = acc
            s_f1s[d] = f1
            s_feats_dict[d] = feats
        mean_s_acc = np.mean(list(s_accs.values()))
        mean_s_f1 = np.mean(list(s_f1s.values()))
        worst_s_acc = np.min(list(s_accs.values()))
        worst_s_f1 = np.min(list(s_f1s.values()))
        src_separability = compute_source_domain_separability(s_feats_dict, seed=6304)
        sharpness = compute_sharpness_proxy(backbone, classifier, source_val, device, seed=6304, rho=0.05)
        t_acc, t_f1, t_true, t_pred, _ = run_evaluation(backbone, classifier, target_eval, device)
        
        results[name] = {
            'photo_acc': s_accs['photo'], 'art_acc': s_accs['art_painting'], 'cartoon_acc': s_accs['cartoon'],
            'mean_s_acc': mean_s_acc, 'worst_s_acc': worst_s_acc,
            'mean_s_f1': mean_s_f1, 'worst_s_f1': worst_s_f1,
            't_acc': t_acc, 't_f1': t_f1,
            'separability': src_separability,
            'sharpness': sharpness
        }
        if name == "ERM":
            baseline_true = t_true
            baseline_pred = t_pred
        else:
            analyze_class_transfers(baseline_true, baseline_pred, t_true, t_pred, method_name=name)

    print("\n" + "="*80)
    print("=== REQUIRED EVIDENCE TABLE: TASK 3 DOMAIN GENERALIZATION ===")
    print("="*80)
    header = "Method  | Photo | Art   | Crtn  | MeanS | WrstS | MeanF1| WrstF1| Sketch| Sk-Δ  | Sk-F1 | SrcSep| Sharp"
    print(header)
    print("-" * len(header))
    erm_t_acc = results["ERM"]['t_acc']
    for name in ["ERM", "DAN-DG", "SAM"]:
        r = results[name]
        delta = r['t_acc'] - erm_t_acc if name != "ERM" else 0.0
        print(f"{name:<8}| {r['photo_acc']:5.1f} | {r['art_acc']:5.1f} | {r['cartoon_acc']:5.1f} | "
              f"{r['mean_s_acc']:5.1f} | {r['worst_s_acc']:5.1f} | {r['mean_s_f1']:5.1f} | {r['worst_s_f1']:5.1f} | "
              f"{r['t_acc']:5.1f} | {delta:+5.2f}| {r['t_f1']:5.1f} | {r['separability']:5.1f} | {r['sharpness']:.4f}")

if __name__ == "__main__":
    evaluate_task3()
