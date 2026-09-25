import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
def compute_per_class_accuracy(y_true, y_pred, num_classes=7):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = (cm.diagonal() / cm.sum(axis=1)) * 100.0
    return per_class_acc, cm

def analyze_class_transfers(erm_true, erm_pred, adapt_true, adapt_pred, method_name="Adaptation"):
    erm_accs, erm_cm = compute_per_class_accuracy(erm_true, erm_pred)
    adapt_accs, adapt_cm = compute_per_class_accuracy(adapt_true, adapt_pred)
    deltas = adapt_accs - erm_accs
    best_class_idx = int(np.argmax(deltas))
    worst_class_idx = int(np.argmin(deltas))
    print(f"\n--- Class Transfer Analysis ({method_name} vs. Source-only) ---")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  Class {name:<10}: Source-only = {erm_accs[idx]:.2f}%, {method_name} = {adapt_accs[idx]:.2f}% (Δ = {deltas[idx]:+.2f}%)")
    print(f"\nLargest Improvement: {CLASS_NAMES[best_class_idx]} ({deltas[best_class_idx]:+.2f}%)")
    print(f"Largest Degradation / Stagnation: {CLASS_NAMES[worst_class_idx]} ({deltas[worst_class_idx]:+.2f}%)")
    row = adapt_cm[worst_class_idx].copy()
    row[worst_class_idx] = 0 # exclude correct prediction
    confused_idx = int(np.argmax(row))
    print(f"Dominant confusion for {CLASS_NAMES[worst_class_idx]}: misclassified as {CLASS_NAMES[confused_idx]} ({row[confused_idx]} times)")
    return {
        "class_names": CLASS_NAMES,
        "source_only_per_class": erm_accs.tolist(),
        "adapt_per_class": adapt_accs.tolist(),
        "deltas": deltas.tolist()
    }
