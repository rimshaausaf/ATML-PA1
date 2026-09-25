import os
import pandas as pd
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
methods = {
    "Source-only": "task2/results/source_only_alpha1.0_losses.csv",
    "DAN (MMD)": "task2/results/dan_alpha1.0_losses.csv",
    "DANN": "task2/results/dann_alpha1.0_losses.csv",
    "CDAN": "task2/results/cdan_alpha1.0_losses.csv"
}
for name, path in methods.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        ax1.plot(df['epoch'], df['class_loss'], label=name, lw=1.8)
        if name != "Source-only":
            ax2.plot(df['epoch'], df['domain_loss'], label=name, lw=1.8)
ax1.set_title("Classification Loss Across Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss (Log Scale)")
ax1.set_yscale("log")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.set_title("Domain / Alignment Loss Across Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Domain Loss (Log Scale)")
ax2.set_yscale("log")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("task2_loss_curves.png", dpi=300)
plt.close()
print("Saved task2_loss_curves.png")

alphas = [0.25, 0.5, 1.0]
val_f1s = []
for a in alphas:
    p = f"task2/results/dann_alpha{a}_losses.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        val_f1s.append(df['mean_f1'].max())
    else:
        val_f1s.append(0.0)

plt.figure(figsize=(6, 4))
plt.plot(alphas, val_f1s, marker='o', color='crimson', lw=2, label="Peak Source Val Macro-F1")
plt.title("DANN Alignment Strength Trade-off")
plt.xlabel(r"Maximum Gradient Reversal Strength ($\alpha$)")
plt.ylabel("Source Val Macro-F1 (%)")
plt.ylim(0, 50)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("task2_dann_study.png", dpi=300)
plt.close()
print("Saved task2_dann_study.png")
