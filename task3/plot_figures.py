import os
import pandas as pd
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
methods = {
    "ERM (Source-only)": "task2/results/source_only_alpha1.0_losses.csv",
    "DAN-DG": "task3/results/dan_dg_lambda1.0_losses.csv",
    "SAM": "task3/results/sam_rho0.05_losses.csv"
}

for name, path in methods.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        ax1.plot(df['epoch'], df['class_loss'], label=name, lw=1.8)
        if name == "DAN-DG" and 'mmd_penalty' in df.columns:
            ax2.plot(df['epoch'], df['mmd_penalty'], label=name, lw=1.8, color='orange')

ax1.set_title("Classification Loss Across Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss (Log Scale)")
ax1.set_yscale("log")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

ax2.set_title("DAN-DG Alignment Penalty")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MMD Penalty")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("task3_loss_curves.png", dpi=300)
plt.close()
print("Saved task3_loss_curves.png")

lambdas = [0.1, 1.0, 10.0]
val_f1s = []
for lam in lambdas:
    p = f"task3/results/dan_dg_lambda{lam}_losses.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        val_f1s.append(df['mean_f1'].max())
    else:
        val_f1s.append(0.0)

plt.figure(figsize=(6, 4))
plt.plot(lambdas, val_f1s, marker='o', color='forestgreen', lw=2, label="Peak Source Val Macro-F1")
plt.title(r"DAN-DG Alignment Strength Trade-off")
plt.xlabel(r"MMD Discrepancy Weight ($\lambda_{DG}$)")
plt.ylabel("Mean Source Val Macro-F1 (%)")
plt.xscale("log")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("task3_controlled_study.png", dpi=300)
plt.close()
print("Saved task3_controlled_study.png")
