import pandas as pd
import matplotlib.pyplot as plt

# Lê o CSV gerado pelo Rust
# df = pd.read_csv("boxplot_3sat.csv", header=None, names=["run", "generation", "best"])
df = pd.read_csv("boxplot_radios.csv", header=None, names=["run", "generation", "best"])

# Garante os tipos corretos
df["run"] = df["run"].astype(int)
df["best"] = df["best"].astype(float)

# Pega apenas o melhor da última geração de cada run
best_final_per_run = [df[df["run"] == r].iloc[-1]["best"] for r in sorted(df["run"].unique())]

plt.figure(figsize=(6, 8))

# Boxplot com uma única vela
plt.boxplot(best_final_per_run,
            patch_artist=True,
            boxprops=dict(facecolor="#f7c89f", color="black"),
            medianprops=dict(color="orange"))

plt.title("Problema dos Rádios - Distribuição dos Melhores Finais", fontsize=14)
plt.ylabel("Fitness")
plt.xticks([1], ["Todas as Runs"])
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
# plt.savefig("boxplot_3sat_final.png", dpi=300, bbox_inches='tight')
plt.savefig("boxplot_radios_final.png", dpi=300, bbox_inches='tight')
plt.show()
