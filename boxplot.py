import pandas as pd
import matplotlib.pyplot as plt
import glob

# Lista de CSVs, um por run
csv_files = sorted(glob.glob("zdt*_run*_metrics.csv"))

spacing_com_runs = []
spacing_sem_runs = []
hv_com_runs = []
hv_sem_runs = []

run_labels = []

for file in csv_files:
    run_num = file.split("_run")[1].split("_")[0]
    run_labels.append(f"Run {run_num}")
    df = pd.read_csv(file)
    
    # Cada vela = todos os valores ao longo das gerações
    spacing_com_runs.append(df["Spacing"].values)
    spacing_sem_runs.append(df["Crowding_Spacing"].values)
    hv_com_runs.append(df["Hypervolume"].values)
    hv_sem_runs.append(df["Crowding_Hypervolume"].values)

# Função para plotar boxplots
def plot_boxplot(data_list, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()

# --- Spacing ---
plot_boxplot(spacing_com_runs, run_labels, "Spacing - Com Crowding", "Spacing", "spacing_com.png")
plot_boxplot(spacing_sem_runs, run_labels, "Spacing - Sem Crowding", "Spacing", "spacing_sem.png")

# --- Hypervolume ---
plot_boxplot(hv_com_runs, run_labels, "Hypervolume - Com Crowding", "Hypervolume", "hv_com.png")
plot_boxplot(hv_sem_runs, run_labels, "Hypervolume - Sem Crowding", "Hypervolume", "hv_sem.png")
