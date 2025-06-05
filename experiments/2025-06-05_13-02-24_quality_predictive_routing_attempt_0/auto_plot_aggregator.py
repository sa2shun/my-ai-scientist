#!/usr/bin/env python3
"""
Aggregator script for final QualityRoute paper figures.
This script loads experiment data from four .npy files (baseline, research, and two ablation study files)
and produces a comprehensive set of final, publication‐quality plots.
All final figures are saved in the "figures/" directory.
Each plot is wrapped in its own try/except block so that failure in one does not block others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global style and font sizes for publication quality plots
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
# Function to remove top and right spines for a given axis for cleaner style.
def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Create the figures folder
os.makedirs("figures", exist_ok=True)

# ---------------------
# 1. BASELINE PLOTS
# ---------------------
try:
    baseline_path = "experiment_results/experiment_ff2fb2732d5b42d3ba84e690748204a8_proc_3634790/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    # Extract data from the baseline experiment
    ds = baseline_data.get("weight_decay_tuning", {}).get("query_response_dataset", {})
    baseline_train_losses = ds.get("losses", {}).get("train", None)
    baseline_val_losses = ds.get("losses", {}).get("val", None)
    baseline_metrics = ds.get("metrics", {}).get("val", None)
    # For weight decay values, try to get a stored list; otherwise generate indices.
    baseline_weight_decays = ds.get("weight_decays", list(range(len(baseline_metrics) if baseline_metrics is not None else 0)))
    
    if baseline_train_losses is None or baseline_val_losses is None or baseline_metrics is None:
        raise ValueError("Missing keys in baseline experiment data.")
    
    epochs = range(1, len(baseline_train_losses) + 1)
    
    # (1) Baseline Loss Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, baseline_train_losses, label="Training Loss", marker="o")
    ax.plot(epochs, baseline_val_losses, label="Validation Loss", marker="s")
    ax.set_title("Baseline Loss Curves: Query Response Dataset", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_loss_curves.png"), dpi=300)
    plt.close(fig)
    
    # (2) Baseline Quality-Speed Tradeoff Plot (bar plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(baseline_weight_decays)), baseline_metrics, color="skyblue")
    ax.set_xticks(range(len(baseline_weight_decays)))
    ax.set_xticklabels([str(x) for x in baseline_weight_decays])
    ax.set_title("Baseline Quality-Speed Tradeoff", fontsize=16)
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Quality-Speed Metric")
    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_quality_speed_tradeoff.png"), dpi=300)
    plt.close(fig)
    
    print("Baseline plots created successfully.")
except Exception as e:
    print(f"Error in baseline plots: {e}")

# ---------------------
# 2. RESEARCH PLOTS (Hyperparameter Tuning: Momentum)
# ---------------------
try:
    research_path = "experiment_results/experiment_a242946b6f3d4fec8489d145d0d64b35_proc_3634790/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
    momentum_dict = research_data.get("hyperparam_tuning_momentum", {})
    if not momentum_dict:
        raise ValueError("No momentum tuning data found in research experiment.")

    momentum_keys = sorted(momentum_dict.keys(), key=lambda x: float(x))
    
    # (3) Research: Individual Momentum Loss Curves in one figure (subplots for each momentum)
    num_momentum = len(momentum_keys)
    fig, axs = plt.subplots(1, num_momentum, figsize=(6*num_momentum, 5))
    if num_momentum == 1:
        axs = [axs]
    for i, m in enumerate(momentum_keys):
        data_m = momentum_dict[m]
        train_losses = data_m.get("losses", {}).get("train", None)
        val_losses = data_m.get("losses", {}).get("val", None)
        if train_losses is None or val_losses is None:
            continue
        epochs_m = range(1, len(train_losses)+1)
        axs[i].plot(epochs_m, train_losses, label="Training Loss", marker="o")
        axs[i].plot(epochs_m, val_losses, label="Validation Loss", marker="s")
        axs[i].set_title(f"Momentum: {m}", fontsize=16)
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        clean_axes(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "research_momentum_loss_curves.png"), dpi=300)
    plt.close(fig)
    
    # (4) Research: Aggregated Quality-Speed Tradeoff across Epochs for different momentum values
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in momentum_keys:
        data_m = momentum_dict[m]
        qs = data_m.get("metrics", {}).get("val", None)
        if qs is None:
            continue
        epochs_m = range(1, len(qs)+1)
        ax.plot(epochs_m, qs, marker="o", label=f"Momentum: {m}")
    ax.set_title("Research Quality-Speed Tradeoff Across Epochs", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Quality-Speed Metric")
    ax.legend()
    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "research_quality_speed_tradeoff.png"), dpi=300)
    plt.close(fig)
    
    print("Research plots created successfully.")
except Exception as e:
    print(f"Error in research plots: {e}")

# ---------------------
# 3. ABLATION - Dataset Diversity Ablation (Linear and Nonlinear)
# ---------------------
try:
    ablation_div_path = "experiment_results/experiment_565ead2cb98440cda6984ae814c1ba4e_proc_3651759/experiment_data.npy"
    ablation_div_data = np.load(ablation_div_path, allow_pickle=True).item()
    dd_data = ablation_div_data.get("dataset_diversity_ablation", {})
    for dataset_type in ["linear", "nonlinear"]:
        ds = dd_data.get(dataset_type, {})
        train_losses = ds.get("losses", {}).get("train", None)
        val_losses = ds.get("losses", {}).get("val", None)
        qs_metrics = ds.get("metrics", {}).get("val", None)
        if train_losses is None or val_losses is None or qs_metrics is None:
            print(f"Missing data for {dataset_type} dataset in ablation.")
            continue
        epochs_ds = range(1, len(train_losses)+1)
        
        # (5) Loss Curves for dataset diversity: {dataset_type} dataset
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs_ds, train_losses, label="Training Loss", marker="o")
        ax.plot(epochs_ds, val_losses, label="Validation Loss", marker="s")
        title_text = f"{dataset_type.capitalize()} Dataset Loss Curves"
        if dataset_type == "nonlinear":
            title_text += " (Appendix)"
        ax.set_title(title_text, fontsize=16)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        clean_axes(ax)
        plt.tight_layout()
        fname = f"ablation_{dataset_type}_loss_curves.png"
        fig.savefig(os.path.join("figures", fname), dpi=300)
        plt.close(fig)
        
        # (6) Quality-Speed Tradeoff for dataset diversity: {dataset_type} dataset
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs_ds, qs_metrics, label="Quality-Speed Tradeoff", marker="o")
        title_text = f"{dataset_type.capitalize()} Dataset Quality-Speed Tradeoff"
        if dataset_type == "nonlinear":
            title_text += " (Appendix)"
        ax.set_title(title_text, fontsize=16)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Tradeoff Metric")
        ax.legend()
        clean_axes(ax)
        plt.tight_layout()
        fname = f"ablation_{dataset_type}_quality_speed_tradeoff.png"
        fig.savefig(os.path.join("figures", fname), dpi=300)
        plt.close(fig)
    print("Ablation (Dataset Diversity) plots created successfully.")
except Exception as e:
    print(f"Error in ablation (dataset diversity) plots: {e}")

# ---------------------
# 4. ABLATION - Multiple Dataset Evaluation Ablation
# ---------------------
try:
    ablation_multi_path = "experiment_results/experiment_12eae2e5710c4d299a181d9cef49c4b6_proc_3651760/experiment_data.npy"
    ablation_multi_data = np.load(ablation_multi_path, allow_pickle=True).item()
    mde_data = ablation_multi_data.get("multiple_dataset_evaluation", {})
    dataset_names = list(mde_data.keys())
    
    # (7) Multiple Dataset Evaluation: Loss Curves (aggregated subplots)
    n_datasets = len(dataset_names)
    cols = min(n_datasets, 3)
    rows = (n_datasets + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    # Make axs iterable as list even if single plot
    if n_datasets == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for i, dname in enumerate(dataset_names):
        ddata = mde_data[dname]
        train_losses = ddata.get("losses", {}).get("train", None)
        val_losses = ddata.get("losses", {}).get("val", None)
        if train_losses is None or val_losses is None:
            continue
        epochs_d = range(1, len(train_losses)+1)
        axs[i].plot(epochs_d, train_losses, label="Training Loss", marker="o")
        axs[i].plot(epochs_d, val_losses, label="Validation Loss", marker="s")
        axs[i].set_title(f"{dname} Loss Curves", fontsize=16)
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        clean_axes(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_multiple_dataset_loss_curves.png"), dpi=300)
    plt.close(fig)
    
    # (8) Multiple Dataset Evaluation: Quality-Speed Tradeoff (aggregated subplots)
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_datasets == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for i, dname in enumerate(dataset_names):
        ddata = mde_data[dname]
        qs = ddata.get("metrics", {}).get("val", None)
        if qs is None:
            continue
        epochs_d = range(1, len(qs)+1)
        axs[i].plot(epochs_d, qs, label="Quality-Speed Tradeoff", marker="o")
        axs[i].set_title(f"{dname} Quality-Speed Tradeoff", fontsize=16)
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Tradeoff Metric")
        axs[i].legend()
        clean_axes(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_multiple_dataset_quality_speed_tradeoff.png"), dpi=300)
    plt.close(fig)
    
    print("Ablation (Multiple Dataset Evaluation) plots created successfully.")
except Exception as e:
    print(f"Error in ablation (multiple dataset evaluation) plots: {e}")

# ---------------------
# 5. AGGREGATED QUALITY-SPEED TRADEOFF COMPARISON
# ---------------------
try:
    # Use three experiments: Baseline (metrics from baseline_data), Research momentum 0.9, and first dataset from multiple evaluation
    # Baseline
    baseline_qs = baseline_metrics
    epochs_base = range(1, len(baseline_qs)+1)
    
    # Research: momentum "0.9"
    research_mom = research_data.get("hyperparam_tuning_momentum", {}).get("0.9", {})
    research_qs = research_mom.get("metrics", {}).get("val", None)
    if research_qs is None:
        raise ValueError("Missing research momentum 0.9 quality-speed data.")
    epochs_research = range(1, len(research_qs)+1)
    
    # Multiple Dataset Evaluation: take the first available dataset
    if len(dataset_names) == 0:
        raise ValueError("No datasets found in multiple dataset evaluation.")
    multi_first = mde_data[dataset_names[0]]
    multi_qs = multi_first.get("metrics", {}).get("val", None)
    if multi_qs is None:
        raise ValueError("Missing quality-speed in multiple dataset evaluation for first dataset.")
    epochs_multi = range(1, len(multi_qs)+1)
    
    # To aggregate, find common epoch length by truncating to minimum length
    min_len = min(len(baseline_qs), len(research_qs), len(multi_qs))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, min_len+1), baseline_qs[:min_len], marker="o", label="Baseline")
    ax.plot(range(1, min_len+1), research_qs[:min_len], marker="s", label="Research (Momentum 0.9)")
    ax.plot(range(1, min_len+1), multi_qs[:min_len], marker="^", label=f"Multiple Eval: {dataset_names[0]}")
    ax.set_title("Aggregated Quality-Speed Tradeoff Comparison", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Tradeoff Metric")
    ax.legend()
    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "aggregated_quality_speed_tradeoff_comparison.png"), dpi=300)
    plt.close(fig)
    
    print("Aggregated quality-speed tradeoff comparison plot created successfully.")
except Exception as e:
    print(f"Error in aggregated quality-speed tradeoff comparison plot: {e}")

# ---------------------
# 6. APPENDIX: Synthetic Data Analysis from Nonlinear Dataset (Histogram)
# ---------------------
try:
    # Histogram of quality-speed tradeoff values for the nonlinear dataset in Dataset Diversity Ablation
    ds_nonlinear = dd_data.get("nonlinear", {})
    qs_nonlinear = ds_nonlinear.get("metrics", {}).get("val", None)
    if qs_nonlinear is None:
        raise ValueError("Missing nonlinear quality-speed metric data for appendix histogram.")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(qs_nonlinear, bins=10, color="lightgreen", edgecolor="black")
    ax.set_title("Appendix: Nonlinear Dataset Quality-Speed Distribution", fontsize=16)
    ax.set_xlabel("Quality-Speed Metric")
    ax.set_ylabel("Frequency")
    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "appendix_nonlinear_quality_speed_histogram.png"), dpi=300)
    plt.close(fig)
    
    print("Appendix synthetic data histogram created successfully.")
except Exception as e:
    print(f"Error in appendix synthetic data plot: {e}")

print("All plots have been generated and saved in the 'figures' directory.")