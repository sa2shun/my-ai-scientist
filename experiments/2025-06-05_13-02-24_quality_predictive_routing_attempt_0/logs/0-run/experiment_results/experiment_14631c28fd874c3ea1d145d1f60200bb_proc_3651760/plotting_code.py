import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses
try:
    train_losses = experiment_data["feature_importance_ablation"][
        "query_response_dataset"
    ]["losses"]["train"]
    val_losses = experiment_data["feature_importance_ablation"][
        "query_response_dataset"
    ]["losses"]["val"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves for Query Response Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "query_response_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plotting quality-speed tradeoff metrics
try:
    quality_speed_tradeoff = experiment_data["feature_importance_ablation"][
        "query_response_dataset"
    ]["metrics"]["val"]
    plt.figure()
    plt.plot(quality_speed_tradeoff, label="Quality-Speed Tradeoff")
    plt.title("Quality-Speed Tradeoff for Query Response Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Tradeoff")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "query_response_quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
