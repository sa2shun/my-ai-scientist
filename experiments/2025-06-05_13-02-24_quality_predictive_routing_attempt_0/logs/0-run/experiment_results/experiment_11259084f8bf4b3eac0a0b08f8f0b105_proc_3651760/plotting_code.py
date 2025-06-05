import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(
        experiment_data["ablation_study"]["removed"]["losses"]["val"], label="Removed"
    )
    plt.plot(
        experiment_data["ablation_study"]["constant"]["losses"]["val"], label="Constant"
    )
    plt.title("Validation Losses per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation losses plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["ablation_study"]["removed"]["metrics"]["val"], label="Removed"
    )
    plt.plot(
        experiment_data["ablation_study"]["constant"]["metrics"]["val"],
        label="Constant",
    )
    plt.title("Quality-Speed Tradeoff per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Quality-Speed Tradeoff")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
