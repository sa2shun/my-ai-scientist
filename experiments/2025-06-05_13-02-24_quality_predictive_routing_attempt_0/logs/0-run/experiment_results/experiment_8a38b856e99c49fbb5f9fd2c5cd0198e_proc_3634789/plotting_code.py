import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation loss curves
try:
    plt.figure()
    for rate in experiment_data["dropout_tuning"]:
        plt.plot(
            experiment_data["dropout_tuning"][rate]["losses"]["train"],
            label=f"Train Loss {rate}",
        )
        plt.plot(
            experiment_data["dropout_tuning"][rate]["losses"]["val"],
            label=f"Val Loss {rate}",
            linestyle="--",
        )
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot quality-speed tradeoff
try:
    plt.figure()
    for rate in experiment_data["dropout_tuning"]:
        plt.plot(
            experiment_data["dropout_tuning"][rate]["metrics"]["val"],
            label=f"Quality-Speed Tradeoff {rate}",
        )
    plt.title("Quality-Speed Tradeoff for Validation Data")
    plt.xlabel("Epochs")
    plt.ylabel("Quality-Speed Tradeoff")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
