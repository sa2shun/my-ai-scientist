import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting training and validation losses
batch_sizes = [16, 32, 64]
for batch_size in batch_sizes:
    try:
        losses = experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
            "losses"
        ]
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_batch_size_{batch_size}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for batch size {batch_size}: {e}")
        plt.close()

# Plotting quality-speed tradeoff
try:
    plt.figure()
    for batch_size in batch_sizes:
        quality_speed = experiment_data["batch_size_tuning"][
            f"batch_size_{batch_size}"
        ]["metrics"]["val"]
        plt.plot(quality_speed, label=f"Batch Size {batch_size}")
    plt.title("Quality-Speed Tradeoff")
    plt.xlabel("Epochs")
    plt.ylabel("Quality-Speed Tradeoff Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
