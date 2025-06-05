import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name in experiment_data["multiple_dataset_evaluation"]:
    try:
        train_losses = experiment_data["multiple_dataset_evaluation"][dataset_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["multiple_dataset_evaluation"][dataset_name][
            "losses"
        ]["val"]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} loss curves: {e}")
        plt.close()

    try:
        quality_metrics = experiment_data["multiple_dataset_evaluation"][dataset_name][
            "metrics"
        ]["val"]
        plt.figure()
        plt.plot(
            range(1, len(quality_metrics) + 1),
            quality_metrics,
            label="Quality-Speed Tradeoff",
        )
        plt.title(f"{dataset_name} Quality-Speed Tradeoff")
        plt.xlabel("Epochs")
        plt.ylabel("Tradeoff Metric")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_quality_speed_tradeoff.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} quality-speed tradeoff: {e}")
        plt.close()
