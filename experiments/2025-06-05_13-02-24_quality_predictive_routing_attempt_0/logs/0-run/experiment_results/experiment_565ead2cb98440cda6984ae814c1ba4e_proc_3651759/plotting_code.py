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
for dataset_name in ["linear", "nonlinear"]:
    try:
        plt.figure()
        epochs = range(
            1,
            len(
                experiment_data["dataset_diversity_ablation"][dataset_name]["losses"][
                    "train"
                ]
            )
            + 1,
        )
        plt.plot(
            epochs,
            experiment_data["dataset_diversity_ablation"][dataset_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["dataset_diversity_ablation"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} losses: {e}")
        plt.close()

    try:
        plt.figure()
        metrics = experiment_data["dataset_diversity_ablation"][dataset_name][
            "metrics"
        ]["val"]
        plt.plot(epochs, metrics, label="Quality-Speed Tradeoff")
        plt.title(f"{dataset_name.capitalize()} Dataset Quality-Speed Tradeoff")
        plt.xlabel("Epochs")
        plt.ylabel("Quality-Speed Metric")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_quality_speed_tradeoff.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} metrics: {e}")
        plt.close()
