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

weight_decays = list(
    experiment_data["weight_decay_tuning"]["query_response_dataset"]["losses"]["train"]
)
train_losses = experiment_data["weight_decay_tuning"]["query_response_dataset"][
    "losses"
]["train"]
val_losses = experiment_data["weight_decay_tuning"]["query_response_dataset"]["losses"][
    "val"
]
metrics = experiment_data["weight_decay_tuning"]["query_response_dataset"]["metrics"][
    "val"
]

try:
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.title("Loss Curves: Query Response Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "query_response_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.bar(range(len(weight_decays)), metrics)
    plt.xticks(range(len(weight_decays)), weight_decays)
    plt.title("Quality-Speed Tradeoff: Query Response Dataset")
    plt.xlabel("Weight Decay")
    plt.ylabel("Quality-Speed Tradeoff")
    plt.savefig(
        os.path.join(working_dir, "query_response_dataset_quality_speed_tradeoff.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
