import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

momentum_values = list(experiment_data["hyperparam_tuning_momentum"].keys())

for momentum in momentum_values:
    try:
        plt.figure()
        plt.plot(
            experiment_data["hyperparam_tuning_momentum"][momentum]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["hyperparam_tuning_momentum"][momentum]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Training and Validation Loss (Momentum: {momentum})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"training_validation_loss_momentum_{momentum}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for momentum {momentum}: {e}")
        plt.close()

try:
    plt.figure()
    for momentum in momentum_values:
        plt.plot(
            experiment_data["hyperparam_tuning_momentum"][momentum]["metrics"]["val"],
            label=f"Momentum: {momentum}",
        )
    plt.title("Quality-Speed Tradeoff across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Quality-Speed Tradeoff")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
