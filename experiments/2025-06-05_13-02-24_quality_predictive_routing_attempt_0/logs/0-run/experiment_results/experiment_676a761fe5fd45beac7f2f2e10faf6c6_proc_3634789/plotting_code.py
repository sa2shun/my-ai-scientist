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
        experiment_data["hyperparam_tuning_epochs"]["quality_route"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_epochs"]["quality_route"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "quality_route_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_epochs"]["quality_route"]["metrics"]["val"],
        label="Quality-Speed Tradeoff",
    )
    plt.title("Quality-Speed Tradeoff Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Tradeoff Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_route_quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
