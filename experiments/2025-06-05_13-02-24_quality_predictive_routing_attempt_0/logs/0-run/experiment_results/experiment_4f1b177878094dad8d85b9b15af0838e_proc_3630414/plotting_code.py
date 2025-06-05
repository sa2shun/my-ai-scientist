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

try:
    plt.figure()
    plt.plot(experiment_data["quality_route"]["losses"]["train"], label="Training Loss")
    plt.plot(experiment_data["quality_route"]["losses"]["val"], label="Validation Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation losses plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["quality_route"]["metrics"]["val"],
        label="Quality-Speed Tradeoff",
    )
    plt.title("Quality-Speed Tradeoff Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Quality-Speed Tradeoff")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
    plt.close()
