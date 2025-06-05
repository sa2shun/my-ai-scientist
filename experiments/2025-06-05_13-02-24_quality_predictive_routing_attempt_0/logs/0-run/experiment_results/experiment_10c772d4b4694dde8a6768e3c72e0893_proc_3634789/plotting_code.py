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

try:
    train_losses = experiment_data["hyperparam_tuning_learning_rate"]["quality_route"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_learning_rate"]["quality_route"][
        "losses"
    ]["val"]
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Curves for Quality Prediction")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_quality_prediction.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")

try:
    metrics = experiment_data["hyperparam_tuning_learning_rate"]["quality_route"][
        "metrics"
    ]["val"]
    plt.figure()
    plt.plot(epochs, metrics, label="Quality-Speed Tradeoff", color="orange")
    plt.title("Quality-Speed Tradeoff")
    plt.xlabel("Epochs")
    plt.ylabel("Tradeoff Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quality_speed_tradeoff.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quality-speed tradeoff plot: {e}")
