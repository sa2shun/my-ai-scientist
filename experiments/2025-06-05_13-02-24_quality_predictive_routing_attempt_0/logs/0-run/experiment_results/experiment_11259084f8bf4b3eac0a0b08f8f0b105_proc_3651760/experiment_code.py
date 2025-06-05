import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Synthetic dataset generation
class QueryResponseDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.features = np.random.rand(num_samples, 5)  # 5 features
        self.quality = np.dot(
            self.features, np.array([0.3, 0.2, 0.5, 0.1, 0.4])
        ) + np.random.normal(0, 0.1, num_samples)
        self.processing_time = (
            np.random.rand(num_samples) * 0.5 + 0.5
        )  # Simulated processing time between 0.5 and 1.0 seconds

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "quality": torch.tensor(self.quality[idx], dtype=torch.float32),
            "processing_time": torch.tensor(
                self.processing_time[idx], dtype=torch.float32
            ),
        }


# Model definition
class QualityPredictor(nn.Module):
    def __init__(self):
        super(QualityPredictor, self).__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        return self.linear(x)


# Training and evaluation
def train_and_evaluate(train_loader, val_loader, momentum, ablation_type):
    model = QualityPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)

    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    predictions = []
    ground_truth = []

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features = batch["features"]
            quality = batch["quality"]
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), quality)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        losses["train"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        total_quality = 0
        total_processing_time = 0
        quality_speed_tradeoff = 0  # Initialize here
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features = batch["features"]
                quality = batch["quality"]
                if ablation_type == "removed":
                    processing_time = None
                elif ablation_type == "constant":
                    processing_time = torch.full_like(
                        batch["processing_time"], 0.75
                    ).to(device)
                else:
                    processing_time = batch["processing_time"]

                outputs = model(features)
                val_loss += criterion(outputs.squeeze(), quality).item()
                total_quality += outputs.sum().item()
                if processing_time is not None:
                    total_processing_time += processing_time.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        losses["val"].append(avg_val_loss)
        if total_processing_time > 0:
            quality_speed_tradeoff = total_quality / total_processing_time
            metrics["val"].append(quality_speed_tradeoff)

        print(
            f"Ablation {ablation_type}, Momentum {momentum}, Epoch {epoch+1}: "
            f"train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}, "
            f"quality_speed_tradeoff = {quality_speed_tradeoff:.4f}"
        )

    return metrics, losses, predictions, ground_truth


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Original dataset
dataset = QueryResponseDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

momentum_values = [0.0, 0.5, 0.9]
experiment_data = {
    "ablation_study": {
        "removed": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "constant": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

for ablation_type in ["removed", "constant"]:
    for momentum in momentum_values:
        metrics, losses, predictions, ground_truth = train_and_evaluate(
            train_loader, val_loader, momentum, ablation_type
        )
        experiment_data["ablation_study"][ablation_type]["metrics"]["val"].extend(
            metrics["val"]
        )
        experiment_data["ablation_study"][ablation_type]["losses"]["val"].extend(
            losses["val"]
        )

# Saving experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
