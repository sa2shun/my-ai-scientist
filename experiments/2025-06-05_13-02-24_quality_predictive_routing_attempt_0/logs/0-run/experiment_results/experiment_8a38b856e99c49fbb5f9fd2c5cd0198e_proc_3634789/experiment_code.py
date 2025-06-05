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


# Model definition with dropout
class QualityPredictor(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(QualityPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


# Training and evaluation function
def train_and_evaluate(dropout_rate):
    model = QualityPredictor(dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            quality = batch["quality"].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), quality)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        experiment_data["dropout_tuning"]["dropout_rate_" + str(dropout_rate)][
            "losses"
        ]["train"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        total_quality = 0
        total_processing_time = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                quality = batch["quality"].to(device)
                processing_time = batch["processing_time"].to(device)
                outputs = model(features)
                val_loss += criterion(outputs.squeeze(), quality).item()
                total_quality += outputs.sum().item()
                total_processing_time += processing_time.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        experiment_data["dropout_tuning"]["dropout_rate_" + str(dropout_rate)][
            "losses"
        ]["val"].append(avg_val_loss)
        quality_speed_tradeoff = (
            total_quality / total_processing_time
        )  # Simplified metric
        experiment_data["dropout_tuning"]["dropout_rate_" + str(dropout_rate)][
            "metrics"
        ]["val"].append(quality_speed_tradeoff)

        print(
            f"Dropout {dropout_rate} Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}, quality_speed_tradeoff = {quality_speed_tradeoff:.4f}"
        )


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset preparation
dataset = QueryResponseDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Experiment data structure
experiment_data = {
    "dropout_tuning": {
        "dropout_rate_0.0": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dropout_rate_0.2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dropout_rate_0.5": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Train and evaluate for different dropout rates
for rate in [0.0, 0.2, 0.5]:
    train_and_evaluate(rate)

# Saving experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
