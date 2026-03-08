# train_lstm.py

import numpy as np
from pathlib import Path
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic_data import AquacultureSyntheticGenerator, SyntheticConfig

LSTM_MODEL_PATH = Path("stress_lstm.pt")
LSTM_META_PATH = Path("stress_lstm_meta.joblib")


class StressLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.lstm(x)
        # out: (batch, timesteps, hidden_size)
        return self.fc(out).squeeze(-1)
        # returns: (batch, timesteps)


def train_lstm():
    print("Generating sequences for LSTM training...")
    cfg = SyntheticConfig(n_samples=1, random_state=42)
    gen = AquacultureSyntheticGenerator(cfg)
    X, y, numeric_cols = gen.generate_sequences(n_sequences=2000, timesteps=24)

    # Normalize features
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Save normalization stats for inference
    joblib.dump(
        {"mean": X_mean, "std": X_std, "cols": numeric_cols},
        LSTM_META_PATH,
    )

    # Train/val split
    split = int(0.8 * len(X_norm))
    X_train = torch.tensor(X_norm[:split])
    y_train = torch.tensor(y[:split])
    X_val   = torch.tensor(X_norm[split:])
    y_val   = torch.tensor(y[split:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )

    input_size = X.shape[2]
    model = StressLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Training LSTM: {input_size} features, 2000 sequences, 24 timesteps each...")
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        print(f"Epoch {epoch+1:02d}/20 | train_loss: {total_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f}")

    torch.save(model.state_dict(), LSTM_MODEL_PATH)
    print(f"Saved LSTM model to {LSTM_MODEL_PATH}")
    return model, X_mean, X_std, numeric_cols


if __name__ == "__main__":
    train_lstm()
