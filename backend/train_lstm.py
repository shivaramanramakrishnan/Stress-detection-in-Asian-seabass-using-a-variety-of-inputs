import numpy as np
from pathlib import Path
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic_data import AquacultureSyntheticGenerator, SyntheticConfig

LSTM_MODEL_PATH = Path("stress_lstm.pt")
LSTM_META_PATH  = Path("stress_lstm_meta.joblib")


class StressLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out)).squeeze(-1)


def train_lstm():
    print("Generating sequences for LSTM training...")
    cfg = SyntheticConfig(n_samples=1, random_state=42)
    gen = AquacultureSyntheticGenerator(cfg)
    X, y, numeric_cols = gen.generate_sequences(n_sequences=2000, timesteps=24)

    # Normalize — shape (n_features,), no spurious batch/timestep dims
    X_mean = X.mean(axis=(0, 1))
    X_std  = X.std(axis=(0, 1)) + 1e-8
    X_norm = (X - X_mean) / X_std

    joblib.dump({"mean": X_mean, "std": X_std, "cols": numeric_cols}, LSTM_META_PATH)

    split   = int(0.8 * len(X_norm))
    X_train = torch.tensor(X_norm[:split])
    y_train = torch.tensor(y[:split])
    X_val   = torch.tensor(X_norm[split:])
    y_val   = torch.tensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    input_size = X.shape[2]
    model      = StressLSTM(input_size=input_size)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode="min", factor=0.5, patience=3)
    criterion  = nn.MSELoss()

    best_val_loss   = float("inf")
    patience_counter = 0

    print(f"Training LSTM: {input_size} features, 2000 sequences, 24 timesteps each...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/50 | train_loss: {total_loss/len(train_loader):.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f}")
                break

    print(f"Saved best LSTM model (val_loss: {best_val_loss:.4f}) to {LSTM_MODEL_PATH}")
    return model, X_mean, X_std, numeric_cols


if __name__ == "__main__":
    train_lstm()