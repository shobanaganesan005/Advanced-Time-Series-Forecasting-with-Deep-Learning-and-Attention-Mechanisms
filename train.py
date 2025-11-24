
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import BaselineLSTM, Seq2SeqAttentionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowedTimeSeries(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        input_window: int = 72,
        horizon: int = 24,
    ):
        self.data = data.astype("float32")
        self.input_window = input_window
        self.horizon = horizon
        self.n = len(data) - input_window - horizon

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_window]
        y = self.data[idx + self.input_window : idx + self.input_window + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def scale_standard(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    mean = train.mean()
    std = train.std().replace(0, 1.0)
    train_s = (train - mean) / std
    val_s = (val - mean) / std
    test_s = (test - mean) / std
    return train_s, val_s, test_s, mean, std


def mase(y_true, y_pred, seasonal_period: int = 24):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    d = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period]).mean()
    errors = np.abs(y_true - y_pred).mean()
    return errors / (d + 1e-8)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (np.abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100.0


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(((y_true - y_pred) ** 2).mean())


@dataclass
class TrainConfig:
    input_window: int = 72
    horizon: int = 24
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2


def train_model(model, train_loader, val_loader, cfg: TrainConfig):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            if isinstance(model, Seq2SeqAttentionModel):
                # teacher forcing: feed ground truth shifted by one step (simple trick)
                pred = model(xb, yb)
            else:
                pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if isinstance(model, Seq2SeqAttentionModel):
                    pred = model(xb, yb)
                else:
                    pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        print(
            f"Epoch {epoch}/{cfg.epochs}: "
            f"train_loss={mean_train:.4f} val_loss={mean_val:.4f}"
        )

    return history


def evaluate_model(model, loader, cfg: TrainConfig):
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if isinstance(model, Seq2SeqAttentionModel):
                pred = model(xb, yb)
            else:
                pred = model(xb)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0).reshape(-1, yb.shape[-1])
    y_pred = np.concatenate(y_pred_all, axis=0).reshape(-1, yb.shape[-1])

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred),
    }
    return metrics


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    cfg_cli = parser.parse_args(args=args)

    cfg = TrainConfig(epochs=cfg_cli.epochs, batch_size=cfg_cli.batch_size)

    df = pd.read_csv(cfg_cli.csv_path, index_col=0)
    train_df, val_df, test_df = train_val_test_split(df)
    train_s, val_s, test_s, mean, std = scale_standard(train_df, val_df, test_df)

    train_data = train_s.values
    val_data = val_s.values
    test_data = test_s.values

    train_ds = WindowedTimeSeries(train_data, cfg.input_window, cfg.horizon)
    val_ds = WindowedTimeSeries(val_data, cfg.input_window, cfg.horizon)
    test_ds = WindowedTimeSeries(test_data, cfg.input_window, cfg.horizon)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    n_features = df.shape[1]

    # Baseline model
    baseline = BaselineLSTM(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        horizon=cfg.horizon,
    ).to(DEVICE)
    print("Training baseline LSTM...")
    hist_base = train_model(baseline, train_loader, val_loader, cfg)
    metrics_base = evaluate_model(baseline, test_loader, cfg)
    print("Baseline metrics:", metrics_base)

    # Attention model
    attn_model = Seq2SeqAttentionModel(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        horizon=cfg.horizon,
    ).to(DEVICE)
    print("Training attention-based seq2seq model...")
    hist_attn = train_model(attn_model, train_loader, val_loader, cfg)
    metrics_attn = evaluate_model(attn_model, test_loader, cfg)
    print("Attention model metrics:", metrics_attn)

    # Save histories and metrics
    results = {
        "baseline": {"history": hist_base, "metrics": metrics_base},
        "attention": {"history": hist_attn, "metrics": metrics_attn},
    }
    import json

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved training history and metrics to results.json")


if __name__ == "__main__":
    main()
