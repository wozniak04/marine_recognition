from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import create_logger

logger = create_logger(__name__)


class ShipLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        out, _ = self.lstm(x)
        center = out.shape[1] // 2
        return self.head(out[:, center, :])


class LSTMTrainer:

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.hidden_size: int = p.get("hidden_size", 128)
        self.num_layers: int = p.get("num_layers", 2)
        self.dropout: float = p.get("dropout", 0.3)
        self.epochs: int = p.get("epochs", 120)
        self.batch_size: int = p.get("batch_size", 64)
        self.lr: float = p.get("learning_rate", 0.002)
        self.patience: int = p.get("patience", 25)
        self.window_size: int = p.get("window_size", 30)
        self.num_classes: int = p.get("num_classes", 4)

        self.feature_cols: list[str] = p.get("feature_cols", [
            "sog_knots", "delta_lat", "delta_lon",
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ShipLSTM | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray | None, gap_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        w = self.window_size
        n = len(X)
        gap_flags = gap_mask if gap_mask is not None else np.zeros(n, dtype=bool)

        # Precompute cumulative gap count to quickly check if a window contains a gap
        gap_cum = np.cumsum(gap_flags.astype(int))

        sequences = []
        labels = []

        for i in range(n - w + 1):
            gaps_in_window = gap_cum[i + w - 1] - (gap_cum[i - 1] if i > 0 else 0)
            if gaps_in_window > 0:
                continue
            sequences.append(X[i : i + w])
            if y is not None:
                labels.append(y[i + w // 2])

        X_seq = np.array(sequences) if sequences else np.empty((0, w, X.shape[1]))
        y_seq = np.array(labels) if labels else None
        return X_seq, y_seq

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = np.nanmean(X, axis=0)
            self.scaler_std = np.nanstd(X, axis=0)
            self.scaler_std[self.scaler_std < 1e-8] = 1.0
        X_norm = (X - self.scaler_mean) / self.scaler_std
        return np.nan_to_num(X_norm, nan=0.0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        gap_train: np.ndarray | None = None,
        gap_val: np.ndarray | None = None,
        class_weights: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        X_train_n = self._normalize(X_train, fit=True)
        X_val_n = self._normalize(X_val)

        X_train_seq, y_train_seq = self._build_sequences(X_train_n, y_train, gap_train)
        X_val_seq, y_val_seq = self._build_sequences(X_val_n, y_val, gap_val)

        if len(X_train_seq) == 0:
            raise ValueError("No valid training sequences")

        logger.info("Sequences: train=%d val=%d", len(X_train_seq), len(X_val_seq))

        n_features = X_train_seq.shape[2]
        self.model = ShipLSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(self.device)

        weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 0.01)

        train_loader = self._make_loader(X_train_seq, y_train_seq, shuffle=True)
        val_loader = self._make_loader(X_val_seq, y_val_seq, shuffle=False) if y_val_seq is not None else None

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            scheduler.step()
            history["train_loss"].append(train_loss)

            if val_loader:
                val_loss, val_acc = self._eval_epoch(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
                        epoch + 1, self.epochs, train_loss, val_loss, val_acc,
                        optimizer.param_groups[0]["lr"],
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return history

    def predict(
        self, X: np.ndarray, gap_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        X_n = self._normalize(X)
        X_seq, _ = self._build_sequences(X_n, None, gap_mask)

        n = len(X)
        predictions = np.full(n, -1, dtype=int)
        confidences = np.full(n, 50.0)

        if len(X_seq) == 0:
            return predictions, confidences

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(X_seq), self.batch_size):
                batch = torch.tensor(X_seq[start:start + self.batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1) * 100.0

        w = self.window_size
        gap_flags = gap_mask if gap_mask is not None else np.zeros(n, dtype=bool)
        gap_cum = np.cumsum(gap_flags.astype(int))
        seq_idx = 0
        for i in range(n - w + 1):
            gaps_in_window = gap_cum[i + w - 1] - (gap_cum[i - 1] if i > 0 else 0)
            if gaps_in_window > 0:
                continue
            center = i + w // 2
            predictions[center] = preds[seq_idx]
            confidences[center] = confs[seq_idx]
            seq_idx += 1

        mask = predictions == -1
        if mask.any():
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) > 0:
                for i in np.where(mask)[0]:
                    nearest = valid_indices[np.abs(valid_indices - i).argmin()]
                    predictions[i] = predictions[nearest]
                    confidences[i] = 30.0

        return predictions, confidences

    def _make_loader(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool) -> DataLoader:
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y, dtype=torch.long))
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            X_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_b)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                X_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(X_b)
                total_loss += criterion(logits, y_b).item() * len(X_b)
                correct += (logits.argmax(1) == y_b).sum().item()
                total += len(y_b)
        return total_loss / total, correct / total

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "lstm_model.pt")
        np.savez(path / "scaler.npz", mean=self.scaler_mean, std=self.scaler_std)
        meta = {
            "feature_cols": self.feature_cols,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "window_size": self.window_size,
        }
        with open(path / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("LSTM saved to %s", path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> LSTMTrainer:
        with open(path / "model_meta.json") as f:
            meta = json.load(f)

        trainer = cls(meta)
        scaler = np.load(path / "scaler.npz")
        trainer.scaler_mean = scaler["mean"]
        trainer.scaler_std = scaler["std"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.device = torch.device(device)

        trainer.model = ShipLSTM(
            input_size=len(meta["feature_cols"]),
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            num_classes=meta["num_classes"],
            dropout=meta["dropout"],
        ).to(trainer.device)

        state_dict = torch.load(path / "lstm_model.pt", map_location=trainer.device, weights_only=True)
        trainer.model.load_state_dict(state_dict)
        trainer.model.eval()

        logger.info("LSTM loaded from %s (features=%s)", path, meta["feature_cols"])
        return trainer
