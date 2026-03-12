"""Train a multi-output regression MLP on the supply-chain dataset."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from applied_active_inference.supply_chain_dataset import (
    FEATURE_COLS,
    TARGET_COLS,
    SupplyChainDataset,
)

N_FEATURES = len(FEATURE_COLS)  # 21
N_TARGETS = len(TARGET_COLS)  # 4


class SupplyChainMLP(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, n_out: int = N_TARGETS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.net(x)
        return torch.concatenate((nn.functional.sigmoid(predictions[:, :2]),
                                  predictions[:, 2:]), axis=-1)

CHECKPOINT_DIR = Path("checkpoints")

def combined_loss(predictions, targets, loss_weights):
    prob_predictions = torch.stack((predictions[:, :2],
                                    1 - predictions[:, :2]), dim=-1)
    prob_targets = torch.stack((targets[:, :2], 1 - targets[:, :2]),
                               dim=-1)
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    per_target_ce = ce_loss(prob_predictions[:, 0], prob_targets[:, 0])
    per_target_ce = torch.stack((per_target_ce,
                                 ce_loss(prob_predictions[:, 1],
                                         prob_targets[:, 1])),
                                dim=-1).mean(dim=0)
    per_target_mse = ((predictions[:, 2:] - targets[:, 2:]) ** 2).mean(dim=0)
    loss = (per_target_ce * loss_weights[:2] +\
            per_target_mse * loss_weights[2:]).mean()
    return loss

def train(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    val_frac: float = 0.2,
    checkpoint_path: Path | str | None = None,
) -> SupplyChainMLP:
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    ds = SupplyChainDataset(normalize=True)
    val_size = int(len(ds) * val_frac)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SupplyChainMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Weight each target by inverse variance so all contribute equally
    all_targets_tensor = ds.targets
    target_var = all_targets_tensor.var(dim=0) + 1e-8
    loss_weights = (1.0 / target_var)
    loss_weights = loss_weights / loss_weights.sum() * N_TARGETS  # normalize so weights sum to N_TARGETS
    loss_weights = loss_weights.to(device)
    print(f"Loss weights: {dict(zip(TARGET_COLS, loss_weights.tolist()))}")

    best_val_loss = float("inf")
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(features), targets, loss_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(features)
        train_loss /= train_size

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                loss = combined_loss(model(features), targets, loss_weights)
                val_loss += loss.item() * len(features)
        val_loss /= val_size

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.4f}")

    # Per-target MAE on validation set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            all_preds.append(model(features.to(device)).cpu())
            all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = (all_preds - all_targets).abs().mean(dim=0)

    print("\nPer-target MAE on validation set:")
    for name, m in zip(TARGET_COLS, mae):
        print(f"  {name:30s} {m:.4f}")

    # Save checkpoint
    if checkpoint_path is None:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        checkpoint_path = CHECKPOINT_DIR / "supply_chain_mlp.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "loss_weights": loss_weights.cpu(),
        "feature_mean": ds._mean,
        "feature_std": ds._std,
        "target_cols": TARGET_COLS,
        "feature_cols": FEATURE_COLS,
        "best_val_loss": best_val_loss,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return model


if __name__ == "__main__":
    train()
