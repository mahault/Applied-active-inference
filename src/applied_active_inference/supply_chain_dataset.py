"""PyTorch Dataset and DataLoader for the Kaggle Logistics & Supply Chain dataset."""

from __future__ import annotations

import kagglehub
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_SLUG = "datasetengineer/logistics-and-supply-chain-dataset"

FEATURE_COLS = [
    "vehicle_gps_latitude",
    "vehicle_gps_longitude",
    "fuel_consumption_rate",
    "eta_variation_hours",
    "traffic_congestion_level",
    "warehouse_inventory_level",
    "loading_unloading_time",
    "handling_equipment_availability",
    "order_fulfillment_status",
    "weather_condition_severity",
    "port_congestion_level",
    "shipping_costs",
    "supplier_reliability_score",
    "lead_time_days",
    "historical_demand",
    "iot_temperature",
    "cargo_condition_status",
    "route_risk_level",
    "customs_clearance_time",
    "driver_behavior_score",
    "fatigue_monitoring_score",
]

TARGET_COLS = [
    "disruption_likelihood_score",
    "delay_probability",
    "risk_classification",
    "delivery_time_deviation",
]

RISK_LABELS = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}


class SupplyChainDataset(Dataset):
    """Tabular dataset wrapping the logistics & supply-chain CSV.

    Parameters
    ----------
    path : str | None
        Path to the CSV file. If ``None``, downloads via ``kagglehub``.
    normalize : bool
        If ``True``, z-score normalize the feature columns.
    """

    def __init__(self, path: str | None = None, normalize: bool = True) -> None:
        if path is None:
            dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
            path = f"{dataset_dir}/dynamic_supply_chain_logistics_dataset.csv"

        df = pd.read_csv(path, parse_dates=["timestamp"])

        # Encode the categorical risk column
        df["risk_classification"] = df["risk_classification"].map(RISK_LABELS)

        features = df[FEATURE_COLS].values.astype(np.float32)
        targets = df[TARGET_COLS].values.astype(np.float32)

        if normalize:
            self._mean = features.mean(axis=0)
            self._std = features.std(axis=0) + 1e-8
            features = (features - self._mean) / self._std
        else:
            self._mean = None
            self._std = None

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.timestamps = df["timestamp"].values

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def get_dataloader(
    path: str | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience function that returns a ready-to-use DataLoader."""
    ds = SupplyChainDataset(path=path, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    loader = get_dataloader()
    features, targets = next(iter(loader))
    print(f"Dataset size : {len(loader.dataset)}")
    print(f"Feature batch: {features.shape}")
    print(f"Target batch : {targets.shape}")
    print(f"Feature cols : {FEATURE_COLS}")
    print(f"Target cols  : {TARGET_COLS}")
