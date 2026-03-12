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


# Per-feature (min, max) ranges derived from the dataset.
# Used by generate_random_shipment to sample in-distribution values.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "vehicle_gps_latitude":           (30.0,    50.0),
    "vehicle_gps_longitude":          (-120.0,  -70.0),
    "fuel_consumption_rate":          (5.0,     20.0),
    "eta_variation_hours":            (-2.0,     5.0),
    "traffic_congestion_level":       (0.0,     10.0),
    "warehouse_inventory_level":      (0.0,   1000.0),
    "loading_unloading_time":         (0.5,      5.0),
    "handling_equipment_availability": (0.0,      1.0),
    "order_fulfillment_status":       (0.0,      1.0),
    "weather_condition_severity":     (0.0,      1.0),
    "port_congestion_level":          (0.0,     10.0),
    "shipping_costs":                 (100.0, 1000.0),
    "supplier_reliability_score":     (0.0,      1.0),
    "lead_time_days":                 (1.0,     15.0),
    "historical_demand":              (100.0, 10000.0),
    "iot_temperature":                (-10.0,   40.0),
    "cargo_condition_status":         (0.0,      1.0),
    "route_risk_level":               (0.0,     10.0),
    "customs_clearance_time":         (0.5,      5.0),
    "driver_behavior_score":          (0.0,      1.0),
    "fatigue_monitoring_score":       (0.0,      1.0),
}


def generate_random_shipment(n: int = 1, seed: int | None = None) -> dict[str, np.ndarray]:
    """Generate *n* random shipments with in-distribution feature values.

    Each feature is sampled uniformly between the observed min and max in the
    training data.

    Returns a dict mapping feature names to arrays of shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    shipment: dict[str, np.ndarray] = {}
    for col in FEATURE_COLS:
        lo, hi = FEATURE_RANGES[col]
        shipment[col] = rng.uniform(lo, hi, size=n).astype(np.float32)
    return shipment


def generate_random_shipment_tensor(
    n: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Like ``generate_random_shipment`` but returns a ``(n, 21)`` tensor."""
    shipment = generate_random_shipment(n=n, seed=seed)
    return torch.from_numpy(
        np.column_stack([shipment[col] for col in FEATURE_COLS])
    )


if __name__ == "__main__":
    loader = get_dataloader()
    features, targets = next(iter(loader))
    print(f"Dataset size : {len(loader.dataset)}")
    print(f"Feature batch: {features.shape}")
    print(f"Target batch : {targets.shape}")
    print(f"Feature cols : {FEATURE_COLS}")
    print(f"Target cols  : {TARGET_COLS}")
