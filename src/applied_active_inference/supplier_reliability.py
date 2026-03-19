"""Supplier reliability analysis.

Models each supplier's delivery reliability using two key metrics from the
dataset:

  - **On-time delivery percentage** (``Supplier_OnTime_Pct``): the fraction
    of shipments that arrive on or before the promised date.
  - **Lead time** (``Lead_Time_Days``): the number of days between order
    placement and delivery.

The module builds a statistical profile per supplier and provides a function
to sample from the *effective* lead-time distribution — the actual number of
days a shipment takes, accounting for the probability of late delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SupplierProfile:
    """Statistical profile of a single supplier's delivery reliability."""

    supplier_id: str
    supplier_name: str

    # Lead-time statistics (days)
    avg_lead_time: float
    std_lead_time: float
    min_lead_time: float
    max_lead_time: float

    # Reliability
    on_time_pct: float          # average on-time delivery percentage (0-100)
    delay_probability: float    # probability of a late delivery (0-1)

    # Capacity / scope
    n_skus: int                 # number of SKUs this supplier serves
    avg_unit_cost: float        # mean unit cost across supplied SKUs
    categories_served: list[str] = field(default_factory=list)


# ── Public API ───────────────────────────────────────────────────────────────

def build_supplier_profiles(df: pd.DataFrame) -> dict[str, SupplierProfile]:
    """Build a reliability profile for every supplier in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned grocery dataset (output of ``load_grocery_data``).

    Returns
    -------
    dict mapping ``Supplier_ID`` -> ``SupplierProfile``.
    """
    profiles: dict[str, SupplierProfile] = {}

    for supplier_id, group in df.groupby("Supplier_ID"):
        lead = group["Lead_Time_Days"]
        on_time = group["Supplier_OnTime_Pct"]

        profiles[str(supplier_id)] = SupplierProfile(
            supplier_id=str(supplier_id),
            supplier_name=group["Supplier_Name"].iloc[0],
            avg_lead_time=float(lead.mean()),
            std_lead_time=float(lead.std()) if len(lead) > 1 else 1.0,
            min_lead_time=float(lead.min()),
            max_lead_time=float(lead.max()),
            on_time_pct=float(on_time.mean()),
            delay_probability=1.0 - float(on_time.mean()) / 100.0,
            n_skus=len(group),
            avg_unit_cost=float(group["Unit_Cost_USD"].mean()),
            categories_served=group["Category"].unique().tolist(),
        )

    return profiles


def effective_lead_time_distribution(
    profile: SupplierProfile,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from the effective lead-time distribution for a supplier.

    The model works in two stages:

    1. **Base lead time** — drawn from Normal(avg, std), clipped to the
       observed [min, max * 1.5] range so we never get impossible values.
    2. **Delay component** — with probability ``delay_probability``, an
       additional 1–5 day delay is added to simulate late deliveries.

    Parameters
    ----------
    profile : supplier whose lead-time distribution we want to sample.
    n_samples : number of Monte-Carlo samples to draw.
    rng : numpy random generator (for reproducibility).

    Returns
    -------
    np.ndarray of shape ``(n_samples,)`` with integer lead-time values (days).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stage 1: base lead time from a clipped Normal
    std = max(profile.std_lead_time, 0.5)  # floor to avoid degenerate σ=0
    base = rng.normal(profile.avg_lead_time, std, size=n_samples)
    base = np.clip(base, profile.min_lead_time, profile.max_lead_time * 1.5)

    # Stage 2: stochastic delay for late deliveries
    is_late = rng.random(size=n_samples) < profile.delay_probability
    delay = rng.integers(1, 6, size=n_samples) * is_late  # 1-5 extra days

    # Ceiling to whole days (shipments arrive on integer day boundaries)
    return np.ceil(base + delay).astype(int)
