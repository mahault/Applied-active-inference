"""Daily warehouse inventory simulation.

Runs a day-by-day simulation across all SKUs to evaluate the reorder
policy defined by :mod:`reorder_engine`.  The simulation loop:

  1. **Receive** — shipments that have reached their arrival day are added
     to on-hand inventory.
  2. **Sell** — daily demand is sampled from the fitted category distribution
     and fulfilled up to the available stock.
  3. **Decide** — the reorder engine evaluates each SKU and may place a new
     order with a supplier.

NumPy arrays are used to hold the inventory state matrix and to vectorise
the sell step across all SKUs within each day.  Scipy is used for
distribution sampling and supplier lead-time modelling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from applied_active_inference.distribution_fitting import FitResult
from applied_active_inference.supplier_reliability import (
    SupplierProfile,
    effective_lead_time_distribution,
)
from applied_active_inference.reorder_engine import make_reorder_decision


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Tuneable parameters for the warehouse simulation."""

    n_days: int = 90                  # number of days to simulate
    service_level_sigmas: float = 5.0 # target service level (5σ)
    seed: int = 42                    # random seed for reproducibility


# ── Internal bookkeeping ─────────────────────────────────────────────────────

@dataclass
class PendingOrder:
    """An order placed but not yet delivered."""

    sku_idx: int          # index into the SKU tensor dimension
    sku_id: str
    supplier_id: str
    quantity: int
    arrival_day: int      # simulation day the shipment arrives


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass
class SimulationResults:
    """Aggregate output of a simulation run."""

    # Per-SKU time-series  (sku_id -> list of daily values)
    daily_inventory: dict[str, list[float]]
    daily_sales: dict[str, list[float]]
    daily_stockouts: dict[str, list[bool]]

    # Order log — each entry is a dict with keys:
    #   day, sku_id, supplier_id, quantity, expected_arrival, actual_arrival
    orders_placed: list[dict]

    # Summary statistics
    total_stockout_days: int
    total_orders_placed: int
    avg_inventory_level: float
    fill_rate: float           # fraction of total demand that was fulfilled


# ── Sampling helper ──────────────────────────────────────────────────────────

def sample_daily_sales(
    fit: FitResult,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *n_days* daily-sales samples from a fitted distribution.

    Uses the scipy distribution identified during the fitting phase.
    Samples are clipped to ≥ 0 (sales cannot be negative).

    Parameters
    ----------
    fit : the ``FitResult`` for the SKU's product category.
    n_days : how many days of sales to generate.
    rng : numpy random generator for reproducibility.

    Returns
    -------
    1-D np.ndarray of length *n_days*.
    """
    dist = getattr(sp_stats, fit.distribution_name)
    samples = dist.rvs(*fit.params, size=n_days, random_state=rng)
    return np.maximum(samples, 0.0)


# ── Main simulation ─────────────────────────────────────────────────────────

def run_simulation(
    df: pd.DataFrame,
    category_fits: dict[str, FitResult],
    supplier_profiles: dict[str, SupplierProfile],
    config: SimulationConfig | None = None,
) -> SimulationResults:
    """Run the full daily warehouse simulation.

    Parameters
    ----------
    df : cleaned grocery DataFrame.
    category_fits : mapping of ``Category`` -> ``FitResult``.
    supplier_profiles : mapping of ``Supplier_ID`` -> ``SupplierProfile``.
    config : simulation configuration (defaults to 90 days, seed 42).

    Returns
    -------
    SimulationResults with per-SKU time-series and summary metrics.
    """
    if config is None:
        config = SimulationConfig()

    rng = np.random.default_rng(config.seed)

    # ── Build SKU index ──────────────────────────────────────────────────
    # We only simulate SKUs whose category has a fitted distribution.
    sku_rows = [
        row for _, row in df.iterrows()
        if row["Category"] in category_fits
    ]
    n_skus = len(sku_rows)

    # Map SKU index <-> SKU_ID / metadata
    sku_ids: list[str] = [row["SKU_ID"] for row in sku_rows]
    sku_categories: list[str] = [row["Category"] for row in sku_rows]
    sku_supplier_ids: list[str] = [str(row["Supplier_ID"]) for row in sku_rows]

    # ── Pre-generate daily demand for every SKU ──────────────────────────
    # Shape: (n_skus, n_days) — each row is one SKU's synthetic sales trace
    demand_matrix = np.zeros((n_skus, config.n_days), dtype=np.float32)
    for i, cat in enumerate(sku_categories):
        demand_matrix[i] = sample_daily_sales(
            category_fits[cat], config.n_days, rng,
        )

    # ── Initialise inventory array from dataset Quantity_On_Hand ─────────
    stock = np.array(
        [float(row["Quantity_On_Hand"]) for row in sku_rows], dtype=np.float32
    )

    # ── Tracking arrays ──────────────────────────────────────────────────
    inv_history = np.zeros((n_skus, config.n_days), dtype=np.float32)
    sales_history = np.zeros((n_skus, config.n_days), dtype=np.float32)
    stockout_history = np.zeros((n_skus, config.n_days), dtype=bool)

    pending_orders: list[PendingOrder] = []
    orders_log: list[dict] = []

    # ── Day-by-day simulation loop ───────────────────────────────────────
    for day in range(config.n_days):

        # 1. RECEIVE — add arriving shipments to on-hand stock
        still_pending: list[PendingOrder] = []
        for order in pending_orders:
            if order.arrival_day <= day:
                stock[order.sku_idx] += order.quantity
            else:
                still_pending.append(order)
        pending_orders = still_pending

        # 2. SELL — vectorised across all SKUs using numpy
        demand_today = demand_matrix[:, day]
        actual_sales = np.minimum(demand_today, stock)
        stock -= actual_sales

        # Record history
        inv_history[:, day] = stock
        sales_history[:, day] = actual_sales
        stockout_history[:, day] = actual_sales < demand_today

        # 3. DECIDE — evaluate reorder for every SKU
        for i in range(n_skus):
            cat = sku_categories[i]

            # Find suppliers that serve this category
            eligible = [
                p for p in supplier_profiles.values()
                if cat in p.categories_served
            ]
            if not eligible:
                continue

            # Sum up units already on order for this SKU
            pending_qty = sum(
                o.quantity for o in pending_orders if o.sku_idx == i
            )

            decision = make_reorder_decision(
                sku_id=sku_ids[i],
                current_stock=float(stock[i]),
                pending_orders=float(pending_qty),
                sales_fit=category_fits[cat],
                eligible_suppliers=eligible,
                current_day=day,
                service_level_sigmas=config.service_level_sigmas,
            )

            if decision.should_reorder:
                # Sample *actual* delivery time from the supplier model
                supplier = supplier_profiles[decision.supplier_id]
                actual_lead = int(
                    effective_lead_time_distribution(supplier, n_samples=1, rng=rng)[0]
                )

                order = PendingOrder(
                    sku_idx=i,
                    sku_id=sku_ids[i],
                    supplier_id=decision.supplier_id,
                    quantity=decision.order_quantity,
                    arrival_day=day + actual_lead,
                )
                pending_orders.append(order)

                orders_log.append({
                    "day": day,
                    "sku_id": sku_ids[i],
                    "supplier_id": decision.supplier_id,
                    "quantity": decision.order_quantity,
                    "expected_arrival": decision.expected_arrival_day,
                    "actual_arrival": order.arrival_day,
                })

    # ── Build results ────────────────────────────────────────────────────
    daily_inventory = {
        sku_ids[i]: inv_history[i].tolist() for i in range(n_skus)
    }
    daily_sales = {
        sku_ids[i]: sales_history[i].tolist() for i in range(n_skus)
    }
    daily_stockouts = {
        sku_ids[i]: stockout_history[i].tolist() for i in range(n_skus)
    }

    # Summary metrics
    total_stockout_days = int(stockout_history.sum())
    total_fulfilled = float(sales_history.sum())
    total_demand = float(demand_matrix.sum())
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0
    avg_inventory = float(inv_history.mean())

    return SimulationResults(
        daily_inventory=daily_inventory,
        daily_sales=daily_sales,
        daily_stockouts=daily_stockouts,
        orders_placed=orders_log,
        total_stockout_days=total_stockout_days,
        total_orders_placed=len(orders_log),
        avg_inventory_level=avg_inventory,
        fill_rate=fill_rate,
    )
