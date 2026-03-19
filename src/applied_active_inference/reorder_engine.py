"""Reorder decision engine.

For each SKU on each simulated day the engine answers three questions:

  1. **Should we reorder?**  Compare the *inventory position* (on-hand +
     on-order) against a dynamically computed reorder point that accounts
     for demand *and* lead-time uncertainty.
  2. **From which supplier?**  Score eligible suppliers on reliability,
     speed, and cost using a weighted multi-criteria model.
  3. **How much?**  Order enough to cover expected demand during the
     supplier's lead time plus a safety buffer of ``service_level_sigmas``
     standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applied_active_inference.distribution_fitting import FitResult
from applied_active_inference.supplier_reliability import SupplierProfile


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ReorderDecision:
    """Outcome of a single reorder evaluation for one SKU on one day."""

    sku_id: str
    should_reorder: bool
    supplier_id: str | None       # chosen supplier (None if no reorder)
    order_quantity: int            # units to order (0 if no reorder)
    expected_arrival_day: int      # estimated arrival day in the simulation
    reasoning: str                 # human-readable explanation


# ── Reorder-point computation ────────────────────────────────────────────────

def compute_reorder_point(
    sales_fit: FitResult,
    lead_time_days: float,
    lead_time_std: float,
    service_level_sigmas: float = 5.0,
) -> float:
    """Compute the reorder point for a SKU.

    Uses the classic formula that compounds demand variance over the lead
    time with lead-time variance scaled by mean demand::

        ROP = d̄ × L̄  +  z × √( L̄ × σ²_d  +  d̄² × σ²_L )

    where
      - d̄  = mean daily sales
      - σ_d = std of daily sales
      - L̄  = mean supplier lead time (days)
      - σ_L = std of supplier lead time
      - z   = desired service level in standard deviations

    Parameters
    ----------
    sales_fit : fitted distribution for this product category.
    lead_time_days : mean lead time from the chosen supplier.
    lead_time_std : standard deviation of supplier lead time.
    service_level_sigmas : number of σ for the safety buffer (default 5).

    Returns
    -------
    float — the reorder point in units.
    """
    d_mean = sales_fit.mean
    d_std = sales_fit.std

    # Expected demand during lead time
    expected_demand = d_mean * lead_time_days

    # Combined uncertainty: demand variance accumulated over lead time,
    # plus lead-time variance amplified by mean demand.
    demand_var_over_lt = lead_time_days * (d_std ** 2)
    lt_var_contribution = (d_mean ** 2) * (lead_time_std ** 2)
    combined_std = np.sqrt(demand_var_over_lt + lt_var_contribution)

    return expected_demand + service_level_sigmas * combined_std


# ── Order-quantity computation ───────────────────────────────────────────────

def compute_order_quantity(
    sales_fit: FitResult,
    lead_time_days: float,
    safety_stock: float,
    current_stock: float,
) -> int:
    """Compute how many units to order.

    The target inventory level is the expected demand during the lead time
    plus a safety buffer.  We order enough to close the gap between the
    target and what we currently hold.

    Parameters
    ----------
    sales_fit : fitted daily-sales distribution.
    lead_time_days : expected lead time from the chosen supplier.
    safety_stock : safety-stock buffer in units.
    current_stock : units currently on hand.

    Returns
    -------
    int — order quantity (non-negative, rounded up).
    """
    target = sales_fit.mean * lead_time_days + safety_stock
    quantity = max(0, int(np.ceil(target - current_stock)))
    return quantity


# ── Supplier selection ───────────────────────────────────────────────────────

def select_best_supplier(
    eligible_suppliers: list[SupplierProfile],
    weights: dict[str, float] | None = None,
) -> SupplierProfile:
    """Select the best supplier using a weighted multi-criteria score.

    Three criteria are evaluated (all normalised so higher = better):

      * **Reliability** — on-time delivery percentage.
      * **Speed** — inverse of mean lead time (shorter is better).
      * **Cost** — inverse of mean unit cost (cheaper is better).

    Parameters
    ----------
    eligible_suppliers : suppliers that serve the required product category.
    weights : dict with keys ``'reliability'``, ``'speed'``, ``'cost'``.
              Defaults to ``{reliability: 0.5, speed: 0.3, cost: 0.2}``.

    Returns
    -------
    The ``SupplierProfile`` with the highest composite score.
    """
    if weights is None:
        weights = {"reliability": 0.5, "speed": 0.3, "cost": 0.2}

    scores: list[float] = []
    for s in eligible_suppliers:
        reliability_score = s.on_time_pct / 100.0            # 0-1
        speed_score = 1.0 / (s.avg_lead_time + 1.0)          # shorter = higher
        cost_score = 1.0 / (s.avg_unit_cost + 0.01)          # cheaper = higher

        composite = (
            weights["reliability"] * reliability_score
            + weights["speed"] * speed_score
            + weights["cost"] * cost_score
        )
        scores.append(composite)

    best_idx = int(np.argmax(scores))
    return eligible_suppliers[best_idx]


# ── Main decision function ───────────────────────────────────────────────────

def make_reorder_decision(
    sku_id: str,
    current_stock: float,
    pending_orders: float,
    sales_fit: FitResult,
    eligible_suppliers: list[SupplierProfile],
    current_day: int,
    service_level_sigmas: float = 5.0,
) -> ReorderDecision:
    """Decide whether to reorder a SKU and, if so, from which supplier.

    The decision compares the *inventory position* (on-hand + on-order)
    against the computed reorder point.  If the position is at or below
    the ROP, an order is placed with the highest-scoring eligible supplier.

    Parameters
    ----------
    sku_id : identifier for the SKU being evaluated.
    current_stock : units currently on hand.
    pending_orders : units already ordered but not yet received.
    sales_fit : fitted daily-sales distribution for this product category.
    eligible_suppliers : suppliers that can fulfil this SKU.
    current_day : current simulation day (used for arrival estimate).
    service_level_sigmas : number of σ for the safety buffer (default 5).

    Returns
    -------
    ReorderDecision describing the action to take.
    """
    # Pick the best supplier first — we need their lead-time stats for the ROP
    supplier = select_best_supplier(eligible_suppliers)

    reorder_point = compute_reorder_point(
        sales_fit,
        lead_time_days=supplier.avg_lead_time,
        lead_time_std=supplier.std_lead_time,
        service_level_sigmas=service_level_sigmas,
    )

    inventory_position = current_stock + pending_orders

    # ── Reorder? ─────────────────────────────────────────────────────────
    if inventory_position <= reorder_point:
        safety_stock = service_level_sigmas * sales_fit.std
        order_qty = compute_order_quantity(
            sales_fit, supplier.avg_lead_time, safety_stock, current_stock,
        )
        order_qty = max(order_qty, 1)  # always order at least 1 unit

        return ReorderDecision(
            sku_id=sku_id,
            should_reorder=True,
            supplier_id=supplier.supplier_id,
            order_quantity=order_qty,
            expected_arrival_day=current_day + int(np.ceil(supplier.avg_lead_time)),
            reasoning=(
                f"Inventory position ({inventory_position:.0f}) <= "
                f"reorder point ({reorder_point:.0f}).  "
                f"Ordering {order_qty} units from {supplier.supplier_name}."
            ),
        )

    # ── No reorder needed ────────────────────────────────────────────────
    return ReorderDecision(
        sku_id=sku_id,
        should_reorder=False,
        supplier_id=None,
        order_quantity=0,
        expected_arrival_day=0,
        reasoning=(
            f"Inventory position ({inventory_position:.0f}) > "
            f"reorder point ({reorder_point:.0f}).  No reorder needed."
        ),
    )
