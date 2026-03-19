"""Hypothesis test: distribution-fitted reorder policy vs static baseline.

Hypothesis
----------
Using per-category probability distributions of daily sales to compute
dynamic reorder points (mean + 5σ across lead-time uncertainty) minimises
average held inventory compared to the dataset's static ``Reorder_Point`` /
``Safety_Stock`` values, while maintaining the same or better fill rate.

The test runs two simulations on *identical* synthetic demand traces:

  1. **Baseline** — the existing static policy from the dataset: reorder
     when on-hand stock falls to or below the recorded ``Reorder_Point``,
     order up to ``Reorder_Point + Safety_Stock``, with no stochastic delay.

  2. **Statistical** — distribution-fitted reorder points that combine
     demand variance over the lead time with lead-time variance, plus
     supplier selection based on reliability / speed / cost scores and a
     stochastic delay model.

The hypothesis is SUPPORTED if the statistical approach achieves a lower
average inventory level with no meaningful reduction in fill rate.
"""

from __future__ import annotations

import numpy as np

from applied_active_inference.data_loader import load_grocery_data
from applied_active_inference.distribution_fitting import fit_categories
from applied_active_inference.supplier_reliability import build_supplier_profiles
from applied_active_inference.simulation import (
    SimulationConfig,
    SimulationResults,
    run_baseline_simulation,
    run_simulation,
)


# ── Reporting helpers ────────────────────────────────────────────────────────

def _min_stock_stats(res: SimulationResults) -> tuple[float, float, float]:
    """Return (mean, median, overall_min) of per-SKU minimum stock levels.

    For each SKU, find the lowest inventory it reached during the simulation.
    Aggregating these per-SKU minimums reveals how close each policy comes
    to running out of stock in the worst case.
    """
    # per-SKU minimum: the floor each SKU's inventory hit over all days
    per_sku_min = np.array([min(daily) for daily in res.daily_inventory.values()])
    return float(per_sku_min.mean()), float(np.median(per_sku_min)), float(per_sku_min.min())


def _print_results(label: str, res: SimulationResults) -> None:
    """Print a one-block summary for a single simulation run."""
    mean_min, median_min, overall_min = _min_stock_stats(res)
    print(f"  {label}")
    print(f"    Fill rate        : {res.fill_rate:.2%}")
    print(f"    Stockout-days    : {res.total_stockout_days:,}")
    print(f"    Orders placed    : {res.total_orders_placed:,}")
    print(f"    Avg inventory    : {res.avg_inventory_level:.1f} units")
    print(f"    Min stock (mean) : {mean_min:.1f} units  (avg lowest point per SKU)")
    print(f"    Min stock (med)  : {median_min:.1f} units")
    print(f"    Min stock (worst): {overall_min:.1f} units  (single lowest across all SKUs)")


def _verdict(baseline: SimulationResults, statistical: SimulationResults) -> str:
    """Return a one-line verdict on whether the hypothesis is supported."""
    inv_reduced = statistical.avg_inventory_level < baseline.avg_inventory_level
    fill_maintained = statistical.fill_rate >= baseline.fill_rate - 0.01  # 1% tolerance

    if inv_reduced and fill_maintained:
        pct = (1 - statistical.avg_inventory_level / baseline.avg_inventory_level) * 100
        return (
            f"SUPPORTED — statistical policy holds {pct:.1f}% less inventory "
            f"with no meaningful loss in fill rate."
        )
    elif inv_reduced:
        return (
            "PARTIALLY SUPPORTED — inventory reduced, but fill rate dropped "
            f"by more than 1% ({baseline.fill_rate:.2%} → {statistical.fill_rate:.2%})."
        )
    else:
        return (
            "NOT SUPPORTED — statistical policy did not reduce average inventory "
            f"({statistical.avg_inventory_level:.1f} vs {baseline.avg_inventory_level:.1f})."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load data, run both simulations, and report the hypothesis verdict."""

    N_DAYS = 90
    SEED = 42
    SIGMAS = 5.0  # service-level target for the statistical policy

    print("=" * 60)
    print("  Hypothesis Test: Statistical vs Baseline Reorder Policy")
    print("=" * 60)

    # ── Prepare ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading and cleaning dataset...")
    df = load_grocery_data()
    print(f"      {len(df):,} SKUs across {df['Category'].nunique()} categories")

    print("\n[2/4] Fitting sales distributions per category...")
    category_fits = fit_categories(df)
    for cat, fit in sorted(category_fits.items()):
        print(
            f"      {cat:15s}  {fit.distribution_name:10s}  "
            f"mean={fit.mean:6.1f}  std={fit.std:5.1f}  "
            f"5σ stock={fit.stock_level_5sigma:7.1f}"
        )

    print("\n[3/4] Building supplier reliability profiles...")
    supplier_profiles = build_supplier_profiles(df)
    for sid, p in sorted(supplier_profiles.items()):
        print(
            f"      {sid}  {p.supplier_name:30s}  "
            f"lead={p.avg_lead_time:.1f}±{p.std_lead_time:.1f}d  "
            f"on-time={p.on_time_pct:.1f}%"
        )

    # ── Run simulations ───────────────────────────────────────────────────
    print(f"\n[4/4] Running {N_DAYS}-day simulations (seed={SEED})...")
    config = SimulationConfig(n_days=N_DAYS, service_level_sigmas=SIGMAS, seed=SEED)

    print("      Running baseline (static reorder point)...")
    baseline = run_baseline_simulation(df, category_fits, config)

    print("      Running statistical (distribution-fitted ROP)...")
    statistical = run_simulation(df, category_fits, supplier_profiles, config)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    _print_results("Baseline (static Reorder_Point / Safety_Stock)", baseline)
    print()
    _print_results(f"Statistical (distribution-fitted, σ={SIGMAS:.0f})", statistical)

    inv_delta = statistical.avg_inventory_level - baseline.avg_inventory_level
    fill_delta = (statistical.fill_rate - baseline.fill_rate) * 100
    print(
        f"\n  Δ inventory : {inv_delta:+.1f} units "
        f"({'↓ less stock' if inv_delta < 0 else '↑ more stock'})"
    )
    print(
        f"  Δ fill rate : {fill_delta:+.2f} pp "
        f"({'↑ improved' if fill_delta > 0 else '↓ reduced'})"
    )

    print("\n" + "=" * 60)
    print("  Verdict")
    print("=" * 60)
    print(f"  {_verdict(baseline, statistical)}")
    print()


if __name__ == "__main__":
    main()
