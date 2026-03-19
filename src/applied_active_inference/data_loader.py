"""Data loading and cleaning for the Grocery Inventory Management dataset.

Handles the Indonesian locale formatting found in the CSV:
  - Comma-decimal numbers:  "28,57"      -> 28.57
  - Currency values:        "$2.084,25"  -> 2084.25
  - Percentage values:      "70,68%"     -> 70.68
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd


# ── Kaggle dataset identifiers ──────────────────────────────────────────────
DATASET_SLUG = "mustofaahmad/inventory-management-grocery-industry"
CSV_FILENAME = "Inventory Management E-Grocery - InventoryData.csv"

# ── Column groups by locale format ──────────────────────────────────────────

# Columns using comma as decimal separator, e.g. "28,57" meaning 28.57
COMMA_DECIMAL_COLS = [
    "Avg_Daily_Sales",
    "Days_of_Inventory",
    "SKU_Churn_Rate",
    "Order_Frequency_per_month",
]

# Columns with $X.XXX,XX format (dot = thousands, comma = decimal)
CURRENCY_COLS = [
    "Unit_Cost_USD",
    "Last_Purchase_Price_USD",
    "Total_Inventory_Value_USD",
]

# Columns with XX,XX% format (comma decimal, percent sign suffix)
PERCENT_COLS = [
    "Supplier_OnTime_Pct",
    "Audit_Variance_Pct",
    "Demand_Forecast_Accuracy_Pct",
]

# Columns to parse as datetime
DATE_COLS = [
    "Received_Date",
    "Last_Purchase_Date",
    "Expiry_Date",
    "Audit_Date",
]


# ── Locale parsing helpers ──────────────────────────────────────────────────

def _parse_comma_decimal(series: pd.Series) -> pd.Series:
    """Convert comma-decimal strings to floats.  '28,57' -> 28.57

    Also handles dot-thousands + comma-decimal, e.g. '2.142,90' -> 2142.90.
    """
    return (
        series.astype(str)
        .str.replace(".", "", regex=False)    # strip thousands separator (dot)
        .str.replace(",", ".", regex=False)   # swap comma -> decimal point
        .astype(float)
    )


def _parse_currency(series: pd.Series) -> pd.Series:
    """Convert Indonesian-locale currency to floats.  '$2.084,25' -> 2084.25"""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)   # strip dollar sign
        .str.replace(".", "", regex=False)    # strip thousands separator (dot)
        .str.replace(",", ".", regex=False)   # swap comma -> decimal point
        .astype(float)
    )


def _parse_percent(series: pd.Series) -> pd.Series:
    """Convert percent strings to floats.  '70,68%' -> 70.68 (keeps as %)."""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )


# ── Main loader ─────────────────────────────────────────────────────────────

def load_grocery_data(path: str | None = None) -> pd.DataFrame:
    """Load and clean the grocery inventory dataset.

    Parameters
    ----------
    path : str or None
        Explicit path to the CSV file.  If *None*, the dataset is
        downloaded (or fetched from cache) via ``kagglehub``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with proper numeric types and parsed dates.
        Currency columns are in USD (float), percentages are 0-100 (float),
        and all comma-decimal columns are standard Python floats.
    """
    # Resolve the CSV path ────────────────────────────────────────────────
    if path is None:
        dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
        path = str(Path(dataset_dir) / CSV_FILENAME)

    # Read raw — locale-formatted columns come in as strings due to quoting
    # noinspection PyArgumentList
    df = pd.read_csv(path)

    # Clean comma-decimal columns ────────────────────────────────────────
    for col in COMMA_DECIMAL_COLS:
        if col in df.columns:
            df[col] = _parse_comma_decimal(df[col])

    # Clean currency columns ─────────────────────────────────────────────
    for col in CURRENCY_COLS:
        if col in df.columns:
            df[col] = _parse_currency(df[col])

    # Clean percent columns ──────────────────────────────────────────────
    for col in PERCENT_COLS:
        if col in df.columns:
            df[col] = _parse_percent(df[col])

    # Parse date columns ─────────────────────────────────────────────────
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
