"""Applied Active Inference — grocery inventory management."""

__version__ = "0.1.0"

from applied_active_inference.data_loader import load_grocery_data
from applied_active_inference.distribution_fitting import (
    FitResult,
    fit_best_distribution,
    fit_categories,
)
from applied_active_inference.supplier_reliability import (
    SupplierProfile,
    build_supplier_profiles,
)
from applied_active_inference.reorder_engine import (
    ReorderDecision,
    make_reorder_decision,
)
from applied_active_inference.simulation import (
    SimulationConfig,
    SimulationResults,
    run_simulation,
)
