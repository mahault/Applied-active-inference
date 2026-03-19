"""Active inference for grocery inventory management.

Parametric inventory simulator + homeostatic preference model + active inference agent.
Designed for the Kaggle E-Grocery Inventory Management dataset (cross-sectional snapshot
of 1000 SKUs). The snapshot parameterizes per-SKU stochastic simulators; the active
inference agent makes daily binary order/hold decisions.

Architecture follows the same pattern as supply_chain_pomdp.ipynb:
  - GrocerySimulator  ↔ StepFunction  (generative process)
  - GroceryPreferences ↔ HomeostaticPreferences  (C matrix)
  - GroceryAIAgent     ↔ ActiveInferenceAgent  (EFE minimization)
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def parse_european_number(val):
    """Parse European/Indonesian number format.

    '28,57'     → 28.57
    '$5,81'     → 5.81
    '70,68%'    → 70.68
    '-7,14%'    → -7.14
    '2.084,25'  → 2084.25   (dot = thousands, comma = decimal)
    """
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace('$', '').replace('%', '').strip()
    # Remove thousands separator (dot), then comma → decimal point
    s = s.replace('.', '').replace(',', '.')
    return float(s)


@dataclass
class SKUParams:
    """Parameters extracted from one dataset row for simulation."""
    sku_id: str
    category: str
    abc_class: str
    avg_daily_sales: float
    lead_time_days: int
    supplier_on_time: float      # fraction [0, 1]
    demand_accuracy: float       # fraction [0, 1]
    reorder_point: float
    safety_stock: float
    quantity_on_hand: float
    quantity_reserved: float
    quantity_committed: float
    unit_cost: float
    days_of_inventory: float
    order_frequency: float

    @property
    def atp(self):
        return max(0.0, self.quantity_on_hand - self.quantity_reserved - self.quantity_committed)

    @property
    def demand_std(self):
        """Demand standard deviation derived from forecast accuracy."""
        return self.avg_daily_sales * max(0.05, 1.0 - self.demand_accuracy)

    @property
    def lead_time_std(self):
        """Lead time variability derived from supplier reliability."""
        return self.lead_time_days * max(0.05, 1.0 - self.supplier_on_time)


def extract_sku_params(row) -> SKUParams:
    """Extract SKUParams from a preprocessed DataFrame row (after number parsing)."""
    return SKUParams(
        sku_id=row['SKU_ID'],
        category=row['Category'],
        abc_class=row['ABC_Class'],
        avg_daily_sales=row['Avg_Daily_Sales'],
        lead_time_days=int(row['Lead_Time_Days']),
        supplier_on_time=row['Supplier_OnTime_Pct'] / 100.0,
        demand_accuracy=row['Demand_Forecast_Accuracy_Pct'] / 100.0,
        reorder_point=row['Reorder_Point'],
        safety_stock=row['Safety_Stock'],
        quantity_on_hand=row['Quantity_On_Hand'],
        quantity_reserved=row['Quantity_Reserved'],
        quantity_committed=row['Quantity_Committed'],
        unit_cost=row['Unit_Cost_USD'],
        days_of_inventory=row['Days_of_Inventory'],
        order_frequency=row['Order_Frequency_per_month'],
    )


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# State vector: [inventory, on_order, days_until_delivery, demand_mean, demand_std]
# Index constants:
INV = 0
ON_ORDER = 1
DAYS_UNTIL = 2
DEMAND_MEAN = 3
DEMAND_STD = 4
STATE_DIM = 5


def make_initial_state(params: SKUParams) -> np.ndarray:
    """Create initial state vector from SKU parameters."""
    return np.array([
        params.atp,
        0.0,                # no pending orders at start
        0.0,                # no delivery countdown
        params.avg_daily_sales,
        params.demand_std,
    ])


# ---------------------------------------------------------------------------
# Generative Process — Parametric Inventory Simulator
# ---------------------------------------------------------------------------

class GrocerySimulator:
    """Parametric daily inventory simulator for a single SKU.

    step(state, action) → next_state  (stochastic single sample)

    This replaces the neural-net StepFunction from supply_chain_pomdp.ipynb
    with analytical inventory dynamics parameterized by the dataset snapshot.

    Uses a separate RNG for simulation vs. planning so that the AI agent's
    internal Monte Carlo doesn't contaminate the simulation trajectory.
    """

    def __init__(self, params: SKUParams):
        self.params = params
        # Separate RNG streams: sim_rng for actual simulation, plan_rng for planning
        self.sim_rng = np.random.default_rng()
        self.plan_rng = np.random.default_rng()

    def set_sim_seed(self, seed: int):
        """Reset the simulation RNG (for reproducible evaluation)."""
        self.sim_rng = np.random.default_rng(seed)

    def _step_with_rng(self, state: np.ndarray, action: int,
                       rng: np.random.Generator) -> np.ndarray:
        """Core transition logic using a specific RNG.

        Order of operations (standard inventory management):
        1. Receive deliveries (morning)
        2. Process demand / sales (throughout day)
        3. Place new orders (end of day)
        """
        inventory = state[INV]
        on_order = state[ON_ORDER]
        days_until = state[DAYS_UNTIL]
        demand_mean = state[DEMAND_MEAN]
        demand_std = state[DEMAND_STD]

        # 1. Delivery arrival check (happens first — morning receiving)
        new_on_order = on_order
        new_days_until = days_until
        new_inventory = inventory
        if days_until >= 0.5 and days_until < 1.5:
            if rng.random() < self.params.supplier_on_time:
                new_inventory += on_order
                new_on_order = 0.0
                new_days_until = 0.0
            else:
                new_days_until = float(rng.integers(1, 4))
        elif days_until >= 1.5:
            new_days_until = days_until - 1.0

        # 2. Demand realization (truncated normal, >= 0)
        demand = max(0.0, rng.normal(demand_mean, demand_std))

        # 3. Inventory depletion (lost sales if insufficient stock)
        new_inventory = max(0.0, new_inventory - demand)

        # 4. Order placement (multiple outstanding orders allowed)
        if action == 1:
            # Order-up-to level S = d*(L+R) + z*sigma*sqrt(L+R), z=1.65 for 95%
            L = self.params.lead_time_days
            R = 1.0  # daily review
            order_up_to = (demand_mean * (L + R)
                           + 1.65 * demand_std * np.sqrt(L + R))
            order_qty = max(0.0, order_up_to - new_inventory - new_on_order)
            if order_qty > 0:
                new_on_order += order_qty
                lt = max(1, int(round(rng.normal(
                    self.params.lead_time_days, self.params.lead_time_std
                ))))
                new_days_until = max(new_days_until, float(lt))

        return np.array([new_inventory, new_on_order, new_days_until,
                         demand_mean, demand_std])

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        """One-day stochastic transition (uses simulation RNG)."""
        return self._step_with_rng(state, action, self.sim_rng)

    def step_distribution(self, state: np.ndarray, action: int,
                          n_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) of next-state distribution via Monte Carlo.

        Uses the planning RNG so it doesn't affect the simulation trajectory.
        Matches the StepFunction.step() API from supply_chain_pomdp.ipynb.
        """
        samples = np.array([
            self._step_with_rng(state, action, self.plan_rng)
            for _ in range(n_samples)
        ])
        return samples.mean(axis=0), samples.std(axis=0) + 1e-3

    def step_distribution_analytical(self, state: np.ndarray, action: int
                                     ) -> tuple[np.ndarray, np.ndarray]:
        """Fast analytical Gaussian approximation of the next-state distribution.

        Avoids Monte Carlo — computes mean and std directly from the dynamics.
        Uses Gaussian approximation (ignores max(0,...) clamp) which is accurate
        when inventory is well above zero.
        """
        inventory = state[INV]
        on_order = state[ON_ORDER]
        days_until = state[DAYS_UNTIL]
        demand_mean = state[DEMAND_MEAN]
        demand_std = state[DEMAND_STD]
        p = self.params

        # Expected delivery
        delivery_mean = 0.0
        if 0.5 <= days_until < 1.5:
            delivery_mean = on_order * p.supplier_on_time

        # Inventory: inv + delivery - demand
        inv_mean = max(0.0, inventory + delivery_mean - demand_mean)
        inv_std = demand_std  # dominated by demand noise

        # On-order after delivery and potential new order
        new_on_order = on_order * (1 - p.supplier_on_time) if 0.5 <= days_until < 1.5 else on_order
        if action == 1:
            L = p.lead_time_days
            order_up_to = demand_mean * (L + 1) + 1.65 * demand_std * np.sqrt(L + 1)
            order_qty = max(0.0, order_up_to - inv_mean - new_on_order)
            new_on_order += order_qty
        on_order_std = demand_std * 0.5  # rough approximation

        # Days until delivery
        if days_until >= 1.5:
            new_days_mean = days_until - 1.0
        elif 0.5 <= days_until < 1.5:
            new_days_mean = (1 - p.supplier_on_time) * 2.0  # expected delay if late
        else:
            new_days_mean = 0.0
        if action == 1 and new_on_order > on_order:
            new_days_mean = max(new_days_mean, float(p.lead_time_days))
        days_std = max(p.lead_time_std, 0.5) if new_days_mean > 0 else 0.1

        mean = np.array([inv_mean, new_on_order, new_days_mean, demand_mean, demand_std])
        std = np.array([inv_std, on_order_std, days_std, 1e-3, 1e-3])
        return mean, std

    def observe(self, state: np.ndarray) -> np.ndarray:
        """Generate noisy observation from true state.

        Returns: [observed_inventory, realized_demand, supplier_signal]
        """
        inventory = state[INV]
        demand_mean = state[DEMAND_MEAN]
        demand_std = state[DEMAND_STD]

        # Noisy inventory count (audit variance)
        obs_inv = max(0.0, inventory + np.random.normal(0, max(1.0, inventory * 0.03)))
        # Realized demand (what we actually observed in sales)
        obs_demand = max(0.0, np.random.normal(demand_mean, demand_std))
        # Supplier reliability signal
        supplier_signal = 1.0 if np.random.random() < self.params.supplier_on_time else 0.0

        return np.array([obs_inv, obs_demand, supplier_signal])


# ---------------------------------------------------------------------------
# Homeostatic Preference Model (C matrix)
# ---------------------------------------------------------------------------

class GroceryPreferences:
    """Homeostatic target distribution for grocery inventory (C matrix).

    The preference encodes the agent's prior beliefs about its own viable
    states — a form of self-evidencing (Friston, 2019). Not reward
    maximization; the system maintains itself within an operationally
    viable regime.

    The target is the ORDER-UP-TO level from classical inventory theory:

        S = d * (L + R) + z * sigma_d * sqrt(L + R)

    where d = avg daily demand, L = lead time, R = review period (1 day),
    z = service-level factor, sigma_d = demand std.

    This is principled: S is the mathematically optimal inventory level for
    a given service level. We encode it as the homeostatic setpoint, and the
    precision reflects the asymmetric cost structure (stockout >> overstock).

    Follows the same pattern as HomeostaticPreferences in supply_chain_pomdp.ipynb.
    """

    def __init__(self, params: SKUParams, service_z: float = 1.65):
        """
        Args:
            params: SKU parameters from the dataset
            service_z: z-score for target service level (1.65 ≈ 95%)
        """
        L = params.lead_time_days
        R = 1.0  # daily review period (we decide every day)
        d = params.avg_daily_sales
        sigma_d = params.demand_std

        # Order-up-to level = expected demand during (L+R) + safety buffer
        self.target_inventory = d * (L + R) + service_z * sigma_d * np.sqrt(L + R)
        self.safety_stock = params.safety_stock
        self.params = params

        # Precision encodes cost asymmetry:
        # - Below target: high precision (risk of stockout, lost sales)
        # - Above target: low precision (just holding cost)
        # Ratio reflects that stockout cost >> holding cost in grocery
        self.precision_below = 10.0    # urgency of being understocked
        self.precision_above = 1.0     # mild cost of overstocking
        self.stockout_penalty = 50.0   # catastrophic — non-viable state

    def compute_deviation(self, state: np.ndarray) -> float:
        """Asymmetric precision-weighted deviation from homeostatic target.

        This is -ln C(o) in the EFE equation: the negative log-preference.
        """
        inventory = state[INV]

        # Normalized distance from target (in units of target)
        delta = (inventory - self.target_inventory) / max(self.target_inventory, 1.0)

        if inventory <= 0:
            return self.stockout_penalty
        elif inventory < self.safety_stock:
            # Below safety stock: near-catastrophic
            frac = 1.0 - inventory / max(self.safety_stock, 1.0)
            return self.stockout_penalty * 0.5 + self.precision_below * frac
        elif delta < 0:
            # Below target but above safety stock: concerning
            return self.precision_below * delta ** 2
        else:
            # Above target: gentle holding cost
            return self.precision_above * delta ** 2

    def log_preference(self, state: np.ndarray) -> float:
        """Log-probability under preference distribution. Higher = more preferred."""
        return -self.compute_deviation(state)

    def in_bounds(self, state: np.ndarray, tolerance: float = 1.0) -> bool:
        """Check if state is within homeostatic bounds."""
        return self.compute_deviation(state) < tolerance


# ---------------------------------------------------------------------------
# Active Inference Agent
# ---------------------------------------------------------------------------

class GroceryAIAgent:
    """Selects daily order/hold decisions by minimizing expected free energy.

    G(π, τ) = -E_q[ln C(o_τ)]        (pragmatic: preference alignment)
              + H[q(o_τ|π)]           (epistemic: ambiguity)

    Follows the same architecture as ActiveInferenceAgent in supply_chain_pomdp.ipynb.
    """

    def __init__(self, simulator: GrocerySimulator, preferences: GroceryPreferences,
                 planning_horizon: int | None = None, n_rollouts: int = 50,
                 use_analytical: bool = True):
        self.simulator = simulator
        self.preferences = preferences
        # Adaptive horizon: at least lead_time + 3 so we can see delivery benefit
        if planning_horizon is None:
            planning_horizon = max(7, simulator.params.lead_time_days + 3)
        self.planning_horizon = planning_horizon
        self.n_rollouts = n_rollouts
        self.use_analytical = use_analytical

    # Indices of dynamic state dimensions (exclude constant demand_mean, demand_std)
    DYNAMIC_DIMS = [INV, ON_ORDER, DAYS_UNTIL]

    def expected_free_energy(self, state: np.ndarray, policy: list[int]) -> float:
        """Compute G(π) = Σ_τ [pragmatic + epistemic] for a policy sequence."""
        current = state.copy()
        G = 0.0
        step_fn = (self.simulator.step_distribution_analytical if self.use_analytical
                   else self.simulator.step_distribution)

        for tau in range(len(policy)):
            action = policy[tau]
            mean, std = step_fn(current, action)

            # Pragmatic: precision-weighted deviation from homeostatic target
            pragmatic = self.preferences.compute_deviation(mean)

            # Epistemic: entropy of predictive distribution over DYNAMIC dims only.
            # Constant params (demand_mean, demand_std) have near-zero variance
            # and would produce -inf entropy, drowning out the pragmatic signal.
            dynamic_std = std[self.DYNAMIC_DIMS]
            dynamic_std = np.maximum(dynamic_std, 1e-3)  # floor to avoid log(0)
            ambiguity = 0.5 * np.sum(np.log(2 * np.pi * np.e * dynamic_std ** 2))

            G += pragmatic + ambiguity

            # Advance using predicted mean (deterministic planning)
            current = mean

        return G

    def select_action(self, state: np.ndarray) -> tuple[int, list[float]]:
        """Evaluate hold vs order by averaging over random policy continuations.

        Returns: (best_action, G_per_action)
        """
        G_per_action = []

        for a0 in range(2):  # 0=hold, 1=order
            G_total = 0.0
            for _ in range(self.n_rollouts):
                continuation = list(np.random.randint(0, 2, self.planning_horizon - 1))
                policy = [a0] + continuation
                G_total += self.expected_free_energy(state, policy)
            G_per_action.append(G_total / self.n_rollouts)

        best = int(np.argmin(G_per_action))
        return best, G_per_action


# ---------------------------------------------------------------------------
# Baseline: Reorder Point Policy
# ---------------------------------------------------------------------------

class ReorderPointAgent:
    """Classical (s, Q) reorder point policy from the dataset's built-in parameters.

    Orders when inventory ≤ reorder_point. This is the statistical baseline
    the collaborator is building with Monte Carlo.
    """

    def __init__(self, params: SKUParams):
        self.reorder_point = params.reorder_point
        self.safety_stock = params.safety_stock
        self.avg_daily_sales = params.avg_daily_sales
        self.lead_time = params.lead_time_days

    def select_action(self, state: np.ndarray) -> tuple[int, list[float]]:
        """Returns (action, info). action: 1=order if inventory ≤ reorder_point."""
        inventory = state[INV]
        if inventory <= self.reorder_point:
            return 1, [0.0, 0.0]  # order
        return 0, [0.0, 0.0]      # hold


# ---------------------------------------------------------------------------
# Simulation Loop
# ---------------------------------------------------------------------------

def run_simulation(agent, simulator: GrocerySimulator, initial_state: np.ndarray,
                   horizon: int = 90, seed: int | None = None) -> dict:
    """Run a single SKU simulation for `horizon` days.

    Both agent types implement select_action(state) → (action, info),
    so this function works for GroceryAIAgent and ReorderPointAgent alike.

    Uses the simulator's sim_rng for reproducible demand/delivery sequences
    that are independent of the agent's planning computations.

    Returns dict with trajectories and metrics.
    """
    if seed is not None:
        simulator.set_sim_seed(seed)

    states = [initial_state.copy()]
    actions = []
    demands = []
    stockout_days = 0
    lost_sales = 0.0

    current = initial_state.copy()

    for t in range(horizon):
        action, _ = agent.select_action(current)
        actions.append(action)

        prev_inv = current[INV]
        current = simulator.step(current, action)

        # Track demand (inventory drop + any delivery that arrived)
        effective_demand = max(0.0, prev_inv - current[INV])
        demands.append(effective_demand)

        if current[INV] <= 0:
            stockout_days += 1
            # Lost sales: demand that couldn't be fulfilled
            lost_sales += max(0.0, current[DEMAND_MEAN] - prev_inv)

        states.append(current.copy())

    states = np.array(states)
    return {
        'states': states,
        'actions': actions,
        'demands': demands,
        'stockout_days': stockout_days,
        'lost_sales': lost_sales,
        'fill_rate': 1.0 - stockout_days / horizon,
        'avg_inventory': states[:, INV].mean(),
        'total_orders': sum(actions),
        'min_inventory': states[:, INV].min(),
    }
