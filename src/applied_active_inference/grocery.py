"""Active inference for grocery inventory management.

Proper active inference implementation with:
  - Belief state: Gaussian posterior q(s) = N(mu, diag(sigma^2)) updated via VFE
  - Observation model: P(o|s) mapping hidden states to noisy observations
  - EFE: Risk + Ambiguity decomposition with closed-form KL divergence
  - Softmax policy selection: P(pi) ~ exp(-gamma * G(pi))

VFE (state estimation / perception):
    F = KL[q(s) || P(s)] - E_q[ln P(o|s)]
    Minimized by Kalman-like precision-weighted prediction error update
    (Baltieri & Isomura 2021: KF = steady-state of VFE gradient descent)

EFE (action selection / planning):
    G(pi, tau) = Risk + Ambiguity - InfoGain + StockoutPenalty
    Risk     = KL[q(o_tau|pi) || P(o)]     (divergence from preferences)
    Ambiguity = E_q[H[P(o|s)]]             (expected observation entropy)
    InfoGain  = H[q(o|pi)] - E[H[P(o|s)]]  (mutual information)

References:
    Parr & Friston (2019), "Generalised Free Energy and Active Inference"
    Millidge et al. (2021), "Whence the Expected Free Energy?"
    Baltieri & Isomura (2021), "Kalman filters as steady-state of VFE gradient descent"
"""

import numpy as np
from dataclasses import dataclass, field


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
        # Separate RNG streams to prevent cross-contamination:
        # sim_rng: actual simulation trajectory (demand, delivery)
        # plan_rng: agent's internal Monte Carlo planning
        # obs_rng: observation noise (so it doesn't alter demand/delivery sequence)
        self.sim_rng = np.random.default_rng()
        self.plan_rng = np.random.default_rng()
        self.obs_rng = np.random.default_rng()

    def set_sim_seed(self, seed: int):
        """Reset simulation and observation RNGs (for reproducible evaluation)."""
        self.sim_rng = np.random.default_rng(seed)
        self.obs_rng = np.random.default_rng(seed + 10000)

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
# Observation Model — P(o|s)
# ---------------------------------------------------------------------------

class ObservationModel:
    """Gaussian observation likelihood P(o|s) for the inventory POMDP.

    o_i = s_i + epsilon_i,  epsilon_i ~ N(0, sigma^2_obs_i)

    Each state dimension has dimension-specific observation noise reflecting
    how accurately that quantity can be measured:
      - INV: inventory audit noise (~3% of initial level)
      - ON_ORDER: well-tracked purchase orders
      - DAYS_UNTIL: delivery tracking system
      - DEMAND_MEAN: observed through noisy daily sales
      - DEMAND_STD: estimated from sales variance (noisier)

    Uses state-independent noise (fixed at init) for tractable Kalman updates.
    """

    def __init__(self, params: SKUParams, obs_rng: np.random.Generator | None = None):
        self.params = params
        self.obs_rng = obs_rng or np.random.default_rng()

        inv_scale = max(params.atp, 10.0)
        self.obs_std = np.array([
            max(1.0, 0.03 * inv_scale),                # INV: audit noise
            max(0.5, 0.01 * inv_scale),                 # ON_ORDER: well-tracked
            0.5,                                         # DAYS_UNTIL: +/-0.5 day
            max(0.5, 0.15 * params.avg_daily_sales),    # DEMAND_MEAN: forecast noise
            max(0.1, 0.20 * params.demand_std),          # DEMAND_STD: volatility noise
        ])
        self.obs_var = self.obs_std ** 2

    def observe(self, true_state: np.ndarray) -> np.ndarray:
        """Sample o ~ P(o|s) = N(s, diag(sigma^2_obs))."""
        noise = self.obs_rng.normal(0.0, self.obs_std)
        obs = true_state + noise
        obs[INV] = max(0.0, obs[INV])
        obs[ON_ORDER] = max(0.0, obs[ON_ORDER])
        obs[DAYS_UNTIL] = max(0.0, obs[DAYS_UNTIL])
        obs[DEMAND_MEAN] = max(0.01, obs[DEMAND_MEAN])
        obs[DEMAND_STD] = max(0.01, obs[DEMAND_STD])
        return obs

    def log_likelihood(self, obs: np.ndarray, state: np.ndarray) -> float:
        """ln P(o|s) = -0.5 * sum [(o-s)^2/var + ln(2*pi*var)]."""
        residual = obs - state
        return -0.5 * np.sum(residual**2 / self.obs_var
                             + np.log(2 * np.pi * self.obs_var))


# ---------------------------------------------------------------------------
# Belief State — q(s) = N(mu, diag(sigma^2))
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """Diagonal Gaussian posterior q(s) = N(mu, diag(sigma^2)).

    Maintained by minimizing Variational Free Energy (VFE) each timestep.
    The VFE steady-state solution is equivalent to a Kalman filter for
    the linear-Gaussian case (Baltieri & Isomura, 2021).

    Predict step (transition model):
        mu^-    = f(mu^+, a)
        sigma^2_- = sigma^2_+ + sigma^2_w

    Update step (observation):
        K_i      = sigma^2_-_i / (sigma^2_-_i + sigma^2_obs_i)
        mu^+_i   = mu^-_i + K_i * (o_i - mu^-_i)
        sigma^2_+_i = (1 - K_i) * sigma^2_-_i
    """
    mu: np.ndarray               # posterior mean, shape (STATE_DIM,)
    sigma2: np.ndarray           # posterior variance (diagonal), shape (STATE_DIM,)
    process_noise: np.ndarray    # sigma^2_w per dimension, shape (STATE_DIM,)

    @staticmethod
    def initialize(initial_obs: np.ndarray, obs_model: 'ObservationModel',
                   params: SKUParams) -> 'BeliefState':
        """Initialize belief from first observation with moderate uncertainty."""
        mu = initial_obs.copy()
        sigma2 = obs_model.obs_var * 2.0  # start uncertain

        # Process noise: how much does the true state change per step?
        process_noise = np.array([
            params.demand_std ** 2,                    # INV: demand variance
            (params.avg_daily_sales * 0.5) ** 2,       # ON_ORDER: order qty
            max(params.lead_time_std, 0.5) ** 2,       # DAYS_UNTIL: lead time
            (params.avg_daily_sales * 0.02) ** 2,      # DEMAND_MEAN: very slow
            (params.demand_std * 0.05) ** 2,           # DEMAND_STD: very slow
        ])

        return BeliefState(mu=mu, sigma2=sigma2, process_noise=process_noise)

    def predict(self, pred_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kalman predict: propagate belief through transition model.

        Returns (predicted_mean, predicted_variance).
        """
        pred_var = self.sigma2 + self.process_noise
        return pred_mean, pred_var

    def update(self, obs: np.ndarray, pred_mean: np.ndarray,
               pred_var: np.ndarray, obs_model: 'ObservationModel') -> float:
        """Kalman update: incorporate observation. Returns VFE.

        VFE = 0.5 * sum [eps_o^2/var_obs + eps_s^2/var_pred
                         + ln(var_obs * var_pred / var_post) - 1]
        """
        obs_var = obs_model.obs_var

        # Kalman gain (element-wise for diagonal covariance)
        K = pred_var / (pred_var + obs_var)

        # Prediction errors
        eps_o = obs - pred_mean                # observation prediction error
        eps_s = self.mu - pred_mean            # state prediction error

        # Posterior update
        self.mu = pred_mean + K * eps_o
        self.sigma2 = np.maximum((1.0 - K) * pred_var, 1e-4)

        # Physical validity floors
        self.mu[INV] = max(0.0, self.mu[INV])
        self.mu[ON_ORDER] = max(0.0, self.mu[ON_ORDER])
        self.mu[DAYS_UNTIL] = max(0.0, self.mu[DAYS_UNTIL])
        self.mu[DEMAND_MEAN] = max(0.01, self.mu[DEMAND_MEAN])
        self.mu[DEMAND_STD] = max(0.01, self.mu[DEMAND_STD])

        # Compute VFE
        vfe = 0.5 * np.sum(
            eps_o**2 / obs_var
            + eps_s**2 / pred_var
            + np.log(obs_var * pred_var / self.sigma2)
            - 1.0
        )
        return vfe

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.sigma2)

    def copy(self) -> 'BeliefState':
        return BeliefState(self.mu.copy(), self.sigma2.copy(),
                           self.process_noise.copy())


# ---------------------------------------------------------------------------
# Gaussian Preference Model — P(o) = N(mu_C, diag(sigma^2_C))
# ---------------------------------------------------------------------------

class GaussianPreferences:
    """Prior preferences P(o) as a proper Gaussian distribution.

    Enables closed-form KL divergence for the EFE risk term:
        Risk = KL[q(o|pi) || P(o)]

    The target is the same order-up-to level S = d*(L+R) + z*sigma*sqrt(L+R)
    as GroceryPreferences. The asymmetric stockout cost is captured by an
    additive penalty based on stockout probability P(inv < 0).

    References:
        Parr & Friston (2019), "Generalised Free Energy and Active Inference"
    """

    def __init__(self, params: SKUParams, service_z: float = 1.65):
        L = params.lead_time_days
        R = 1.0
        d = params.avg_daily_sales
        sigma_d = params.demand_std

        self.target_inventory = d * (L + R) + service_z * sigma_d * np.sqrt(L + R)
        self.safety_stock = params.safety_stock
        self.stockout_penalty = 50.0
        self.params = params

        # Convert precision to variance: var = target^2 / precision
        # Effective precision: harmonic mean weighted 70% toward below-target
        precision_below = 10.0
        precision_above = 1.0
        var_below = max(1.0, self.target_inventory ** 2 / precision_below)
        var_above = max(1.0, self.target_inventory ** 2 / precision_above)
        w_below = 0.7
        effective_var_inv = 1.0 / (w_below / var_below + (1 - w_below) / var_above)

        self.mu_C = np.array([
            self.target_inventory,
            0.0,
            0.0,
            d,
            sigma_d,
        ])

        self.sigma2_C = np.array([
            effective_var_inv,
            (d * L) ** 2 * 0.5 + 1.0,
            (L * 0.5) ** 2 + 1.0,
            (d * 0.5) ** 2 + 1.0,
            (sigma_d * 0.5) ** 2 + 1.0,
        ])

    def kl_divergence(self, mu_q: np.ndarray, sigma2_q: np.ndarray) -> float:
        """KL[N(mu_q, Sigma_q) || N(mu_C, Sigma_C)] — the Risk term.

        For diagonal Gaussians:
        KL = 0.5 * sum [sigma2_q/sigma2_C + (mu_C - mu_q)^2/sigma2_C
                        - 1 + ln(sigma2_C / sigma2_q)]
        """
        sigma2_q_safe = np.maximum(sigma2_q, 1e-8)
        kl = 0.5 * np.sum(
            sigma2_q_safe / self.sigma2_C
            + (self.mu_C - mu_q) ** 2 / self.sigma2_C
            - 1.0
            + np.log(self.sigma2_C / sigma2_q_safe)
        )
        return max(0.0, kl)

    def stockout_risk_penalty(self, mu_inv: float, sigma_inv: float) -> float:
        """Asymmetric penalty: stockout_penalty * P(inventory < 0).

        Uses logistic approximation of Phi(-z) to avoid scipy dependency.
        """
        if sigma_inv < 1e-6:
            return self.stockout_penalty if mu_inv <= 0 else 0.0
        z = mu_inv / sigma_inv
        if z > 6:
            return 0.0
        stockout_prob = 1.0 / (1.0 + np.exp(1.7 * z))
        return self.stockout_penalty * stockout_prob

    def compute_deviation(self, state: np.ndarray) -> float:
        """Backward-compatible deviation for plotting. Higher = worse."""
        delta = state - self.mu_C
        deviation = 0.5 * np.sum(delta ** 2 / self.sigma2_C)
        deviation += self.stockout_risk_penalty(state[INV], np.sqrt(self.sigma2_C[INV]) * 0.1)
        return deviation

    def log_preference(self, state: np.ndarray) -> float:
        return -self.compute_deviation(state)

    def in_bounds(self, state: np.ndarray, tolerance: float = 1.0) -> bool:
        return self.compute_deviation(state) < tolerance


# ---------------------------------------------------------------------------
# Active Inference Agent — Proper EFE with Belief Updating
# ---------------------------------------------------------------------------

class GroceryAIAgent:
    """Active inference agent with VFE perception and EFE action selection.

    Perception (each timestep):
        1. Predict: propagate belief through transition model
        2. Update: incorporate observation via precision-weighted Kalman update
        3. Compute VFE for monitoring

    Planning (action selection):
        G(pi, tau) = Risk + Ambiguity - InfoGain + StockoutPenalty
        Risk      = KL[q(o_tau|pi) || P(o)]
        Ambiguity = E_q[H[P(o|s)]] = 0.5 * sum ln(2*pi*e * sigma2_obs)
        InfoGain  = 0.5 * sum ln((sigma2_pred + sigma2_obs) / sigma2_obs)
        Stockout  = penalty * P(inv < 0)

    Action: P(a) ~ exp(-gamma * G(a))  (softmax policy selection)
    """

    def __init__(self, simulator: GrocerySimulator,
                 preferences: GaussianPreferences,
                 obs_model: ObservationModel,
                 planning_horizon: int | None = None,
                 n_rollouts: int = 50,
                 gamma: float = 4.0,
                 use_analytical: bool = True):
        self.simulator = simulator
        self.preferences = preferences
        self.obs_model = obs_model
        if planning_horizon is None:
            planning_horizon = max(7, simulator.params.lead_time_days + 3)
        self.planning_horizon = planning_horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.use_analytical = use_analytical

        # Precompute constant ambiguity term
        self._constant_ambiguity = 0.5 * np.sum(
            np.log(2 * np.pi * np.e * obs_model.obs_var)
        )

        self.belief = None

    def init_belief(self, initial_obs: np.ndarray):
        """Initialize belief from first observation."""
        self.belief = BeliefState.initialize(
            initial_obs, self.obs_model, self.simulator.params)

    def update_belief(self, action: int, obs: np.ndarray) -> float:
        """Perception step: predict from action, update from observation.

        Returns VFE for this step.
        """
        # Predict: use transition model to get expected next state
        pred_mean, pred_std = self.simulator.step_distribution_analytical(
            self.belief.mu, action)
        pred_mean_b, pred_var = self.belief.predict(pred_mean)
        # Update: incorporate observation
        vfe = self.belief.update(obs, pred_mean_b, pred_var, self.obs_model)
        return vfe

    def expected_free_energy(self, belief_mu: np.ndarray,
                             belief_sigma2: np.ndarray,
                             process_noise: np.ndarray,
                             policy: list[int]) -> float:
        """G(pi) = sum_tau [Risk + Ambiguity - InfoGain + Stockout]."""
        mu = belief_mu.copy()
        sigma2 = belief_sigma2.copy()
        G = 0.0

        for tau in range(len(policy)):
            action = policy[tau]
            pred_mean, _ = self.simulator.step_distribution_analytical(mu, action)

            # Propagate belief uncertainty
            pred_var = sigma2 + process_noise

            # Predicted observation variance: state uncertainty + obs noise
            obs_pred_var = pred_var + self.obs_model.obs_var

            # --- Risk: KL[q(o|pi) || P(o)] ---
            risk = self.preferences.kl_divergence(pred_mean, obs_pred_var)

            # --- Ambiguity: E_q[H[P(o|s)]] ---
            ambiguity = self._constant_ambiguity

            # --- Info gain: H[q(o|pi)] - E[H[P(o|s)]] ---
            # = 0.5 * sum ln(obs_pred_var / obs_var)
            info_gain = 0.5 * np.sum(np.log(
                np.maximum(obs_pred_var, 1e-8) / self.obs_model.obs_var))

            # --- Stockout penalty ---
            inv_std = np.sqrt(max(obs_pred_var[INV], 1e-8))
            stockout = self.preferences.stockout_risk_penalty(
                pred_mean[INV], inv_std)

            G += risk + ambiguity - info_gain + stockout

            # Advance
            mu = pred_mean
            sigma2 = pred_var

        return G

    def select_action(self, state: np.ndarray = None) -> tuple[int, list[float]]:
        """Evaluate hold vs order by minimizing EFE with softmax selection.

        The state argument is accepted for API compatibility with
        ReorderPointAgent but ignored (uses internal belief).
        """
        belief_mu = self.belief.mu.copy()
        belief_sigma2 = self.belief.sigma2.copy()
        process_noise = self.belief.process_noise

        G_per_action = []
        for a0 in range(2):
            G_total = 0.0
            for _ in range(self.n_rollouts):
                continuation = list(np.random.randint(0, 2,
                                                      self.planning_horizon - 1))
                policy = [a0] + continuation
                G_total += self.expected_free_energy(
                    belief_mu, belief_sigma2, process_noise, policy)
            G_per_action.append(G_total / self.n_rollouts)

        # Softmax: P(a) ~ exp(-gamma * G(a))
        G_arr = np.array(G_per_action)
        log_probs = -self.gamma * G_arr
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()

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
                   horizon: int = 90, seed: int | None = None,
                   obs_model: ObservationModel | None = None) -> dict:
    """Run a single SKU simulation for `horizon` days.

    For GroceryAIAgent: uses obs_model to generate noisy observations,
    agent updates belief state via VFE before selecting actions.
    For ReorderPointAgent: passes true state directly (backward compatible).

    Returns dict with trajectories, metrics, and optionally VFE/belief traces.
    """
    if seed is not None:
        simulator.set_sim_seed(seed)

    states = [initial_state.copy()]
    actions = []
    demands = []
    stockout_days = 0
    lost_sales = 0.0
    vfe_trace = []
    belief_trace = []

    current = initial_state.copy()
    is_ai = isinstance(agent, GroceryAIAgent) and obs_model is not None

    # Initialize belief from first observation
    if is_ai:
        obs_model.obs_rng = simulator.obs_rng
        first_obs = obs_model.observe(current)
        agent.init_belief(first_obs)
        belief_trace.append(agent.belief.mu.copy())

    for t in range(horizon):
        # Action selection (AI uses belief, baseline uses true state)
        action, _ = agent.select_action(current)
        actions.append(action)

        prev_inv = current[INV]
        current = simulator.step(current, action)

        # Perception: observe + belief update (AI agent only)
        if is_ai:
            obs = obs_model.observe(current)
            vfe = agent.update_belief(action, obs)
            vfe_trace.append(vfe)
            belief_trace.append(agent.belief.mu.copy())

        effective_demand = max(0.0, prev_inv - current[INV])
        demands.append(effective_demand)

        if current[INV] <= 0:
            stockout_days += 1
            lost_sales += max(0.0, current[DEMAND_MEAN] - prev_inv)

        states.append(current.copy())

    states = np.array(states)
    result = {
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

    if vfe_trace:
        result['vfe_trace'] = np.array(vfe_trace)
        result['belief_trace'] = np.array(belief_trace)
        result['avg_vfe'] = np.mean(vfe_trace)

    return result
