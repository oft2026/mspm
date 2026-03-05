import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SAMPortfolioEnv(gym.Env):
    """Multi-asset portfolio environment for SAM.

    State: profound state V_t^+ of shape (f, m*, n)

    Action: allocation weights a_t of shape (m*,), summing to 1.0
            Index 0 = cash, indices 1..m = assets

    Reward (Eq. 8 from paper):
        r_t* = ln(a_t . y_t - beta * sum|a_i,t - w_i,t| - phi * sigma_t^2)
        where:
            y_t = price relative vector (cash=1, asset_i = p_i,t / p_i,t-1)
            w_i,t = weights after price fluctuation (before rebalancing)
            sigma_t^2 = portfolio variance over trailing window
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        profound_states: np.ndarray,
        price_relatives: np.ndarray,
        commission: float = 0.0025,
        risk_scaling: float = 0.001,
        initial_value: float = 10000.0,
        window: int = 50,
    ):
        """
        Args:
            profound_states: shape (T, f, m*, n) - pre-computed states
            price_relatives: shape (T, m*) - daily price relatives
                             Index 0 = cash (=1.0), rest = assets
        """
        super().__init__()

        self.profound_states = profound_states.astype(np.float32)
        self.price_relatives = price_relatives.astype(np.float32)
        self.commission = commission
        self.risk_scaling = risk_scaling
        self.initial_value = initial_value
        self.var_window = window

        self.num_steps = len(profound_states)
        self.num_assets_cash = price_relatives.shape[1]

        # Gymnasium spaces
        f, m_star, n = profound_states.shape[1:]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(f, m_star, n),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets_cash,),
            dtype=np.float32,
        )

        # State tracking (initialized in reset)
        self.current_step = 0
        self.portfolio_value = initial_value
        self.current_weights = np.zeros(self.num_assets_cash, dtype=np.float32)
        self.current_weights[0] = 1.0
        self.return_history: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        """Reset to start of the dataset."""
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.current_weights = np.zeros(self.num_assets_cash, dtype=np.float32)
        self.current_weights[0] = 1.0
        self.return_history = []
        return self.profound_states[0].copy(), {}

    def step(self, action: np.ndarray):
        """Execute rebalancing action.

        Args:
            action: allocation weights (m*,) summing to 1.0

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.clip(action, 1e-6, 1.0).astype(np.float32)
        action = action / action.sum()  # Normalize to sum = 1

        y_t = self.price_relatives[self.current_step]

        # Portfolio return: a_t . y_t
        port_return = float(np.dot(action, y_t))

        # Weights after price fluctuation (before rebalancing)
        # w_i,t = (y_i,t * a_i,t) / sum(y * a)
        weighted = y_t * action
        denom = weighted.sum()
        if denom > 1e-10:
            weights_after = weighted / denom
        else:
            # Fallback: equal weights (portfolio effectively wiped out)
            weights_after = np.full_like(action, 1.0 / len(action))

        # Transaction cost: beta * sum|a_i,t - w_i,t-1|
        turnover = float(np.sum(np.abs(action - self.current_weights)))
        transaction_cost = self.commission * turnover

        # Portfolio variance over trailing window
        self.return_history.append(y_t.copy())
        variance = self._compute_portfolio_variance()

        # Reward: ln(port_return - transaction_cost - phi * variance)
        inner = port_return - transaction_cost - self.risk_scaling * variance
        if inner > 1e-10:
            reward = float(np.log(inner))
        else:
            # Large penalty proportional to how negative inner is
            reward = float(-10.0 + min(inner, 0.0) * 100.0)

        # Update portfolio value
        self.portfolio_value *= port_return - transaction_cost

        # Update weights
        self.current_weights = weights_after

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.num_steps

        next_state = (
            self.profound_states[self.current_step].copy()
            if not terminated
            else self.profound_states[-1].copy()
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "port_return": port_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "variance": variance,
            "weights": weights_after.copy(),
        }

        return next_state, reward, terminated, False, info

    def _compute_portfolio_variance(self) -> float:
        """Compute portfolio variance over trailing window.

        sigma_t^2 = (1/n) * sum_{t-n+1}^{t} sum_{i=1}^{m*} (y_i,t - mean_y_i)^2
        """
        if len(self.return_history) < 2:
            return 0.0

        window = self.return_history[-self.var_window :]
        returns_arr = np.array(window)
        var_per_asset = np.var(returns_arr, axis=0)
        return float(np.sum(var_per_asset))
