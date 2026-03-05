import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EAMTradingEnv(gym.Env):
    """Single-asset trading environment for EAM.

    State: rolling window of normalized features, shape (num_features, window_length).

    Actions: 0=buy (open long), 1=close (exit position), 2=skip (hold)

    Reward (Eq. 3 from paper):
        r_t = 100 * sum_{i=entry}^{t} [close_i/close_{i-1} - 1 - beta]
              if position is open after action
        r_t = 0 otherwise
        (beta is charged per day of holding)

    Constraints:
        - No short selling
        - Must close before opening new position
        - Invalid actions are treated as skip
    """

    metadata = {"render_modes": []}

    BUY = 0
    CLOSE = 1
    SKIP = 2

    def __init__(
        self,
        states: np.ndarray,
        close_prices: np.ndarray,
        commission: float = 0.0025,
    ):
        """
        Args:
            states: pre-computed rolling window states, shape (T, f, n).
            close_prices: adjusted close prices aligned with states,
                          shape (T,). states[i] uses prices up to close_prices[i].
        """
        super().__init__()

        self.states = states.astype(np.float32)
        self.close_prices = close_prices
        self.commission = commission
        self.num_steps = len(states)

        # Gymnasium spaces
        num_features, window_length = states.shape[1], states.shape[2]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features, window_length),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # buy, close, skip

        self.current_step = 0
        self.position_open = False
        self.entry_step = -1

    def reset(self, *, seed=None, options=None):
        """Reset environment to the beginning."""
        super().reset(seed=seed)
        self.current_step = 0
        self.position_open = False
        self.entry_step = -1
        return self.states[0].copy(), {}

    def step(self, action: int):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        # Determine effective action (with constraints) WITHOUT updating state
        effective_action = self._get_effective_action(action)

        # Compute reward BEFORE updating position state,
        # so CLOSE actions can still access entry_step.
        reward = self._compute_reward(effective_action)

        # Now apply the action to update position state
        self._apply_action(effective_action)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1

        next_state = (
            self.states[self.current_step].copy()
            if not terminated
            else self.states[-1].copy()
        )

        info = {
            "position_open": self.position_open,
            "effective_action": effective_action,
            "close_price": self.close_prices[self.current_step],
        }

        return next_state, reward, terminated, False, info

    def _get_effective_action(self, action: int) -> int:
        """Determine effective action after applying constraints (no state change)."""
        if action == self.BUY:
            if not self.position_open:
                return self.BUY
            return self.SKIP  # Already in position
        elif action == self.CLOSE:
            if self.position_open:
                return self.CLOSE
            return self.SKIP  # No position to close
        else:
            return self.SKIP

    def _apply_action(self, effective_action: int):
        """Update position state after reward has been computed."""
        if effective_action == self.BUY:
            self.position_open = True
            self.entry_step = self.current_step
        elif effective_action == self.CLOSE:
            self.position_open = False
            self.entry_step = -1

    def _compute_reward(self, effective_action: int) -> float:
        """Compute reward per Equation 3 of the paper."""
        if effective_action == self.CLOSE:
            return self._cumulative_return_reward()

        if self.position_open:
            return self._cumulative_return_reward()

        return 0.0

    def _cumulative_return_reward(self) -> float:
        """Compute 100 * sum_{i=entry}^{t} (daily_ret_i - beta) per Eq. 3.

        Commission beta is charged per day of holding, not once per trade.
        """
        if self.entry_step < 0 or self.entry_step >= self.current_step:
            return 0.0

        cum_return = 0.0
        num_days = self.current_step - self.entry_step
        for i in range(self.entry_step + 1, self.current_step + 1):
            daily_ret = (
                self.close_prices[i] / self.close_prices[i - 1] - 1.0
            )
            cum_return += daily_ret

        return 100.0 * (cum_return - num_days * self.commission)
