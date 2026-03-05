import numpy as np
import torch


class RolloutBuffer:
    """On-policy rollout storage for PPO with GAE-lambda advantage computation."""

    def __init__(
        self,
        rollout_length: int,
        state_shape: tuple,
        num_assets_cash: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_assets_cash = num_assets_cash

        # Pre-allocate storage
        self.states = np.zeros((rollout_length, *state_shape), dtype=np.float32)
        self.actions = np.zeros(
            (rollout_length, num_assets_cash), dtype=np.float32
        )
        self.raw_actions = np.zeros(
            (rollout_length, num_assets_cash), dtype=np.float32
        )
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.raw_actions[self.pos] = raw_action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos == self.rollout_length:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float):
        """Compute GAE-lambda advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self, mini_batch_size: int, device: torch.device = torch.device("cpu")
    ):
        """Yield shuffled mini-batches as tensors."""
        indices = np.arange(self.rollout_length)
        np.random.shuffle(indices)

        for start in range(0, self.rollout_length, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]

            yield (
                torch.FloatTensor(self.states[batch_idx]).to(device),
                torch.FloatTensor(self.actions[batch_idx]).to(device),
                torch.FloatTensor(self.raw_actions[batch_idx]).to(device),
                torch.FloatTensor(self.returns[batch_idx]).to(device),
                torch.FloatTensor(self.advantages[batch_idx]).to(device),
                torch.FloatTensor(self.log_probs[batch_idx]).to(device),
            )

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False
