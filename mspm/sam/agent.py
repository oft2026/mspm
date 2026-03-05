from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mspm.sam.policy_network import SAMPolicyNetwork
from mspm.sam.value_network import SAMValueNetwork
from mspm.sam.rollout_buffer import RolloutBuffer


class SAMAgent:
    """PPO agent for portfolio allocation.

    PPO with clipped surrogate objective:
        L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    Total loss = -L_clip + c1 * L_value - c2 * H[pi]
    """

    def __init__(
        self,
        num_features: int,
        num_assets: int,
        window: int = 50,
        conv_channels: list[int] = None,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        rollout_length: int = 256,
        device: str = "cpu",
    ):
        if conv_channels is None:
            conv_channels = [32, 64]

        self.num_assets_cash = num_assets + 1  # +1 for cash
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_length = rollout_length
        self.device = torch.device(device)

        # Networks
        self.policy = SAMPolicyNetwork(
            num_features=num_features,
            num_assets_cash=self.num_assets_cash,
            window=window,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.value_net = SAMValueNetwork(
            num_features=num_features,
            num_assets_cash=self.num_assets_cash,
            window=window,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
        )

        # State shape for rollout buffer: (f, m*, n)
        state_shape = (num_features, self.num_assets_cash, window)
        self.rollout_buffer = RolloutBuffer(
            rollout_length=rollout_length,
            state_shape=state_shape,
            num_assets_cash=self.num_assets_cash,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Sample action from policy.

        Returns: (action, raw_action, value, log_prob)
        - action: softmax allocation weights
        - raw_action: pre-softmax samples (for PPO update)
        """
        with torch.no_grad():
            state_t = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )

            # Get action from policy
            means, log_std = self.policy(state_t)
            std = log_std.exp()

            if deterministic:
                raw_action = means
            else:
                dist = torch.distributions.Normal(means, std)
                raw_action = dist.rsample()

            action = F.softmax(raw_action, dim=-1)

            # Log prob of raw action
            dist = torch.distributions.Normal(means, std)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            # Value estimate
            value = self.value_net(state_t)

        return (
            action.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            value.item(),
            log_prob.item(),
        )

    def update(self) -> dict:
        """Run PPO update over multiple epochs on collected rollout."""
        # Compute advantages
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.rollout_buffer.get_batches(
                self.mini_batch_size, self.device
            ):
                (
                    states,
                    actions,
                    raw_actions,
                    returns,
                    advantages,
                    old_log_probs,
                ) = batch

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Evaluate current policy on old actions
                new_log_probs, entropy = self.policy.evaluate_action(
                    states, raw_actions
                )

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.value_net(states).squeeze(-1)
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def get_allocation(self, state: np.ndarray) -> np.ndarray:
        """Deterministic action for evaluation."""
        action, _, _, _ = self.select_action(state, deterministic=True)
        return action

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
