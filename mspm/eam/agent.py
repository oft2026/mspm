import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mspm.eam.dueling_dqn import DuelingDQN
from mspm.eam.replay_buffer import NStepReplayBuffer


class EAMAgent:
    """Dueling Double DQN agent with n-step Bellman for EAM.

    Key features:
    - Double DQN: online net selects actions, target net evaluates
    - Dueling architecture: separate value and advantage streams
    - 2-step Bellman: uses 2-step returns for reduced variance
    - Epsilon-greedy exploration with linear decay
    """

    def __init__(
        self,
        num_features: int = 7,
        num_actions: int = 3,
        hidden_dim: int = 256,
        resnet_channels: list[int] = None,
        num_residual_blocks: int = 2,
        resnet_kernel_size: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        replay_buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ):
        if resnet_channels is None:
            resnet_channels = [64, 128, 256]

        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        self.online_net = DuelingDQN(
            in_features=num_features,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            resnet_channels=resnet_channels,
            num_residual_blocks=num_residual_blocks,
            resnet_kernel_size=resnet_kernel_size,
        ).to(self.device)

        self.target_net = DuelingDQN(
            in_features=num_features,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            resnet_channels=resnet_channels,
            num_residual_blocks=num_residual_blocks,
            resnet_kernel_size=resnet_kernel_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )
        self.replay_buffer = NStepReplayBuffer(
            capacity=replay_buffer_size, n_step=n_step, gamma=gamma
        )

        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_t = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(self) -> float | None:
        """Sample batch, compute Double DQN loss with n-step returns."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size, self.device)
        )

        # Current Q values
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Online net selects action
            next_actions = self.online_net(next_states).argmax(dim=1)
            # Target net evaluates
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + (self.gamma**self.n_step) * next_q * (
                1.0 - dones
            )

        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Linear epsilon decay."""
        self.total_steps += 1
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def update_target_network(self):
        """Hard copy online -> target."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def generate_signals(self, states: np.ndarray) -> np.ndarray:
        """Run greedy policy to produce trading signals for SAM.

        Simulates the environment logic to ensure valid action sequences.
        Returns array of actions {0=buy, 1=close, 2=skip}.
        """
        self.online_net.eval()
        signals = np.zeros(len(states), dtype=np.int64)
        position_open = False

        with torch.no_grad():
            for i in range(len(states)):
                state_t = (
                    torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                )
                q_values = self.online_net(state_t)
                action = int(q_values.argmax(dim=1).item())

                # Apply constraints
                if action == 0 and position_open:
                    action = 2  # Can't buy again
                elif action == 1 and not position_open:
                    action = 2  # Nothing to close

                # Update position tracking
                if action == 0:
                    position_open = True
                elif action == 1:
                    position_open = False

                signals[i] = action

        self.online_net.train()
        return signals

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]

    def load_backbone_from(self, foundational_path: str):
        """Transfer learning: load only the ResNet backbone weights.

        The dueling head (value/advantage streams) is kept randomly
        initialized so the agent can adapt to the new asset.
        """
        checkpoint = torch.load(
            foundational_path, map_location=self.device, weights_only=True
        )
        # Extract only backbone weights
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in checkpoint["online_net"].items()
            if k.startswith("backbone.")
        }
        self.online_net.backbone.load_state_dict(backbone_state)
        self.target_net.backbone.load_state_dict(backbone_state)
