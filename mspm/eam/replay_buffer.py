import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class NStepReplayBuffer:
    """Experience replay buffer with n-step return computation.

    For 2-step Bellman unrolling:
        R_t^(2) = r_t + gamma * r_{t+1}
        Target: R_t^(2) + gamma^2 * Q_target(s_{t+2}, argmax Q_online(s_{t+2},.))
    """

    def __init__(
        self, capacity: int = 100000, n_step: int = 2, gamma: float = 0.99
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.n_step_buffer: deque = deque(maxlen=n_step)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition, computing n-step return when buffer is full."""
        self.n_step_buffer.append(
            Transition(state, action, reward, next_state, done)
        )

        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_return = 0.0
            last_trans = self.n_step_buffer[-1]
            for i, trans in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma**i) * trans.reward
                if trans.done:
                    last_trans = trans
                    break

            self.buffer.append(
                Transition(
                    self.n_step_buffer[0].state,
                    self.n_step_buffer[0].action,
                    n_step_return,
                    last_trans.next_state,
                    last_trans.done,
                )
            )
            # Remove only the processed (oldest) transition;
            # remaining ones are flushed by the `if done:` block below.
            self.n_step_buffer.popleft()

        # Flush remaining transitions on episode end
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_return = 0.0
                for i, trans in enumerate(self.n_step_buffer):
                    n_step_return += (self.gamma**i) * trans.reward
                last = self.n_step_buffer[-1]
                self.buffer.append(
                    Transition(
                        self.n_step_buffer[0].state,
                        self.n_step_buffer[0].action,
                        n_step_return,
                        last.next_state,
                        True,
                    )
                )
                self.n_step_buffer.popleft()

    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, ...]:
        """Sample a random batch and return as tensors."""
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(device)
        actions = torch.LongTensor([t.action for t in batch]).to(device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in batch])
        ).to(device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
