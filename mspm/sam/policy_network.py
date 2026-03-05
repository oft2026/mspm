import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SAMPolicyNetwork(nn.Module):
    """PPO actor network for portfolio allocation.

    Input: profound state V_t^+ of shape (batch, f, m*, n)
           where f=features, m*=assets+cash, n=window length

    Architecture:
    - Conv2d layers processing the (f, m*, n) tensor
    - ReLU after each conv except last
    - Flatten -> FC -> m* means for normal distributions
    - Softmax -> allocation weights summing to 1.0

    Output: Normal distribution parameters for each asset allocation.
    """

    def __init__(
        self,
        num_features: int,
        num_assets_cash: int,
        window: int = 50,
        conv_channels: list[int] = None,
        hidden_dim: int = 256,
        init_log_std: float = -1.0,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        self.num_assets_cash = num_assets_cash

        # Conv2d layers: input channels = num_features (f)
        # Spatial dims: height = m*, width = n
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_features, conv_channels[0], kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((num_assets_cash, 1)),
        )

        # After pool: (batch, conv_channels[-1], m*, 1)
        flat_size = conv_channels[-1] * num_assets_cash

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_assets_cash),
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.full((num_assets_cash,), init_log_std)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_means, log_std)."""
        out = self.conv_layers(state)
        out = out.flatten(1)
        means = self.fc(out)
        return means, self.log_std.expand_as(means)

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample allocation weights from Normal -> Softmax.

        Returns: (action, log_prob, entropy)
        - action: (batch, m*) allocation weights summing to 1
        - log_prob: (batch,) log probability of the sampled action
        - entropy: (batch,) distribution entropy
        """
        means, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(means, std)

        if deterministic:
            raw_action = means
        else:
            raw_action = dist.rsample()

        # Softmax to get allocation weights summing to 1
        action = F.softmax(raw_action, dim=-1)

        # Log probability (of the raw action before softmax)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_action(
        self, state: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for a given raw action (before softmax).

        Used during PPO update to compute importance sampling ratio.
        """
        means, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(means, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
