import torch
import torch.nn as nn


class SAMValueNetwork(nn.Module):
    """PPO critic network: estimates state value V(s).

    Input: profound state (batch, f, m*, n)
    Output: scalar value (batch, 1)
    """

    def __init__(
        self,
        num_features: int,
        num_assets_cash: int,
        window: int = 50,
        conv_channels: list[int] = None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_features, conv_channels[0], kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((num_assets_cash, 1)),
        )

        flat_size = conv_channels[-1] * num_assets_cash

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value estimate (batch, 1)."""
        out = self.conv_layers(state)
        out = out.flatten(1)
        return self.fc(out)
