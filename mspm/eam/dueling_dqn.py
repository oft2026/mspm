import torch
import torch.nn as nn

from mspm.eam.resnet1d import ResNet1D


class DuelingDQN(nn.Module):
    """Dueling DQN combining ResNet1D backbone with value/advantage streams.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))

    Input shape:  (batch, num_features, window_length)  e.g. (B, 7, 50)
    Output shape: (batch, num_actions)                   e.g. (B, 3)
    """

    def __init__(
        self,
        in_features: int,
        num_actions: int = 3,
        hidden_dim: int = 256,
        resnet_channels: list[int] = None,
        num_residual_blocks: int = 2,
        resnet_kernel_size: int = 3,
    ):
        super().__init__()
        if resnet_channels is None:
            resnet_channels = [64, 128, 256]

        self.backbone = ResNet1D(
            in_features=in_features,
            channels=resnet_channels,
            num_blocks=num_residual_blocks,
            kernel_size=resnet_kernel_size,
            feature_dim=hidden_dim,
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, num_actions)
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
