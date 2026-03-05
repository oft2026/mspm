import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Single residual block with 1D convolutions.

    Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+ skip) -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ResNet1D(nn.Module):
    """1D ResNet feature extractor for time-series data.

    Input shape:  (batch, in_features, seq_len)  e.g. (B, 7, 50)
    Output shape: (batch, feature_dim)            e.g. (B, 256)
    """

    def __init__(
        self,
        in_features: int,
        channels: list[int] = None,
        num_blocks: int = 2,
        kernel_size: int = 3,
        feature_dim: int = 256,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(in_features, channels[0], kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual block groups
        layers = []
        in_ch = channels[0]
        for out_ch in channels:
            for b in range(num_blocks):
                stride = 2 if b == 0 and out_ch != channels[0] else 1
                layers.append(
                    ResidualBlock1D(in_ch, out_ch, kernel_size, stride=stride)
                )
                in_ch = out_ch
        self.res_blocks = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial(x)
        out = self.res_blocks(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc(out)
        return out
