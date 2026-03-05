from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    rolling_window: int = 50
    features: list[str] = field(
        default_factory=lambda: ["adj_close", "open", "high", "low", "volume"]
    )
    eam_train_start: str = "2009-01-01"
    eam_train_end: str = "2015-12-31"
    eam_predict_start: str = "2016-01-01"
    eam_predict_end: str = "2020-12-31"
    sam_train_start: str = "2016-01-01"
    sam_train_end: str = "2018-12-31"
    sam_val_start: str = "2019-01-01"
    sam_val_end: str = "2019-12-31"
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2020-12-31"


@dataclass
class EAMConfig:
    resnet_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    resnet_kernel_size: int = 3
    num_residual_blocks: int = 2
    num_actions: int = 3
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step: int = 2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50000
    replay_buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    num_episodes: int = 200
    commission: float = 0.0025
    foundational_ticker: str = "AAPL"
    transfer_finetune_episodes: int = 50


@dataclass
class SAMConfig:
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    conv_kernel_size: list[int] = field(default_factory=lambda: [1, 3])
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    rollout_length: int = 256
    num_updates: int = 500
    commission: float = 0.0025
    risk_scaling: float = 0.001


@dataclass
class PortfolioConfig:
    tickers: list[str] = field(default_factory=list)


@dataclass
class MSPMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    eam: EAMConfig = field(default_factory=EAMConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    portfolios: dict[str, PortfolioConfig] = field(default_factory=dict)
    initial_capital: float = 10000.0
    seed: int = 42


def load_config(path: str = "configs/default.yaml") -> MSPMConfig:
    schema = OmegaConf.structured(MSPMConfig)
    file_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, file_cfg)
    # Prevent silent typos: reject unknown keys in top-level and sub-configs
    OmegaConf.set_struct(merged, True)
    return OmegaConf.to_object(merged)
