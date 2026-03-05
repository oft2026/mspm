# MSPM: Modularized and Scalable Multi-Agent RL for Portfolio Management

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An unofficial PyTorch reproduction of the paper:

> Zhenhan Huang, Fumihide Tanaka. **MSPM: A modularized and scalable multi-agent reinforcement learning-based system for financial portfolio management.** *PLOS ONE*, 17(2): e0263689, 2022.
> [[DOI]](https://doi.org/10.1371/journal.pone.0263689)

## Architecture Overview

MSPM is a two-stage multi-agent framework for portfolio management:

1. **EAM (Evolving Agent Module)**: Each stock is assigned an independent EAM agent that learns single-asset trading signals. EAMs use a **Dueling Double DQN** with a **1D ResNet** backbone and **2-step Bellman** updates. A foundational agent is first trained on AAPL, then transferred to other tickers via fine-tuning.

2. **SAM (Strategic Agent Module)**: A portfolio-level agent that takes the EAM signals as input and learns optimal portfolio allocation across multiple assets. SAM uses **PPO** (Proximal Policy Optimization) with Conv2D policy and value networks.

## Project Structure

```
mspm/
├── configs/
│   └── default.yaml          # All hyperparameters and data settings
├── mspm/
│   ├── data/
│   │   ├── downloader.py     # OHLCV data download via yfinance
│   │   └── preprocessor.py   # Feature engineering & normalization
│   ├── eam/
│   │   ├── agent.py          # EAM agent (Dueling Double DQN)
│   │   ├── dueling_dqn.py    # Dueling DQN network
│   │   ├── environment.py    # Single-asset trading environment
│   │   ├── replay_buffer.py  # N-step replay buffer
│   │   └── resnet1d.py       # 1D ResNet backbone
│   ├── sam/
│   │   ├── agent.py          # SAM agent (PPO)
│   │   ├── environment.py    # Portfolio environment
│   │   ├── policy_network.py # Conv2D policy network
│   │   ├── value_network.py  # Conv2D value network
│   │   └── rollout_buffer.py # PPO rollout buffer
│   └── utils/
│       ├── config.py         # Configuration dataclasses
│       ├── device.py         # Device selection & seed utilities
│       ├── metrics.py        # Performance metrics (Sortino, MDD, etc.)
│       └── portfolio.py      # CRP & BAH baselines
├── scripts/
│   ├── train_eam.py          # Step 1: Train EAM agents
│   ├── generate_signals.py   # Step 2: Generate trading signals
│   ├── train_sam.py          # Step 3: Train SAM portfolio agent
│   └── backtest.py           # Step 4: Backtest & evaluate
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone https://github.com/oft2026/mspm.git
cd mspm
pip install -e .
```

## Quick Start

The full pipeline consists of 4 steps:

```bash
# Step 1: Train EAM agents (foundational on AAPL + transfer to others)
python scripts/train_eam.py

# Step 2: Generate trading signals from trained EAMs
python scripts/generate_signals.py

# Step 3: Train SAM portfolio agents
python scripts/train_sam.py

# Step 4: Backtest and evaluate against CRP/BAH baselines
python scripts/backtest.py
```

## Configuration

All hyperparameters are centralized in [`configs/default.yaml`](configs/default.yaml). Key parameters:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `data` | `rolling_window` | 50 | Lookback window for feature construction |
| `eam` | `num_episodes` | 200 | Training episodes for foundational EAM |
| `eam` | `transfer_finetune_episodes` | 50 | Fine-tuning episodes for transferred EAMs |
| `eam` | `n_step` | 2 | N-step Bellman update |
| `eam` | `commission` | 0.0025 | Transaction cost rate |
| `sam` | `num_updates` | 500 | PPO update iterations |
| `sam` | `clip_epsilon` | 0.2 | PPO clipping parameter |
| `portfolios` | `portfolio_a..d` | various | Stock ticker combinations |

## Differences from the Paper

- **Sentiment data**: The paper uses sentiment features from financial news. This reproduction fills the sentiment channel with zeros, as the original sentiment dataset is not publicly available.
- **Data source**: OHLCV data is fetched from Yahoo Finance via `yfinance`, which may differ slightly from the data used in the original paper.
- **Evaluation period**: Default config uses 2020 as the backtest period. Results may vary from the paper due to data source differences.
- **Hyperparameters**: The paper only provides a few key parameters (commission rate, rolling window, risk scaling, n-step, initial capital). Most training hyperparameters (learning rates, network architecture, epsilon schedule, PPO epochs, etc.) are not disclosed in the paper or its S1 Appendix. This reproduction uses common default values for DQN and PPO.

## Citation

If you find this code useful, please cite the original paper:

```bibtex
@article{huang2022mspm,
  title={{MSPM}: A modularized and scalable multi-agent reinforcement learning-based system for financial portfolio management},
  author={Huang, Zhenhan and Tanaka, Fumihide},
  journal={PLOS ONE},
  volume={17},
  number={2},
  pages={e0263689},
  year={2022},
  doi={10.1371/journal.pone.0263689}
}
```

## Disclaimer

**Fair warning: the entire codebase is the result of agentic coding, a.k.a. vibe coding — so use at your own risk, and definitely don't bet your retirement fund on it.**

This software is provided for **educational and research purposes only**. It is not financial advice. The authors are not responsible for any financial losses incurred from using this system.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
