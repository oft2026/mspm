"""
EAM Training Script.

1. Download OHLCV data for all unique tickers (2009-2015)
2. Train foundational EAM on AAPL
3. Transfer weights to EAMs for each other ticker and fine-tune
4. Save all EAM models

Usage:
    python scripts/train_eam.py [--config configs/default.yaml]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mspm.data.downloader import OHLCVDownloader
from mspm.data.preprocessor import FeaturePreprocessor
from mspm.eam.agent import EAMAgent
from mspm.eam.environment import EAMTradingEnv
from mspm.utils.config import load_config
from mspm.utils.device import get_device, set_seed


def get_all_tickers(config) -> list[str]:
    """Collect all unique tickers from all portfolios."""
    tickers = set()
    for portfolio in config.portfolios.values():
        for t in portfolio.tickers:
            tickers.add(t)
    return sorted(tickers)


def train_eam_on_ticker(
    ticker: str,
    config,
    downloader: OHLCVDownloader,
    preprocessor: FeaturePreprocessor,
    foundational_path: str = None,
    num_episodes: int = None,
) -> EAMAgent:
    """Train an EAM agent for a single ticker."""
    eam_cfg = config.eam

    # Download data
    df = downloader.download(
        ticker,
        start=config.data.eam_train_start,
        end=config.data.eam_train_end,
    )

    # Build states and get close prices
    states = preprocessor.build_eam_states(df, window=config.data.rolling_window)
    close_prices = preprocessor.get_close_prices(df)
    # Align close prices with rolling windows
    # states[i] uses data from day i to day i+window-1
    # The "current" close price for state i is close_prices[i + window - 1]
    window = config.data.rolling_window
    aligned_close = close_prices[window - 1 :]
    assert len(aligned_close) == len(states)

    # Create agent
    device = get_device()
    agent = EAMAgent(
        num_features=7,  # 5 OHLCV + 2 sentiment placeholder
        num_actions=eam_cfg.num_actions,
        hidden_dim=eam_cfg.hidden_dim,
        resnet_channels=eam_cfg.resnet_channels,
        num_residual_blocks=eam_cfg.num_residual_blocks,
        resnet_kernel_size=eam_cfg.resnet_kernel_size,
        learning_rate=eam_cfg.learning_rate,
        gamma=eam_cfg.gamma,
        n_step=eam_cfg.n_step,
        epsilon_start=eam_cfg.epsilon_start,
        epsilon_end=eam_cfg.epsilon_end,
        epsilon_decay_steps=eam_cfg.epsilon_decay_steps,
        replay_buffer_size=eam_cfg.replay_buffer_size,
        batch_size=eam_cfg.batch_size,
        target_update_freq=eam_cfg.target_update_freq,
        device=device,
    )

    # Transfer learning if not foundational
    if foundational_path is not None:
        print(f"  Transferring weights from foundational EAM...")
        agent.load_backbone_from(foundational_path)

    episodes = num_episodes or eam_cfg.num_episodes
    env = EAMTradingEnv(states, aligned_close, commission=eam_cfg.commission)

    best_reward = -float("inf")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update()
            agent.decay_epsilon()

            if agent.total_steps % eam_cfg.target_update_freq == 0:
                agent.update_target_network()

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(
                f"  [{ticker}] Episode {episode+1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

        if total_reward > best_reward:
            best_reward = total_reward

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train EAM agents")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    downloader = OHLCVDownloader(cache_dir="data/raw")
    preprocessor = FeaturePreprocessor()

    all_tickers = get_all_tickers(config)
    foundational_ticker = config.eam.foundational_ticker

    print(f"All tickers: {all_tickers}")
    print(f"Foundational ticker: {foundational_ticker}")

    # Ensure checkpoint directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Step 1: Train foundational EAM on AAPL
    print(f"\n{'='*60}")
    print(f"Training foundational EAM on {foundational_ticker}...")
    print(f"{'='*60}")

    foundational_agent = train_eam_on_ticker(
        foundational_ticker, config, downloader, preprocessor
    )
    foundational_path = f"checkpoints/eam_{foundational_ticker}.pt"
    foundational_agent.save(foundational_path)
    print(f"Saved foundational EAM to {foundational_path}")

    # Step 2: Transfer and fine-tune for other tickers
    other_tickers = [t for t in all_tickers if t != foundational_ticker]
    for ticker in other_tickers:
        print(f"\n{'='*60}")
        print(f"Training transferred EAM on {ticker}...")
        print(f"{'='*60}")

        agent = train_eam_on_ticker(
            ticker,
            config,
            downloader,
            preprocessor,
            foundational_path=foundational_path,
            num_episodes=config.eam.transfer_finetune_episodes,
        )
        save_path = f"checkpoints/eam_{ticker}.pt"
        agent.save(save_path)
        print(f"Saved EAM for {ticker} to {save_path}")

    print("\nAll EAM agents trained successfully!")


if __name__ == "__main__":
    main()
