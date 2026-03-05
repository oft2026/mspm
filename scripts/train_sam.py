"""
SAM Training Script.

1. Load profound states and price relatives
2. Split into train/val/test periods
3. Train PPO agent on training data
4. Validate and save best model
5. Save SAM model for each portfolio

Usage:
    python scripts/train_sam.py [--config configs/default.yaml] [--portfolio portfolio_a]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mspm.sam.agent import SAMAgent
from mspm.sam.environment import SAMPortfolioEnv
from mspm.utils.config import load_config
from mspm.utils.device import get_device, set_seed
from mspm.utils.metrics import compute_all_metrics


def evaluate_agent(agent: SAMAgent, env: SAMPortfolioEnv) -> dict:
    """Run deterministic evaluation on environment."""
    state, _ = env.reset()
    portfolio_values = [env.portfolio_value]

    while True:
        action = agent.get_allocation(state)
        state, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        if terminated or truncated:
            break

    return compute_all_metrics(np.array(portfolio_values))


def train_portfolio(portfolio_name: str, config):
    """Train SAM for a single portfolio."""
    sam_cfg = config.sam
    device = get_device()

    # Load pre-computed data
    profound_states = np.load(
        f"data/processed/{portfolio_name}_profound.npy"
    )
    price_relatives = np.load(
        f"data/processed/{portfolio_name}_price_rel.npy"
    )
    splits = np.load(
        f"data/processed/{portfolio_name}_splits.npy", allow_pickle=True
    ).item()

    train_end = splits["train_end"]
    val_end = splits["val_end"]

    assert profound_states.shape[0] == price_relatives.shape[0], (
        f"Data misaligned: profound_states has {profound_states.shape[0]} steps "
        f"but price_relatives has {price_relatives.shape[0]}. "
        f"Re-run generate_signals.py to fix alignment."
    )

    print(f"Profound states shape: {profound_states.shape}")
    print(f"Price relatives shape: {price_relatives.shape}")
    print(f"Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-end")

    # Split data
    # Align: profound_states[t] -> price_relatives[t] gives return for that day
    train_states = profound_states[:train_end]
    train_pr = price_relatives[:train_end]

    val_states = profound_states[train_end:val_end]
    val_pr = price_relatives[train_end:val_end]

    num_features = profound_states.shape[1]
    num_assets_cash = profound_states.shape[2]
    num_assets = num_assets_cash - 1  # minus cash
    window = profound_states.shape[3]

    print(
        f"Features: {num_features}, Assets+Cash: {num_assets_cash}, "
        f"Window: {window}"
    )

    # Create environments
    train_env = SAMPortfolioEnv(
        train_states,
        train_pr,
        commission=sam_cfg.commission,
        risk_scaling=sam_cfg.risk_scaling,
        initial_value=config.initial_capital,
        window=config.data.rolling_window,
    )

    val_env = SAMPortfolioEnv(
        val_states,
        val_pr,
        commission=sam_cfg.commission,
        risk_scaling=sam_cfg.risk_scaling,
        initial_value=config.initial_capital,
        window=config.data.rolling_window,
    )

    # Create agent
    agent = SAMAgent(
        num_features=num_features,
        num_assets=num_assets,
        window=window,
        conv_channels=sam_cfg.conv_channels,
        hidden_dim=sam_cfg.hidden_dim,
        learning_rate=sam_cfg.learning_rate,
        gamma=sam_cfg.gamma,
        gae_lambda=sam_cfg.gae_lambda,
        clip_epsilon=sam_cfg.clip_epsilon,
        entropy_coeff=sam_cfg.entropy_coeff,
        value_loss_coeff=sam_cfg.value_loss_coeff,
        max_grad_norm=sam_cfg.max_grad_norm,
        ppo_epochs=sam_cfg.ppo_epochs,
        mini_batch_size=sam_cfg.mini_batch_size,
        rollout_length=sam_cfg.rollout_length,
        device=device,
    )

    best_sortino = -float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for update_idx in range(sam_cfg.num_updates):
        # Collect rollout
        state, _ = train_env.reset()
        agent.rollout_buffer.reset()

        for step in range(sam_cfg.rollout_length):
            action, raw_action, value, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            agent.rollout_buffer.add(
                state, action, raw_action, reward, value, log_prob, done
            )

            if done:
                state, _ = train_env.reset()
            else:
                state = next_state

        # Compute last value for GAE
        with torch.no_grad():
            state_t = (
                torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            )
            last_value = agent.value_net(state_t).item()

        agent.rollout_buffer.compute_returns_and_advantages(last_value)

        # PPO update
        metrics = agent.update()

        # Periodic validation
        if (update_idx + 1) % 20 == 0:
            val_metrics = evaluate_agent(agent, val_env)
            print(
                f"Update {update_idx+1}/{sam_cfg.num_updates} | "
                f"PL: {metrics['policy_loss']:.4f} | "
                f"VL: {metrics['value_loss']:.4f} | "
                f"Ent: {metrics['entropy']:.4f} | "
                f"Val ARR: {val_metrics['ARR']:.4f} | "
                f"Val Sortino: {val_metrics['Sortino']:.4f}"
            )

            if val_metrics["Sortino"] > best_sortino:
                best_sortino = val_metrics["Sortino"]
                save_path = f"checkpoints/sam_{portfolio_name}_best.pt"
                agent.save(save_path)
                print(f"  -> New best model saved (Sortino: {best_sortino:.4f})")

    # Save final model
    agent.save(f"checkpoints/sam_{portfolio_name}_final.pt")
    print(f"\nTraining complete for {portfolio_name}!")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train SAM agents")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--portfolio", type=str, default=None,
        help="Train specific portfolio (e.g., portfolio_a). "
             "If not set, trains all portfolios.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.portfolio:
        portfolios = [args.portfolio]
    else:
        portfolios = list(config.portfolios.keys())

    for port_name in portfolios:
        print(f"\n{'='*60}")
        print(f"Training SAM for {port_name}: {config.portfolios[port_name].tickers}")
        print(f"{'='*60}")
        train_portfolio(port_name, config)

    print("\nAll SAM agents trained!")


if __name__ == "__main__":
    main()
