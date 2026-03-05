"""
Backtesting and Evaluation Script.

1. Load trained SAM model
2. Run on test period (2020)
3. Compute metrics (DRR, ARR, Sortino, MaxDrawdown)
4. Run CRP and BAH baselines
5. Generate comparison table and performance plot

Usage:
    python scripts/backtest.py [--config configs/default.yaml] [--portfolio portfolio_a]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mspm.sam.agent import SAMAgent
from mspm.sam.environment import SAMPortfolioEnv
from mspm.utils.config import load_config
from mspm.utils.device import get_device, set_seed
from mspm.utils.metrics import compute_all_metrics
from mspm.utils.portfolio import buy_and_hold, constant_rebalanced_portfolio


def backtest_portfolio(portfolio_name: str, config):
    """Run backtesting for a single portfolio."""
    sam_cfg = config.sam
    device = get_device()

    # Load pre-computed data
    profound_states = np.load(
        f"data/processed/{portfolio_name}_profound.npy"
    )
    price_relatives = np.load(
        f"data/processed/{portfolio_name}_price_rel.npy"
    )
    close_prices_arr = np.load(
        f"data/processed/{portfolio_name}_close.npy"
    )
    splits = np.load(
        f"data/processed/{portfolio_name}_splits.npy", allow_pickle=True
    ).item()

    val_end = splits["val_end"]
    tickers = config.portfolios[portfolio_name].tickers

    # Test period data
    test_states = profound_states[val_end:]
    test_pr = price_relatives[val_end:]
    test_close = close_prices_arr[val_end:]

    print(f"Test period: {len(test_states)} days")
    print(f"Tickers: {tickers}")

    num_features = profound_states.shape[1]
    num_assets_cash = profound_states.shape[2]
    num_assets = num_assets_cash - 1
    window = profound_states.shape[3]

    # ---- MSPM Agent ----
    test_env = SAMPortfolioEnv(
        test_states,
        test_pr,
        commission=sam_cfg.commission,
        risk_scaling=sam_cfg.risk_scaling,
        initial_value=config.initial_capital,
        window=config.data.rolling_window,
    )

    agent = SAMAgent(
        num_features=num_features,
        num_assets=num_assets,
        window=window,
        conv_channels=sam_cfg.conv_channels,
        hidden_dim=sam_cfg.hidden_dim,
        device=device,
    )

    # Try loading best model, fallback to final
    model_path = f"checkpoints/sam_{portfolio_name}_best.pt"
    if not os.path.exists(model_path):
        model_path = f"checkpoints/sam_{portfolio_name}_final.pt"
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    # Run MSPM
    state, _ = test_env.reset()
    mspm_values = [test_env.portfolio_value]
    allocations = []

    while True:
        action = agent.get_allocation(state)
        state, reward, terminated, truncated, info = test_env.step(action)
        mspm_values.append(info["portfolio_value"])
        allocations.append(action)
        if terminated or truncated:
            break

    mspm_values = np.array(mspm_values)
    mspm_metrics = compute_all_metrics(mspm_values)

    # ---- Baselines ----
    # Prepare close prices dict for baselines
    close_dict = {}
    for i, ticker in enumerate(tickers):
        close_dict[ticker] = test_close[:, i]

    crp_values = constant_rebalanced_portfolio(
        close_dict,
        initial_value=config.initial_capital,
        commission=sam_cfg.commission,
    )
    crp_metrics = compute_all_metrics(crp_values)

    bah_values = buy_and_hold(
        close_dict,
        initial_value=config.initial_capital,
        commission=sam_cfg.commission,
    )
    bah_metrics = compute_all_metrics(bah_values)

    # ---- Print Results ----
    print(f"\n{'='*70}")
    print(f"  Backtest Results for {portfolio_name}: {tickers}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'MSPM':>12} {'CRP':>12} {'BAH':>12}")
    print(f"{'-'*56}")

    for metric in ["DRR", "ARR", "Sortino", "MaxDrawdown"]:
        print(
            f"{metric:<20} "
            f"{mspm_metrics[metric]:>12.4f} "
            f"{crp_metrics[metric]:>12.4f} "
            f"{bah_metrics[metric]:>12.4f}"
        )

    print(f"\nFinal Portfolio Value:")
    print(f"  MSPM: ${mspm_values[-1]:,.2f}")
    print(f"  CRP:  ${crp_values[-1]:,.2f}")
    print(f"  BAH:  ${bah_values[-1]:,.2f}")

    # ---- Plot ----
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Portfolio value over time
    ax1.plot(mspm_values, label="MSPM", linewidth=2)
    ax1.plot(crp_values, label="CRP", linewidth=1, linestyle="--")
    ax1.plot(bah_values, label="BAH", linewidth=1, linestyle=":")
    ax1.set_title(f"Portfolio Value - {portfolio_name} ({', '.join(tickers)})")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Allocation weights over time
    alloc_arr = np.array(allocations)
    labels = ["Cash"] + tickers
    ax2.stackplot(
        range(len(alloc_arr)),
        *[alloc_arr[:, i] for i in range(alloc_arr.shape[1])],
        labels=labels,
        alpha=0.8,
    )
    ax2.set_title(f"Asset Allocation - {portfolio_name}")
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Weight")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{portfolio_name}_backtest.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to results/{portfolio_name}_backtest.png")

    return {
        "portfolio": portfolio_name,
        "tickers": tickers,
        "mspm": mspm_metrics,
        "crp": crp_metrics,
        "bah": bah_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest MSPM")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--portfolio", type=str, default=None,
        help="Backtest specific portfolio. If not set, runs all.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.portfolio:
        portfolios = [args.portfolio]
    else:
        portfolios = list(config.portfolios.keys())

    all_results = []
    for port_name in portfolios:
        print(f"\n{'#'*70}")
        print(f"# Backtesting {port_name}")
        print(f"{'#'*70}")
        results = backtest_portfolio(port_name, config)
        all_results.append(results)

    # Summary table
    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print("  SUMMARY: MSPM ARR Across All Portfolios")
        print(f"{'='*80}")
        print(
            f"{'Portfolio':<15} {'Tickers':<25} "
            f"{'MSPM ARR':>10} {'CRP ARR':>10} {'BAH ARR':>10}"
        )
        print(f"{'-'*75}")
        for r in all_results:
            print(
                f"{r['portfolio']:<15} {str(r['tickers']):<25} "
                f"{r['mspm']['ARR']:>10.4f} "
                f"{r['crp']['ARR']:>10.4f} "
                f"{r['bah']['ARR']:>10.4f}"
            )


if __name__ == "__main__":
    main()
