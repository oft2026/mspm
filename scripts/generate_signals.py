"""
Signal Generation Script.

1. Load trained EAM models
2. Run each EAM on prediction-period data (2016-2020) with greedy policy
3. Generate trading signals for each asset at each timestep
4. Build signal-comprised tensors and profound states per portfolio
5. Save profound states for SAM training

Usage:
    python scripts/generate_signals.py [--config configs/default.yaml]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mspm.data.downloader import OHLCVDownloader
from mspm.data.preprocessor import FeaturePreprocessor
from mspm.eam.agent import EAMAgent
from mspm.utils.config import load_config
from mspm.utils.device import get_device, set_seed


def get_all_tickers(config) -> list[str]:
    tickers = set()
    for portfolio in config.portfolios.values():
        for t in portfolio.tickers:
            tickers.add(t)
    return sorted(tickers)


def find_date_index(dates: np.ndarray, target: str) -> int:
    """Find the first index where date >= target date."""
    target_dt = np.datetime64(target)
    indices = np.where(dates >= target_dt)[0]
    if len(indices) == 0:
        return len(dates)
    return int(indices[0])


def main():
    parser = argparse.ArgumentParser(description="Generate EAM signals")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    downloader = OHLCVDownloader(cache_dir="data/raw")
    preprocessor = FeaturePreprocessor()
    device = get_device()

    all_tickers = get_all_tickers(config)
    window = config.data.rolling_window
    eam_cfg = config.eam

    os.makedirs("data/processed", exist_ok=True)

    # Generate signals for each ticker
    signals_dict = {}
    windows_dict = {}
    close_dict = {}
    dates_dict = {}  # Store aligned dates for date-based splitting

    for ticker in all_tickers:
        print(f"Generating signals for {ticker}...")

        # Download prediction period data
        df = downloader.download(
            ticker,
            start=config.data.eam_predict_start,
            end=config.data.eam_predict_end,
        )

        # Build states
        states = preprocessor.build_eam_states(df, window=window)
        close_prices = preprocessor.get_close_prices(df)
        aligned_close = close_prices[window - 1 :]

        # Aligned dates: each rolling window state[i] ends at date[i + window - 1]
        raw_dates = df.index.values
        aligned_dates = raw_dates[window - 1 :]

        # Build price-only rolling windows (5 features) for signal-comprised tensor
        features = preprocessor.extract_features(df)
        price_windows = preprocessor.build_rolling_windows(features, window)

        # Load trained EAM
        agent = EAMAgent(
            num_features=7,
            num_actions=eam_cfg.num_actions,
            hidden_dim=eam_cfg.hidden_dim,
            resnet_channels=eam_cfg.resnet_channels,
            num_residual_blocks=eam_cfg.num_residual_blocks,
            resnet_kernel_size=eam_cfg.resnet_kernel_size,
            device=device,
        )
        checkpoint_path = f"checkpoints/eam_{ticker}.pt"
        agent.load(checkpoint_path)

        # Generate greedy signals
        signals = agent.generate_signals(states)
        print(
            f"  {ticker}: {len(signals)} signals | "
            f"Buy: {(signals==0).sum()} | "
            f"Close: {(signals==1).sum()} | "
            f"Skip: {(signals==2).sum()}"
        )

        signals_dict[ticker] = signals
        windows_dict[ticker] = price_windows
        close_dict[ticker] = aligned_close
        dates_dict[ticker] = aligned_dates

    # Build profound states for each portfolio
    for port_name, port_cfg in config.portfolios.items():
        tickers = port_cfg.tickers
        print(f"\nBuilding profound state for {port_name}: {tickers}")

        # Find common length (all should be same if same date range)
        min_len = min(len(signals_dict[t]) for t in tickers)

        # Build signal-comprised tensors per asset
        per_asset_tensors = []
        for ticker in tickers:
            sig_tensor = preprocessor.build_signal_comprised_tensor(
                windows_dict[ticker][:min_len],
                signals_dict[ticker][:min_len],
                window=window,
            )
            per_asset_tensors.append(sig_tensor)

        # Build profound state: (T, f, m+1, window) where m+1 includes cash
        profound_state = preprocessor.build_profound_state(
            per_asset_tensors, window=window
        )
        print(f"  Profound state shape: {profound_state.shape}")

        # Compute price relatives for the portfolio
        close_prices_port = {
            t: close_dict[t][:min_len] for t in tickers
        }
        price_relatives = preprocessor.compute_price_relatives(close_prices_port)
        print(f"  Price relatives shape: {price_relatives.shape}")

        # IMPORTANT: Align profound_states with price_relatives.
        # price_relatives[i] = close[i+1] / close[i], so it describes the return
        # from day i to day i+1. The SAM agent observes state[i] and then
        # experiences return price_relatives[i]. Therefore:
        #   profound_state[0..N-2] aligns with price_relatives[0..N-2]
        #   profound_state[N-1] has no corresponding price_relative.
        # Trim the last profound state to match.
        n_pr = price_relatives.shape[0]
        profound_state = profound_state[:n_pr]
        assert profound_state.shape[0] == price_relatives.shape[0], (
            f"Shape mismatch: profound {profound_state.shape[0]} vs "
            f"price_rel {price_relatives.shape[0]}"
        )
        print(f"  Aligned profound state shape: {profound_state.shape}")

        # Use dates from first ticker for split computation
        dates = dates_dict[tickers[0]][:n_pr]

        # Date-based split using actual calendar boundaries
        train_end = find_date_index(dates, config.data.sam_val_start)
        val_end = find_date_index(dates, config.data.backtest_start)

        split_info = {
            "train_end": train_end,
            "val_end": val_end,
            "total": n_pr,
        }

        # Save
        np.save(f"data/processed/{port_name}_profound.npy", profound_state)
        np.save(f"data/processed/{port_name}_price_rel.npy", price_relatives)
        # Save all min_len close prices (one more than n_pr) so baselines
        # in backtest can reconstruct the same number of price relatives.
        np.save(
            f"data/processed/{port_name}_close.npy",
            np.column_stack(
                [close_prices_port[t] for t in tickers]
            ),
        )
        np.save(f"data/processed/{port_name}_splits.npy", split_info)

        print(
            f"  Split (date-based): "
            f"train=0:{train_end} ({train_end} days), "
            f"val={train_end}:{val_end} ({val_end - train_end} days), "
            f"test={val_end}:{n_pr} ({n_pr - val_end} days)"
        )

    print("\nAll signals and profound states generated!")


if __name__ == "__main__":
    main()
