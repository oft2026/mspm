import numpy as np
import pandas as pd


def constant_rebalanced_portfolio(
    close_prices: dict[str, np.ndarray],
    initial_value: float = 10000.0,
    commission: float = 0.0025,
) -> np.ndarray:
    """CRP baseline: equal-weight 1/N rebalancing daily.

    Args:
        close_prices: dict mapping ticker -> array of daily close prices.
        initial_value: starting portfolio value.
        commission: transaction cost rate.

    Returns:
        Array of portfolio values over time.
    """
    tickers = list(close_prices.keys())
    n_assets = len(tickers)
    n_days = len(next(iter(close_prices.values())))

    # Stack price relatives: p_t / p_{t-1}
    price_relatives = np.ones((n_days, n_assets))
    for i, ticker in enumerate(tickers):
        prices = close_prices[ticker]
        price_relatives[1:, i] = prices[1:] / prices[:-1]

    target_weight = 1.0 / n_assets
    portfolio_values = np.zeros(n_days)
    portfolio_values[0] = initial_value

    weights = np.full(n_assets, target_weight)

    for t in range(1, n_days):
        # Portfolio return before rebalancing
        port_return = np.dot(weights, price_relatives[t])
        # Weights after price change (before rebalancing)
        new_weights = (weights * price_relatives[t]) / port_return
        # Transaction cost for rebalancing back to equal weight
        turnover = np.sum(np.abs(target_weight - new_weights))
        cost = commission * turnover
        portfolio_values[t] = portfolio_values[t - 1] * port_return * (1 - cost)
        weights = np.full(n_assets, target_weight)

    return portfolio_values


def buy_and_hold(
    close_prices: dict[str, np.ndarray],
    initial_value: float = 10000.0,
    commission: float = 0.0025,
) -> np.ndarray:
    """BAH baseline: buy equal amounts on day 1, hold forever.

    Args:
        close_prices: dict mapping ticker -> array of daily close prices.
        initial_value: starting portfolio value.
        commission: transaction cost rate.

    Returns:
        Array of portfolio values over time.
    """
    tickers = list(close_prices.keys())
    n_assets = len(tickers)
    n_days = len(next(iter(close_prices.values())))

    # Initial allocation: equal dollars into each asset after commission
    alloc_per_asset = (initial_value * (1 - commission)) / n_assets

    # Number of shares (fractional) purchased on day 0
    shares = {}
    for ticker in tickers:
        shares[ticker] = alloc_per_asset / close_prices[ticker][0]

    portfolio_values = np.zeros(n_days)
    for t in range(n_days):
        total = sum(
            shares[ticker] * close_prices[ticker][t] for ticker in tickers
        )
        portfolio_values[t] = total

    return portfolio_values
