import numpy as np


def daily_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """Compute daily log returns from portfolio value series."""
    return np.log(portfolio_values[1:] / portfolio_values[:-1])


def daily_rate_of_return(portfolio_values: np.ndarray) -> float:
    """DRR: average of daily exponential returns."""
    log_rets = daily_returns(portfolio_values)
    return float(np.mean(np.exp(log_rets)))


def accumulated_rate_of_return(portfolio_values: np.ndarray) -> float:
    """ARR = p_T / p_0."""
    return float(portfolio_values[-1] / portfolio_values[0])


def sortino_ratio(
    portfolio_values: np.ndarray, risk_free_rate: float = 0.0
) -> float:
    """Sortino ratio: (mean_return - Rf) / downside_std."""
    log_rets = daily_returns(portfolio_values)
    exp_rets = np.exp(log_rets)
    excess = exp_rets - (1.0 + risk_free_rate)
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(excess) / downside_std)


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough decline as a fraction."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return float(np.max(drawdown))


def compute_all_metrics(portfolio_values: np.ndarray) -> dict:
    """Compute all performance metrics."""
    return {
        "DRR": daily_rate_of_return(portfolio_values),
        "ARR": accumulated_rate_of_return(portfolio_values),
        "Sortino": sortino_ratio(portfolio_values),
        "MaxDrawdown": max_drawdown(portfolio_values),
    }
