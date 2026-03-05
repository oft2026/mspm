import os
from pathlib import Path

import pandas as pd
import yfinance as yf


class OHLCVDownloader:
    """Download and cache OHLCV data from yfinance."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self, ticker: str, start: str, end: str, force: bool = False
    ) -> pd.DataFrame:
        """Download OHLCV data for a single ticker with local parquet caching.

        Returns DataFrame with columns:
            Date (index), Open, High, Low, Close, Adj Close, Volume
        """
        cache_file = self.cache_dir / f"{ticker}_{start}_{end}.parquet"

        if cache_file.exists() and not force:
            return pd.read_parquet(cache_file)

        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_parquet(cache_file)
        return df

    def download_all(
        self, tickers: list[str], start: str, end: str, force: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Download data for multiple tickers."""
        result = {}
        for ticker in tickers:
            result[ticker] = self.download(ticker, start, end, force)
        return result
