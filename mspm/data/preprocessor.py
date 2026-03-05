import numpy as np
import pandas as pd


class FeaturePreprocessor:
    """Normalize OHLCV features and construct rolling windows."""

    FEATURE_COLS = ["Adj Close", "Open", "High", "Low", "Volume"]

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract OHLCV features from raw DataFrame.

        Returns array of shape (num_days, num_features=5).
        """
        features = df[self.FEATURE_COLS].values.astype(np.float32)
        return features

    def normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize a single window by dividing by the first day's values.

        Input: (num_features, window_length)
        Output: (num_features, window_length) - relative to day 0
        """
        first_day = window[:, 0:1].copy()
        first_day[first_day == 0] = 1.0  # avoid division by zero
        return window / first_day

    def build_rolling_windows(
        self, features: np.ndarray, window: int = 50
    ) -> np.ndarray:
        """Build normalized rolling windows from feature array.

        Input: (num_days, num_features)
        Output: (num_steps, num_features, window) where num_steps = num_days - window + 1
        """
        num_days, num_features = features.shape
        num_steps = num_days - window + 1

        if num_steps <= 0:
            raise ValueError(
                f"Not enough days ({num_days}) for window size {window}"
            )

        windows = np.zeros((num_steps, num_features, window), dtype=np.float32)
        for i in range(num_steps):
            w = features[i : i + window].T  # (num_features, window)
            windows[i] = self.normalize_window(w)

        return windows

    def build_eam_states(
        self, df: pd.DataFrame, window: int = 50
    ) -> np.ndarray:
        """Build EAM state tensors from raw DataFrame.

        Extracts OHLCV features + zero sentiment placeholders.
        Returns shape (num_steps, num_features=7, window).
        """
        features = self.extract_features(df)
        num_days = features.shape[0]

        # Add 2 sentiment placeholder columns (zeros)
        sentiment = np.zeros((num_days, 2))
        all_features = np.concatenate([features, sentiment], axis=1)

        return self.build_rolling_windows(all_features, window)

    def get_close_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Extract adjusted close prices."""
        return df["Adj Close"].values.astype(np.float32)

    def build_signal_comprised_tensor(
        self,
        rolling_windows: np.ndarray,
        signals: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Stack EAM trading signals onto price rolling windows.

        Input rolling_windows: (num_steps, num_features, window)
        Input signals: (num_steps,) with values in {0=buy, 1=close, 2=skip}
        Output: (num_steps, num_features+3, window) - original + 3 one-hot signal rows
        """
        num_steps = rolling_windows.shape[0]
        assert len(signals) == num_steps

        # One-hot encode signals: (num_steps, 3)
        one_hot = np.eye(3, dtype=rolling_windows.dtype)[signals]
        # Repeat across window: (num_steps, 3, window)
        signal_rows = np.repeat(one_hot[:, :, np.newaxis], window, axis=2)

        return np.concatenate([rolling_windows, signal_rows], axis=1)

    def build_profound_state(
        self,
        per_asset_tensors: list[np.ndarray],
        window: int = 50,
    ) -> np.ndarray:
        """Stack per-asset 2D signal-comprised tensors into 3D profound state.

        Input: list of m arrays, each (num_steps, f, window)
        Output: (num_steps, f, m+1, window) where +1 is cash

        Cash asset has price features = 1.0 (constant) and signal = [0,0,1] (skip).
        The profound state is arranged as: [cash, asset_1, ..., asset_m]
        """
        num_steps = per_asset_tensors[0].shape[0]
        num_features = per_asset_tensors[0].shape[1]
        num_assets = len(per_asset_tensors)

        # Validate all assets have the same shape
        for i, t in enumerate(per_asset_tensors):
            assert t.shape == (num_steps, num_features, window), (
                f"Asset {i} shape {t.shape} != expected "
                f"({num_steps}, {num_features}, {window})"
            )

        # Cash tensor: price features = 1.0, signal one-hot = [0, 0, 1] (skip)
        # Signal one-hot occupies the last 3 rows of the feature dimension
        n_price_features = num_features - 3  # OHLCV (+ sentiment placeholders)
        cash = np.zeros((num_steps, num_features, window),
                        dtype=per_asset_tensors[0].dtype)
        cash[:, :n_price_features, :] = 1.0   # constant price
        cash[:, -1, :] = 1.0                  # skip signal: [0, 0, 1]

        # Stack: (num_steps, f, m+1, window)
        all_tensors = [cash] + per_asset_tensors
        profound = np.stack(all_tensors, axis=2)

        return profound

    def compute_price_relatives(
        self,
        close_prices: dict[str, np.ndarray],
        start_idx: int = 0,
    ) -> np.ndarray:
        """Compute price relative vector y_t = p_t / p_{t-1}.

        Returns: (num_days-1, m+1) where first column is cash (=1.0).
        """
        tickers = list(close_prices.keys())
        n_days = len(next(iter(close_prices.values()))) - start_idx

        # price relatives for each asset
        relatives = []
        for ticker in tickers:
            prices = close_prices[ticker][start_idx:]
            rel = prices[1:] / prices[:-1]
            relatives.append(rel)

        relatives = np.column_stack(relatives)  # (n_days-1, m)

        # Prepend cash column (always 1.0)
        cash = np.ones((relatives.shape[0], 1))
        return np.concatenate([cash, relatives], axis=1)  # (n_days-1, m+1)
