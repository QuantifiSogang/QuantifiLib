import numpy as np
import pandas as pd

from typing import Optional, Union

class MetaLabeling:
    """
    Meta labeling class for refining primary signal (bins) using actual return outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' column.
    signal : pd.Series
        Primary model output with values {-1, 0, +1}, index-aligned with data.
    """

    def __init__(self, data: pd.DataFrame, signal: pd.Series):
        self.data = data
        self.signal = signal

    def _get_vertical_barrier(self, days: int = 5) -> pd.Series:
        """
        Set vertical barrier at fixed time horizon.

        Parameters
        ----------
        days : int
            Number of days ahead to evaluate outcome.

        Returns
        -------
        barrier : pd.Series
            Index-aligned series of timestamps for signal expiration.
        """
        # For each signal time, find the timestamp N days later
        barrier = self.signal.index.to_series().apply(
            lambda x: self.data.index[self.data.index.get_loc(x) + days]
            if self.data.index.get_loc(x) + days < len(self.data)
            else pd.NaT  # If exceeding data range, return NaT
        )
        return barrier

    def get_meta_label(self, barrier: Optional[pd.Series] = None, barrier_days: int = 5) -> pd.DataFrame:
        """
        Generate meta labels and realized returns at vertical barrier horizon.

        Parameters
        ----------
        barrier : Optional[pd.Series]
            Series of timestamps indicating when to evaluate each signal. If None, will be computed.
        barrier_days : int
            If barrier is None, this value determines the days to look ahead for the barrier.

        Returns
        -------
        meta : pd.DataFrame
            Columns: ['bins', 'meta_label', 'return']
        """
        signal = self.signal.dropna()

        # Generate vertical barrier if not provided
        if barrier is None:
            barrier = self._get_vertical_barrier(days=barrier_days)

        close = self.data['Close']
        meta_label = []
        realized_return = []

        # Evaluate each signal at the barrier point
        for t0 in signal.index:
            t1 = barrier.loc[t0]
            if pd.isna(t1) or t1 not in close.index:
                meta_label.append(None)
                realized_return.append(None)
                continue

            ret = close.loc[t1] / close.loc[t0] - 1
            pred = signal.loc[t0]

            if pred == 0:
                meta_label.append(None)
                realized_return.append(None)
            else:
                correct = int((ret > 0 and pred == 1) or (ret < 0 and pred == -1))
                meta_label.append(correct)
                realized_return.append(ret)

        # Compile results
        meta = pd.DataFrame({
            'bins': signal,
            'meta_label': meta_label,
            'actual_ret': realized_return
        }, index=signal.index)

        meta = meta.dropna()
        meta['meta_label'] = meta['meta_label'].astype(int)

        return meta.sort_index()