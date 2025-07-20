import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

class BaseLabel(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def add_vertical_barrier(
            self,
            t_events: pd.Series,
            close: pd.Series = None,
            num_days: int = 0,
            num_hours: int = 0,
            num_minutes: int = 0,
            num_seconds: int = 0
    ) -> pd.Series :
        """
        Add a vertical barrier (time-based event horizon) to the given event timestamps.

        Parameters
        ----------
        t_events : pd.Series
            Series of datetime indices representing event start times.
        close : pd.Series, optional
            Series of price data indexed by datetime, used to locate valid time points.
            If None, tries to use 'Close' or 'Adj Close' column from self.data.
        num_days, num_hours, num_minutes, num_seconds : int
            Time delta components for how far the vertical barrier should be placed from the event.

        Returns
        -------
        vertical_barriers : pd.Series
            Series with event timestamps as index and corresponding vertical barrier timestamps as values.
        """
        self._check_close_columns()

        timedelta = pd.Timedelta(
            days=num_days,
            hours=num_hours,
            minutes=num_minutes,
            seconds=num_seconds
        )

        barrier_timestamps = []
        barrier_indices = []

        for t in t_events:
            loc = close.index.searchsorted(t + timedelta)
            if loc < len(close.index):
                barrier_indices.append(t)
                barrier_timestamps.append(close.index[loc])

        return pd.Series(data = barrier_timestamps, index = barrier_indices)

    def _check_ohlcv_columns(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        cols = [col.lower() for col in self.data.columns]
        missing = [col for col in required if col not in cols]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

    def _check_close_columns(self, check_col):
        # Infer close column if not provided
        if check_col is None:
            candidates = ['Close', 'Adj Close', 'close', 'adj close']
            for col in candidates:
                if col in self.data.columns:
                    close = self.data[col]
                    break
            else:
                raise ValueError("No 'Close' or 'Adj Close' column found in self.data. Please pass `close` manually.")

    @abstractmethod
    def get_labels(self, *args, **kwargs) -> Union[pd.Series, pd.DataFrame] :
        pass