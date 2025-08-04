import numpy as np
import pandas as pd

from .base_label import BaseLabel

class MultiLabeling(BaseLabel):
    """
    Combine multiple labeling strategies into a single ensemble strategy.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data
    strategies : list
        List of labeling strategy instances (must implement get_labels())
    method : str
        Combination method: 'vote' (default), 'average'
    """

    def __init__(
            self, data: pd.DataFrame,
            *strategies,
            method: str = 'vote'
        ) :
        super().__init__(data)
        self.strategies = strategies
        self.method = method

    def get_labels(self) -> pd.Series:
        """
        Combine labels from all strategies.

        Returns
        -------
        combined : pd.Series
            Final label series with values in {-1, 0, +1}
        """
        all_labels = []

        # Generate labels from each strategy
        for strat in self.strategies:
            strat.data = self.data  # Inject data (in case it was not passed before)
            label = strat.get_labels().rename(str(type(strat).__name__))  # Rename by strategy name
            all_labels.append(label)

        # Combine labels into DataFrame
        df = pd.concat(all_labels, axis=1).dropna()

        # Combine method 1: majority voting
        if self.method == 'vote':
            combined = df.sum(axis=1).clip(-1, 1)

        # Combine method 2: average and threshold
        elif self.method == 'average':
            combined = (df.mean(axis=1)).apply(lambda x: 1 if x > 0.1 else -1 if x < -0.1 else 0)

        else:
            raise ValueError("method must be 'vote' or 'average'")

        combined.name = 'bins'
        return combined