import numpy as np
import pandas as pd
from typing import Union
from quantifilib.strategy.base_label import BaseLabel

class BollingerBandLabeling(BaseLabel):
    def __init__(
            self,
            data: pd.DataFrame,
            upper_multiplier : float,
            lower_multiplier : float,
            window : int = 20,
            method : str = 'ewm',
            price_col_name : str = 'Close',
        ) -> None:
        super().__init__(data = data)
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.window = window
        self.method = method
        self.price_col_name = price_col_name

    def get_bollinger_band(
            self
        ) -> pd.DataFrame :
        """
        Compute Bollinger Bands for the given price series.

        Parameters
        ----------
        price_col_name : str
            Column name of price series.
        window : int
            Rolling window for moving average and std.
        method : str
            Smoothing method ('ewm' or 'sma').

        Returns
        -------
        bands : pd.DataFrame
            DataFrame with columns ['mid', 'upper', 'lower']
        """

        price = self.data[self.price_col_name]
        if self.method == 'ewm':
            ma = price.ewm(span=self.window).mean()
            std = price.ewm(span=self.window).std()
        else:
            ma = price.rolling(self.window).mean()
            std = price.rolling(self.window).std()

        upper = ma + self.upper_multiplier * std
        lower = ma - self.lower_multiplier * std

        return pd.DataFrame({'ma': ma, 'upper': upper, 'lower': lower})

    def get_labels(
            self,
        ) -> Union[pd.DataFrame,pd.Series] :
        """
        Generate buy/sell signals based on SMA crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        bollinger = self.get_bollinger_band()
        price = self.data[self.price_col_name]

        sell = ((price < bollinger['upper']) & (price.shift(1) >= bollinger['upper'].shift(1))).astype(int) * (-1)
        buy = ((price >= bollinger['lower']) & (price.shift(1) < bollinger['lower'].shift(1))).astype(int)

        labels = sell + buy
        labels.name = 'bins'

        return labels

class RSILabeling(BaseLabel):
    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 14,
        upper_threshold: float = 70,
        lower_threshold: float = 30,
        price_col_name: str = 'Close',
    ) -> None:
        super().__init__(data=data)
        self.window = window
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.price_col_name = price_col_name

    def get_rsi(self) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Returns
        -------
        rsi : pd.Series
            RSI values.
        """
        price = self.data[self.price_col_name]
        delta = price.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on RSI thresholds.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        rsi = self.get_rsi()

        buy = ((rsi < self.lower_threshold) & (rsi.shift(1) >= self.lower_threshold)).astype(int)
        sell = ((rsi > self.upper_threshold) & (rsi.shift(1) <= self.upper_threshold)).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels

class StochasticOscillatorLabeling(BaseLabel):
    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 14,
        smooth_k: int = 3,
        upper_threshold: float = 80,
        lower_threshold: float = 20,
        price_col_name: str = 'Close',
    ) -> None:
        super().__init__(data=data)
        self.window = window
        self.smooth_k = smooth_k
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.price_col_name = price_col_name

    def get_stochastic_k(self) -> pd.Series:
        """
        Calculate %K line of the Stochastic Oscillator.

        Returns
        -------
        k : pd.Series
            Smoothed %K values.
        """
        price = self.data[self.price_col_name]
        low_min = price.rolling(window=self.window).min()
        high_max = price.rolling(window=self.window).max()

        raw_k = 100 * (price - low_min) / (high_max - low_min)
        smooth_k = raw_k.rolling(window=self.smooth_k).mean()

        return smooth_k

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on Stochastic Oscillator %K.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        k = self.get_stochastic_k()

        buy = ((k < self.lower_threshold) & (k.shift(1) >= self.lower_threshold)).astype(int)
        sell = ((k > self.upper_threshold) & (k.shift(1) <= self.upper_threshold)).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels