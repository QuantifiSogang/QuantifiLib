import numpy as np
import pandas as pd

from typing import Union
from quantifilib.strategy.base_label import BaseLabel

class MovingAverageLabeling(BaseLabel):
    def __init__(
        self,
        data: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 60,
        price_col_name: str = 'Close',
    ) -> None:
        super().__init__(data = data)
        self.short_window = short_window
        self.long_window = long_window
        self.price_col_name = price_col_name

    def get_moving_averages(self, method:str = 'sma') -> pd.DataFrame:
        """
        Compute short and long moving averages.

        Returns
        -------
        pd.DataFrame with 'short_ma' and 'long_ma'
        """
        price = self.data[self.price_col_name]

        if method == 'sma':
            short_ma = price.rolling(window=self.short_window).mean()
            long_ma = price.rolling(window=self.long_window).mean()

        elif method == 'ema':
            short_ma = price.ewm(window=self.short_window).mean()
            long_ma = price.ewm(window=self.long_window).mean()

        else : raise ValueError(f'Moving average method {method} not supported. available options: sma, ema')

        return pd.DataFrame({
            'short_ma': short_ma,
            'long_ma': long_ma
        })

    def get_labels(self, method: str = 'sma') -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on SMA crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        ma_df = self.get_moving_averages(method = method)
        short_ma = ma_df['short_ma']
        long_ma = ma_df['long_ma']

        # golden cross
        buy = ((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))).astype(int)

        # dead cross
        sell = ((short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels

class MACDLabeling(BaseLabel):
    def __init__(
        self,
        data: pd.DataFrame,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        method: str = 'ema',
        price_col_name: str = 'Close',
    ) -> None:
        super().__init__(data=data)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.method = method
        self.price_col_name = price_col_name

    def get_macd(self) -> pd.DataFrame:
        """
        Calculate MACD and Signal lines using specified method (ema or sma).

        Returns
        -------
        macd_df : pd.DataFrame
            Columns: ['macd', 'signal']
        """
        price = self.data[self.price_col_name]

        if self.method == 'ema':
            short_ma = price.ewm(span=self.short_window, adjust=False).mean()
            long_ma = price.ewm(span=self.long_window, adjust=False).mean()
            macd_line = short_ma - long_ma
            signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()

        elif self.method == 'sma':
            short_ma = price.rolling(window=self.short_window).mean()
            long_ma = price.rolling(window=self.long_window).mean()
            macd_line = short_ma - long_ma
            signal_line = macd_line.rolling(window=self.signal_window).mean()

        else:
            raise ValueError(f"Method '{self.method}' not supported. Use 'ema' or 'sma'.")

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line
        })

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on MACD crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        macd_df = self.get_macd()
        macd = macd_df['macd']
        signal = macd_df['signal']

        buy = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
        sell = ((macd < signal) & (macd.shift(1) >= signal.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels