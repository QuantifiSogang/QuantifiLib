import numpy as np
import pandas as pd
from quantifilib.strategy.base_label import BaseTechnicalLabel

class MomentumBasedLabeling(BaseTechnicalLabel) :
    """
    Labeling with Technical Indicators based on momentum strategies
    """
    def __init__(self, long_only : bool = False, short_only : bool = False) :
        super().__init__()
        self.long_only = long_only
        self.short_only = short_only

    def get_simple_moving_average_labels(
            self, short_window:int = 5,
            long_window:int = 20
        ) -> pd.Series :
        """
        labeling with simple moving average
        :param short_window: short window size
        :param log_window: long window size
        :return: labels
        """
        sma_short = self.data['Close'].rolling(window=short_window).mean()
        sma_long = self.data['Close'].rolling(window=long_window).mean()

        sell_signal = ((sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))).astype(int) * (-1)
        buy_signal = ((sma_short >= sma_long) & (sma_short.shift(1) < sma_long.shift(1))).astype(int)

        if self.long_only :
            buy_signal.name = f'SMA_({short_window},{long_window})'
            return buy_signal
        elif self.short_only :
            sell_signal.name = f'SMA_({short_window},{long_window})'
            return sell_signal
        else :
            signal = sell_signal + buy_signal
            signal.name = f'SMA_({short_window},{long_window})'
            return signal

    def get_exponential_moving_average_labels(
            self, short_window:int = 5,
            long_window:int = 20
        ) -> pd.Series :
        """
        labeling with exponential moving average
        :param short_window: short window size
        :param log_window: long window size
        :return: labels
        """
        ema_short = self.data['Close'].ewm(window=short_window).mean()
        ema_long = self.data['Close'].ewm(window=long_window).mean()

        sell_signal = ((ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))).astype(int) * (-1)
        buy_signal = ((ema_short >= ema_long) & (ema_short.shift(1) < ema_long.shift(1))).astype(int)

        if self.long_only :
            buy_signal.name = f'EMA_({short_window},{long_window})'
            return buy_signal
        elif self.short_only :
            sell_signal.name = f'EMA_({short_window},{long_window})'
            return sell_signal
        else :
            signal = sell_signal + buy_signal
            signal.name = f'EMA_({short_window},{long_window})'
            return signal