import numpy as np
import pandas as pd
from quantifilib.features.technical_indicator.base import BaseFeature

class TrendFeatures(BaseFeature):
    def __init__(self, data : pd.DataFrame) :
        super().__init__(data)
        self._check_ohlcv_columns()
        self.data.columns = [col.lower() for col in data.columns]

    def simple_moving_average(
            self, price_col : str = 'close',
            window : int = 20
        ) -> pd.Series :
        """
        Computes the simple moving average of a given dataframe.
        :param price_col: base column to compute the simple moving average for
        :param window: window size to compute the simple moving average for
        :return: simple moving average dataframe
        """
        sma = self.data[price_col.lower()].rolling(window = window)
        sma.name = f'SMA_{window}'
        return sma

    def exponential_moving_average(
            self, price_col : str = 'close',
            window : int = 20
        ) -> pd.Series :
        """
        Computes the exponential moving average of a given dataframe.
        :param price_col: base column to compute the exponential moving average for
        :param window: window size to compute the exponential moving average for
        :return: exponential moving average dataframe
        """
        ema = self.data[price_col.lower()].ewm(span = window).mean()
        ema.name = f'EMA_{window}'
        return ema

