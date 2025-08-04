import numpy as np
import pandas as pd
from typing import Union
from quantifilib.strategy.base_label import BaseLabel

class ADLLabeling(BaseLabel):
    """
    Labeling class based on Accumulation/Distribution Line (ADL) momentum.

    This class calculates the ADL using price and volume data,
    and labels points where the slope of ADL changes direction.

    Signals:
        +1: ADL slope turns upward (from decreasing to increasing)
        -1: ADL slope turns downward (from increasing to decreasing)
         0: No significant change
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def get_adl(self) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line (ADL).

        Returns
        -------
        adl : pd.Series
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']

        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
        mfv = mfm * volume
        adl = mfv.cumsum()

        return adl

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on ADL slope crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}
        """
        adl = self.get_adl()
        adl_diff = adl.diff()

        buy = ((adl_diff > 0) & (adl_diff.shift(1) <= 0)).astype(int)
        sell = ((adl_diff < 0) & (adl_diff.shift(1) >= 0)).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'
        return labels

class OBVLabeling(BaseLabel):
    """
    Labeling class based on OBV short/long moving average crossovers.

    This class calculates the On-Balance Volume (OBV) and generates
    buy/sell signals when the short-term OBV moving average crosses
    above or below the long-term moving average.

    Signals:
        +1: Short OBV MA crosses above long OBV MA (bullish crossover)
        -1: Short OBV MA crosses below long OBV MA (bearish crossover)
         0: No crossover
    """
    def __init__(
        self,
        data: pd.DataFrame,
        short_window: int = 10,
        long_window: int = 20
    ) -> None:
        super().__init__(data=data)
        self.short_window = short_window
        self.long_window = long_window

    def get_obv(self) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        Returns
        -------
        obv : pd.Series
        """
        close = self.data['Close']
        volume = self.data['Volume']

        direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        obv = (volume * direction).fillna(0).cumsum()

        return obv

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on OBV short/long MA crossovers.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}
        """
        obv = self.get_obv()
        obv_short = obv.rolling(window=self.short_window).mean()
        obv_long = obv.rolling(window=self.long_window).mean()

        buy = ((obv_short > obv_long) & (obv_short.shift(1) <= obv_long.shift(1))).astype(int)
        sell = ((obv_short < obv_long) & (obv_short.shift(1) >= obv_long.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels

class MFILabeling(BaseLabel):
    """
    Labeling class based on Money Flow Index (MFI).

    This class generates signals based on MFI crossing overbought/oversold thresholds.

    Signals:
        +1: MFI crosses above oversold threshold (e.g., 20) → Buy
        -1: MFI crosses below overbought threshold (e.g., 80) → Sell
         0: Otherwise
    """
    def __init__(self, data: pd.DataFrame, window: int = 14, overbought: int = 80, oversold: int = 20) -> None:
        super().__init__(data=data)
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def get_mfi(self) -> pd.Series:
        """
        Calculate the Money Flow Index (MFI).

        Returns
        -------
        mfi : pd.Series
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']

        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Positive or negative money flow
        price_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(price_diff > 0, 0.0)
        negative_flow = raw_money_flow.where(price_diff < 0, 0.0)

        pos_mf_sum = positive_flow.rolling(self.window).sum()
        neg_mf_sum = negative_flow.rolling(self.window).sum()

        mfi = 100 - (100 / (1 + (pos_mf_sum / neg_mf_sum.replace(0, 1e-9))))
        return mfi

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on MFI thresholds.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}
        """
        mfi = self.get_mfi()

        buy = ((mfi > self.oversold) & (mfi.shift(1) <= self.oversold)).astype(int)
        sell = ((mfi < self.overbought) & (mfi.shift(1) >= self.overbought)).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels