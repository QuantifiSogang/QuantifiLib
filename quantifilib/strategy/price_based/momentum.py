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
    """
    Labeling class based on MACD (Moving Average Convergence Divergence) crossover signals.

    This class computes the MACD and signal line using either EMA or SMA,
    and generates trading signals when crossovers occur.

    Signals:
        +1: MACD crosses above the signal line (bullish crossover)
        -1: MACD crosses below the signal line (bearish crossover)
         0: No crossover
    """
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

class AroonLabeling(BaseLabel):
    """
    Labeling class based on Aroon indicator crossover signals.

    This class calculates the Aroon Up and Aroon Down values,
    and generates trading signals based on their crossovers.

    Signals:
        +1: Aroon Up crosses above Aroon Down (bullish signal)
        -1: Aroon Up crosses below Aroon Down (bearish signal)
         0: No crossover
    """
    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 25,
        price_col_name: str = 'Close',
    ) -> None:
        super().__init__(data=data)
        self.window = window
        self.price_col_name = price_col_name

    def get_aroon(self) -> pd.DataFrame:
        """
        Calculate Aroon Up and Aroon Down.

        Returns
        -------
        aroon_df : pd.DataFrame
            Columns: ['aroon_up', 'aroon_down']
        """
        price = self.data[self.price_col_name]

        rolling_high_idx = price.rolling(self.window).apply(lambda x: x.argmax(), raw=True)
        rolling_low_idx = price.rolling(self.window).apply(lambda x: x.argmin(), raw=True)

        aroon_up = 100 * (self.window - (self.window - 1 - rolling_high_idx)) / self.window
        aroon_down = 100 * (self.window - (self.window - 1 - rolling_low_idx)) / self.window

        return pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down
        })

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on Aroon crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}, named 'bins'.
        """
        aroon_df = self.get_aroon()
        up = aroon_df['aroon_up']
        down = aroon_df['aroon_down']

        buy = ((up > down) & (up.shift(1) <= down.shift(1))).astype(int)
        sell = ((up < down) & (up.shift(1) >= down.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels

class ADXLabeling(BaseLabel):
    """
    Labeling class based on Average Directional Index (ADX) and DI crossovers.

    This class computes the ADX along with +DI and -DI,
    and generates trading signals when +DI and -DI cross each other.

    Signals:
        +1: +DI crosses above -DI (bullish trend signal)
        -1: +DI crosses below -DI (bearish trend signal)
         0: No crossover
    """
    def __init__(self, data: pd.DataFrame, window: int = 14) -> None:
        super().__init__(data=data)
        self.window = window

    def get_adx(self) -> pd.DataFrame:
        """
        Calculate ADX, +DI, and -DI.

        Returns
        -------
        pd.DataFrame with columns ['adx', 'plus_di', 'minus_di']
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(self.window).mean()

        plus_di = 100 * (plus_dm.rolling(self.window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.window).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(self.window).mean()

        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on +DI and -DI crossover.

        Returns
        -------
        labels : pd.Series
        """
        adx_df = self.get_adx()
        plus_di = adx_df['plus_di']
        minus_di = adx_df['minus_di']

        buy = ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))).astype(int)
        sell = ((plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'
        return labels

class ParabolicSARLabeling(BaseLabel):
    """
    Labeling class based on Parabolic SAR (Stop and Reverse).

    This class generates buy/sell signals when the price crosses the SAR.

    Signals:
        +1: Price crosses above SAR (bullish reversal)
        -1: Price crosses below SAR (bearish reversal)
         0: No crossover
    """
    def __init__(self, data: pd.DataFrame, step: float = 0.02, max_step: float = 0.2):
        super().__init__(data=data)
        self.step = step
        self.max_step = max_step

    def get_sar(self) -> pd.Series:
        """
        Calculate the Parabolic SAR series.

        Returns
        -------
        sar : pd.Series
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        sar = pd.Series(index=close.index, dtype='float64')

        # Initial settings
        trend = True  # True = uptrend, False = downtrend
        af = self.step
        ep = low.iloc[0]
        sar.iloc[0] = low.iloc[0]

        for i in range(1, len(close)):
            prev_sar = sar.iloc[i - 1]

            if trend:
                sar_val = prev_sar + af * (ep - prev_sar)
                sar_val = min(sar_val, low.iloc[i - 1], low.iloc[i])
                if close.iloc[i] < sar_val:
                    trend = False
                    sar_val = ep
                    ep = high.iloc[i]
                    af = self.step
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + self.step, self.max_step)
            else:
                sar_val = prev_sar + af * (ep - prev_sar)
                sar_val = max(sar_val, high.iloc[i - 1], high.iloc[i])
                if close.iloc[i] > sar_val:
                    trend = True
                    sar_val = ep
                    ep = low.iloc[i]
                    af = self.step
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + self.step, self.max_step)

            sar.iloc[i] = sar_val

        return sar

    def get_labels(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate buy/sell signals based on SAR crossover.

        Returns
        -------
        labels : pd.Series
            Series with values {-1, 0, +1}
        """
        sar = self.get_sar()
        close = self.data['Close']

        buy = ((close > sar) & (close.shift(1) <= sar.shift(1))).astype(int)
        sell = ((close < sar) & (close.shift(1) >= sar.shift(1))).astype(int) * (-1)

        labels = buy + sell
        labels.name = 'bins'

        return labels