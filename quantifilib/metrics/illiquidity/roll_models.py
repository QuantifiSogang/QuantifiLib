import numpy as np
import pandas as pd

class RollModel:
    def __init__(self, close_prices : pd.Series, window : int = 20) -> None:
        self.close_prices = close_prices
        self.window = window
    def roll_measure(self) -> pd.Series :
        price_diff = self.close_prices.diff()
        price_diff_lag = price_diff.shift(1)
        return 2 * np.sqrt(abs(price_diff.rolling(window = self.window).cov(price_diff_lag)))
    def roll_impact(self, dollar_volume : pd.Series) -> pd.Series :
        roll_measure = self.roll_measure()
        return roll_measure / dollar_volume
def roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))
def roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    roll_measure_ = roll_measure(close_prices, window)
    return roll_measure_ / dollar_volume