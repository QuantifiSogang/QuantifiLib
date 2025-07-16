import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class BarbasedLambda :
    def __init__(self, close : pd.Series, volume : pd.Series,
                 dollar_volume: pd.Series, window : int = 20):
        self.close = close
        self.volume = volume
        self.window = window
        self.dollar_volume = dollar_volume
    def kyle(self) -> pd.Series :
        close_diff = self.close.diff()
        close_diff_sign = np.sign(close_diff)
        close_diff_sign.replace(0, method='pad', inplace=True)
        volume_mult_trade_signs = self.volume * close_diff_sign
        return (close_diff / volume_mult_trade_signs).rolling(window = self.window).mean()
    def amihud(self) -> pd.Series :
        returns_abs = np.log(self.close / self.close.shift(1)).abs()
        return (returns_abs / self.dollar_volume).rolling(window = self.window).mean()
    def hasbrouck(self) -> pd.Series :
        log_ret = np.log(self.close / self.close.shift(1))
        log_ret_sign = np.sign(log_ret).replace(0, method='pad')
        signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(self.dollar_volume)
        return (log_ret / signed_dollar_volume_sqrt).rolling(window = self.window).mean()

class TradebasedLambda :
    def __init__(self, price_diff : list, log_ret : list,
                 volume : list, dollar_volume : list, aggressor_flags : list) -> float:
        self.price_diff = price_diff
        self.log_ret = log_ret
        self.volume = volume
        self.dollar_volume = dollar_volume
        self.aggressor_flags = aggressor_flags
    def kyle(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        signed_volume = np.array(self.volume) * np.array(self.aggressor_flags)
        X = np.array(signed_volume).reshape(-1, 1)
        y = np.array(self.price_diff)
        model.fit(X, y)
        return model.coef_[0]
    def amihud(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        X = np.array(self.dollar_volume).reshape(-1, 1)
        y = np.abs(np.array(self.log_ret))
        model.fit(X, y)
        return model.coef_[0]
    def hasbrouck(self):
        model = LinearRegression(fit_intercept=False, copy_X=False)
        X = (np.sqrt(np.array(self.dollar_volume)) * np.array(self.aggressor_flags)).reshape(-1, 1)
        y = np.abs(np.array(self.log_ret))
        model.fit(X, y)
        return model.coef_[0]

def bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    close_diff = close.diff()
    close_diff_sign = np.sign(close_diff)
    close_diff_sign.replace(0, method='pad', inplace=True)
    volume_mult_trade_signs = volume * close_diff_sign
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()
def bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    returns_abs = np.log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()
def bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = np.sign(log_ret).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()
def trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    model.fit(X, y)
    return model.coef_[0]
def trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]
def trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> float:
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]