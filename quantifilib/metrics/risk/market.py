import numpy as np
import pandas as pd

def daily_volatility(close: pd.Series, lookback: int = 100) -> pd.Series:
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1],
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    df0 = df0.ewm(span = lookback).std()
    return df0

def parkinson_volatility(high: pd.Series, low : pd.Series, window : int = 20) -> pd.Series :
    ret = np.log(high / low)
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window = window).mean())

def garman_klass_volatility(series, window=21):
    """
    Function to calculate Garman-Klass volatility
    """
    a = (np.log(series['High'] / series['Low']) ** 2).rolling(window=window).mean() * 0.5
    b = (2 * np.log(2) - 1) * (np.log(series['Close'] / series['Open']) ** 2).rolling(window=window).mean()
    return np.sqrt(a - b)


def rogers_satchell_volatility(series, window=21):
    """
    Function to calculate Rogers-Satchell volatility
    """
    a = (np.log(series['High'] / series['Close']) * np.log(series['High'] / series['Open'])).rolling(
        window=window).mean()
    b = (np.log(series['Low'] / series['Close']) * np.log(series['Low'] / series['Open'])).rolling(window=window).mean()
    return np.sqrt(a + b)

def yang_zhang_volatility(series, window=21):
    """
    Function to calculate Yang-Zhang volatility
    """
    a = (np.log(series['Open'] / series['Close'].shift(1))).rolling(window=window).mean()
    vol_open = ((np.log(series['Open'] / series['Close'].shift(1)) - a) ** 2).rolling(window=window).mean()
    b = (np.log(series['Close'] / series['Open'])).rolling(window=window).mean()
    vol_close = ((np.log(series['Close'] / series['Open']) - b) ** 2).rolling(window=window).mean()
    vol_rogers_satchell = rogers_satchell_volatility(series, window)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_volatility = np.sqrt(
        vol_open + k * vol_close + (1 - k) * (vol_rogers_satchell ** 2)
    )

    return yz_volatility

def intrinsic_entropy(series, total_volume, window=21):
    h_co = - (
            np.log(series['Open'] / series['Close'].shift(1)) *
            (series['Volume'] / total_volume) *
            np.log(series['Volume'].shift(1) / total_volume)
    ).rolling(window=window).mean()

    h_oc = - (
            np.log(series['Close'] / series['Open']) *
            (series['Volume'] / total_volume) *
            np.log(series['Volume'] / total_volume)
    ).rolling(window=window).mean()

    h_ohlc = - (
            (
                    (np.log(series['Open'] / series['High']) * np.log(series['High'] / series['Close'])) +
                    (np.log(series['Low'] / series['Open']) * np.log(series['Low'] / series['Close']))
            ) * (series['Volume'] / total_volume) * np.log(series['Volume'] / total_volume)
    ).rolling(window=window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    h = np.abs(h_co + k * h_oc + (1 - k) * h_ohlc)
    return h