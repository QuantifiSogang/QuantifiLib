import numpy as np
import pandas as pd

class CorwinSchultz :
    def __init__(self, high : pd.Series, low : pd.Series) -> None:
        self.high = high
        self.low = low
    def beta(self, window : int) -> pd.Series:
        ret = np.log(self.high / self.low)
        high_low_ret = ret ** 2
        beta = high_low_ret.rolling(window=2).sum()
        beta = beta.rolling(window=window).mean()
        return beta
    def gamma(self) -> pd.Series:
        high_max = self.high.rolling(window = 2).max()
        low_min = self.low.rolling(window = 2).min()
        gamma = np.log(high_max / low_min) ** 2
        return gamma
    def alpha(self, window : int) -> pd.Series:
        den = 3 - 2 * 2 ** .5
        alpha = (2 ** .5 - 1) * (self.beta(window = window) ** .5) / den
        alpha -= (self.gamma() / den) ** .5
        alpha[alpha < 0] = 0
        return alpha
    def corwin_schultz_estimator(self, window : int = 20) -> pd.Series :
        alpha_ = self.alpha(window = window)
        spread = 2 * (np.exp(alpha_) - 1) / (1 + np.exp(alpha_))
        start_time = pd.Series(self.high.index[0:spread.shape[0]], index=spread.index)
        spread = pd.concat([spread, start_time], axis=1)
        spread.columns = ['Spread', 'Start_Time']
        return spread.Spread
    def becker_parkinson_vol(self, window: int = 20) -> pd.Series:
        Beta = self.beta(window = window)
        Gamma = self.gamma()
        k2 = (8 / np.pi) ** 0.5
        den = 3 - 2 * 2 ** .5
        sigma = (2 ** -0.5 - 1) * Beta ** 0.5 / (k2 * den)
        sigma += (Gamma / (k2 ** 2 * den)) ** 0.5
        sigma[sigma < 0] = 0
        return sigma

def _beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta

def _gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma

def _alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0
    return alpha

def corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    beta_ = _beta(high, low, window)
    gamma_ = _gamma(high, low)
    alpha_ = _alpha(beta_, gamma_)
    spread = 2 * (np.exp(alpha_) - 1) / (1 + np.exp(alpha_))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread

def becker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    Beta = _beta(high, low, window)
    Gamma = _gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * Beta ** 0.5 / (k2 * den)
    sigma += (Gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma