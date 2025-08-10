from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from ..utils import _annualization_factor, _ensure_df
import numpy as np
import pandas as pd

@dataclass
class ReturnSnapshot :
    """
    Data container for expected return and covariance snapshot.

    Attributes
    ----------
    mu : Series
        Expected returns at the snapshot date.
    cov : DataFrame, optional
        Covariance matrix at the snapshot date.
    """
    mu : pd.Series
    cov : Optional[pd.DataFrame] = None

def to_returns(
        prices : pd.DataFrame,
        method : Literal['log','simple'] = 'simple',
        period : int = 1
    ) -> pd.DataFrame:
    """
    Convert price series into returns.

    Parameters
    ----------
    prices : DataFrame
        Price series with shape (T x N).
    method : {"log", "simple"}, default="log"
        - 'log': r_t = log(P_t / P_{t-1})
        - 'simple': r_t = P_t / P_{t-1} - 1
    dropna : bool, default=True
        Whether to drop NaN rows after calculation.

    Returns
    -------
    DataFrame
        Return series with the same shape as `prices` (minus one row).
    """
    px = _ensure_df(prices).sort_index()
    if method == "log":
        rets = np.log(px / px.shift(period))
    elif method == "simple":
        rets = px.pct_change(period)
    else :
        raise ValueError(f"Unsupported method '{method}'. method must be 'log' or 'simple'")

    return rets

def aggregate_returns(
        returns: pd.DataFrame,
        to: Literal["D", "W", "M", "Q", "A"],
        method: Literal["log", "simple"] = "simple"
    ) -> pd.DataFrame:
        """
        Aggregate returns to a higher frequency.

        Parameters
        ----------
        returns : DataFrame
            Return series.
        to : {"D", "W", "M", "Q", "A"}
            Target frequency.
        from_freq : {"D", "W", "M", "Q", "A"}, optional
            Original frequency (not used in this basic implementation).
        method : {"log", "simple"}, default="log"
            Aggregation method for returns.

        Returns
        -------
        DataFrame
            Aggregated return series.
        """
        r = _ensure_df(returns).sort_index()
        if method == "log":
            agg = r.resample(to).sum()
        else:
            agg = (1.0 + r).resample(to).prod() - 1.0
        return agg.dropna(how="all")

def get_simple_mean(
        returns : pd.DataFrame,
        freq: Literal["D","W","M","Q","A"] = "D",
        annualize : bool = True,
    ) -> pd.Series :
    """
    Sample mean estimator for expected returns.

    Parameters
    ----------
    returns : DataFrame
        Return series.
    freq : {"D", "W", "M", "Q", "A"}, default="D"
        Frequency of the data.
    annualize : bool, default=True
        Whether to annualize the mean returns.

    Returns
    -------
    Series
        Expected returns per asset.
    """
    r = _ensure_df(returns)
    mu = r.mean()
    if annualize :
        af = _annualization_factor(freq)
        mu = mu * af
    return mu

def get_exponential_weighted_mean(
        returns : pd.DataFrame,
        window : float = 63,
        freq : Literal["D","W","M","Q","A"] = "D",
        annualize : bool = True
    ) -> pd.Series :
    """
    Exponentially weighted mean estimator.

    Parameters
    ----------
    returns : DataFrame
        Return series.
    halflife : float, default=63
        Halflife parameter for exponential weighting.
    freq : {"D", "W", "M", "Q", "A"}, default="D"
        Data frequency.
    annualize : bool, default=True
        Whether to annualize the mean returns.

    Returns
    -------
    Series
        Expected returns per asset.
    """
    r = _ensure_df(returns).dropna(how="all")
    mu = r.ewm(halflife=window, adjust=False).mean().iloc[-1]
    if annualize:
        af = _annualization_factor(freq)
        mu = mu * af
    return mu

def winsorize(s : pd.Series, p : float = 0.01) -> pd.Series :
    """
    Clip extreme values at the given percentile p.
    :param s:
    :param p:
    :return:
    """
    low, high = s.quantile(p), s.quantile(1 - p)
    return s.clip(low, high)

def robust_mean_huber(
        returns : pd.DataFrame,
        c : float = 1.345,
        max_iter: int = 100,
        tol: float = 1e-8,
        freq: Literal["D", "W", "M", "Q", "A"] = "D",
        annualize : bool = True
    ) -> pd.Series :
    """
    Robust mean estimator using the Huber M-estimator.

    Parameters
    ----------
    returns : DataFrame
        Return series.
    c : float, default=1.345
        Huber tuning constant.
    max_iter : int, default=100
        Maximum number of iterations for convergence.
    tol : float, default=1e-8
        Convergence tolerance.
    freq : {"D", "W", "M", "Q", "A"}, default="D"
        Data frequency.
    annualize : bool, default=True
        Whether to annualize the mean returns.

    Returns
    -------
    Series
        Robustly estimated expected returns per asset.
    """
    r = _ensure_df(returns)
    out = {}
    for col in r :
        x = r[col].dropna().values
        if x.size == 0:
            out[col] = np.nan
            continue
        m = np.median(x)
        s = 1.4826 * np.median(np.abs(x - m)) + 1e-12
        mu = float(np.mean(x))
        for _ in range(max_iter) :
            z = (x - mu) / s
            w = np.clip(c / np.maximum(np.abs(z), 1e-12), 0, 1)
            mu_new = np.sum(w * x) / np.sum(w)
            if abs(mu_new - mu) < tol:
                break
            mu = mu_new
        out[col] = mu
    mu = pd.Series(out, index = r.columns)
    if annualize :
        mu = mu * _annualization_factor(freq)
    return mu

def james_stein_mean(
        returns : pd.DataFrame,
        freq : Literal["D","W","M","Q","A"] = "D",
        annualize : bool = True,
        grand_target : Literal['cross_section', 'zero'] = 'cross_section',
    ) -> pd.Series :
    """
    James–Stein shrinkage estimator for expected returns.

    Parameters
    ----------
    returns : DataFrame
        Return series.
    freq : {"D", "W", "M", "Q", "A"}, default="D"
        Data frequency.
    annualize : bool, default=True
        Whether to annualize the mean returns.
    grand_target : {"cross_section", "zero"}, default="cross_section"
        Target mean to shrink towards.

    Returns
    -------
    Series
        Shrinkage-adjusted expected returns per asset.
    """
    r = _ensure_df(returns)
    T = r.shape[0]
    mu_hat = r.mean()
    if grand_target == "cross_section":
        theta = mu_hat.mean()
    elif grand_target == "zero":
        theta = 0.0
    else:
        raise ValueError("grand_target must be 'cross_section' or 'zero'")
    var_hat = r.var(ddof=1) / max(T, 1)
    numer = (len(mu_hat) - 2) * var_hat
    denom = (mu_hat - theta).pow(2).sum() + 1e-16
    b = float(numer.mean() / denom)
    b = np.clip(b, 0.0, 1.0)
    mu_js = theta + (1 - b) * (mu_hat - theta)
    if annualize:
        mu_js = mu_js * _annualization_factor(freq)
    return mu_js

def rolling_mean(
        returns : pd.DataFrame,
        window : int,
        min_periods : Optional[int] = None
    ) -> pd.DataFrame :
    """
    Rolling mean for time-varying expected returns.

    Parameters
    ----------
    returns : DataFrame
        Return series.
    window : int
        Rolling window size.
    min_periods : int, optional
        Minimum periods required for a value.

    Returns
    -------
    DataFrame
        Rolling mean returns.
    """
    r = _ensure_df(returns)
    return r.rolling(window = window, min_periods = min_periods or window).mean()