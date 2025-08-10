from __future__ import annotations
from typing import Optional, Tuple, Union, Literal
from ..utils import _FREQ2AF, _ensure_df, _annualization_factor
from ..utils import covariance_to_correlation, correlation_to_covariace
import numpy as np
import pandas as pd

def get_sample_covariance(
        returns : pd.DataFrame,
        ddof: int = 1,
        annualize: bool = False,
        freq: Literal['D','W','M','Q','A'] = 'D'
    ) -> pd.DataFrame:
    """
    Sample covariance estimator.

    Parameters
    ----------
    returns : DataFrame (T x N)
        Return series.
    ddof : int, default=1
        Delta degrees of freedom.
    annualize : bool, default=False
        Whether to annualize covariance.
    freq : {"D","W","M","Q","A"}, default="D"
        Data frequency, used if `annualize=True`.

    Returns
    -------
    DataFrame (N x N)
        Sample covariance matrix.
    """
    r = _ensure_df(returns)
    s = r.cov(ddof=ddof)
    if annualize:
        af = _annualization_factor(freq)
        s = s * af
    return s

def get_exponential_weighted_covariance(
        returns: pd.DataFrame,
        halflife: float = 63,
        adjust: bool = False,
        min_periods: Optional[int] = None,
        annualize: bool = False,
        freq: Literal["D", "W", "M", "Q", "A"] = "D",
    ) -> pd.DataFrame :
    """
    Exponentially weighted covariance using pandas EWM.

    Notes
    -----
    This uses pandas' `ewm(...).cov()` path. For a DataFrame input, the
    result is a DataFrame with a MultiIndex on rows (time, column). We
    pick the last timestamp's block to form the N x N covariance.

    Parameters
    ----------
    returns : DataFrame (T x N)
    halflife : float, default=63
    adjust : bool, default=False
    min_periods : int, optional
    annualize : bool, default=False
    freq : {"D","W","M","Q","A"}, default="D"

    Returns
    -------
    DataFrame (N x N)
        EWM covariance at the last timestamp.
    """
    r = _ensure_df(returns)
    ewm_cov = r.ewm(halflife=halflife, adjust=adjust, min_periods=min_periods).cov()

    # Slice last timestamp block -> index is assets, columns are assets
    cov_last = ewm_cov.xs(r.index[-1], level=0)
    cov_last = cov_last.reindex(index=r.columns, columns=r.columns)
    if annualize:
        af = _annualization_factor(freq)
        cov_last = cov_last * af
    return cov_last

def linear_shrinkage_to_target(
        S: pd.DataFrame,
        T: pd.DataFrame,
        alpha: float,
    ) -> pd.DataFrame:
    """
    Generic linear shrinkage: S* (alpha) + (1-alpha) * T.

    Parameters
    ----------
    S : DataFrame
        Sample covariance.
    T : DataFrame
        Target covariance.
    alpha : float
        Shrinkage intensity in [0, 1]. alpha=1 keeps S; alpha=0 returns T.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    S = _ensure_df(S)
    T = T.reindex_like(S)
    return alpha * S + (1.0 - alpha) * T

def diagonal_target(S: pd.DataFrame) -> pd.DataFrame:
    """Diagonal target with original variances preserved."""
    S = _ensure_df(S)
    D = np.diag(np.diag(S.values))
    return pd.DataFrame(D, index=S.index, columns=S.columns)


def identity_target(S: pd.DataFrame) -> pd.DataFrame:
    """Identity target scaled by average variance."""
    S = _ensure_df(S)
    avg_var = float(np.trace(S.values) / S.shape[0])
    I = np.eye(S.shape[0]) * avg_var
    return pd.DataFrame(I, index=S.index, columns=S.columns)


def constant_correlation_target(S: pd.DataFrame) -> pd.DataFrame:
    """
    Constant-correlation target per Elton–Gruber style:
    keep individual variances; set all off-diagonal correlations to their average.
    """
    S = _ensure_df(S)
    std = np.sqrt(np.diag(S.values))
    # Compute average correlation (off-diagonal)
    with np.errstate(invalid="ignore", divide="ignore"):
        Corr = covariance_to_correlation(S).values
    n = S.shape[0]
    if n <= 1:
        return S.copy()
    mask = ~np.eye(n, dtype=bool)
    rho_bar = Corr[mask].mean()
    # Build target
    T = np.outer(std, std) * rho_bar
    np.fill_diagonal(T, std**2)
    return pd.DataFrame(T, index=S.index, columns=S.columns)

def ledoit_wolf(
    returns: pd.DataFrame,
    target: Literal["identity", "diagonal"] = "identity",
    annualize: bool = False,
    freq: Literal["D", "W", "M", "Q", "A"] = "D",
) -> Tuple[pd.DataFrame, float]:
    """
    Ledoit–Wolf shrinkage estimator (optimal alpha) to a simple target.

    Implementation follows Ledoit & Wolf (2004) for covariance matrices.
    We provide two typical targets:
      - "identity": avg variance * I
      - "diagonal": diag(S)

    Returns
    -------
    (cov_shrunk, alpha_opt)
    """
    X = _ensure_df(returns).values  # T x N
    Tn, N = X.shape
    if Tn <= 1:
        raise ValueError("Not enough observations for covariance estimation.")
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (Tn - 1)  # sample covariance (ddof=1)

    if target == "identity":
        F = np.eye(N) * (np.trace(S) / N)
    elif target == "diagonal":
        F = np.diag(np.diag(S))
    else:
        raise ValueError("target must be 'identity' or 'diagonal'.")

    # Compute pi-hat (variance of sample cov elements)
    X2 = Xc**2
    pi_hat = 0.0
    for t in range(Tn):
        xt = Xc[t][:, None]
        Pi_t = (xt @ xt.T) - S
        pi_hat += (Pi_t**2).sum()
    pi_hat /= Tn

    # Compute gamma-hat (distance between S and target)
    gamma_hat = ((S - F) ** 2).sum()

    # Compute rho-hat (covariance between S and F)
    # For diagonal target, Ledoit–Wolf uses a closed-form; for identity target, similar form:
    if target == "diagonal":
        # See Ledoit & Wolf (2004), section for diagonal target
        var_s = ((S - np.diag(np.diag(S))) ** 2).sum()
        rho_hat = pi_hat - var_s
    else:  # identity
        # As a simple and common approximation:
        rho_hat = pi_hat

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 1.0
    alpha = max(0.0, min(1.0, kappa / Tn))

    S_hat = alpha * S + (1 - alpha) * F
    cov = pd.DataFrame(S_hat, index=returns.columns, columns=returns.columns)
    if annualize:
        af = _annualization_factor(freq)
        cov = cov * af
    return cov, float(alpha)


def oracle_approx_shrinkage(
    returns: pd.DataFrame,
    annualize: bool = False,
    freq: Literal["D", "W", "M", "Q", "A"] = "D",
) -> Tuple[pd.DataFrame, float]:
    """
    Oracle Approximating Shrinkage (OAS) estimator to identity target.

    Chen, Wiesel, and Hero (2010). This is a well-known closed-form
    shrinkage intensity to an identity-scaled target.

    Returns
    -------
    (cov_shrunk, alpha_opt)
    """
    X = _ensure_df(returns).values  # T x N
    Tn, N = X.shape
    if Tn <= 1:
        raise ValueError("Not enough observations for covariance estimation.")
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (Tn - 1)

    mu = np.trace(S) / N
    a2 = (S**2).sum() / N
    # OAS shrinkage intensity (to μI)
    num = (1 - 2 / N) * a2 + (mu**2)
    den = (Tn + 1 - 2 / N) * (a2 - (mu**2))
    if den == 0:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, num / den))

    F = np.eye(N) * mu
    S_hat = alpha * F + (1 - alpha) * S
    cov = pd.DataFrame(S_hat, index=returns.columns, columns=returns.columns)
    if annualize:
        af = _annualization_factor(freq)
        cov = cov * af
    return cov, float(alpha)

def has_covariance(
    returns: pd.DataFrame,
    lags: int,
    kernel: Literal["bartlett"] = "bartlett",
    center: bool = True,
    annualize: bool = False,
    freq: Literal["D", "W", "M", "Q", "A"] = "D",
) -> pd.DataFrame:
    """
    Heteroskedasticity- and Autocorrelation-Consistent (HAC) covariance.
    Newey–West with Bartlett kernel.

    Σ = Γ(0) + Σ_{ℓ=1..L} w_ℓ ( Γ(ℓ) + Γ(ℓ)^T ),
    where Γ(ℓ) = E[(x_t - μ)(x_{t-ℓ} - μ)^T], and w_ℓ = 1 - ℓ/(L+1).

    Parameters
    ----------
    returns : DataFrame (T x N)
    lags : int
        Maximum lag L.
    kernel : {"bartlett"}, default="bartlett"
    center : bool, default=True
        If True, subtract sample mean before computing auto-covariances.
    annualize : bool, default=False
    freq : {"D","W","M","Q","A"}, default="D"

    Returns
    -------
    DataFrame (N x N)
        HAC covariance matrix.
    """
    X = _ensure_df(returns).values  # T x N
    Tn, N = X.shape
    if Tn <= 1:
        raise ValueError("Not enough observations for covariance estimation.")
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # Γ(0)
    Gamma0 = (X.T @ X) / Tn

    # Weighted sum of lagged auto-covariances
    HAC = Gamma0.copy()
    for ell in range(1, lags + 1):
        if kernel == "bartlett":
            w = 1.0 - ell / (lags + 1.0)
        else:
            raise ValueError("Only 'bartlett' kernel is implemented.")
        X1 = X[ell:, :]
        X2 = X[:-ell, :]
        Gamma_ell = (X1.T @ X2) / Tn
        HAC += w * (Gamma_ell + Gamma_ell.T)

    cov = pd.DataFrame(HAC, index=returns.columns, columns=returns.columns)
    if annualize:
        af = _annualization_factor(freq)
        cov = cov * af
    return cov

def shrink_covariance(
        returns: pd.DataFrame,
        method: Literal["lw_identity", "lw_diagonal", "oas"] = "lw_identity",
        annualize: bool = False,
        freq: Literal["D", "W", "M", "Q", "A"] = "D",
    ) -> Tuple[pd.DataFrame, float]:
    """
    Convenience wrapper for common shrinkage estimators.

    Parameters
    ----------
    method : {"lw_identity", "lw_diagonal", "oas"}
        - lw_identity: Ledoit–Wolf to identity target
        - lw_diagonal: Ledoit–Wolf to diagonal target
        - oas: Oracle Approximating Shrinkage to identity

    Returns
    -------
    (cov, alpha)
        Covariance matrix and shrinkage intensity.
    """
    if method == "lw_identity":
        return ledoit_wolf(returns, target="identity", annualize=annualize, freq=freq)
    if method == "lw_diagonal":
        return ledoit_wolf(returns, target="diagonal", annualize=annualize, freq=freq)
    if method == "oas":
        return oracle_approx_shrinkage(returns, annualize=annualize, freq=freq)
    raise ValueError("Unknown method.")