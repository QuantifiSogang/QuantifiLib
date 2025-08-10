from typing import Literal
import numpy as np
import pandas as pd
from pandas import DataFrame

_FREQ2AF = {
    "D" : 252,
    "W" : 52,
    "M" : 12,
    "Q" : 4,
    "A" : 1
}

def _annualization_factor(freq: Literal["D", "W", "M", "Q", "A"]) -> int:
    """Return the annualization factor given the data frequency."""
    if freq not in _FREQ2AF:
        raise ValueError(f"Unsupported freq '{freq}'. Use one of {list(_FREQ2AF)}.")
    return _FREQ2AF[freq]

def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the input is a DataFrame with no duplicate columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a DataFrame, not {type(df)}")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate asset columns.")
    return df

def covariance_to_correlation(covariance:pd.DataFrame) -> pd.DataFrame :
    """Convert a covariance matrix to a correlation matrix."""
    covariance = _ensure_df(covariance)
    std = np.sqrt(np.diag(covariance.values))
    inv = np.where(std > 0, 1.0/std, 0.0)
    d_inv = np.diag(inv)
    corr = d_inv @ covariance @ d_inv
    out = pd.DataFrame(corr, index = covariance.index, columns = covariance.columns)
    return out.clip(-1.0,1.0)

def correlation_to_covariace(corr:pd.DataFrame, vol: pd.Series) -> pd.DataFrame :
    """Convert correlation matrix and vol vector to covariance matrix."""
    corr = _ensure_df(corr)
    vol = vol.reindex(corr.index).astype(float)
    D = np.diag(vol.values)
    cov = D @ corr.values @ D
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)

def regularize_min_eig(covariance: pd.DataFrame, min_eig: float = 1e-10) -> pd.DataFrame:
    """
    Ensure positive semi-definiteness by lifting the minimum eigenvalue.

    Adds a ridge term if the minimum eigenvalue is below `min_eig`.
    """
    covariance = _ensure_df(covariance)
    vals, vecs = np.linalg.eigh(covariance.values)
    lift = max(0.0, float(min_eig - vals.min()))
    if lift > 0:
        covariance = covariance + np.eye(covariance.shape[0]) * lift
    return covariance