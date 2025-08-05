import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Tuple, Optional, Union

from .base_generator import BaseGenerator

class ARCHGenerator(BaseGenerator):
    """
    ARCH(p) time-series generator.

    Parameters (via from_params)
    ----------------------------
    params : dict
        {
          'omega': float (>0),
          'alpha': tuple[float, ...]  # length p
        }
    u : str
        Innovation distribution for z_t: 'normal' | 'student_t' | 'uniform'
    u_params : tuple
        Parameters for z_t.
        - 'normal': (mu, sigma)  [default (0.0, 1.0)]
        - 'student_t': (df,)     [df>2; standardized to unit variance]
        - 'uniform': ()          [uses U(-sqrt(3), sqrt(3)) -> unit variance]
    Notes
    -----
    • Stationarity / finite variance requires sum(alpha) < 1, then
      long-run Var(y_t) = omega / (1 - sum(alpha)).
    """

    def __init__(self, omega: float, alpha: np.ndarray, noise_fn: Callable[[int], np.ndarray]):
        super().__init__()
        self.omega = float(omega)
        self.alpha = np.array(alpha, dtype=float)
        self.noise_fn = noise_fn
        if self.omega <= 0:
            raise ValueError("omega must be > 0.")
        if np.any(self.alpha < 0):
            raise ValueError("All alpha_i must be >= 0 for ARCH.")
        if self.alpha.sum() >= 1:
            print("[WARN] sum(alpha) >= 1; unconditional variance may not exist and series can explode.")

    @classmethod
    def from_params(
        cls,
        params: dict,
        u: str = 'normal',
        u_params: Optional[Tuple[float, ...]] = None
    ) -> 'ARCHGenerator':
        omega = params.get('omega', None)
        alpha = params.get('alpha', ())
        if omega is None:
            raise ValueError("params must include 'omega'.")

        if u == 'normal':
            mu, sigma = (0.0, 1.0) if not u_params else u_params
            noise_fn = lambda n: np.random.normal(loc=mu, scale=sigma, size=n)
        elif u == 'student_t':
            df = 8.0 if not u_params else float(u_params[0])
            scale = np.sqrt((df - 2.0) / df)
            noise_fn = lambda n: np.random.standard_t(df, size=n) * scale
        elif u == 'uniform':
            a = -np.sqrt(3.0);
            b = np.sqrt(3.0)
            noise_fn = lambda n: np.random.uniform(low=a, high=b, size=n)
        else:
            raise ValueError("u must be one of {'normal','student_t','uniform'}.")

        return cls(omega=omega, alpha=np.array(alpha, dtype=float), noise_fn=noise_fn)

    def generate_simulation(
            self,
            n: int = 252,
            burn_in: int = 300,
            n_series: int = 1
        ) -> pd.DataFrame:
        """
        Simulate ARCH(p) series.

        Returns
        -------
        pd.DataFrame
            shape (n, n_series), business-day datetime index, columns 'series_i'.
        """
        p = len(self.alpha)
        total = n + burn_in
        out = np.zeros((n, n_series))

        var_uncond = self.omega / max(1e-12, (1.0 - self.alpha.sum())) if self.alpha.sum() < 1 else self.omega

        for s in range(n_series):
            y = np.zeros(total)
            sig2 = np.ones(total) * var_uncond
            z = self.noise_fn(total)

            for t in range(total):
                s2 = self.omega
                for i in range(1, p + 1):
                    if t - i >= 0:
                        s2 += self.alpha[i - 1] * (y[t - i] ** 2)
                sig2[t] = max(s2, 1e-12)
                y[t] = np.sqrt(sig2[t]) * z[t]

            out[:, s] = y[burn_in:]

        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        cols = [f"series_{i}" for i in range(n_series)]
        return pd.DataFrame(out, index=index, columns=cols)

    def get_conditional_volatility(
            self,
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            start_var: Optional[float] = None,
            ret: str = "sigma"
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Compute conditional volatility path given returns under ARCH(p).

        Parameters
        ----------
        y : Series | DataFrame | np.ndarray
            Observed (or simulated) returns. If 2D, each column is treated as an independent series.
        start_var : float, optional
            Initial variance. If None, uses the unconditional variance when sum(alpha) < 1, else omega.
        ret : {"sigma", "sig2"}
            Return conditional standard deviation ("sigma") or variance ("sig2").

        Returns
        -------
        Same type/shape as input y, containing sigma_t (or sigma_t^2).
        """
        was_series = isinstance(y, pd.Series)
        if isinstance(y, pd.Series):
            idx = y.index
            Y = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            idx = y.index
            Y = y
        else:
            Y = np.atleast_2d(y)
            idx = None
            if Y.shape[0] == 1 and y.ndim == 1:
                Y = Y.T  # (n,) -> (n,1)

        n, k = (Y.shape[0], 1) if isinstance(Y, pd.Series) else Y.shape
        p = len(self.alpha)

        ab_sum = self.alpha.sum()
        uncond = (self.omega / (1.0 - ab_sum)) if ab_sum < 1.0 else self.omega
        if start_var is None:
            start_var = uncond

        sig2 = np.zeros((n, k)) + start_var
        Yvals = Y.values if isinstance(Y, pd.DataFrame) else (Y.values if isinstance(Y, pd.Series) else Y)

        for j in range(k):
            for t in range(n):
                s2 = self.omega
                for i in range(1, p + 1):
                    if t - i >= 0:
                        s2 += self.alpha[i - 1] * (Yvals[t - i, j] ** 2)
                sig2[t, j] = max(s2, 1e-12)

        if ret == "sig2":
            out = sig2
        elif ret == "sigma":
            out = np.sqrt(sig2)
        else:
            raise ValueError("ret must be 'sigma' or 'sig2'.")

        if idx is not None:
            cols = Y.columns if isinstance(Y, pd.DataFrame) else [Y.name or "series_0"]
            out_df = pd.DataFrame(out, index=idx, columns=cols)
            return out_df.iloc[:, 0] if was_series else out_df
        return out.squeeze() if out.shape[1] == 1 else out

class GARCHGenerator(BaseGenerator):
    """
    GARCH(p, q) time-series generator.

    Model
    -----
    y_t = sigma_t * z_t,      z_t ~ i.i.d.
    sigma_t^2 = omega + sum_{i=1}^p alpha_i * y_{t-i}^2 + sum_{j=1}^q beta_j * sigma_{t-j}^2

    Parameters (via from_params)
    ----------------------------
    params : dict
        {
          'omega': float (>0),
          'alpha': tuple[float, ...],   # ARCH coefficients, length p (p >= 0)
          'beta' : tuple[float, ...],   # GARCH coefficients, length q (q >= 0)
        }
        If 'beta' is empty, the process reduces to ARCH(p).

    u : str
        Innovation distribution for z_t. One of:
        - 'normal'    : Gaussian N(mu, sigma^2)
        - 'student_t' : Student-t with df; standardized to unit variance
        - 'uniform'   : U(mu - sqrt(3)*sigma, mu + sqrt(3)*sigma) -> unit variance when (mu=0, sigma=1)

    u_params : tuple
        Parameters for the innovation distribution.
        - normal    : (mu, sigma).  **Recommended** (0.0, 1.0) so that volatility scaling is driven by sigma_t.
        - student_t : (df,) with df > 2. Internally rescaled to unit variance.
        - uniform   : (mu, sigma) interpreted as mean/STD surrogate; interval is set to keep variance = sigma^2.

    Notes
    -----
    • Weak stationarity (finite unconditional variance) typically requires sum(alpha) + sum(beta) < 1.
    • The unconditional variance, if it exists, is omega / (1 - sum(alpha) - sum(beta)).
    """

    def __init__(self, omega: float, alpha: np.ndarray, beta: np.ndarray,
                 noise_fn: Callable[[int], np.ndarray]):
        super().__init__()
        self.omega = float(omega)
        self.alpha = np.array(alpha, dtype=float)
        self.beta  = np.array(beta,  dtype=float)
        self.noise_fn = noise_fn

        if self.omega <= 0:
            raise ValueError("omega must be > 0.")
        if (self.alpha < 0).any() or (self.beta < 0).any():
            raise ValueError("All alpha_i and beta_j must be >= 0.")
        s = self.alpha.sum() + self.beta.sum()
        if s >= 1:
            print(f"[WARN] sum(alpha)+sum(beta)={s:.3f} >= 1; "
                  "unconditional variance may not exist and the process can be explosive.")

    @classmethod
    def from_params(
            cls,
            params: dict,
            u: str = "normal",
            u_params: Optional[Tuple[float, ...]] = None
    ) -> "GARCHGenerator":
        """
        Construct a GARCHGenerator from parameter dictionaries and a chosen innovation distribution.
        """
        omega = params.get("omega", None)
        alpha = params.get("alpha", ())
        beta = params.get("beta", ())
        if omega is None:
            raise ValueError("params must include 'omega'.")

        alpha = np.array(alpha, dtype=float)
        beta = np.array(beta, dtype=float)

        if u == "normal":
            mu, sigma = (0.0, 1.0) if not u_params else u_params
            noise_fn = lambda n: np.random.normal(loc=mu, scale=sigma, size=n)
        elif u == "student_t":
            df = 8.0 if not u_params else float(u_params[0])
            if df <= 2:
                raise ValueError("student_t requires df > 2.")
            scale = np.sqrt((df - 2.0) / df)
            noise_fn = lambda n: np.random.standard_t(df, size=n) * scale
        elif u == "uniform":
            mu, sigma = (0.0, 1.0) if not u_params else u_params
            a = mu - np.sqrt(3.0) * sigma
            b = mu + np.sqrt(3.0) * sigma
            noise_fn = lambda n: np.random.uniform(low=a, high=b, size=n)
        else:
            raise ValueError("u must be one of {'normal','student_t','uniform'}.")

        return cls(omega=omega, alpha=alpha, beta=beta, noise_fn=noise_fn)

    def generate_simulation(
        self,
        n: int = 252,
        burn_in: int = 300,
        n_series: int = 1,
        start_var: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Simulate a GARCH(p, q) process and return a DataFrame with a business-day index.

        Parameters
        ----------
        n : int
            Number of observations to return (after burn-in).
        burn_in : int
            Warm-up length to mitigate initialization effects.
        n_series : int
            Number of independent series to simulate.
        start_var : Optional[float]
            Initial variance for sigma^2. If None, uses unconditional variance
            when sum(alpha)+sum(beta) < 1, otherwise uses omega.

        Returns
        -------
        pd.DataFrame
            Shape (n, n_series) with columns 'series_0', 'series_1', ...
        """
        p, q = len(self.alpha), len(self.beta)
        total = n + burn_in

        ab_sum = self.alpha.sum() + self.beta.sum()
        uncond_var = (self.omega / (1.0 - ab_sum)) if ab_sum < 1.0 else self.omega
        if start_var is None:
            start_var = uncond_var

        out = np.zeros((n, n_series))

        for s in range(n_series):
            y = np.zeros(total)
            sig2 = np.ones(total) * start_var
            z = self.noise_fn(total)

            for t in range(total):
                s2 = self.omega
                for i in range(1, p + 1):
                    if t - i >= 0:
                        s2 += self.alpha[i - 1] * (y[t - i] ** 2)
                for j in range(1, q + 1):
                    if t - j >= 0:
                        s2 += self.beta[j - 1] * sig2[t - j]

                sig2[t] = max(s2, 1e-12)
                y[t] = np.sqrt(sig2[t]) * z[t]

            out[:, s] = y[burn_in:]

        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        cols = [f"series_{i}" for i in range(n_series)]
        return pd.DataFrame(out, index=index, columns=cols)

    def get_conditional_volatility(
            self,
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            start_var: Optional[float] = None,
            ret: str = "sigma"
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Compute conditional volatility path given returns under GARCH(p, q).

        Parameters
        ----------
        y : Series | DataFrame | np.ndarray
            Observed (or simulated) returns. If 2D, each column is treated as an independent series.
        start_var : float, optional
            Initial variance. If None, uses the unconditional variance when sum(alpha)+sum(beta) < 1, else omega.
        ret : {"sigma", "sig2"}
            Return conditional standard deviation ("sigma") or variance ("sig2").

        Returns
        -------
        Same type/shape as input y, containing sigma_t (or sigma_t^2).
        """
        was_series = isinstance(y, pd.Series)
        if isinstance(y, pd.Series):
            idx = y.index
            Y = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            idx = y.index
            Y = y
        else:
            Y = np.atleast_2d(y)
            idx = None
            if Y.shape[0] == 1 and y.ndim == 1:
                Y = Y.T

        n, k = (Y.shape[0], 1) if isinstance(Y, pd.Series) else Y.shape
        p, q = len(self.alpha), len(self.beta)

        ab_sum = self.alpha.sum() + self.beta.sum()
        uncond = (self.omega / (1.0 - ab_sum)) if ab_sum < 1.0 else self.omega
        if start_var is None:
            start_var = uncond

        sig2 = np.zeros((n, k)) + start_var
        Yvals = Y.values if isinstance(Y, pd.DataFrame) else (Y.values if isinstance(Y, pd.Series) else Y)

        for j in range(k):
            for t in range(n):
                s2 = self.omega
                # ARCH terms
                for i in range(1, p + 1):
                    if t - i >= 0:
                        s2 += self.alpha[i - 1] * (Yvals[t - i, j] ** 2)
                # GARCH terms
                for h in range(1, q + 1):
                    if t - h >= 0:
                        s2 += self.beta[h - 1] * sig2[t - h, j]
                sig2[t, j] = max(s2, 1e-12)

        if ret == "sig2":
            out = sig2
        elif ret == "sigma":
            out = np.sqrt(sig2)
        else:
            raise ValueError("ret must be 'sigma' or 'sig2'.")

        if idx is not None:
            cols = Y.columns if isinstance(Y, pd.DataFrame) else [Y.name or "series_0"]
            out_df = pd.DataFrame(out, index=idx, columns=cols)
            return out_df.iloc[:, 0] if was_series else out_df
        return out.squeeze() if out.shape[1] == 1 else out