import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.special import gamma

from .base_generator import BaseGenerator
from datetime import datetime
from typing import Callable, Tuple, Optional

class ARMAGenerator(BaseGenerator):
    """
    Generates Time Series Data using ARMA model.
    """
    def __init__(self, ar: np.ndarray, ma: np.ndarray, noise_fn: Callable[[int], np.ndarray]):
        super().__init__()
        self.ar = ar
        self.ma = ma
        self.noise_fn = noise_fn

    @classmethod
    def from_params(
            cls, params: dict,
            u: str = 'normal',
            u_params : Tuple = (0, 0.01),
        ) -> 'ARMAGenerator':
        """
        Create ARMA generator from parameters.

        Parameters
        ----------
        params : dict
            AR and MA parameters. e.g., {'p': [...], 'q': [...]}
        u : str
            Distribution type: 'normal' or 'uniform'
        mu : float
            Mean of the noise
        sigma : float
            Standard deviation of the noise

        Returns
        -------
        ARMAGenerator
        """
        mu, sigma = u_params
        ar = np.array(params.get('p', []))
        ma = np.array(params.get('q', []))

        if u == 'normal':
            noise_fn = lambda n: np.random.normal(loc=mu, scale=sigma, size=n)
        elif u == 'uniform':
            low = mu - np.sqrt(3) * sigma
            high = mu + np.sqrt(3) * sigma
            noise_fn = lambda n: np.random.uniform(low=low, high=high, size=n)
        else:
            raise ValueError(f"Unknown noise type: {u}")

        return cls(ar, ma, noise_fn)

    def generate_simulation(
            self, n: int = 252,
            burn_in: int = 100,
            n_series: int = 1
        ) -> pd.DataFrame:
        """
        Generate simulated ARMA time series.

        Parameters
        ----------
        n : int
            Number of time steps per series.
        burn_in : int
            Number of warm-up periods to discard.
        n_series : int
            Number of independent series to simulate.

        Returns
        -------
        pd.DataFrame
            Simulated time series with shape (n, n_series) and business-day index.
        """
        max_lag = max(len(self.ar), len(self.ma))
        total_len = n + burn_in

        all_series = np.zeros((n, n_series))

        for s in range(n_series):
            y = np.zeros(total_len)
            e = self.noise_fn(total_len)

            for t in range(max_lag, total_len):
                ar_term = np.dot(self.ar, y[t - len(self.ar):t][::-1]) if len(self.ar) else 0
                ma_term = np.dot(self.ma, e[t - len(self.ma):t][::-1]) if len(self.ma) else 0
                y[t] = ar_term + ma_term + e[t]

            all_series[:, s] = y[burn_in:]

        # Business-day index
        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        columns = [f"series_{i}" for i in range(n_series)]

        return pd.DataFrame(all_series, index=index, columns=columns)

class SARIMAGenerator(BaseGenerator):
    """
    Generates Time Series Data using SARIMAX model.
    """
    def __init__(
        self,
        ar: Tuple[float, ...],
        i: int,
        ma: Tuple[float, ...],
        sar: Tuple[float, ...],
        si: int,
        sma: Tuple[float, ...],
        s: int,
        noise_fn: Callable[[int], np.ndarray]
    ):
        super().__init__()
        self.ar = ar
        self.i = i
        self.ma = ma
        self.sar = sar
        self.si = si
        self.sma = sma
        self.s = s
        self.noise_fn = noise_fn

    @classmethod
    def from_params(
        cls,
        params: dict,
        seasonal_params: dict = {},
        u: str = 'normal',
        u_params: Tuple[float, float] = (0, 0.01)
    ) -> 'SARIMAGenerator':
        """
        Initialize SARIMAGenerator from parameter dictionaries.

        Parameters
        ----------
        params : dict
            Dictionary with keys: 'p', 'i', 'q'
        seasonal_params : dict
            Dictionary with keys: 'P', 'I', 'Q', 's'
        u : str
            Noise distribution type: 'normal' or 'uniform'
        u_params : tuple
            (mean, std) for noise distribution

        Returns
        -------
        SARIMAGenerator
        """
        mu, sigma = u_params

        ar = params.get('p', ())
        i = params.get('i', (0,))
        ma = params.get('q', ())

        sar = seasonal_params.get('P', ())
        si = seasonal_params.get('I', (0,))
        sma = seasonal_params.get('Q', ())
        s = seasonal_params.get('s', 0)

        i = i[0] if isinstance(i, tuple) else i
        si = si[0] if isinstance(si, tuple) else si

        if u == 'normal':
            noise_fn = lambda n: np.random.normal(mu, sigma, n)
        elif u == 'uniform':
            low = mu - np.sqrt(3) * sigma
            high = mu + np.sqrt(3) * sigma
            noise_fn = lambda n: np.random.uniform(low, high, n)
        else:
            raise ValueError(f"Unsupported noise type: {u}")

        return cls(ar, i, ma, sar, si, sma, s, noise_fn)

    def generate_simulation(
        self,
        n: int = 252,
        n_series: int = 100
    ) -> pd.DataFrame:
        """
        Generate SARIMA simulated time series.

        Parameters
        ----------
        n : int
            Number of time steps per series.
        n_series : int
            Number of series to simulate.

        Returns
        -------
        pd.DataFrame
            Simulated time series with business-day index.
        """
        results = np.zeros((n, n_series))

        order = (len(self.ar), self.i, len(self.ma))
        seasonal_order = (len(self.sar), self.si, len(self.sma), self.s)

        for i in range(n_series):
            noise = self.noise_fn(n)

            model = SARIMAX(
                endog=np.zeros(n),
                order=order,
                seasonal_order=seasonal_order,
                measurement_error=True
            )
            sim = model.simulate(params=np.zeros(model.k_params), nsimulations=n, measurement_shocks=noise)
            results[:, i] = sim

        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        columns = [f"series_{i}" for i in range(n_series)]

        return pd.DataFrame(results, index=index, columns=columns)

class VARMAGenerator(BaseGenerator):
    """
    Generates Time Series Data using VARMA model.
    """
    def __init__(
        self,
        ar_matrices: Optional[np.ndarray],  # shape: (k, k, p)
        ma_matrices: Optional[np.ndarray],  # shape: (k, k, q)
        noise_fn: Callable[[int, int], np.ndarray]  # input: (n_steps, k)
    ):
        super().__init__()
        self.ar_matrices = ar_matrices
        self.ma_matrices = ma_matrices
        self.noise_fn = noise_fn

    @classmethod
    def from_params(
        cls,
        params: dict,
        u: str = 'normal',
        u_params: tuple = (0, 0.01)
    ) -> 'VARMAGenerator':
        """
        Create VARMA generator from matrix parameters.

        Parameters
        ----------
        params : dict
            Dictionary with 'ar' and optionally 'ma'.
            - ar: np.ndarray of shape (k, k, p)
            - ma: np.ndarray of shape (k, k, q)
        u : str
            Distribution type: 'normal' or 'uniform'
        u_params : tuple
            Mean and std of distribution

        Returns
        -------
        VARMAGenerator
        """
        mu, sigma = u_params
        ar_matrices = params.get('ar', None)
        ma_matrices = params.get('ma', None)

        if ar_matrices is not None:
            ar_matrices = np.asarray(ar_matrices)
        if ma_matrices is not None:
            ma_matrices = np.asarray(ma_matrices)

        def noise_fn(n, k):
            if u == 'normal':
                return np.random.normal(mu, sigma, size=(n, k))
            elif u == 'uniform':
                low = mu - np.sqrt(3) * sigma
                high = mu + np.sqrt(3) * sigma
                return np.random.uniform(low, high, size=(n, k))
            else:
                raise ValueError(f"Unknown distribution type: {u}")

        return cls(ar_matrices, ma_matrices, noise_fn)

    def generate_simulation(
        self,
        n: int = 252,
        burn_in: int = 100
    ) -> pd.DataFrame:
        """
        Simulate multivariate VAR or VARMA process.

        Parameters
        ----------
        n : int
            Number of simulation steps.
        burn_in : int
            Initial periods to discard.

        Returns
        -------
        pd.DataFrame
            Simulated multivariate time series.
        """
        p = self.ar_matrices.shape[2] if self.ar_matrices is not None else 0
        q = self.ma_matrices.shape[2] if self.ma_matrices is not None else 0
        k = self.ar_matrices.shape[0] if self.ar_matrices is not None else self.ma_matrices.shape[0]
        total = n + burn_in

        # Prepare noise
        e = self.noise_fn(total, k)
        y = np.zeros((total, k))

        # Generate series
        for t in range(max(p, q), total):
            ar_term = np.zeros(k)
            ma_term = np.zeros(k)

            # AR term
            if p > 0:
                for lag in range(1, p + 1):
                    ar_term += self.ar_matrices[:, :, lag - 1] @ y[t - lag]

            # MA term
            if q > 0:
                for lag in range(1, q + 1):
                    ma_term += self.ma_matrices[:, :, lag - 1] @ e[t - lag]

            y[t] = ar_term + ma_term + e[t]

        # Convert to DataFrame
        result = y[burn_in:]
        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        columns = [f"series_{i}" for i in range(k)]

        return pd.DataFrame(result, index=index, columns=columns)

class ARFIMAGenerator(BaseGenerator):
    def __init__(self, ar: np.ndarray, ma: np.ndarray, d: float, noise_fn: Callable[[int], np.ndarray]):
        """
        Parameters
        ----------
        ar : np.ndarray
            AR coefficients (e.g., np.array([0.9, -0.3]))
        ma : np.ndarray
            MA coefficients (e.g., np.array([0.4, 0.2]))
        d : float
            Fractional differencing parameter (0 < d < 1 for long memory)
        noise_fn : Callable
            Function generating white noise of length n
        """
        super().__init__()
        self.ar = ar
        self.ma = ma
        self.d = d
        self.noise_fn = noise_fn

    @classmethod
    def from_params(
        cls,
        params: dict,
        d: float,
        u: str = 'normal',
        u_params: tuple = (0, 0.01)
    ) -> 'ARFIMAGenerator':
        ar = np.array(params.get('p', []))
        ma = np.array(params.get('q', []))
        mu, sigma = u_params

        if u == 'normal':
            noise_fn = lambda n: np.random.normal(loc=mu, scale=sigma, size=n)
        elif u == 'uniform':
            low = mu - np.sqrt(3) * sigma
            high = mu + np.sqrt(3) * sigma
            noise_fn = lambda n: np.random.uniform(low=low, high=high, size=n)
        else:
            raise ValueError(f"Unknown noise type: {u}")

        return cls(ar, ma, d, noise_fn)

    def _fracdiff_weights(self, d: float, size: int, tol: float = 1e-10) -> np.ndarray:
        """
        Generate fractional differencing weights using binomial expansion.

        Parameters
        ----------
        d : float
            Differencing parameter
        size : int
            Number of weights to compute

        Returns
        -------
        np.ndarray
            Fractional differencing weights
        """
        w = [1.0]
        for k in range(1, size):
            w.append(w[-1] * (d + k - 1) / k)
            if abs(w[-1]) < tol:  # tail truncate for speed/num stability
                break
        return np.array(w)

    def generate_simulation(
        self,
        n: int = 252,
        burn_in: int = 100,
        n_series: int = 1
    ) -> pd.DataFrame:
        """
        Simulate ARFIMA time series.

        Parameters
        ----------
        n : int
            Length of the series (excluding burn-in)
        burn_in : int
            Number of initial steps to discard
        n_series : int
            Number of independent time series

        Returns
        -------
        pd.DataFrame
            Simulated ARFIMA series
        """
        max_lag = max(len(self.ar), len(self.ma), 100)
        total_len = n + burn_in
        all_series = np.zeros((n, n_series))

        for s in range(n_series):
            # 1) stationary ARMA part
            x = np.zeros(total_len)
            e = self.noise_fn(total_len)
            for t in range(max_lag, total_len):
                ar_term = np.dot(self.ar, x[t - len(self.ar):t][::-1]) if len(self.ar) else 0
                ma_term = np.dot(self.ma, e[t - len(self.ma):t][::-1]) if len(self.ma) else 0
                x[t] = ar_term + ma_term + e[t]

            # 2) fractional INTEGRATION: y_t = (1 - B)^(-d) x_t
            w = self._fracdiff_weights(self.d, total_len + 1)
            # causal convolution: y[t] = sum_{k=0}^{t} w_k * x[t-k]
            y = np.convolve(x, w, mode='full')[:total_len]

            all_series[:, s] = y[burn_in:]

        index = pd.date_range(start=datetime.today().date(), periods=n, freq='B')
        columns = [f'series_{i}' for i in range(n_series)]
        return pd.DataFrame(all_series, index=index, columns=columns)