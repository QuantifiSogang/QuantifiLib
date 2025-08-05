import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Optional, Tuple

from .base_generator import BaseGenerator

class GBMGenerator(BaseGenerator):
    """
    Geometric Brownian Motion (GBM) simulator.

    SDE
    ---
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Discretization (Euler / exact for GBM)
    --------------------------------------
        log(S_{t+dt} / S_t) ~ N( (mu - 0.5*sigma^2) * dt,  (sigma^2 * dt) )

    Parameters (via from_params)
    ----------------------------
    params : dict
        {
            'mu'   : float,   # drift
            'sigma': float,   # volatility (>= 0)
            'S0'   : float,   # initial price (> 0)
            'dt'   : float    # time step, default 1/252 for business days
        }
    u : str
        Innovation distribution for the Brownian shocks z_t. One of:
        - 'normal'    : N(0,1) (recommended)
        - 'student_t' : standardized to unit variance
        - 'uniform'   : U(-sqrt(3), sqrt(3)) -> unit variance
    u_params : tuple
        Distribution parameters for z_t.
        - normal    : (mu_z, sigma_z) -> defaults to (0,1)
        - student_t : (df,) with df > 2; internally rescaled to unit variance
        - uniform   : (mu_z, sigma_z) used to set interval so Var= sigma_z^2

    Notes
    -----
    • For GBM, unit-variance shocks (z_t ~ i.i.d with Var=1) are standard. The
      time/volatility scaling is applied externally via sigma * sqrt(dt).
    • Output can be 'price', 'log_return', or 'return' (arithmetic), controlled
      via `return_type`.
    """

    def __init__(self, mu: float, sigma: float, S0: float, dt: float,
                 z_fn: Callable[[int, int], np.ndarray]):
        """
        Store model parameters and a shock generator.

        Parameters
        ----------
        mu : float
            Drift parameter.
        sigma : float
            Volatility parameter (>= 0).
        S0 : float
            Initial price (> 0).
        dt : float
            Time step (e.g., 1/252).
        z_fn : Callable
            Function that generates shocks z of shape (n_steps, n_series).
        """
        super().__init__()
        if S0 <= 0:
            raise ValueError("S0 must be > 0.")
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.S0 = float(S0)
        self.dt = float(dt)
        self.z_fn = z_fn

    @classmethod
    def from_params(
        cls,
        params: dict,
        u: str = "normal",
        u_params: Optional[Tuple[float, ...]] = None
    ) -> "GBMGenerator":
        """
        Construct a GBM generator from parameter dictionaries and a chosen shock distribution.

        Parameters
        ----------
        params : dict
            Keys: 'mu' (float), 'sigma' (float), 'S0' (float), 'dt' (float, default 1/252).
        u : str
            Shock distribution for z_t in N(0,1)-like scale.
        u_params : tuple, optional
            Parameters for the chosen distribution.

        Returns
        -------
        GeometricBrownianProcess
        """
        mu = float(params.get("mu", 0.0))
        sigma = float(params.get("sigma", 0.2))
        S0 = float(params.get("S0", 100.0))
        dt = float(params.get("dt", 1.0 / 252.0))

        # Build shock generator z_t of shape (n_steps, n_series)
        if u == "normal":
            m, s = (0.0, 1.0) if not u_params else u_params
            z_fn = lambda n, k: np.random.normal(loc=m, scale=s, size=(n, k))
        elif u == "student_t":
            df = 8.0 if not u_params else float(u_params[0])
            if df <= 2:
                raise ValueError("student_t requires df > 2.")
            scale = np.sqrt((df - 2.0) / df)  # standardize to unit variance
            z_fn = lambda n, k: np.random.standard_t(df, size=(n, k)) * scale
        elif u == "uniform":
            m, s = (0.0, 1.0) if not u_params else u_params
            a, b = m - np.sqrt(3.0) * s, m + np.sqrt(3.0) * s
            z_fn = lambda n, k: np.random.uniform(low=a, high=b, size=(n, k))
        else:
            raise ValueError("u must be one of {'normal','student_t','uniform'}.")

        return cls(mu=mu, sigma=sigma, S0=S0, dt=dt, z_fn=z_fn)

    def generate_simulation(
        self,
        n: int = 252,
        n_series: int = 1,
        return_type: str = "price"
    ) -> pd.DataFrame:
        """
        Generate GBM paths and return a DataFrame with a business-day index.

        Parameters
        ----------
        n : int
            Number of time steps (e.g., trading days).
        n_series : int
            Number of independent paths to simulate.
        return_type : {"price", "log_return", "return"}
            Output type:
            - "price"      : Price levels S_t, starting at S0 (default).
            - "log_return" : log(S_t / S_{t-1}) increments.
            - "return"     : arithmetic returns (S_t - S_{t-1}) / S_{t-1}.

        Returns
        -------
        pd.DataFrame
            Shape (n, n_series) with columns 'series_0', 'series_1', ...
            Business-day datetime index starting from today.
        """
        # Draw shocks z ~ i.i.d. with unit variance (per design of z_fn)
        Z = self.z_fn(n, n_series)

        # Exact discretization for GBM log-returns
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        vol   = self.sigma * np.sqrt(self.dt)
        log_r = drift + vol * Z  # shape: (n, n_series)

        if return_type == "log_return":
            data = log_r
        elif return_type == "return":
            # arithmetic returns from log returns: r = exp(log_r) - 1
            data = np.exp(log_r) - 1.0
        elif return_type == "price":
            # build price paths: S_t = S0 * exp(cumsum(log_r))
            log_price = np.cumsum(log_r, axis=0)
            data = self.S0 * np.exp(log_price)
        else:
            raise ValueError("return_type must be one of {'price','log_return','return'}.")

        # Business-day index from today
        index = pd.date_range(start=datetime.today().date(), periods=n, freq="B")
        cols = [f"series_{i}" for i in range(n_series)]
        return pd.DataFrame(data, index=index, columns=cols)

class OUGenerator(BaseGenerator):
    """
    Ornstein–Uhlenbeck (OU) process generator.

    SDE
    ---
        dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

    Exact discretization
    --------------------
        X_{t+dt} = theta + (X_t - theta) * exp(-kappa * dt)
                   + sigma * sqrt((1 - exp(-2*kappa*dt)) / (2*kappa)) * Z_t

        where Z_t are i.i.d. shocks with mean 0 and unit variance.

    Parameters (via from_params)
    ----------------------------
    params : dict
        {
            'kappa': float >= 0,    # mean-reversion speed
            'theta': float,         # long-run mean level
            'sigma': float >= 0,    # diffusion volatility
            'x0'   : Optional[float],  # initial level (if None, see stationary_start)
            'dt'   : float > 0         # time step, e.g., 1/252 for business days
        }
    u : str
        Shock distribution for Z_t (unit-variance scale):
        - 'normal'    : N(0,1) (recommended)
        - 'student_t' : standardized to unit variance
        - 'uniform'   : U(-sqrt(3), sqrt(3)) -> unit variance
    u_params : tuple, optional
        Parameters for the chosen distribution:
        - normal    : (mu_z, sigma_z) default (0,1)
        - student_t : (df,) with df > 2; internally re-scaled to unit variance
        - uniform   : (mu_z, sigma_z) -> interval centered at mu_z with Var = sigma_z^2

    Notes
    -----
    • If 'x0' is None and stationary_start=True in generate_simulation, the initial
      value will be drawn from the stationary distribution N(theta, sigma^2/(2*kappa)).
    • As kappa -> 0, the OU reduces to a Brownian motion with drift 0; the variance
      term smoothly becomes sigma * sqrt(dt).
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        x0: Optional[float],
        dt: float,
        z_fn: Callable[[int, int], np.ndarray],
    ):
        """
        Store model parameters and a shock generator.

        Parameters
        ----------
        kappa : float
            Mean-reversion speed (>= 0).
        theta : float
            Long-run mean level.
        sigma : float
            Diffusion volatility (>= 0).
        x0 : Optional[float]
            Initial level. If None, will be chosen at simulation time depending
            on stationary_start flag.
        dt : float
            Time step (e.g., 1/252 for business days).
        z_fn : Callable
            Function that generates shocks Z of shape (n_steps, n_series).
        """
        super().__init__()
        if kappa < 0:
            raise ValueError("kappa must be >= 0.")
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.x0 = None if x0 is None else float(x0)
        self.dt = float(dt)
        self.z_fn = z_fn

    @classmethod
    def from_params(
        cls,
        params: dict,
        u: str = "normal",
        u_params: Optional[Tuple[float, ...]] = None,
    ) -> "OUGenerator":
        """
        Construct an OUGenerator from parameter dictionaries and a chosen shock distribution.

        Parameters
        ----------
        params : dict
            Keys: 'kappa' (float >=0), 'theta' (float), 'sigma' (float >=0),
                  'x0' (optional float), 'dt' (float >0, default 1/252).
        u : str
            Shock distribution type: {'normal','student_t','uniform'}.
        u_params : tuple, optional
            Distribution parameters (see class docstring).

        Returns
        -------
        OUGenerator
        """
        kappa = float(params.get("kappa", 1.0))
        theta = float(params.get("theta", 0.0))
        sigma = float(params.get("sigma", 0.2))
        x0 = params.get("x0", None)
        dt = float(params.get("dt", 1.0 / 252.0))

        # Build Z shock generator of shape (n_steps, n_series), unit-variance by default
        if u == "normal":
            m, s = (0.0, 1.0) if not u_params else u_params
            z_fn = lambda n, k: np.random.normal(loc=m, scale=s, size=(n, k))
        elif u == "student_t":
            df = 8.0 if not u_params else float(u_params[0])
            if df <= 2:
                raise ValueError("student_t requires df > 2.")
            scale = np.sqrt((df - 2.0) / df)  # standardize to unit variance
            z_fn = lambda n, k: np.random.standard_t(df, size=(n, k)) * scale
        elif u == "uniform":
            m, s = (0.0, 1.0) if not u_params else u_params
            a, b = m - np.sqrt(3.0) * s, m + np.sqrt(3.0) * s
            z_fn = lambda n, k: np.random.uniform(low=a, high=b, size=(n, k))
        else:
            raise ValueError("u must be one of {'normal','student_t','uniform'}.")

        return cls(kappa=kappa, theta=theta, sigma=sigma, x0=x0, dt=dt, z_fn=z_fn)

    def _exact_step_coeffs(self) -> Tuple[float, float]:
        """
        Compute exact discretization coefficients for one time step.

        Returns
        -------
        (phi, sig_eff)
            phi     : exp(-kappa * dt)
            sig_eff : sigma * sqrt((1 - phi^2) / (2*kappa))  if kappa > 0
                      sigma * sqrt(dt)                       if kappa is ~ 0
        """
        phi = np.exp(-self.kappa * self.dt)
        if self.kappa > 1e-12:
            var = (1.0 - phi * phi) / (2.0 * self.kappa)
            sig_eff = self.sigma * np.sqrt(var)
        else:
            # Limit as kappa -> 0
            sig_eff = self.sigma * np.sqrt(self.dt)
        return phi, sig_eff

    def generate_simulation(
        self,
        n: int = 252,
        n_series: int = 1,
        return_type: str = "level",
        stationary_start: bool = False,
    ) -> pd.DataFrame:
        """
        Generate OU paths and return a DataFrame with a business-day index.

        Parameters
        ----------
        n : int
            Number of time steps (e.g., trading days).
        n_series : int
            Number of independent paths to simulate.
        return_type : {"level", "increment"}
            Output type:
            - "level"     : X_t levels (default).
            - "increment" : X_t - X_{t-1} (first differences).
        stationary_start : bool
            If True and x0 is None, draw initial X_0 from the stationary distribution
            N(theta, sigma^2/(2*kappa)). If kappa == 0, falls back to X_0 = theta.

        Returns
        -------
        pd.DataFrame
            Shape (n, n_series) with columns 'series_0', 'series_1', ...
            Business-day datetime index starting from today.
        """
        phi, sig_eff = self._exact_step_coeffs()

        # Initialize X_0
        if self.x0 is not None:
            X0 = float(self.x0)
        elif stationary_start and self.kappa > 1e-12:
            std_stat = self.sigma / np.sqrt(2.0 * self.kappa)
            X0 = self.theta + std_stat * np.random.normal()
        else:
            # Default to long-run mean if no x0 is provided
            X0 = self.theta

        # Draw shocks
        Z = self.z_fn(n, n_series)  # unit-variance shocks

        # Allocate and simulate
        X = np.zeros((n, n_series))
        X_prev = np.full(shape=(n_series,), fill_value=X0, dtype=float)

        # Iterate exact recursion
        for t in range(n):
            # X_t = theta + (X_{t-1} - theta)*phi + sig_eff * Z_t
            X_t = self.theta + (X_prev - self.theta) * phi + sig_eff * Z[t, :]
            X[t, :] = X_t
            X_prev = X_t

        # Prepare output
        if return_type == "level":
            data = X
        elif return_type == "increment":
            # first difference with zeros at the first step (or drop first row if preferred)
            data = np.vstack([X[0, :] - X0, X[1:, :] - X[:-1, :]])
        else:
            raise ValueError("return_type must be one of {'level','increment'}.")

        index = pd.date_range(start=datetime.today().date(), periods=n, freq="B")
        cols = [f"series_{i}" for i in range(n_series)]
        return pd.DataFrame(data, index=index, columns=cols)

class JumpDiffusionGenerator(BaseGenerator):
    """
    Merton Jump-Diffusion (GBM with Poisson lognormal jumps) generator.

    Continuous-time model
    ---------------------
        dS_t / S_t = (mu - lambda * k) dt + sigma dW_t + (J_t - 1) dN_t

        where
          • N_t ~ Poisson(lambda * t) is a Poisson process (jump arrivals),
          • J_t = exp(Y),  Y ~ Normal(jump_mu, jump_sigma^2)  (lognormal jump size),
          • k = E[J - 1] = exp(jump_mu + 0.5 * jump_sigma^2) - 1 (drift compensation term).

    Discrete-time (per step dt)
    ---------------------------
        log(S_{t+dt}/S_t) = (mu - 0.5*sigma^2 - comp * lambda * k) * dt
                            + sigma * sqrt(dt) * Z_t
                            + sum_{i=1}^{N_t} Y_i,
        with Z_t ~ i.i.d unit-variance shocks, N_t ~ Poisson(lambda*dt),
        Y_i ~ Normal(jump_mu, jump_sigma^2).

        Sum of Y_i given N_t = n is Normal(n*jump_mu, n*jump_sigma^2).

    Parameters (via from_params)
    ----------------------------
    params : dict
        {
            'mu'        : float,   # continuous drift
            'sigma'     : float,   # diffusion volatility (>=0)
            'S0'        : float,   # initial price (>0)
            'dt'        : float,   # time step (e.g., 1/252)
            'lambda'    : float,   # jump intensity per unit time (>=0)
            'jump_mu'   : float,   # mean of log-jump Y
            'jump_sigma': float,   # std of log-jump Y (>=0)
        }
    u : str
        Shock distribution for Brownian part Z_t (unit-variance scale):
        - 'normal' (default), 'student_t' (standardized), or 'uniform' (unit variance).
    u_params : tuple, optional
        Parameters for Z_t distribution:
        - normal    : (mu_z, sigma_z) default (0,1)
        - student_t : (df,) with df>2, rescaled to unit variance
        - uniform   : (mu_z, sigma_z) -> interval centered at mu_z with Var = sigma_z^2

    Notes
    -----
    • When lambda = 0, the model reduces to GBM.
    • Drift compensation (comp=True) ensures the expected jump effect on drift is neutralized.
    """

    def __init__(
            self,mu: float,
            sigma: float,
            S0: float,
            dt: float,
            lam: float,
            jump_mu: float,
            jump_sigma: float,
            z_fn: Callable[[int, int], np.ndarray]
        ):
        super().__init__()
        # Basic checks
        if S0 <= 0:      raise ValueError("S0 must be > 0.")
        if sigma < 0:    raise ValueError("sigma must be >= 0.")
        if dt <= 0:      raise ValueError("dt must be > 0.")
        if lam < 0:      raise ValueError("lambda must be >= 0.")
        if jump_sigma < 0: raise ValueError("jump_sigma must be >= 0.")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.S0 = float(S0)
        self.dt = float(dt)
        self.lam = float(lam)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)
        self.z_fn = z_fn  # unit-variance shocks of shape (n_steps, n_series)

        # Precompute compensation term k = E[J-1]
        self.k = np.exp(self.jump_mu + 0.5 * self.jump_sigma ** 2) - 1.0

    @classmethod
    def from_params(
        cls,
        params: dict,
        u: str = "normal",
        u_params: Optional[Tuple[float, ...]] = None
    ) -> "JumpDiffusionGenerator":
        """
        Construct a JumpDiffusionGenerator from parameter dicts and a shock distribution.

        Parameters
        ----------
        params : dict
            Keys: 'mu','sigma','S0','dt','lambda','jump_mu','jump_sigma'.
        u : str
            Shock distribution for Brownian part Z_t in unit-variance scale.
        u_params : tuple, optional
            Parameters for chosen Z_t distribution.

        Returns
        -------
        JumpDiffusionGenerator
        """
        mu          = float(params.get("mu", 0.0))
        sigma       = float(params.get("sigma", 0.2))
        S0          = float(params.get("S0", 100.0))
        dt          = float(params.get("dt", 1.0 / 252.0))
        lam         = float(params.get("lambda", 0.1))
        jump_mu     = float(params.get("jump_mu", -0.02))
        jump_sigma  = float(params.get("jump_sigma", 0.10))

        # Build unit-variance shock generator for Z_t
        if u == "normal":
            m, s = (0.0, 1.0) if not u_params else u_params
            z_fn = lambda n, k: np.random.normal(loc=m, scale=s, size=(n, k))
        elif u == "student_t":
            df = 8.0 if not u_params else float(u_params[0])
            if df <= 2:
                raise ValueError("student_t requires df > 2.")
            scale = np.sqrt((df - 2.0) / df)  # standardize to unit variance
            z_fn = lambda n, k: np.random.standard_t(df, size=(n, k)) * scale
        elif u == "uniform":
            m, s = (0.0, 1.0) if not u_params else u_params
            a, b = m - np.sqrt(3.0) * s, m + np.sqrt(3.0) * s
            z_fn = lambda n, k: np.random.uniform(low=a, high=b, size=(n, k))
        else:
            raise ValueError("u must be one of {'normal','student_t','uniform'}.")

        return cls(mu=mu, sigma=sigma, S0=S0, dt=dt,
                   lam=lam, jump_mu=jump_mu, jump_sigma=jump_sigma,
                   z_fn=z_fn)

    def generate_simulation(
        self,
        n: int = 252,
        n_series: int = 1,
        return_type: str = "price",
        compensate_drift: bool = True
    ) -> pd.DataFrame:
        """
        Generate Merton jump-diffusion paths.

        Parameters
        ----------
        n : int
            Number of time steps (e.g., trading days).
        n_series : int
            Number of independent paths to simulate.
        return_type : {"price", "log_return", "return"}
            Output type:
            - "price"      : price levels S_t starting at S0 (default).
            - "log_return" : log(S_t / S_{t-1}) increments.
            - "return"     : arithmetic returns (S_t - S_{t-1}) / S_{t-1}.
        compensate_drift : bool
            If True, subtract lambda * k in drift (recommended).

        Returns
        -------
        pd.DataFrame
            Shape (n, n_series) with business-day datetime index and columns 'series_i'.
        """
        # Brownian shocks Z ~ unit variance (per z_fn design)
        Z = self.z_fn(n, n_series)

        # Poisson jump counts per step
        lam_dt = self.lam * self.dt
        N = np.random.poisson(lam=lam_dt, size=(n, n_series))

        # Sum of log-jumps per step: conditional Normal(N*mu_J, N*sigma_J^2)
        # For N=0, sum is 0.
        jump_log = np.zeros_like(Z)
        has_jump = N > 0
        if np.any(has_jump):
            mean_jump = N * self.jump_mu
            std_jump  = np.sqrt(N) * self.jump_sigma
            # To avoid std=0 warnings when N>0 but jump_sigma=0, clip std non-negative
            std_jump = np.where(std_jump > 0, std_jump, 0.0)
            jump_log[has_jump] = np.random.normal(
                loc=mean_jump[has_jump],
                scale=std_jump[has_jump]
            )

        # Drift term with optional compensation
        comp = (self.lam * self.k) if compensate_drift else 0.0
        drift = (self.mu - comp - 0.5 * self.sigma ** 2) * self.dt
        vol   = self.sigma * np.sqrt(self.dt)

        # Log-returns
        log_r = drift + vol * Z + jump_log  # (n, n_series)

        # Map to requested output
        if return_type == "log_return":
            data = log_r
        elif return_type == "return":
            data = np.exp(log_r) - 1.0
        elif return_type == "price":
            log_price = np.cumsum(log_r, axis=0)
            data = self.S0 * np.exp(log_price)
        else:
            raise ValueError("return_type must be one of {'price','log_return','return'}.")

        # Business-day index
        index = pd.date_range(start=datetime.today().date(), periods=n, freq="B")
        cols = [f"series_{i}" for i in range(n_series)]
        return pd.DataFrame(data, index=index, columns=cols)