from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Literal
import numpy as np
import pandas as pd

from .base import BaseValuation

class DCFValuation(BaseValuation):
    """
    Discounted Cash Flow (DCF) valuation.

    Supports:
    ---------
    • Mode: 'fcff' (Firm) or 'fcfe' (Equity)
    • Terminal value: 'g' (Gordon / perpetual growth) or 'exit' (exit multiple)
    • Mid-year convention for discounting

    Expected inputs to `evaluate()` (minimal):
    -----------------------------------------
    Common:
      - horizon arrays of length T (numpy-like or list)
      - shares_out : float
      - net_debt   : float  (only used in FCFF mode to bridge to equity value)
      - wacc       : float  (FCFF mode) or ke (FCFE mode)
      - g          : float  (terminal perpetual growth, when terminal='g')
      - OR exit_multiple : float and exit_metric : float (or array), when terminal='exit'

    FCFF mode: (one of)
      A) provide 'fcff' (array length T)
      B) OR provide components to compute FCFF:
           'ebit', 'dep', 'capex', 'delta_nwc', 'tax_rate' (scalar or array)
         where:  NOPAT = EBIT * (1 - tax_rate)
                 FCFF  = NOPAT + Dep - CAPEX - ΔNWC

    FCFE mode: (one of)
      A) provide 'fcfe' (array length T)
      B) OR provide:
           • 'fcff' and (optionally) 'at_interest' (after-tax interest) and 'net_borrowing'
             FCFE ≈ FCFF - at_interest + net_borrowing
           • OR provide components to derive FCFE directly if you prefer.

    Returns (dict):
    ---------------
      {
        'firm_value'    : float or None,
        'equity_value'  : float,
        'value_per_share': float,
        'breakdown': {
            'PV_FCF'       : float,
            'PV_Terminal'  : float,
            'FCF'          : np.ndarray,   # fcff or fcfe along horizon
            'TV'           : float,
            'terminal_method': 'g' or 'exit'
        }
      }

    Notes
    -----
    • For FCFF mode, discount at WACC; bridge to equity via net debt.
    • For FCFE mode, discount at cost of equity (ke); result is equity directly.
    • Mid-year convention shifts discount exponent by 0.5 years.
    """

    def __init__(
        self,
        mode: Literal["fcff", "fcfe"] = "fcff",
        terminal: Literal["g", "exit"] = "g",
        mid_year: bool = True,
    ):
        self.mode = mode
        self.terminal = terminal
        self.mid_year = mid_year

    @classmethod
    def from_params(cls, params: Dict) -> "DCFValuation":
        """
        Construct DCFValuation from configuration parameters.

        Parameters
        ----------
        params : dict
          Keys:
            - mode     : 'fcff' | 'fcfe'
            - terminal : 'g' | 'exit'
            - mid_year : bool
        """
        return cls(
            mode=params.get("mode", "fcff"),
            terminal=params.get("terminal", "g"),
            mid_year=bool(params.get("mid_year", True)),
        )

    # ---------- helpers ----------
    def _to_array(self, x, T: Optional[int] = None) -> np.ndarray:
        """
        Convert scalar/list/array to 1-D numpy array (length T if provided).

        If x is scalar and T is given, broadcast to length T.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.asarray(x, dtype=float).ravel()
        if T is None:
            return np.asarray([float(x)], dtype=float)
        return np.full(shape=(T,), fill_value=float(x), dtype=float)

    def _pv_factors(self, r: float, T: int) -> np.ndarray:
        """
        Present value factors for periods 1..T.
        Applies mid-year shift if enabled.
        """
        t = np.arange(1, T + 1, dtype=float)
        if self.mid_year:
            t -= 0.5
        return (1.0 + float(r)) ** (-t)

    def _terminal_gordon(self, fcf_T: float, g: float, r: float) -> float:
        """
        Gordon growth terminal value at period T (value as of end of year T).
          TV = FCF_{T+1} / (r - g) = fcf_T * (1+g) / (r - g)
        """
        eps = 1e-10
        if r - g <= eps:
            # prevent division by ~0; caller should ensure r > g
            r = g + 1e-6
        return fcf_T * (1.0 + g) / (r - g)

    def _terminal_exit(self, exit_multiple: float, exit_metric: float) -> float:
        """
        Exit multiple terminal value:
          TV = exit_multiple * exit_metric
        """
        return float(exit_multiple) * float(exit_metric)

    # ---------- core ----------
    def evaluate(self, inputs: Dict) -> Dict:
        """
        Run DCF valuation.

        Parameters
        ----------
        inputs : dict
          Common:
            - shares_out : float
          FCFF mode:
            - wacc      : float
            - net_debt  : float
            - g         : float (if terminal='g')
            - OR exit_multiple, exit_metric (if terminal='exit')
            - Either 'fcff' (array) OR components: 'ebit','dep','capex','delta_nwc','tax_rate'
          FCFE mode:
            - ke        : float
            - g         : float (if terminal='g')
            - OR exit_multiple, exit_metric (if terminal='exit')
            - Either 'fcfe' (array),
              OR 'fcff' plus optional 'at_interest' and 'net_borrowing'.

        Returns
        -------
        dict  (see class docstring)
        """
        # -------- prepare FCF path --------
        if self.mode == "fcff":
            fcf = inputs.get("fcff", None)
            if fcf is None:
                # compute FCFF from components
                required = ["ebit", "dep", "capex", "delta_nwc", "tax_rate"]
                missing = [k for k in required if k not in inputs]
                if missing:
                    raise ValueError(f"Missing inputs for FCFF: {missing}")
                # infer horizon length
                T = len(self._to_array(inputs["ebit"]))
                ebit      = self._to_array(inputs["ebit"])
                dep       = self._to_array(inputs["dep"], T)
                capex     = self._to_array(inputs["capex"], T)
                delta_nwc = self._to_array(inputs["delta_nwc"], T)
                tax_rate  = self._to_array(inputs["tax_rate"], T)
                nopat = ebit * (1.0 - tax_rate)
                fcf = nopat + dep - capex - delta_nwc
            else:
                fcf = self._to_array(fcf)
            T = len(fcf)
        else:  # FCFE
            fcf = inputs.get("fcfe", None)
            if fcf is None:
                # derive from FCFF if provided
                if "fcff" not in inputs:
                    raise ValueError("Provide 'fcfe' array or 'fcff' with adjustments for FCFE mode.")
                fcff = self._to_array(inputs["fcff"])
                T = len(fcff)
                at_int = self._to_array(inputs.get("at_interest", 0.0), T)     # after-tax interest
                net_b  = self._to_array(inputs.get("net_borrowing", 0.0), T)   # new debt - repayments
                fcf = fcff - at_int + net_b
            else:
                fcf = self._to_array(fcf)
                T = len(fcf)

        # -------- rate & PV factors --------
        if self.mode == "fcff":
            if "wacc" not in inputs:
                raise ValueError("FCFF mode requires 'wacc'.")
            r = float(inputs["wacc"])
        else:
            if "ke" not in inputs:
                raise ValueError("FCFE mode requires 'ke'.")
            r = float(inputs["ke"])

        pv_f = self._pv_factors(r, T)

        # -------- terminal value --------
        if self.terminal == "g":
            if "g" not in inputs:
                raise ValueError("Terminal 'g' requires 'g' (perpetual growth).")
            g = float(inputs["g"])
            tv = self._terminal_gordon(fcf[-1], g, r)
            # discount TV to present (end of year T, apply mid-year shift if enabled)
            t_exp = T - (0.5 if self.mid_year else 0.0)
            pv_tv = tv / ((1.0 + r) ** t_exp)
        else:  # 'exit'
            if "exit_multiple" not in inputs or "exit_metric" not in inputs:
                raise ValueError("Terminal 'exit' requires 'exit_multiple' and 'exit_metric'.")
            exit_multiple = float(inputs["exit_multiple"])
            # exit_metric can be array; use last if sequence
            exit_metric = inputs["exit_metric"]
            if isinstance(exit_metric, (list, tuple, np.ndarray)):
                exit_metric = float(np.asarray(exit_metric).ravel()[-1])
            else:
                exit_metric = float(exit_metric)
            tv = self._terminal_exit(exit_multiple, exit_metric)
            t_exp = T - (0.5 if self.mid_year else 0.0)
            pv_tv = tv / ((1.0 + r) ** t_exp)

        # -------- PV of explicit FCF --------
        pv_fcf = float(np.dot(fcf, pv_f))

        # -------- bridge to equity --------
        firm_value = None
        if self.mode == "fcff":
            firm_value = pv_fcf + pv_tv
            if "net_debt" not in inputs:
                raise ValueError("FCFF mode requires 'net_debt' to bridge to equity value.")
            equity_value = firm_value - float(inputs["net_debt"])
        else:
            # FCFE discounted at cost of equity gives equity directly
            equity_value = pv_fcf + pv_tv

        if "shares_out" not in inputs:
            raise ValueError("Missing 'shares_out' to compute value per share.")
        shares_out = max(float(inputs["shares_out"]), 1e-9)
        value_per_share = equity_value / shares_out

        return {
            "firm_value": firm_value,
            "equity_value": float(equity_value),
            "value_per_share": float(value_per_share),
            "breakdown": {
                "PV_FCF": float(pv_fcf),
                "PV_Terminal": float(pv_tv),
                "FCF": fcf.astype(float),
                "TV": float(tv),
                "terminal_method": self.terminal,
            },
        }