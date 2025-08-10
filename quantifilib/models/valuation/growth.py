from typing import Dict, Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd

from .base import BaseValuation
from .utils import *

class GordonGrowthValuation(BaseValuation):
    """
    Single-stage Dividend Discount Model (Gordon Growth) valuation engine.

    Formula
    -------
    Value_per_share = D1 / (r - g)
      where:
        D1 : next-period dividend per share
        r  : cost of equity (required return)
        g  : perpetual dividend growth rate
      Constraint: r > g

    Inputs to evaluate()
    --------------------
    Common (required):
      - shares_out        : float
      - cost_of_equity    : float  (aliases: 'r', 'ke')
      - growth            : float  (alias: 'g')

    Dividend inputs (any one path below):
      A) 'd1' (next dividend per share)              # direct
      B) 'dps' (trailing DPS) + growth               # D1 = dps * (1+g)
      C) 'dividends_next' (total next dividend)      # will be divided by shares_out
      D) 'dividends' (trailing total) + growth       # D1_total = dividends * (1+g)

    Returns
    -------
    dict
      {
        'equity_value'    : float,
        'value_per_share' : float,
        'meta'            : dict   # timing, method, r, g, path_used
      }

    Notes
    -----
    • Negative dividends are allowed but unusual; user responsibility.
    • If you supply totals (company-level) we convert to per-share using shares_out.
    """

    def __init__(self):
        # Kept minimal to mirror the MultiplesValuation structure/feel.
        pass

    @classmethod
    def from_params(cls, params: Dict) -> "GordonGrowthValuation":
        """
        Build from a configuration dict (kept for symmetry with other engines).
        No persistent parameters are required for single-stage DDM.
        """
        return cls()

    # ---------- core ----------
    def evaluate(self, inputs: Dict) -> Dict:
        """
        Compute equity value and value per share with the Gordon Growth model.

        Parameters
        ----------
        inputs : dict
            See class docstring for accepted keys.

        Returns
        -------
        dict
            Keys: 'equity_value', 'value_per_share', 'meta'
        """
        # Required common inputs
        if "shares_out" not in inputs:
            raise ValueError("Missing 'shares_out'.")
        shares_out = max(float(inputs["shares_out"]), 1e-12)

        r = pick_first(inputs, ["cost_of_equity", "r", "ke"])
        g = pick_first(inputs, ["growth", "g"])
        check_r_g(r, g)

        # Determine next dividend per share D1 (priority order)
        path_used = None
        d1 = pick_first(inputs, ["d1", "dps_next", "dividend_next", "dividend_per_share_next"])
        if d1 is not None:
            path_used = "d1"
        else:
            dps = pick_first(inputs, ["dps", "dividend", "dividend_per_share"])
            if dps is not None:
                d1 = dps * (1.0 + g)
                path_used = "dps*(1+g)"
            else:
                div_next_total = pick_first(inputs, ["dividends_next", "dividend_total_next"])
                if div_next_total is not None:
                    d1 = div_next_total / shares_out
                    path_used = "dividends_next / shares_out"
                else:
                    div_total = pick_first(inputs, ["dividends", "dividend_total"])
                    if div_total is None:
                        raise ValueError(
                            "Provide one of: 'd1' | 'dps'+growth | 'dividends_next' | 'dividends'+growth."
                        )
                    d1 = (div_total * (1.0 + g)) / shares_out
                    path_used = "dividends*(1+g)/shares_out"

        # Gordon formula
        v_ps = d1 / (r - g)
        equity_value = v_ps * shares_out

        return {
            "equity_value": float(equity_value),
            "value_per_share": float(v_ps),
            "meta": {
                "model": "GordonGrowth (single-stage DDM)",
                "r": float(r),
                "g": float(g),
                "path": path_used,
                "timing": inputs.get("timing", None),
            },
        }

    def evaluate_df(
        self,
        df: pd.DataFrame,
        shares_out: Union[float, pd.Series],
        *,
        r: Optional[Union[float, pd.Series]] = None,
        g: Optional[Union[float, pd.Series]] = None,
        # Optional column name aliases if your dataframe uses different headers
        col_dps: str = "DPS",
        col_dps_next: str = "DPS_next",
        col_div_total: str = "Dividends",
        col_div_total_next: str = "DividendsNext",
        col_r: str = "CostOfEquity",
        col_g: str = "Growth",
    ) -> pd.DataFrame:
        """
        Vectorized valuation over a time index.

        Inputs precedence to construct D1 per share at each timestamp:
          1) col_dps_next
          2) col_dps + growth
          3) col_div_total_next / shares_out
          4) col_div_total * (1+growth) / shares_out

        Parameters
        ----------
        df : DataFrame
            Time-indexed fundamentals and parameters.
        shares_out : float | Series
            Shares outstanding (scalar or time series aligned to df.index).
        r : float | Series, optional
            Cost of equity override (else use df[col_r]).
        g : float | Series, optional
            Growth override (else use df[col_g]).
        col_* : str
            Column names used to pull inputs from df.

        Returns
        -------
        DataFrame
            Columns: ['value_per_share','equity_value','r','g','path']
        """
        df = df.copy().sort_index()
        idx = df.index

        # Align shares_out, r, g to index
        sh = as_series(shares_out, idx, "shares_out")
        r_series = as_series(r, idx, "r") if r is not None else df.get(col_r)
        g_series = as_series(g, idx, "g") if g is not None else df.get(col_g)

        if r_series is None or g_series is None:
            raise ValueError(f"Missing required r/g: supply args or ensure columns '{col_r}' and '{col_g}' exist.")

        r_series = r_series.astype(float)
        g_series = g_series.astype(float)

        # Validate r > g everywhere; raise if violated
        bad = r_series <= g_series
        if bad.any():
            first_bad = bad.idxmax()
            raise ValueError(f"Invalid r<=g at {first_bad}: r={r_series.loc[first_bad]}, g={g_series.loc[first_bad]}")

        # Build D1 per share (priority)
        if col_dps_next in df.columns:
            d1_ps = df[col_dps_next].astype(float)
            path = pd.Series("DPS_next", index=idx)
        elif col_dps in df.columns:
            d1_ps = df[col_dps].astype(float) * (1.0 + g_series)
            path = pd.Series("DPS*(1+g)", index=idx)
        elif col_div_total_next in df.columns:
            d1_ps = df[col_div_total_next].astype(float) / sh
            path = pd.Series("DividendsNext / shares_out", index=idx)
        elif col_div_total in df.columns:
            d1_ps = (df[col_div_total].astype(float) * (1.0 + g_series)) / sh
            path = pd.Series("Dividends*(1+g) / shares_out", index=idx)
        else:
            raise ValueError(
                "DataFrame must contain one of: "
                f"'{col_dps_next}' | '{col_dps}'+{col_g} | '{col_div_total_next}' | '{col_div_total}'+{col_g}"
            )

        # Gordon formula (vectorized)
        v_ps = d1_ps / (r_series - g_series)
        equity = v_ps * sh

        out = pd.DataFrame(
            {
                "target_price": v_ps.astype(float),
                "equity_value": equity.astype(float),
                "r": r_series.astype(float),
                "g": g_series.astype(float),
                "path": path,
            },
            index=idx,
        )
        return out

