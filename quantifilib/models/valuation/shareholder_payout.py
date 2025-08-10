from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

from .base import BaseValuation
from .utils import *

class TwoStageDDMValuation(BaseValuation):
    """
    Two-stage Dividend Discount Model.

    Stage 1 (t=1..N): high growth g_h
    Terminal at t=N: PV of stable stage using Gordon with g_s, based on D_{N+1} = D_0*(1+g_h)^N*(1+g_s)

    Inputs (evaluate):
      - shares_out
      - cost_of_equity (r)
      - g_high, g_stable, N_years
      - dividend inputs: d0 | d1 | dps (+g_high) | dividends/dividends_next (total)

    Constraints:
      - r > g_stable (terminal well-defined)
    """

    @classmethod
    def from_params(cls, params: Dict) -> "TwoStageDDMValuation":
        return cls()

    def _d0_per_share(self, inputs: Dict, shares_out: float) -> float:
        # First try D1, then back out D0 if g_high provided
        d1 = pick_first(inputs, ["d1", "dps_next", "dividend_per_share_next"])
        if d1 is not None:
            g_h = pick_first(inputs, ["g_high", "growth_high"])
            if g_h is None:
                # If D1 directly supplied but no g_h, treat it as D0 (conservative)
                return d1
            return d1 / (1.0 + g_h)

        d0 = pick_first(inputs, ["d0", "dps", "dividend_per_share"])
        if d0 is not None:
            return d0

        # Totals path
        div_next_total = pick_first(inputs, ["dividends_next", "dividend_total_next"])
        if div_next_total is not None:
            g_h = pick_first(inputs, ["g_high", "growth_high"])
            if g_h is None:
                # treat dividends_next as total D0
                return (div_next_total / shares_out)
            return (div_next_total / shares_out) / (1.0 + g_h)

        div_total = pick_first(inputs, ["dividends", "dividend_total"])
        if div_total is not None:
            return (div_total / shares_out)

        raise ValueError("Provide one of: d0 | d1 | dps | dividends[_next].")

    def evaluate(self, inputs: Dict) -> Dict:
        if "shares_out" not in inputs:
            raise ValueError("Missing 'shares_out'.")
        shares_out = max(float(inputs["shares_out"]), 1e-12)

        r = pick_first(inputs, ["cost_of_equity", "r", "ke"])
        g_h = pick_first(inputs, ["g_high", "growth_high"])
        g_s = pick_first(inputs, ["g_stable", "growth_stable"])
        N = int(pick_first(inputs, ["N", "N_years", "years"]) or 0)
        if r is None or g_h is None or g_s is None or N <= 0:
            raise ValueError("Require r, g_high, g_stable, N_years>0.")

        if r <= g_s:
            raise ValueError(f"[Two-Stage] invalid r<=g_stable (r={r}, g_s={g_s}).")

        D0 = self._d0_per_share(inputs, shares_out)  # per share
        # Stage 1 PV of dividends
        pv_stage1 = 0.0
        for t in range(1, N + 1):
            Dt = D0 * ((1.0 + g_h) ** t)
            pv_stage1 += Dt / ((1.0 + r) ** t)

        # Terminal (stable) value at N using GGM on D_{N+1}
        D_N1 = D0 * ((1.0 + g_h) ** N) * (1.0 + g_s)
        tv_N = D_N1 / (r - g_s)
        pv_terminal = tv_N / ((1.0 + r) ** N)

        v_ps = pv_stage1 + pv_terminal
        equity = v_ps * shares_out

        return {
            "equity_value": float(equity),
            "value_per_share": float(v_ps),
            "meta": {
                "model": "TwoStageDDM",
                "r": float(r),
                "g_high": float(g_h),
                "g_stable": float(g_s),
                "N": int(N),
                "timing": inputs.get("timing", None),
            },
        }

    def evaluate_df(
        self,
        df: pd.DataFrame,
        shares_out: Union[float, pd.Series],
        *,
        col_d0: str = "DPS",
        col_d1: str = "DPS_next",
        col_r: str = "CostOfEquity",
        col_gh: str = "GrowthHigh",
        col_gs: str = "GrowthStable",
        col_N: str = "NYears",
    ) -> pd.DataFrame:
        df = df.copy().sort_index()
        idx = df.index

        sh = as_series(shares_out, idx, "shares_out").astype(float)
        r = df[col_r].astype(float)
        g_h = df[col_gh].astype(float)
        g_s = df[col_gs].astype(float)
        N = df[col_N].astype(int)

        if (r <= g_s).any():
            t = (r <= g_s).idxmax()
            raise ValueError(f"[Two-Stage] r<=g_stable at {t}: r={r.loc[t]}, g_s={g_s.loc[t]}")

        # D0 per share: prefer D1 if provided, back out; else DPS as D0
        if col_d1 in df.columns and df[col_d1].notna().any():
            d1 = df[col_d1].astype(float).fillna(method="ffill")
            d0 = d1 / (1.0 + g_h)
            path = pd.Series("D1_backout", index=idx)
        elif col_d0 in df.columns:
            d0 = df[col_d0].astype(float)
            path = pd.Series("D0_direct", index=idx)
        else:
            raise ValueError(f"Missing '{col_d0}' or '{col_d1}' for dividends per share.")

        # Vectorized loop per row (N differs by row)
        out = []
        for t0 in idx:
            rr, gh, gs, nn, d0_t, sh_t = r.loc[t0], g_h.loc[t0], g_s.loc[t0], int(N.loc[t0]), float(d0.loc[t0]), float(sh.loc[t0])
            # Stage 1
            terms = [(d0_t * ((1.0 + gh) ** t)) / ((1.0 + rr) ** t) for t in range(1, nn + 1)]
            pv_stage1 = float(np.sum(terms)) if terms else 0.0
            # Terminal
            d_n1 = d0_t * ((1.0 + gh) ** nn) * (1.0 + gs)
            tv_n = d_n1 / (rr - gs)
            pv_terminal = tv_n / ((1.0 + rr) ** nn)
            v_ps = pv_stage1 + pv_terminal
            equity = v_ps * sh_t
            out.append((v_ps, equity))

        out = pd.DataFrame(out, columns=["target_price", "equity_value"], index=idx)
        out["r"] = r
        out["g_high"] = g_h
        out["g_stable"] = g_s
        out["N"] = N
        out["path"] = path
        return out

class HModelValuation(BaseValuation):
    """
    H-Model (Fuller's H-model) — linear fade from g_high to g_stable over N years.

    Approximation formula:
      P0 = D0 * (1+g_s) / (r - g_s)  +  D0 * H * (g_high - g_stable) / (r - g_s),
      where H = N/2, r > g_s

    Inputs:
      - shares_out, r, g_high, g_stable, N
      - D0 per share via: d0 | dps | (if only d1 given, D0 = d1/(1+g_high))

    Notes:
      - This is an approximation; use TwoStageDDM for exact PV by year.
    """

    @classmethod
    def from_params(cls, params: Dict) -> "HModelValuation":
        return cls()

    def _d0_per_share(self, inputs: Dict, shares_out: float, g_high: float) -> float:
        d0 = pick_first(inputs, ["d0", "dps", "dividend_per_share"])
        if d0 is not None:
            return d0
        d1 = pick_first(inputs, ["d1", "dps_next", "dividend_per_share_next"])
        if d1 is not None and g_high is not None:
            return d1 / (1.0 + g_high)
        div_total = pick_first(inputs, ["dividends", "dividend_total"])
        if div_total is not None:
            return div_total / shares_out
        raise ValueError("Provide D0 via d0/dps OR d1 back-out OR total dividends.")

    def evaluate(self, inputs: Dict) -> Dict:
        if "shares_out" not in inputs:
            raise ValueError("Missing 'shares_out'.")
        shares_out = max(float(inputs["shares_out"]), 1e-12)

        r = pick_first(inputs, ["cost_of_equity", "r", "ke"])
        g_h = pick_first(inputs, ["g_high", "growth_high"])
        g_s = pick_first(inputs, ["g_stable", "growth_stable"])
        N = int(pick_first(inputs, ["N", "N_years", "years"]) or 0)
        if r is None or g_h is None or g_s is None or N <= 0:
            raise ValueError("Require r, g_high, g_stable, N_years>0.")
        if r <= g_s:
            raise ValueError(f"[H-Model] invalid r<=g_stable (r={r}, g_s={g_s}).")

        D0 = self._d0_per_share(inputs, shares_out, g_h)
        H = N / 2.0
        v_ps = D0 * ((1.0 + g_s) / (r - g_s)) + D0 * H * (g_h - g_s) / (r - g_s)
        equity = v_ps * shares_out

        return {
            "equity_value": float(equity),
            "value_per_share": float(v_ps),
            "meta": {
                "model": "HModel",
                "r": float(r),
                "g_high": float(g_h),
                "g_stable": float(g_s),
                "N": int(N),
                "timing": inputs.get("timing", None),
            },
        }

    def evaluate_df(
        self,
        df: pd.DataFrame,
        shares_out: Union[float, pd.Series],
        *,
        col_d0: str = "DPS",
        col_d1: str = "DPS_next",
        col_r: str = "CostOfEquity",
        col_gh: str = "GrowthHigh",
        col_gs: str = "GrowthStable",
        col_N: str = "NYears",
    ) -> pd.DataFrame:
        df = df.copy().sort_index()
        idx = df.index

        sh = as_series(shares_out, idx, "shares_out").astype(float)
        r = df[col_r].astype(float)
        g_h = df[col_gh].astype(float)
        g_s = df[col_gs].astype(float)
        N = df[col_N].astype(int)

        if (r <= g_s).any():
            t = (r <= g_s).idxmax()
            raise ValueError(f"[H-Model] r<=g_stable at {t}: r={r.loc[t]}, g_s={g_s.loc[t]}")

        if col_d0 in df.columns:
            D0 = df[col_d0].astype(float)
            path = pd.Series("D0_direct", index=idx)
        elif col_d1 in df.columns:
            D0 = (df[col_d1].astype(float) / (1.0 + g_h))
            path = pd.Series("D1_backout", index=idx)
        else:
            raise ValueError(f"Missing '{col_d0}' or '{col_d1}' for dividends per share (D0).")

        H = N.astype(float) / 2.0
        v_ps = D0 * ((1.0 + g_s) / (r - g_s)) + D0 * H * (g_h - g_s) / (r - g_s)
        equity = v_ps * sh

        out = pd.DataFrame(
            {
                "target_price": v_ps.astype(float),
                "equity_value": equity.astype(float),
                "r": r,
                "g_high": g_h,
                "g_stable": g_s,
                "N": N,
                "path": path,
            },
            index=idx,
        )
        return out