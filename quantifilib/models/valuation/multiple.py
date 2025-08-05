from typing import Dict, Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd

from .base import BaseValuation

class MultiplesValuation(BaseValuation):
    """
    Multiples-based (Relative) valuation engine.

    Supported multiples (case-insensitive)
    --------------------------------------
    Equity-based:
      - 'PE'       : Price / Earnings
      - 'PB'       : Price / Book
      - 'P/CF'     : Price / Operating Cash Flow
      - 'P/FCF'    : Price / Free Cash Flow
      - 'PS'       : Price / Sales  (a.k.a. P/S)
    Enterprise-based:
      - 'EV/EBITDA': Enterprise Value / EBITDA
      - 'EV/EBIT'  : Enterprise Value / EBIT
      - 'EV/Sales' : Enterprise Value / Sales

    Multiple source
    ---------------
      • Provide a numeric 'multiple' directly, OR
      • Provide 'peer_multiples' array and choose an aggregator:
          - agg = 'median' | 'mean' | ('percentile', q in [0,100])
      The chosen statistic becomes the multiple used.

    Inputs to evaluate()
    --------------------
    Common:
      - shares_out : float
      - timing     : Optional[str]  # e.g., 'forward', 'ttm' (for labeling only)

    Equity-based multiples (per-share OR total input accepted):
      - For PE : either 'eps' (per share) OR 'net_income' (total)
      - For PB : 'bvps' (per share) OR 'book_value' (total)
      - For P/CF: 'cfps' (per share) OR 'cash_flow' (total)
      - For P/FCF: 'fcfps' (per share) OR 'fcf' (total)
      - For PS : 'sps' (per share) OR 'sales' (total)

    Enterprise-based multiples require the denominator and a bridge to equity:
      - EV/EBITDA : 'ebitda' (total)
      - EV/EBIT   : 'ebit'   (total)
      - EV/Sales  : 'sales'  (total)
      Bridge (optional but recommended):
        - net_debt           : float (defaults to 0.0)
        - minority_interest  : float (defaults to 0.0)
        - preferred_equity   : float (defaults to 0.0)
        - associates         : float (add back; defaults to 0.0)
        Equity value = EV - net_debt - minority_interest - preferred_equity + associates

    Returns
    -------
    dict
      {
        'multiple_used'   : float,
        'equity_value'    : float,
        'value_per_share' : float,
        'bridge'          : dict   # for EV-based: EV and adjustments
        'meta'            : dict   # timing, method, etc.
      }

    Notes
    -----
    • This is a *relative valuation*: results depend on the multiple source.
    • Negative or near-zero denominators are guarded; raises ValueError.
    """

    def __init__(
        self,
        ratio: str,
        agg: Union[str, tuple] = "median",
        multiple: Optional[float] = None,
        peer_multiples: Optional[Sequence[float]] = None,
    ):
        """
        Configure the multiples valuation.

        Parameters
        ----------
        ratio : str
            One of {'PE','PB','P/CF','P/FCF','PS','EV/EBITDA','EV/EBIT','EV/Sales'} (case-insensitive).
        agg : 'median' | 'mean' | ('percentile', q)
            Aggregation for peer multiples when multiple is not directly provided.
        multiple : float, optional
            Fixed multiple to use (overrides peers if provided).
        peer_multiples : sequence of floats, optional
            Peer multiples set; used with 'agg' to derive the statistic.
        """
        self.ratio = ratio.upper()
        self.agg = agg
        self.multiple = float(multiple) if multiple is not None else None
        self.peer_multiples = np.asarray(peer_multiples, dtype=float) if peer_multiples is not None else None

        # Validate supported ratios
        self._equity_ratios = {"PE", "PB", "P/CF", "P/FCF", "PS"}
        self._ev_ratios = {"EV/EBITDA", "EV/EBIT", "EV/SALES"}
        if self.ratio not in (self._equity_ratios | self._ev_ratios):
            raise ValueError(f"Unsupported ratio '{ratio}'.")
        if isinstance(self.agg, tuple):
            if not (len(self.agg) == 2 and self.agg[0] == "percentile"):
                raise ValueError("Tuple agg must be ('percentile', q).")

    @classmethod
    def from_params(cls, params: Dict) -> "MultiplesValuation":
        """
        Build from a configuration dictionary.

        Expected keys
        -------------
          - ratio            : str
          - agg              : 'median' | 'mean' | ('percentile', q)
          - multiple         : float (optional)
          - peer_multiples   : sequence of floats (optional)
        """
        return cls(
            ratio=params.get("ratio"),
            agg=params.get("agg", "median"),
            multiple=params.get("multiple", None),
            peer_multiples=params.get("peer_multiples", None),
        )

    # ---------- helpers ----------
    def _choose_multiple(self) -> float:
        """
        Select the multiple to use: fixed value takes precedence; otherwise
        aggregate peer_multiples according to self.agg.
        """
        if self.multiple is not None:
            return float(self.multiple)
        if self.peer_multiples is None or self.peer_multiples.size == 0:
            raise ValueError("No multiple provided and no peer_multiples to aggregate.")
        arr = self.peer_multiples[~np.isnan(self.peer_multiples)]
        if arr.size == 0:
            raise ValueError("peer_multiples contains only NaNs.")
        if self.agg == "median":
            return float(np.median(arr))
        if self.agg == "mean":
            return float(np.mean(arr))
        if isinstance(self.agg, tuple) and self.agg[0] == "percentile":
            q = float(self.agg[1])
            if not (0.0 <= q <= 100.0):
                raise ValueError("percentile q must be in [0,100].")
            return float(np.percentile(arr, q))
        raise ValueError("Unsupported agg; use 'median' | 'mean' | ('percentile', q).")

    @staticmethod
    def _safe_get(inputs: Dict, keys: Sequence[str]) -> Optional[float]:
        """
        Return the first present key from 'keys' in inputs (as float), else None.
        """
        for k in keys:
            if k in inputs and inputs[k] is not None:
                return float(inputs[k])
        return None

    @staticmethod
    def _check_denominator(x: float, name: str):
        """
        Guard against invalid denominators for multiples.
        """
        if x is None:
            raise ValueError(f"Missing denominator '{name}'.")
        if abs(x) < 1e-12:
            raise ValueError(f"Denominator '{name}' is zero or too close to zero.")
        # Negative denominators are allowed in practice but often not meaningful;
        # raise to avoid misleading outputs.
        # Comment the next two lines if you deliberately want to allow negatives.
        if x < 0:
            raise ValueError(f"Denominator '{name}' is negative; multiple may be not meaningful.")

    @staticmethod
    def _as_series(x: Union[float, int, pd.Series], idx: pd.Index, name: str) -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.reindex(idx)
        else:
            s = pd.Series(float(x), index=idx)
        s.name = name
        return s

    def _choose_multiple_series(
            self,
            df: pd.DataFrame,
            idx: pd.Index,
            multiple: Optional[Union[float, pd.Series]] = None,
            peer_multiples: Optional[pd.DataFrame] = None,
            agg: Union[str, Tuple[str, float]] = "median",
    ) -> pd.Series:
        # Priority: explicit arg -> df['multiple'] -> peers row-agg
        if multiple is not None:
            return self._as_series(multiple, idx, "multiple_used")
        if "multiple" in df.columns:
            return df["multiple"].astype(float)
        if peer_multiples is not None:
            peers = peer_multiples.reindex(idx)
            if isinstance(agg, tuple) and agg[0] == "percentile":
                q = float(agg[1])
                m = peers.apply(lambda row: np.nanpercentile(row.values.astype(float), q), axis=1)
            elif agg == "mean":
                m = peers.mean(axis=1)
            else:  # default median
                m = peers.median(axis=1)
            return m.rename("multiple_used").astype(float)
        raise ValueError("No multiple provided: pass `multiple`, df['multiple'], or `peer_multiples`.")

    # ---------- core ----------
    def evaluate(self, inputs: Dict) -> Dict:
        """
        Compute target equity value and value per share using the configured multiple.

        Parameters
        ----------
        inputs : dict
          Common:
            - shares_out : float
            - timing     : Optional[str]  # e.g., 'forward', 'ttm' (for labeling only)

          Equity-based (provide per-share OR total; per-share takes precedence):
            - PE : 'eps' or 'net_income'
            - PB : 'bvps' or 'book_value'
            - P/CF : 'cfps' or 'cash_flow'
            - P/FCF: 'fcfps' or 'fcf'
            - PS : 'sps' or 'sales'

          Enterprise-based (totals only) + EV→Equity bridge:
            - EV/EBITDA : 'ebitda'
            - EV/EBIT   : 'ebit'
            - EV/Sales  : 'sales'
            Bridge fields (optional):
              'net_debt', 'minority_interest', 'preferred_equity', 'associates'

        Returns
        -------
        dict with keys:
          - multiple_used
          - equity_value
          - value_per_share
          - bridge (for EV-based ratios)
          - meta
        """
        if "shares_out" not in inputs:
            raise ValueError("Missing 'shares_out'.")
        shares_out = max(float(inputs["shares_out"]), 1e-9)

        multiple_used = self._choose_multiple()
        bridge_info = {}

        # ---- Equity-based multiples ----
        if self.ratio in self._equity_ratios:
            if self.ratio == "PE":
                # Price = multiple * EPS  (or equity = multiple * NetIncome)
                eps = self._safe_get(inputs, ["eps"])
                if eps is not None:
                    price = multiple_used * eps
                    equity_value = price * shares_out
                else:
                    ni = self._safe_get(inputs, ["net_income"])
                    self._check_denominator(ni, "net_income")
                    equity_value = multiple_used * ni
                    price = equity_value / shares_out

            elif self.ratio == "PB":
                # Price = multiple * BVPS (or equity = multiple * BookValue)
                bvps = self._safe_get(inputs, ["bvps"])
                if bvps is not None:
                    price = multiple_used * bvps
                    equity_value = price * shares_out
                else:
                    bv = self._safe_get(inputs, ["book_value"])
                    self._check_denominator(bv, "book_value")
                    equity_value = multiple_used * bv
                    price = equity_value / shares_out

            elif self.ratio == "P/CF":
                # Price = multiple * CFPS (or equity = multiple * CashFlow)
                cfps = self._safe_get(inputs, ["cfps"])
                if cfps is not None:
                    price = multiple_used * cfps
                    equity_value = price * shares_out
                else:
                    cf = self._safe_get(inputs, ["cash_flow"])
                    self._check_denominator(cf, "cash_flow")
                    equity_value = multiple_used * cf
                    price = equity_value / shares_out

            elif self.ratio == "P/FCF":
                # Price = multiple * FCFPS (or equity = multiple * FCF)
                fcfps = self._safe_get(inputs, ["fcfps"])
                if fcfps is not None:
                    price = multiple_used * fcfps
                    equity_value = price * shares_out
                else:
                    fcf = self._safe_get(inputs, ["fcf"])
                    self._check_denominator(fcf, "fcf")
                    equity_value = multiple_used * fcf
                    price = equity_value / shares_out

            elif self.ratio == "PS":
                # Price = multiple * Sales per Share (or equity = multiple * Sales)
                sps = self._safe_get(inputs, ["sps"])
                if sps is not None:
                    price = multiple_used * sps
                    equity_value = price * shares_out
                else:
                    sales = self._safe_get(inputs, ["sales"])
                    self._check_denominator(sales, "sales")
                    equity_value = multiple_used * sales
                    price = equity_value / shares_out

            else:
                raise ValueError(f"Unsupported equity ratio '{self.ratio}'.")

        # ---- Enterprise-based multiples ----
        else:
            # Determine denominator
            if self.ratio == "EV/EBITDA":
                denom = self._safe_get(inputs, ["ebitda"])
                name = "ebitda"
            elif self.ratio == "EV/EBIT":
                denom = self._safe_get(inputs, ["ebit"])
                name = "ebit"
            elif self.ratio == "EV/SALES":
                denom = self._safe_get(inputs, ["sales"])
                name = "sales"
            else:
                raise ValueError(f"Unsupported EV ratio '{self.ratio}'.")

            self._check_denominator(denom, name)

            # Implied EV
            ev = multiple_used * denom

            # Bridge to equity
            net_debt = float(inputs.get("net_debt", 0.0))
            minority = float(inputs.get("minority_interest", 0.0))
            preferred = float(inputs.get("preferred_equity", 0.0))
            associates = float(inputs.get("associates", 0.0))  # add-back

            equity_value = ev - net_debt - minority - preferred + associates
            price = equity_value / shares_out

            bridge_info = {
                "EV": float(ev),
                "net_debt": net_debt,
                "minority_interest": minority,
                "preferred_equity": preferred,
                "associates": associates,
            }

        return {
            "multiple_used": float(multiple_used),
            "equity_value": float(equity_value),
            "value_per_share": float(price),
            "bridge": bridge_info,
            "meta": {
                "ratio": self.ratio,
                "timing": inputs.get("timing", None),
                "source": "fixed" if self.multiple is not None else "peers",
                "agg": self.agg if self.multiple is None else None,
            },
        }

    def evaluate_df(
            self,
            df: pd.DataFrame,
            shares_out: Union[float, pd.Series],
            *,
            multiple: Optional[Union[float, pd.Series]] = None,
            peer_multiples: Optional[pd.DataFrame] = None,
            agg: Union[str, Tuple[str, float]] = "median",
    ) -> pd.DataFrame:
        """
        Compute a target price time-series from time-series multiples and fundamentals.

        Parameters
        ----------
        df : DataFrame
            Time-indexed inputs (see class docstring for required columns per ratio).
        shares_out : float | Series
            Shares outstanding (scalar or time series aligned to df.index).
        multiple : float | Series, optional
            Multiple to use (scalar or time series). Overrides df['multiple'] and peers.
        peer_multiples : DataFrame, optional
            Peer multiples time series (columns = peers). Row-wise aggregation is used.
        agg : 'median' | 'mean' | ('percentile', q)
            Aggregator when peer_multiples is provided.

        Returns
        -------
        DataFrame
            Columns: ['target_price','equity_value','multiple_used', ...EV bridge columns if applicable]
        """
        df = df.copy()
        df = df.sort_index()
        idx = df.index

        # multiple series
        m = self._choose_multiple_series(df, idx, multiple, peer_multiples, agg)
        if m.isna().any():
            m = m.fillna(method="ffill").fillna(method="bfill")

        # shares outstanding series
        sh = self._as_series(shares_out, idx, "shares_out")

        ratio = self.ratio.upper()
        eq_ratios = {"PE", "PB", "P/CF", "P/FCF", "PS"}
        ev_ratios = {"EV/EBITDA", "EV/EBIT", "EV/SALES"}
        out_cols = {}

        def pick(df: pd.DataFrame, first: str, fallback: str) -> pd.Series:
            if first in df.columns:
                return df[first].astype(float)
            if fallback in df.columns:
                return df[fallback].astype(float)
            raise ValueError(f"Missing required column: '{first}' or '{fallback}'.")

        if ratio in eq_ratios:
            if ratio == "PE":
                # price = multiple * EPS  (or equity = multiple * NetIncome)
                if "EPS" in df.columns:
                    price = m * df["EPS"].astype(float)
                    equity = price * sh
                else:
                    ni = pick(df, "NetIncome", "EPS")  # fallback EPS if NI missing
                    if (ni <= 0).any():
                        raise ValueError("NetIncome must be positive for P/E-based valuation.")
                    equity = m * ni
                    price = equity / sh

            elif ratio == "PB":
                if "BVPS" in df.columns:
                    price = m * df["BVPS"].astype(float)
                    equity = price * sh
                else:
                    bv = pick(df, "BookValue", "BVPS")
                    if (bv <= 0).any():
                        raise ValueError("BookValue must be positive for P/B-based valuation.")
                    equity = m * bv
                    price = equity / sh

            elif ratio == "P/CF":
                if "CFPS" in df.columns:
                    price = m * df["CFPS"].astype(float)
                    equity = price * sh
                else:
                    cf = pick(df, "CashFlow", "CFPS")
                    if (cf <= 0).any():
                        raise ValueError("CashFlow must be positive for P/CF-based valuation.")
                    equity = m * cf
                    price = equity / sh

            elif ratio == "P/FCF":
                if "FCFPS" in df.columns:
                    price = m * df["FCFPS"].astype(float)
                    equity = price * sh
                else:
                    fcf = pick(df, "FCF", "FCFPS")
                    if (fcf <= 0).any():
                        raise ValueError("FCF must be positive for P/FCF-based valuation.")
                    equity = m * fcf
                    price = equity / sh

            elif ratio == "PS":
                if "SPS" in df.columns:
                    price = m * df["SPS"].astype(float)
                    equity = price * sh
                else:
                    sales = pick(df, "Sales", "SPS")
                    if (sales <= 0).any():
                        raise ValueError("Sales must be positive for P/S-based valuation.")
                    equity = m * sales
                    price = equity / sh

            out = pd.DataFrame(
                {"target_price": price.astype(float), "equity_value": equity.astype(float),
                 "multiple_used": m.astype(float)},
                index=idx,
            )
            return out

        elif ratio in ev_ratios:
            if ratio == "EV/EBITDA":
                denom = pick(df, "EBITDA", "EBIT")
                denom_name = "EBITDA"
            elif ratio == "EV/EBIT":
                denom = pick(df, "EBIT", "EBITDA")
                denom_name = "EBIT"
            else:  # EV/SALES
                denom = pick(df, "Sales", "Revenue")  # allow 'Revenue' alias
                denom_name = "Sales"

            if (denom <= 0).any():
                raise ValueError(f"{denom_name} must be positive for EV-based multiples.")

            ev = m * denom

            # EV → Equity bridge (defaults 0 if columns missing)
            net_debt = df.get("NetDebt", 0.0)
            minority = df.get("MinorityInterest", 0.0)
            preferred = df.get("PreferredEquity", 0.0)
            associates = df.get("Associates", 0.0)

            for c in [net_debt, minority, preferred, associates]:
                if isinstance(c, pd.Series):
                    c.index = idx

            equity = ev - pd.Series(net_debt, index=idx) - pd.Series(minority, index=idx) \
                     - pd.Series(preferred, index=idx) + pd.Series(associates, index=idx)
            price = equity / sh

            out = pd.DataFrame(
                {
                    "target_price": price.astype(float),
                    "equity_value": equity.astype(float),
                    "multiple_used": m.astype(float),
                    "EV": ev.astype(float),
                    "NetDebt": pd.Series(net_debt, index=idx).astype(float),
                    "MinorityInterest": pd.Series(minority, index=idx).astype(float),
                    "PreferredEquity": pd.Series(preferred, index=idx).astype(float),
                    "Associates": pd.Series(associates, index=idx).astype(float),
                },
                index=idx,
            )
            return out

        else:
            raise ValueError(f"Unsupported ratio '{ratio}'.")