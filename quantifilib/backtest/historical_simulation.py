import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Sequence


class HistoricalSimulation:
    """
    Historical backtesting engine with rich market-microstructure features.

    Core capabilities
    -----------------
    • Rebalance on given dates with target weights.
    • Separate execution and valuation pricing:
        - Either pass `data` (valuation) and optional `exec_prices` (execution),
        - Or pass `ohlc` dict and pick fields via `valuation_field` / `execution_field`.
    • Trading frictions:
        - Slippage (bps) -> side-specific price adjustment.
        - Commission table: variable bps + fixed fee + minimum per trade.
        - Trade cap per ticker as % of portfolio value.
    • Share rounding:
        - Integer/lot-based sizing and tick-size snapping on prices.
        - Cash account managed; scale buys to avoid negative cash.
    • Cash interest:
        - Daily accrual: deposit for positive cash, borrow for negative cash.
    • Mark-to-market daily with frozen shares between rebalances.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Valuation price panel (e.g., close). Used if `ohlc` not provided.
    weights_df : pd.DataFrame
        Target weights on rebalance dates. Aligned to `rebalance_dates` x tickers of price panel.
    rebalance_dates : Sequence[pd.Timestamp]
        Rebalance dates (must exist in valuation price index).
    initial_value : float, default 1000.0
        Starting portfolio cash (base currency).
    normalize_weights : bool, default True
        Normalize each weight row to sum to 1 (zero-sum rows remain zero).
    fill_method : {'ffill','bfill',None}, default None
        Optional fill for valuation panel (and exec panel if provided).

    # Execution/valuation separation
    use_next_open : bool, default False
        If True, trade at next session's execution prices (e.g., next open).
    exec_prices : pd.DataFrame, optional
        Execution price panel when not using `ohlc`. Required if `use_next_open=True`.
    ohlc : Dict[str, pd.DataFrame], optional
        Dict with any subset of {'open','high','low','close'} as DataFrames.
        If provided, use `valuation_field`/`execution_field` to select price source.
    valuation_field : str, default 'close'
        Which OHLC field to use for daily mark-to-market.
    execution_field : str, default 'open' if use_next_open else 'close'
        Which OHLC field to use for executions.

    # Trading frictions & sizing
    slippage_bps : float, default 0.0
        Buy price = px*(1+slip), sell price = px*(1-slip).
    commission : Dict[str, float], optional
        {'variable_bps': float, 'fixed': float, 'min_per_trade': float}. Missing keys default to 0.
    allow_partial_shares : bool, default True
        If False, round to lot multiples and manage cash accordingly.
    min_lot : int or Dict[str,int], default 1
        Minimum lot size per trade (1 = integer shares). May be scalar or per-ticker dict.
    tick_size : float or Dict[str,float], default 0.0
        Tick size for price snapping. Buy -> ceil to tick, sell -> floor to tick.
    cap_trade_pct : float, default 1.0
        Per-ticker cap on notional traded as a fraction of portfolio value at rebalance.

    # Cash interest
    deposit_rate_annual : float, default 0.0
        Annual deposit rate (applied when cash >= 0).
    borrow_rate_annual : float, default 0.0
        Annual borrowing rate (applied when cash < 0).

    Attributes (after run)
    ----------------------
    index_series : pd.Series
        Daily portfolio value time series on valuation index.
    cash_series : pd.Series
        Daily cash balance (after accrual & trades).
    shares_record : pd.DataFrame
        Share counts snapshot at execution moments (rows: exec dates).
    trade_ledger : pd.DataFrame
        Executed trades per (date,ticker) with prices, notionals, fees.
    shares_ledger : pd.DataFrame
        Holdings snapshots per (date,ticker) at execution moments.
    logs : List[str]
        Validation and informational messages.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame],
        weights_df: pd.DataFrame,
        rebalance_dates: Union[List[pd.Timestamp], pd.DatetimeIndex],
        *,
        initial_value: float = 1000.0,
        normalize_weights: bool = True,
        fill_method: Optional[str] = None,
        # execution/valuation
        use_next_open: bool = False,
        exec_prices: Optional[pd.DataFrame] = None,
        ohlc: Optional[Dict[str, pd.DataFrame]] = None,
        valuation_field: str = "close",
        execution_field: Optional[str] = None,
        # frictions & sizing
        slippage_bps: float = 0.0,
        commission: Optional[Dict[str, float]] = None,
        allow_partial_shares: bool = True,
        min_lot: Union[int, Dict[str, int]] = 1,
        tick_size: Union[float, Dict[str, float]] = 0.0,
        cap_trade_pct: float = 1.0,
        # cash interest
        deposit_rate_annual: float = 0.0,
        borrow_rate_annual: float = 0.0,
    ):
        self.data = None if data is None else data.copy()
        self.ohlc = None if ohlc is None else {k: v.copy() for k, v in ohlc.items()}
        self.weights_df = weights_df.copy()
        self.rebalance_dates = pd.DatetimeIndex(rebalance_dates).sort_values()
        self.initial_value = float(initial_value)
        self.normalize_weights = bool(normalize_weights)
        self.fill_method = fill_method

        self.use_next_open = bool(use_next_open)
        self.exec_prices = None if exec_prices is None else exec_prices.copy()
        self.valuation_field = valuation_field
        self.execution_field = (
            execution_field if execution_field is not None
            else ("open" if use_next_open else "close")
        )

        self.slippage_bps = float(slippage_bps)
        self.commission = {
            "variable_bps": 0.0,
            "fixed": 0.0,
            "min_per_trade": 0.0,
        }
        if commission:
            self.commission.update({k: float(v) for k, v in commission.items()})

        self.allow_partial_shares = bool(allow_partial_shares)
        self.min_lot = min_lot
        self.tick_size = tick_size
        self.cap_trade_pct = float(cap_trade_pct)

        self.deposit_rate_annual = float(deposit_rate_annual)
        self.borrow_rate_annual = float(borrow_rate_annual)

        # results
        self.index_series: Optional[pd.Series] = None
        self.cash_series: Optional[pd.Series] = None
        self.shares_record: Optional[pd.DataFrame] = None
        self.trade_ledger: Optional[pd.DataFrame] = None
        self.shares_ledger: Optional[pd.DataFrame] = None
        self.logs: List[str] = []

    @staticmethod
    def weights_long_to_wide(
            weights_long: pd.DataFrame,
            price_cols: pd.Index,
            rebalance_dates: pd.DatetimeIndex,
            agg: str = 'last'
        ) -> pd.DataFrame:
        wl = weights_long.copy()
        wl['date'] = pd.to_datetime(wl['date'])
        wide = (wl.pivot_table(index='date', columns='ticker', values='weight', aggfunc=agg)
                .reindex(rebalance_dates)
                .reindex(columns=price_cols)
                .fillna(0.0)
                .sort_index())
        return wide

    # ------------------------- validation & alignment ------------------------- #
    def _log(self, msg: str):
        self.logs.append(msg)

    def _valuation_panel(self) -> pd.DataFrame:
        """
        Return the valuation price DataFrame used for MTM each day.
        Priority: ohlc[valuation_field] -> data
        """
        if self.ohlc is not None and self.valuation_field in self.ohlc:
            return self.ohlc[self.valuation_field]
        if self.data is not None:
            return self.data
        raise ValueError("[FATAL] No valuation price panel available.")

    def _execution_panel(self) -> pd.DataFrame:
        """
        Return the execution price DataFrame used for trading.
        Priority: ohlc[execution_field] -> exec_prices -> data
        """
        if self.ohlc is not None and self.execution_field in self.ohlc:
            return self.ohlc[self.execution_field]
        if self.exec_prices is not None:
            return self.exec_prices
        if self.data is not None:
            return self.data
        raise ValueError("[FATAL] No execution price panel available.")

    def _validate_inputs(self) -> None:
        """Align inputs and check for common issues."""
        val_px = self._valuation_panel().astype(float)
        exe_px = self._execution_panel().astype(float)

        # Optional fill
        if self.fill_method in ("ffill", "bfill"):
            val_px = getattr(val_px, self.fill_method)()
            exe_px = getattr(exe_px, self.fill_method)()

        self.val_px = val_px
        self.exe_px = exe_px

        # Align columns
        cols = self.val_px.columns
        # Align execution panel columns to valuation panel
        self.exe_px = self.exe_px.reindex(columns=cols)

        # weights: align to rebalances and columns
        original_cols = set(self.weights_df.columns)
        self.weights_df = self.weights_df.reindex(index=self.rebalance_dates, columns=cols).fillna(0.0)
        dropped = original_cols - set(cols)
        if dropped:
            self._log(f"[WARN] Dropping weight columns not in price panel: {sorted(dropped)}")

        # check rebalance dates present
        if not self.rebalance_dates.isin(self.val_px.index).all():
            missing = self.rebalance_dates[~self.rebalance_dates.isin(self.val_px.index)]
            raise ValueError(f"[FATAL] Rebalance dates missing in valuation index: {missing.tolist()}")

        # When using next open, ensure an execution date exists strictly after each rebalance
        if self.use_next_open:
            for dt in self.rebalance_dates:
                pos = self.exe_px.index.searchsorted(dt, side="right")
                if pos >= len(self.exe_px.index):
                    raise ValueError(f"[FATAL] No next execution row after {dt} in execution panel index.")

        # Price sanity checks
        if (self.val_px <= 0).any().any():
            self._log("[WARN] Non-positive valuation prices found.")
        if (self.exe_px <= 0).any().any():
            self._log("[WARN] Non-positive execution prices found.")

        # Weight sanity checks
        wsum = self.weights_df.sum(axis=1)
        not_one = ~np.isclose(wsum, 1.0) & ~(np.isclose(wsum, 0.0))
        if not_one.any():
            bad = wsum[not_one]
            self._log(f"[WARN] Some weight rows do not sum to 1 (and not 0). Sample: {bad.head().to_dict()}")

    # ------------------------- microstructure helpers ------------------------- #
    def _daily_rate(self, annual: float) -> float:
        """Convert annual simple rate to daily compounding factor - 1, using 252 days."""
        return (1.0 + annual) ** (1.0 / 252.0) - 1.0

    def _snap_tick(self, price: float, side: str, tick: float) -> float:
        """Snap price to tick size: buy -> ceil to tick, sell -> floor to tick."""
        if tick <= 0:
            return float(price)
        if side == "buy":
            return np.ceil(price / tick) * tick
        else:
            return np.floor(price / tick) * tick

    def _lot_round(self, shares: float, lot: int) -> float:
        """Round shares to nearest lot multiple (toward zero change)."""
        if lot <= 1:
            return np.floor(shares)  # integer shares
        return np.floor(shares / lot) * lot

    def _eff_price(self, px: float, side: str, tick: float) -> float:
        """Apply slippage and tick snap to the execution price."""
        slip = self.slippage_bps / 1e4
        raw = px * (1.0 + slip if side == "buy" else 1.0 - slip)
        return self._snap_tick(raw, side, tick)

    def _get_scalar_or_map(self, spec: Union[float, int, Dict[str, Any]], ticker: str) -> float:
        """Return scalar or per-ticker mapped value."""
        if isinstance(spec, dict):
            return float(spec.get(ticker, list(spec.values())[0])) if spec else 0.0
        return float(spec)

    def _normalize_row(self, w: pd.Series) -> pd.Series:
        if not self.normalize_weights:
            return w.fillna(0.0)
        s = w.sum(skipna=True)
        if np.isclose(s, 0.0):
            return w.fillna(0.0)
        return (w / s).fillna(0.0)

    # ------------------------------ core run ------------------------------ #
    def run(self) -> "HistoricalSimulation":
        """
        Execute the simulation. Results are stored on the instance attributes.
        """
        self._validate_inputs()

        idx = self.val_px.index
        cols = self.val_px.columns

        index_series = pd.Series(index=idx, dtype=float)
        cash_series = pd.Series(0.0, index=idx, dtype=float)
        shares_record = pd.DataFrame(0.0, index=[], columns=cols)

        trade_rows: List[Dict[str, Any]] = []
        hold_rows: List[Dict[str, Any]] = []

        prev_shares = pd.Series(0.0, index=cols)
        cash = float(self.initial_value)

        # precompute cash daily factors
        dep_d = self._daily_rate(self.deposit_rate_annual)
        bor_d = self._daily_rate(self.borrow_rate_annual)

        # daily MTM & accrual helper
        def apply_mtm_and_accrual(dates: pd.DatetimeIndex, shares_vec: pd.Series, cash_balance: float):
            nonlocal index_series, cash_series
            if len(dates) == 0:
                return cash_balance
            # iterate each day to accrue interest on cash
            for d in dates:
                # MTM at valuation price on d
                port_val_ex_cash = float((self.val_px.loc[d] * shares_vec).sum())
                # apply one-day accrual on cash BEFORE storing end-of-day cash
                if cash_balance >= 0:
                    cash_balance *= (1.0 + dep_d)
                else:
                    cash_balance *= (1.0 + bor_d)
                index_series.loc[d] = port_val_ex_cash + cash_balance
                cash_series.loc[d] = cash_balance
            return cash_balance

        # main loop
        for i, dt in enumerate(self.rebalance_dates):
            # MTM from previous exec to today's dt-1 (handled by previous iteration)
            # Value at dt close BEFORE trading
            port_value_dt = float((self.val_px.loc[dt] * prev_shares).sum() + cash)

            # desired target weights -> target notional
            w = self._normalize_row(self.weights_df.loc[dt])
            target_value = w * port_value_dt

            # choose execution date and price snapshot
            if self.use_next_open:
                pos = self.exe_px.index.searchsorted(dt, side="right")
                exec_dt = self.exe_px.index[pos]
            else:
                exec_dt = dt
            px_exec_row = self.exe_px.loc[exec_dt]

            # initial target shares (pre-rounding)
            with np.errstate(divide="ignore", invalid="ignore"):
                tgt_shares = (target_value / px_exec_row.replace(0.0, np.nan)).fillna(0.0)

            delta = tgt_shares - prev_shares

            # cap per ticker notional vs portfolio value
            if self.cap_trade_pct < 1.0 - 1e-12:
                cap_abs = self.cap_trade_pct * port_value_dt
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    side = "buy" if d > 0 else "sell"
                    tick = self._get_scalar_or_map(self.tick_size, tic)
                    px_eff = self._eff_price(float(px_exec_row[tic]), side, tick)
                    notional = abs(d) * px_eff
                    if notional > cap_abs > 0:
                        scale = cap_abs / notional
                        delta[tic] = d * scale
                        self._log(f"[INFO] {dt.date()} {tic}: trade capped to {self.cap_trade_pct:.2%} PV.")

            # share rounding (lot / integer policy)
            if not self.allow_partial_shares or (isinstance(self.min_lot, dict) or self.min_lot != 1):
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    lot = int(self._get_scalar_or_map(self.min_lot, tic))
                    if d > 0:
                        delta[tic] = self._lot_round(d, lot)
                    else:
                        # for sells, rounding toward zero exposure change => ceil in magnitude
                        delta[tic] = -self._lot_round(-d, lot)

            # price/tick & cost computation
            buy_notional = 0.0
            sell_notional = 0.0
            variable_bps = self.commission.get("variable_bps", 0.0) / 1e4
            fixed_fee = self.commission.get("fixed", 0.0)
            min_fee = self.commission.get("min_per_trade", 0.0)

            trade_cost_total = 0.0
            eff_px_map: Dict[str, float] = {}

            # first pass: compute notionals and fees, and tentative cash
            for tic in cols:
                d = float(delta[tic])
                if np.isclose(d, 0.0):
                    continue
                side = "buy" if d > 0 else "sell"
                tick = self._get_scalar_or_map(self.tick_size, tic)
                epx = self._eff_price(float(px_exec_row[tic]), side, tick)
                eff_px_map[tic] = epx
                notional = abs(d) * epx
                # per-trade commission: variable + fixed, with minimum
                fee = max(notional * variable_bps + fixed_fee, min_fee)
                trade_cost_total += fee
                if d > 0:
                    buy_notional += notional
                else:
                    sell_notional += notional

            tentative_cash = cash + sell_notional - buy_notional - trade_cost_total

            # scale down buys if cash deficit
            if tentative_cash < -1e-8 and buy_notional > 0:
                scale = max(0.0, (cash + sell_notional - trade_cost_total) / buy_notional)
                self._log(f"[INFO] {dt.date()} scaled buys by {scale:.4f} to avoid negative cash.")
                for tic in cols:
                    if delta[tic] > 0:
                        delta[tic] *= scale
                        # re-apply lot rounding if needed
                        if not self.allow_partial_shares or (isinstance(self.min_lot, dict) or self.min_lot != 1):
                            lot = int(self._get_scalar_or_map(self.min_lot, tic))
                            delta[tic] = self._lot_round(delta[tic], lot)

                # recompute notional & fees after scaling
                buy_notional = 0.0
                sell_notional = 0.0
                trade_cost_total = 0.0
                for tic in cols:
                    d = float(delta[tic])
                    if np.isclose(d, 0.0):
                        continue
                    side = "buy" if d > 0 else "sell"
                    epx = eff_px_map.get(tic, float(px_exec_row[tic]))
                    notional = abs(d) * epx
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    trade_cost_total += fee
                    if d > 0:
                        buy_notional += notional
                    else:
                        sell_notional += notional
                tentative_cash = cash + sell_notional - buy_notional - trade_cost_total

            # accrue cash & MTM from last written day up to (but excluding) exec_dt
            # find previous day in index to continue from
            # previous iteration has written up to prior next_dt; we now ensure interval [prev_written+1 : exec_dt-1]
            # For simplicity, we write from the previous dt (or start) up to the day before exec_dt here,
            # but we prevent overwriting by assigning only the fresh dates.
            # In practice, we rely on monotonically increasing assignment segments.

            # apply execution: update cash & holdings at exec_dt
            cash = tentative_cash
            new_shares = prev_shares + delta

            # record ledgers at execution moment
            exec_label_date = exec_dt
            for tic in cols:
                # trade row
                d = float(delta[tic])
                if not np.isclose(d, 0.0):
                    side = "buy" if d > 0 else "sell"
                    px_eff = eff_px_map[tic]
                    notional = abs(d) * px_eff
                    fee = max(notional * variable_bps + fixed_fee, min_fee)
                    trade_rows.append({
                        "date": exec_label_date,
                        "ticker": tic,
                        "side": side,
                        "shares_delta": d,
                        "exec_price_eff": px_eff,
                        "notional": notional,
                        "commission": fee,
                    })
                # holdings snapshot
                hold_rows.append({
                    "date": exec_label_date,
                    "ticker": tic,
                    "price_ref": float(px_exec_row[tic]),
                    "shares": float(new_shares[tic]),
                    "amount": float(new_shares[tic] * float(px_exec_row[tic])),
                })

            # append shares_record row
            shares_record.loc[exec_label_date, cols] = new_shares.values

            # MTM path intervals:
            # - if use_next_open=False: new shares from [dt : next_dt]
            # - if use_next_open=True : prev_shares on [dt : exec_dt-1], then new shares on [exec_dt : next_dt]
            next_dt = (self.rebalance_dates[i + 1] if i < len(self.rebalance_dates) - 1 else self.val_px.index[-1])

            if not self.use_next_open:
                interval = self.val_px.loc[dt:next_dt].index
                cash = apply_mtm_and_accrual(interval, new_shares, cash)
            else:
                left = self.val_px.loc[dt:exec_dt].index
                if len(left) > 1:
                    cash = apply_mtm_and_accrual(left[:-1], prev_shares, cash)  # up to exec_dt-1 with old shares
                right = self.val_px.loc[exec_dt:next_dt].index
                cash = apply_mtm_and_accrual(right, new_shares, cash)

            # carry forward
            prev_shares = new_shares.copy()

        # finalize
        self.index_series = index_series.astype(float)
        self.cash_series = cash_series.astype(float)
        self.shares_record = shares_record.sort_index().astype(float)
        self.trade_ledger = pd.DataFrame(trade_rows).sort_values("date").reset_index(drop=True)
        self.shares_ledger = pd.DataFrame(hold_rows).sort_values("date").reset_index(drop=True)

        return self

    # ------------------------------ helpers ------------------------------ #
    def get_results(self) -> Dict[str, Any]:
        """Return all result objects."""
        if any(obj is None for obj in [self.index_series, self.cash_series, self.shares_record]):
            raise RuntimeError("Run the simulation first with `.run()`.")
        return {
            "index_series": self.index_series,
            "cash_series": self.cash_series,
            "shares_record": self.shares_record,
            "trade_ledger": self.trade_ledger,
            "shares_ledger": self.shares_ledger,
            "logs": self.logs,
        }

    def to_nav(self) -> pd.Series:
        """Return NAV series normalized to 1.0 at the first non-NaN value."""
        s = self.index_series.dropna()
        if s.empty:
            return self.index_series * np.nan
        return self.index_series / float(s.iloc[0])