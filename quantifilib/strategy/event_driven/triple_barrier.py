import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
from quantifilib.utils.multiprocess.process_job import mp_pandas_obj
from quantifilib.strategy.base_label import BaseLabel
from quantifilib.metrics.risk.market import daily_volatility

@dataclass
class Condition(ABC) :
    factor : float
    barrier : float

    @abstractmethod
    def evaluate(self, value : float) -> bool :
        pass

@dataclass
class PositiveCondition(Condition) :
    def evaluate(self, value : float) -> bool :
        return value > self.factor * self.barrier

@dataclass
class NegativeCondition(Condition) :
    def evaluate(self, value : float) -> bool :
        return value < self.factor * self.barrier * (-1)

@dataclass
class BarrierConditions:
    """
    A class that generates and manages barrier conditions used for different labeling techniques.
    Those conditions can be used to generate labels like this. Example for n=1:
        y =
            -1 if r_{t,t+n} < -barrier,
             1 if  r_{t,t+n} > -barrier,
             0 else
    Different n will add conditions by multiples of barrier up to n-multiples.

    Attributes:
    n (int): The number of barrier conditions to be generated for negative and positive barriers.
    barrier (float): The threshold value for the barrier.
    conditions (Dict[int, Condition): A dictionary holding condition functions for various barrier levels.
        Keys are sorted numerically.
    """
    n: int
    barrier: float
    conditions: Dict[int, Condition] = field(default_factory=dict)

    def __post_init__(self):
        """
        Calculate the conditions after the instance has been initialized.
        """
        self.generate_conditions()
        self.sort_conditions()

    def generate_conditions(self):
        """
        Generates barrier conditions based on the specified number of conditions and threshold values.
        """
        for i in range(1, self.n + 1):
            self.conditions[-i] = NegativeCondition(factor=i, barrier=self.barrier)
            self.conditions[i] = PositiveCondition(factor=i, barrier=self.barrier)

    def sort_conditions(self):
        """
        Sorts the generated conditions in ascending order based on their keys.
        """
        self.conditions = dict(sorted(self.conditions.items()))

    def __str__(self):
        """
        String representation of the BarrierConditions object, showing conditions in a readable format.
        """
        condition_strings = []
        for key, condition in self.conditions.items():
            condition_strings.append(f"\t{key}: \t{condition}")

        conditions_str = "\n\t".join(condition_strings)
        return f"BarrierConditions(n={self.n}, barrier={self.barrier}):\n\tConditions={{\n\t{conditions_str}\n\t}}"

class MultiBarrierLabeling(BaseLabel) :
    """
    Apply the Barrier Method to financial returns data inspired by Marcos López de Prado.

    The goal is to vectorize this method with pandas/numpy to reduce for-loops and add additional functionality:
    - multiple barriers: Instead of having a barrier at +/-1%, we have a 2nd, 3rd,... at +/-2%, +/-3% etc.
    - intermediate steps are returned as a pd.DataFrame that can be used for features of models:
        - cumulative_returns
        - time_since_last_crossing
    - transition probabilities of the generated labels
    - plot_at_date as plotting capabilities and a quick sanity check

    Attributes:
    returns (pandas.Series): Series of returns.
    n (int): Window size for the barrier method.
    barrier (float): Barrier value to determine a label. E.g. for barrier=0.1 the label is 1 if the
                     timeseries has a future return greater than 10% somewhere between the future 1 to n observations.
    center (bool, optional): Center the returns by their mean to denominate an above-average return.
                             Defaults to True.
    """
    pass

class TripleBarrierLabeling(BaseLabel) :
    """
    Triple‑Barrier labeling generator (López de Prado, 2018).

    The class converts a price series into “meta‑labels” for supervised
    learning.  For each event time *t* it asks:

    ▸ Will a position opened at *t* hit its profit‑taking barrier
      (+pt·σ), its stop‑loss barrier (–sl·σ), or the vertical time
      barrier first?
    ▸ Was the outcome ultimately positive, negative, or inconclusive?

    A *1* label means the profit‑taking barrier was reached first
    (good trade), *–1* means the stop‑loss barrier fired first
    (bad trade), and *0* means neither barrier was touched before the
    vertical barrier (ignore / hold‑out).

    Parameters
    ----------
    data : pd.DataFrame
        Price dataframe indexed by datetime.  Must contain at least one
        close column specified by `price_col_name`.
    pt_sl : tuple[float, float], default (1.0, 1.0)
        Multiples of the target volatility that define the upper (pt)
        and lower (sl) horizontal barriers.
    min_ret : float, default 0.0
        Minimum target return (volatility) required for an event to be
        considered.  Smaller moves are ignored.
    vol_lookback : int, default 20
        Rolling window length (in bars) used to estimate the daily
        volatility σₜ.
    vertical_barrier : dict or None, default ``{"days": 1}``
        Distance of the vertical barrier expressed as a dictionary with
        keys ``"days"``, ``"hours"``, ``"minutes"``, ``"seconds"``.
        Example: ``{"days": 0, "hours": 6}`` sets a 6‑hour barrier.
        If *all* values are zero or the argument is ``None``, the
        vertical barrier is disabled.
    num_threads : int, default 4
        Number of worker processes used by ``mp_pandas_obj`` when it
        evaluates the horizontal barriers.
    price_col_name : str, default ``"Close"``
        Column name to treat as the closing price.
    side_prediction : pd.Series or None, optional
        Series with ex‑ante trade direction forecasts (+1 long / –1
        short).  When supplied the method produces *meta‑labels*:
        returns are signed by the forecast direction and trades that go
        against the forecast are set to label 0.  If omitted, the method
        reverts to the symmetric (direction‑agnostic) triple‑barrier
        scheme.
    drop_min_pct : float, default 0.05
        Minimum class proportion required in the final label set.
        Iteratively drops the smallest class until the condition is
        satisfied.  Set to 0 to disable re‑balancing.

    Returns
    -------
    pd.DataFrame
        • ``ret``  – realised return between *t* and the first barrier
        • ``trgt`` – ex‑ante target volatility σₜ
        • ``bin``  – final label in {‑1, 0, 1}
        • ``side`` – (optional) original forecast direction

    Notes
    -----
    Implementation closely follows:

    *Marcos López de Prado, “Advances in Financial Machine Learning”,
    Chapter 3 (Triple‑Barrier Method).*

    Heavy lifting—barrier evaluation, multiprocessing, and class
    balancing—is delegated to helper functions such as
    ``get_events()``, ``meta_labeling()``, etc., supplied separately.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        pt_sl: Tuple[float, float] = (1, 1),
        min_ret: float = 0.0,
        vol_lookback: int = 20,
        vertical_barrier: Optional[dict] = None,
        num_threads: int = 1,
        price_col_name: str = "Close",
        side_prediction: Optional[pd.Series] = None,
    ) -> None:
        super().__init__(data = data)

        self._check_ohlcv_columns()

        self.pt_sl = pt_sl
        self.min_ret = min_ret
        self.vol_lookback = vol_lookback
        self.vertical_barrier = vertical_barrier or {"days": 1}
        self.num_threads = num_threads
        self.price_col_name = price_col_name
        self.side_prediction = side_prediction

        self.close = self.data[self.price_col_name]
        self.target = daily_volatility(self.close, self.vol_lookback)

    def _get_vertical_barriers(self, t_events: pd.Index) -> Union[pd.Series, bool]:
        if all(v == 0 for v in self.vertical_barrier.values()):
            return False
        return add_vertical_barrier(
            t_events=t_events,
            close=self.close,
            num_days=self.vertical_barrier.get("days", 0),
            num_hours=self.vertical_barrier.get("hours", 0),
            num_minutes=self.vertical_barrier.get("minutes", 0),
            num_seconds=self.vertical_barrier.get("seconds", 0),
        )

    @staticmethod
    def _apply_pt_sl_on_t1(close, events, pt_sl, molecule):
        """Vectorised first‑touch evaluation for a slice (molecule)."""
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)

        pt_mult, sl_mult = pt_sl
        profit_taking = pt_mult * events_['trgt'] if pt_mult > 0 else pd.Series(index=events.index)
        stop_loss     = -sl_mult * events_['trgt'] if sl_mult > 0 else pd.Series(index=events.index)

        for loc, vb in events_['t1'].fillna(close.index[-1]).items():
            prices = close[loc: vb]
            cum_ret = (prices / close[loc] - 1) * events_.at[loc, 'side']
            out.loc[loc, 'sl'] = cum_ret[cum_ret < stop_loss[loc]].index.min()
            out.loc[loc, 'pt'] = cum_ret[cum_ret > profit_taking[loc]].index.min()
        return out

    def _add_vertical_barrier(self, t_events: pd.Index) -> Union[pd.Series, bool]:
        if all(v == 0 for v in self.vertical_barrier.values()):
            return pd.Series(pd.NaT, index=t_events)
        td = pd.Timedelta(
            f"{self.vertical_barrier.get('days',0)} days, "
            f"{self.vertical_barrier.get('hours',0)} hours, "
            f"{self.vertical_barrier.get('minutes',0)} minutes, "
            f"{self.vertical_barrier.get('seconds',0)} seconds"
        )
        nearest = self.close.index.searchsorted(t_events + td)
        nearest = nearest[nearest < self.close.shape[0]]
        return pd.Series(self.close.index[nearest], index=t_events[:nearest.shape[0]])

    def _get_events(self, t_events: pd.Index) -> pd.DataFrame:
        target = self.target.loc[t_events]
        target = target[target > self.min_ret]

        v_barriers = self._add_vertical_barrier(t_events)

        if self.side_prediction is None:
            side_ = pd.Series(1.0, index=target.index)
            pt_sl_ = [self.pt_sl[0], self.pt_sl[0]]
        else:
            side_ = self.side_prediction.loc[target.index]
            pt_sl_ = list(self.pt_sl[:2])

        events = pd.concat({'t1': v_barriers, 'trgt': target, 'side': side_}, axis=1).dropna(subset=['trgt'])

        first_touch = mp_pandas_obj(
            func=self._apply_pt_sl_on_t1,
            pd_obj=('molecule', events.index),
            num_threads=self.num_threads,
            close=self.close,
            events=events,
            pt_sl=pt_sl_
        )
        for idx in events.index:
            events.loc[idx, 't1'] = first_touch.loc[idx].dropna().min()

        if self.side_prediction is None:
            events = events.drop('side', axis=1)

        events['pt'], events['sl'] = self.pt_sl
        return events

    @staticmethod
    def _barrier_touched(out_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        label = []
        for dt, vals in out_df.iterrows():
            ret, tgt = vals['ret'], vals['trgt']
            pt_hit = ret >  tgt * events.at[dt, 'pt']
            sl_hit = ret < -tgt * events.at[dt, 'sl']
            label.append( 1 if (ret > 0 and pt_hit) else
                         (-1 if (ret < 0 and sl_hit) else 0))
        out_df['bin'] = label
        return out_df

    def get_labels(self, ret_type:str='percentage', drop_min_pct : float = None) -> pd.DataFrame:
        events = self._get_events(
            t_events = self.target.dropna().index
        )

        ev = events.dropna(subset=['t1'])
        idx_union = ev.index.union(ev['t1']).drop_duplicates()
        prices = self.close.reindex(idx_union, method='bfill')

        out = pd.DataFrame(index=ev.index)

        if ret_type == 'percentage':
            out['ret'] = (prices.loc[ev['t1']].values / prices.loc[ev.index].values - 1)
        elif ret_type == 'log' :
            out['ret'] = np.log(prices.loc[ev['t1']].values / prices.loc[ev.index].values)
        else : raise ValueError('ret_type must be either "percentage" or "log"')

        out['trgt'] = ev['trgt'] # volatility target

        if 'side' in ev:
            out['ret'] *= ev['side']

        out = self._barrier_touched(out, events)

        if 'side' in ev:
            out.loc[out['ret'] <= 0, 'bin'] = 0

        # out['ret'] = np.exp(out['ret']) - 1

        if 'side' in events:
            out['side'] = events['side']

        if drop_min_pct is not None :
            out = self._drop_labels(out, drop_min_pct)

        return out

    @staticmethod
    def _drop_labels(events: pd.DataFrame, min_pct: float) -> pd.DataFrame:
        while True:
            freq = events['bin'].value_counts(normalize=True)
            if freq.min() > min_pct or freq.shape[0] < 3:
                break
            events = events[events['bin'] != freq.idxmin()]
        return events

def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).items():
        closing_prices = close[loc: vertical_barrier]
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()
        out.loc[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()
    return out
def add_vertical_barrier(t_events, close, num_days = 0, num_hours = 0, num_minutes = 0, num_seconds = 0):
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
    nearest_index = close.index.searchsorted(t_events + timedelta)
    nearest_index = nearest_index[nearest_index < close.shape[0]]
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]
    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers
def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times = False,
               side_prediction=None):
    target = target.loc[t_events]
    target = target[target > min_ret]
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.loc[target.index]
        pt_sl_ = pt_sl[:2]
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      close=close,
                                      events=events,
                                      pt_sl=pt_sl_)
    for ind in events.index:
        events.loc[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()
    if side_prediction is None:
        events = events.drop('side', axis=1)
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]
    return events
def barrier_touched(out_df, events):
    store = []
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']

        pt_level_reached = ret > target * events.loc[date_time, 'pt']
        sl_level_reached = ret < -target * events.loc[date_time, 'sl']

        if ret > 0.0 and pt_level_reached:
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            store.append(-1)
        else:
            store.append(0)
    out_df['bin'] = store
    return out_df
def meta_labeling(triple_barrier_events, close):
    events_ = triple_barrier_events.dropna(subset = ['t1'])
    all_dates = events_.index.union(other = events_['t1'].values).drop_duplicates()
    prices = close.reindex(all_dates, method = 'bfill')

    out_df = pd.DataFrame(index = events_.index)
    out_df['ret'] = np.log(prices.loc[events_['t1'].values].values / prices.loc[events_.index])
    out_df['trgt'] = events_['trgt']

    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']
    out_df = barrier_touched(out_df, triple_barrier_events)
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0
    out_df['ret'] = np.exp(out_df['ret']) - 1
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']
    return out_df
def drop_labels(events, min_pct = 0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events