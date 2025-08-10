from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

def as_series(x: Union[float, int, pd.Series], idx: pd.Index, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.reindex(idx)
    else:
        s = pd.Series(float(x), index=idx)
    s.name = name
    return s

def pick_first(inputs: Dict, keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in inputs and inputs[k] is not None:
            return float(inputs[k])
    return None

def check_r_g(r: float, g: float, ctx: str = "GGM"):
    if r is None:
        raise ValueError("Missing 'cost_of_equity' (or alias 'r'/'ke').")
    if g is None:
        raise ValueError("Missing 'growth' (or alias 'g').")
    if r <= g:
        raise ValueError(f"[{ctx}] invalid r<=g (r={r}, g={g}). Requires r > g.")