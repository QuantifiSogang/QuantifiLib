import pandas_datareader as pdr
import pandas as pd
from typing import List, Union
from .loader_base import BaseDataLoader

class FREDLoader(BaseDataLoader):
    """
    Loader for FRED (Federal Reserve Economic Data) using pandas-datareader.
    Each symbol corresponds to a FRED series code (e.g., CPIAUCSL, FEDFUNDS).
    """

    def load(
        self,
        symbols: Union[str, List[str]],
        start: str,
        end: str = None,
        **kwargs
    ) -> pd.DataFrame:
        if isinstance(symbols, str):
            symbols = [symbols]

        frames = []
        for sym in symbols:
            try:
                df = pdr.DataReader(sym, 'fred', start, end, **kwargs)
                df.columns = [sym]  # rename 'Value' → symbol
                frames.append(df)
            except Exception as e:
                print(f"[FREDLoader] Failed to load {sym}: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result.index.name = 'Date'
        return result