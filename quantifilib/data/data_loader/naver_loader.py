import pandas_datareader.data as web
import pandas as pd
from typing import List
from .loader_base import BaseDataLoader

class NaverFinanceLoader(BaseDataLoader):
    """
    Naver Finance data loader using pandas-datareader.
    Symbol should be in KRX format (e.g., '005930.KQ', '000660.KQ').
    """

    def load(
            self, symbols: List[str],
            start: str,
            end: str = None,
            **kwargs
        ) -> pd.DataFrame :
        """
        Load Naver finance data.
        :param symbols: Ticker Symbols
        :param start: start date in 'yyyy-mm-dd'
        :param end: end date in 'yyyy-mm-dd'
        :param kwargs:
        :return:
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            single_symbol = True
        else:
            single_symbol = len(symbols) == 1

        frames = []
        for symbol in symbols:
            try:
                df = web.DataReader(symbol, data_source='naver', start=start, end=end, **kwargs)
                frames.append(df)
            except Exception as e:
                print(f"[NaverFinanceLoader] Failed to load {symbol}: {e}")

        if not frames:
            return pd.DataFrame()

        if single_symbol:
            return frames[0]

        # MultiIndex [(Symbol, Field)] → transpose → [(Field, Symbol)]
        data = pd.concat(frames, axis=1, keys=symbols)
        data.columns = data.columns.swaplevel(0, 1)  # (Symbol, Field) → (Field, Symbol)
        data = data.sort_index(axis=1, level=0)
        data = data.astype(float)
        data.index = pd.to_datetime(data.index)
        data['Volume'] = data['Volume'].astype(int)

        return data