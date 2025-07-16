import yfinance as yf
import pandas as pd
from typing import List
from .loader_base import BaseDataLoader

class YahooFinanceLoader(BaseDataLoader):
    """
    Yahoo finance based data loader using yfinance library
    """

    def load(
            self, symbols : List[str],
            start : str,
            end : str = None,
            **kwargs
        ) -> pd.DataFrame :
        """
        Loads data from Yahoo Finance API
        :param symbols: Ticker Symbols
        :param start: start date in 'yyyy-mm-dd'
        :param end: end date in 'yyyy-mm-dd'
        :param kwargs:
        :return:
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        data = yf.download(
            tickers = symbols,
            start = start,
            end = end,
            **kwargs
        )

        # # tidy multiindex format
        # if not isinstance(data.columns, pd.MultiIndex):
        #     symbol = symbols[0]
        #     data.columns = pd.MultiIndex.from_product([[symbol], data.columns])

        return data