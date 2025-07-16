from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class BaseDataLoader(ABC):
    """
    Abstract base class for financial data loaders.
    All concrete loaders (e.g., YFinanceLoader, FREDLoader) should inherit this.
    """

    @abstractmethod
    def load(
            self, symbols: List[str],
            start:str,
            end:str
        ) -> pd.DataFrame :
        """
                Load financial data for the given symbols and date range.

                Parameters:
                - symbols: list of tickers or codes
                - start: start date in 'YYYY-MM-DD'
                - end: end date in 'YYYY-MM-DD'

                Returns:
                - pd.DataFrame: multi-index (symbol, datetime) DataFrame
                """
        pass