import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

class BaseLabel(ABC):
    def __init__(self, data : Union[pd.Series, pd.DataFrame, np.ndarray]):
        self.data = data

    @abstractmethod
    def get_labels(self) -> pd.Series:
        pass

class BaseTechnicalLabel:
    def __init__(self, data: Union[pd.Series, pd.DataFrame]):
        self.data = data.copy()

    def _check_ohlcv_columns(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        cols = [col.lower() for col in self.data.columns]
        missing = [col for col in required if col not in cols]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")