import numpy as np
import pandas as pd

class BaseFeature:
    def __init__(self, data : pd.DataFrame):
        self.data = data.copy()

    def _check_ohlcv_columns(self):
        required = ['open', 'high', 'low', 'close', 'volume']
        cols = [col.lower() for col in self.data.columns]
        missing = [col for col in required if col not in cols]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")