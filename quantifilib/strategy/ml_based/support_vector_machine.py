import numpy as np
import pandas as pd
from typing import Union
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from quantifilib.strategy.base_label import BaseLabel


class SVMLabeling(BaseLabel):
    """
    SVM-based trend direction labeling
    """

    def __init__(self, data: Union[pd.Series, pd.DataFrame]):
        super().__init__(data)

    def _extract_features(self, window: int) -> pd.DataFrame:
        """
        Create rolling window features (e.g., return, volatility)
        """
        df = pd.DataFrame(index=self.data.index)
        price = self.data if isinstance(self.data, pd.Series) else self.data['close']
        df['return'] = price.pct_change(window)
        df['volatility'] = price.pct_change().rolling(window).std()
        df = df.dropna()
        return df

    def get_labels(self, window: int = 10, pred_horizon: int = 5) -> pd.DataFrame:
        """
        Generate labels using SVM on past window of return/volatility.
        Predicts direction of return over future pred_horizon days.
        """
        price = self.data if isinstance(self.data, pd.Series) else self.data['close']
        features = self._extract_features(window)
        features['future_return'] = price.shift(-pred_horizon) / price - 1
        features['bin'] = np.sign(features['future_return'])
        features = features.dropna()

        # Filter only -1 and +1 for binary SVM
        features = features[features['bin'] != 0]

        X = features[['return', 'volatility']]
        y = features['bin']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train SVM
        model = SVC(kernel='rbf', C=1.0)
        model.fit(X_scaled, y)

        # Predict full dataset (even in-sample)
        features['pred_bin'] = model.predict(X_scaled)

        return features[['bin', 'pred_bin']]