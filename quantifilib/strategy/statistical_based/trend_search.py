import statsmodels.api as sm
import pandas as pd
import numpy as np
from quantifilib.strategy.base_label import BaseLabel

class TrendSearchLabeling(BaseLabel):
    """
    Labeling with Trend Search Method using statsmodels
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _linear_trend_t_values(close : np.array) :
        x = np.ones((close.shape[0], 2))
        x[:, 1] = np.arange(close.shape[0])
        ols = sm.OLS(close, x).fit()
        return ols.tvalues[1]

    def get_labels(self, span : list) :
        """
        get labels for trend search methods
        :param span: time range of trend search such as [1, 20, 1] (start, end, stepsize)
        :return: labeling and t value -> pd.DataFrame
        """
        index = self.data.index
        out = pd.DataFrame(index = index, columns = ['t1','tVal','bin'])
        horizons = range(*span)
        for dt0 in index :
            df0 = pd.Series(dtype = float)
            iloc0 = self.data.index.get_loc(dt0)
            if iloc0 + max(horizons) > self.data.shape[0] : continue
            for horizon in horizons :
                dt1 = self.data.index[iloc0 + horizon -1]
                df1 = self.data.loc[dt0 : dt1]
                df0.loc[dt1] = self._linear_trend_t_values(df1.values)
            dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
            out.loc[dt0, ['t1','tVal','bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
        out['t1'] = pd.to_datetime(out['t1'])
        out['bin'] = pd.to_numeric(out['bin'], downcast = 'signed')

        return out.dropna(subset = ['bin'])