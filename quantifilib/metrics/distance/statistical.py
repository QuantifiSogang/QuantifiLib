import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_distance import BaseDistance

class StatisticalDistance(BaseDistance) :
    """
    distance matrix using statistical metrics
    """
    def __init__(self, data:pd.DataFrame, verbose:bool = True) :
        super().__init__(data = data, verbose = verbose)

    @staticmethod
    def _corr_dist_long_only(x, y):
        """
        Compute the correlation distance suitable for long-only portfolios.

        This metric is defined as:
            distance = sqrt(0.5 * (1 - correlation))

        It treats perfectly negatively correlated assets (rho = -1) as having distance = 1,
        and perfectly positively correlated assets (rho = 1) as having distance = 0.

        This is useful when clustering similar assets in a long-only context, where negative correlation
        is not utilized for hedging.

        :param x: First time series
        :param y: Second time series
        :return: Correlation-based distance between x and y
        """
        return np.sqrt(0.5 * (1 - np.corrcoef(x, y)[0, 1]))

    @staticmethod
    def _corr_dist_long_short(x, y):
        """
        Compute the correlation distance suitable for long-short portfolios.

        This metric is defined as:
            distance = sqrt(1 - correlation)

        It distinguishes between positive and negative correlations more strongly,
        treating perfectly negatively correlated assets (rho = -1) as having distance = sqrt(2),
        which reflects their usefulness in long-short hedging strategies.

        Use this distance when the investment universe includes both long and short positions.

        :param x: First time series
        :param y: Second time series
        :return: Correlation-based distance between x and y
        """
        return np.sqrt(1 - np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _euclidean_dist(x, y):
        """
        Compute Euclidean distance between two time series.

        This metric measures the straight-line distance in high-dimensional space,
        and is sensitive to scale.

        Use only when data is standardized (e.g., returns or z-score normalized).

        :param x: First time series
        :param y: Second time series
        :return: Euclidean distance
        """
        return np.linalg.norm(x - y)

    @staticmethod
    def _cosine_dist(x, y):
        """
        Compute cosine distance between two time series.

        Measures the angle between two vectors, insensitive to magnitude.

        :param x: First time series
        :param y: Second time series
        :return: Cosine distance (1 - cosine similarity)
        """
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    @staticmethod
    def _dtw_dist(x, y):
        """
        Compute Dynamic Time Warping (DTW) distance between two series.

        Captures similarity with time alignment shifts, useful for asynchronous patterns.

        :param x: First time series
        :param y: Second time series
        :return: DTW distance
        """
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1]  # match
                )

        return dtw_matrix[n, m]

    @staticmethod
    def _mahalanobis_dist(x, y, cov_inv=None):
        """
        Compute the Mahalanobis distance between two time series.

        This considers the covariance structure of the data, providing
        a scale-invariant, correlation-aware distance.

        :param x: First time series
        :param y: Second time series
        :param cov_inv: Inverse covariance matrix (optional)
        :return: Mahalanobis distance
        """
        delta = x - y
        if cov_inv is None:
            # Default: assume identity matrix
            return np.linalg.norm(delta)
        return np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))

    def get_correlation_based_distance(self, long_only:bool = True) -> pd.DataFrame :
        """
        Compute a correlation-based distance matrix between all pairs of time series.

        Uses either the long-only or long-short correlation distance metric:
        - Long-only: sqrt(0.5 * (1 - correlation))
        - Long-short: sqrt(1 - correlation)

        :param long_only: If True, use long-only correlation distance. If False, use long-short version.
        :return: Symmetric DataFrame of distances with shape (n_assets, n_assets)
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        if long_only :
            for i in iterator:
                for j in range(n):
                    dist[i, j] = self._corr_dist_long_only(self.data.iloc[:, i], self.data.iloc[:, j])
        else :
            for i in iterator:
                for j in range(n):
                    dist[i, j] = self._corr_dist_long_short(self.data.iloc[:, i], self.data.iloc[:, j])

        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_euclidian_distance(self) -> pd.DataFrame:
        """
        Compute the Euclidean distance matrix between all pairs of time series.

        Measures the straight-line (L2 norm) distance in raw feature space. Assumes
        time series are aligned.

        :return: Symmetric DataFrame of Euclidean distances between time series
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator:
            for j in range(n):
                dist[i, j] = self._euclidean_dist(self.data.iloc[:, i], self.data.iloc[:, j])

        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_cosine_distance(self) -> pd.DataFrame:
        """
        Compute the cosine distance matrix between all pairs of time series.

        Cosine distance is defined as 1 - cosine similarity. It measures
        angular dissimilarity, focusing on direction rather than magnitude.

        :return: Symmetric DataFrame of cosine distances between time series
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator:
            for j in range(n):
                dist[i, j] = self._cosine_dist(self.data.iloc[:, i], self.data.iloc[:, j])

        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_mahalanobis_distance(self) -> pd.DataFrame:
        """
        Compute the Mahalanobis distance matrix between all pairs of time series.

        Mahalanobis distance accounts for the covariance structure of the data,
        measuring how many standard deviations separate two points.

        :return: Symmetric DataFrame of Mahalanobis distances between time series
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator:
            for j in range(n):
                dist[i, j] = self._mahalanobis_dist(self.data.iloc[:, i], self.data.iloc[:, j])

        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_dynamic_time_warping_distance(self) -> pd.DataFrame:
        """
        Compute the Dynamic Time Warping (DTW) distance matrix between all pairs of time series.

        DTW is a non-linear measure that aligns time series with elastic shifting of time dimension.
        Useful for comparing time series that are similar but out of phase.

        :return: Symmetric DataFrame of DTW distances between time series
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator:
            for j in range(n):
                dist[i, j] = self._dtw_dist(self.data.iloc[:, i], self.data.iloc[:, j])

        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist