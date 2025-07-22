import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import gaussian_kde
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

from .base_distance import BaseDistance
from tqdm import tqdm

class Entropy(BaseDistance) :
    """
    Entropy-based distance and similarity metrics for time series data.

    Includes Jensen-Shannon divergence, variation of information, and mutual information
    as non-linear and information-theoretic measures of distance or dependence.
    """
    def __init__(self, data:pd.DataFrame, verbose : bool = True) :
        super().__init__(data = data, verbose = verbose)

    @staticmethod
    def num_bins(n_obs, corr=None):
        """
        Estimate the optimal number of bins for histogram-based entropy calculations.

        Uses rules-of-thumb from information theory literature, adapting based on the number of
        observations and (optionally) correlation between variables.

        Parameters:
        - n_obs (int): Number of observations in the time series
        - corr (float or None): Optional correlation value between the two time series

        Returns:
        - int: Optimal number of bins
        """
        if corr is None:
            z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs ** 2) ** 0.5) ** (1 / 3)
            b = round(z / 6 + 2 / (3 * z) + 1 / 3)
        else:
            b = round(2 ** 0.5 * (1 + (1 + 24 * n_obs * (1 - corr ** 2)) ** 0.5) ** 0.5)
        return int(b)

    @staticmethod
    def _js_divergence(x, y):
        """
        Compute the Jensen-Shannon (JS) divergence between two time series.

        JS divergence is a symmetric and smoothed version of Kullback-Leibler divergence.
        It is based on kernel density estimates of the distributions of the series.

        Parameters:
        - x (array-like): First time series
        - y (array-like): Second time series

        Returns:
        - float: JS divergence value (always non-negative and bounded)
        """
        kde1, kde2 = gaussian_kde(x), gaussian_kde(y)
        estimate = np.linspace(
            min(np.min(x), np.min(y)),
            max(np.max(x), np.max(y)),
            len(x)
        )
        pdf1, pdf2 = kde1(estimate), kde2(estimate)
        m = 0.5 * (pdf1 + pdf2)
        return 0.5 * np.sum(rel_entr(pdf1, m)) + 0.5 * np.sum(rel_entr(pdf2, m))

    @staticmethod
    def _var_info(x, y, norm=False):
        """
        Compute the Variation of Information (VI) between two time series.

        VI is an information-theoretic metric that quantifies the shared and unique information
        between two distributions. Lower values indicate greater similarity.

        Parameters:
        - x (array-like): First time series
        - y (array-like): Second time series
        - norm (bool): Whether to normalize the result by joint entropy

        Returns:
        - float: Variation of Information (lower is more similar)
        """
        b_xy = Entropy.num_bins(len(x), np.corrcoef(x, y)[0, 1])
        c_xy = np.histogram2d(x, y, bins=b_xy)[0]
        i_xy = mutual_info_score(None, None, contingency=c_xy)
        hx = ss.entropy(np.histogram(x, bins=b_xy)[0])
        hy = ss.entropy(np.histogram(y, bins=b_xy)[0])
        v_xy = hx + hy - (2 * i_xy)
        if norm:
            h_xy = hx + hy - i_xy
            v_xy /= h_xy
        return v_xy

    @staticmethod
    def _mutual_info(x, y, norm=False):
        """
        Compute the mutual information between two time series using histogram binning.

        Mutual information captures both linear and non-linear dependencies between variables.

        Parameters:
        - x (array-like): First time series
        - y (array-like): Second time series
        - norm (bool): Whether to normalize the mutual information by the sum of entropies

        Returns:
        - float: Mutual information value (higher is more dependent)
        """
        b_xy = Entropy.num_bins(len(x), np.corrcoef(x, y)[0, 1])
        c_xy = np.histogram2d(x, y, bins=b_xy)[0]
        i_xy = mutual_info_score(None, None, contingency=c_xy)
        if norm:
            hx = ss.entropy(np.histogram(x, bins=b_xy)[0])
            hy = ss.entropy(np.histogram(y, bins=b_xy)[0])
            i_xy /= (hx + hy)
        return i_xy

    def get_jensen_shannon_divergence(self) -> pd.DataFrame:
        """
        Compute the Jensen-Shannon divergence between each pair of time series in the dataset.

        Jensen-Shannon divergence is a symmetric and smoothed version of KL divergence,
        useful for comparing the similarity between two probability distributions.
        It is bounded between 0 and 1 and can be interpreted as a measure of dissimilarity.

        :return: A symmetric DataFrame of shape (n_assets, n_assets) containing JS divergence values.
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator:
            for j in range(n):
                dist[i, j] = Entropy._js_divergence(self.data.iloc[:, i], self.data.iloc[:, j])
        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_variational_information(self, normalize:bool = False) -> pd.DataFrame:
        """
        Compute the Variation of Information (VI) between each pair of time series.

        VI measures the amount of information lost and gained between two random variables.
        Smaller values indicate higher similarity. Can be optionally normalized by joint entropy.

        :param normalize: If True, normalize the VI by joint entropy to keep values between 0 and 1.
        :return: A symmetric DataFrame of shape (n_assets, n_assets) containing VI values.
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in iterator :
            for j in range(n):
                dist[i, j] = Entropy._var_info(self.data.iloc[:, i], self.data.iloc[:, j], normalize)
        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist

    def get_mutual_information(self, normalize:bool = False) -> pd.DataFrame:
        """
        Compute the Mutual Information (MI) between each pair of time series.

        MI quantifies the amount of information shared between two variables.
        Higher values indicate stronger dependence. Can be optionally normalized by total entropy.

        :param normalize: If True, normalize the MI by the sum of marginal entropies (range: 0 to 1).
        :return: A symmetric DataFrame of shape (n_assets, n_assets) containing MI values.
        """
        n = len(self.data.columns)
        dist = np.zeros((n, n))

        iterator = tqdm(range(n)) if self.verbose else range(n)

        for i in range(n):
            for j in range(n):
                dist[i, j] = Entropy._mutual_info(self.data.iloc[:, i], self.data.iloc[:, j], normalize)
        dist = pd.DataFrame(dist, columns=self.data.columns, index=self.data.columns)
        return dist