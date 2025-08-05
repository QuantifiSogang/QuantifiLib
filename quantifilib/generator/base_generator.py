from abc import ABC, abstractmethod
import numpy as np

class BaseGenerator(ABC):
    """
    Abstract base class for time series generators using user-defined parameters.

    Subclasses must implement:
        - from_params(): classmethod to initialize from manual parameters
        - generate_simulation(): instance method to simulate time series
    """

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict, **kwargs) -> 'BaseGenerator':
        """
        Construct a generator from manually defined model parameters.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters (e.g., AR or MA coefficients).
        kwargs : optional additional config (e.g., noise type)

        Returns
        -------
        BaseGenerator instance
        """
        pass

    @abstractmethod
    def generate_simulation(self, n: int) -> np.ndarray:
        """
        Generate synthetic time series of length n.

        Parameters
        ----------
        n : int
            Length of the series to generate.

        Returns
        -------
        np.ndarray
            Simulated time series values.
        """
        pass