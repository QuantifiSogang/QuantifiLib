from abc import ABC, abstractmethod
from typing import Dict

class BaseValuation(ABC):
    """
    Abstract base class for valuation models.

    Subclasses must implement:
      - from_params(): configure a valuation model instance
      - evaluate():   compute value(s) from provided inputs

    Design notes
    ------------
    • Keep model configuration at construction time (e.g., choice of mode,
      terminal method, mid-year convention).
    • Pass projections/market inputs to evaluate() as a plain dict to keep API simple.
    """

    @classmethod
    @abstractmethod
    def from_params(cls, params: Dict) -> "BaseValuation":
        """
        Build a valuation model instance from a parameter dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, inputs: Dict) -> Dict:
        """
        Run the valuation and return a dictionary with key results and breakdown.
        """
        raise NotImplementedError