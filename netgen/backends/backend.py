"""
The base back-end class.
"""

from abc import ABC
from typing import Any

from pandas import DataFrame


class BackEnd(ABC):
    """
    The base back-end class.
    """

    def report(self, results: DataFrame, item: Any) -> None:
        """
        Reports some classification results.

        :param results: the classification results to report
        :param item: the object that triggered the classification
        """

        pass
