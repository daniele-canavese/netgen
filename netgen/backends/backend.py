"""
The base back-end class.
"""

from abc import ABC

from pandas import DataFrame


class BackEnd(ABC):
    """
    The base back-end class.
    """

    def report(self, results: DataFrame) -> None:
        """
        Reports some classification results.

        :param results: the classification results to report
        """

        pass
