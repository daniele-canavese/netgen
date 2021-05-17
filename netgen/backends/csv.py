"""
The CSV back-end.
"""
from typing import Any

from pandas import DataFrame

from netgen.backends.backend import BackEnd


class CSVBackEnd(BackEnd):
    """
    The CSV back-end.
    """

    def __init__(self) -> None:
        """
        Create the back-end.
        """

        self.__first = True

    def report(self, results: DataFrame, item: Any) -> None:
        """
        Reports some classification results.

        :param results: the classification results to report
        :param item: the object that triggered the classification
        """

        if self.__first:
            print(results.to_csv(index=False), end="", flush=True)
            self.__first = False
        else:
            print(results.to_csv(index=False, header=False), end="", flush=True)
