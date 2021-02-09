"""
Base traffic analysis class.
"""
from abc import ABC
from typing import Sequence

from pandas import DataFrame


class Analyzer(ABC):
    """
    Abstract base class for all the traffic analyzers.
    """

    def analyze(self, file: str) -> Sequence[DataFrame]:
        """
        Analyzes a capture file.

        :param file: the file name to analyze
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        pass

    def sniff(self, interface: str) -> Sequence[DataFrame]:
        """
        Sniffs an interface.

        :param interface: the name of the interface to sniff
        :return: a list of dataframes where each dataframe contains the time steps of a flow
        """

        pass
