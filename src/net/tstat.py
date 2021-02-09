"""
Tstat class.
"""

from collections import Sequence

from pandas import DataFrame

from .analysis import Analyzer


class TstatAnalyzer(Analyzer):
    """
    The tstat analyzer.
    """

    def __init__(self, configuration: str, packets: int) -> None:
        """
        Creates the analyzer.

        :param configuration: the name of the configuration file for tstat
        :param packets: the number of packets after which the analysis functions will return
        """

        self.__packets = packets

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
