"""
The UI back-end.
"""
from colorsys import hls_to_rgb
from typing import Any
from typing import Sequence

from blessed import Terminal
from pandas import DataFrame

from netgen.backends.backend import BackEnd


class UIBackEnd(BackEnd):
    """
    The CSV back-end.
    """

    def __init__(self, classes: Sequence[str]) -> None:
        """
        Create the back-end.
        :param classes: the list of supported classes
        """

        self.__terminal = Terminal()
        self.__first = True
        self.__labels = {}
        self.__labels = {}
        self.__rows = []

        for index, label in enumerate(classes):
            r, g, b = hls_to_rgb(index / len(classes) + 0.1, 0.7, 0.9)
            self.__labels[label] = (self.__terminal.color_rgb(int(r * 255), int(g * 255),
                                                              int(b * 255)) + "%15s" % label + self.__terminal.normal)

        print(self.__terminal.enter_fullscreen(), end="", flush=True)
        print(self.__terminal.move_yx(0, 0), end="", flush=True)

    def report(self, results: DataFrame, item: Any) -> None:
        """
        Reports some classification results.

        :param results: the classification results to report
        :param item: the object that triggered the classification
        """

        columns = results.columns
        if "probability" not in columns:
            return

        if "target" in columns:
            results.drop("target", axis=1, inplace=True)
        index = results.columns.to_list().index("probability") - 1
        results = results.iloc[:, 0: index + 2]

        if self.__first:
            print(self.__terminal.move_xy(0, 0) +
                  self.__terminal.gold(self.__terminal.bold("  ".join(["%15s" % i for i in results.columns]))))
            self.__first = False

        for _, row in results.iterrows():
            identifier = " ".join(row[0:index].astype(str))
            inferred = row[index]
            probability = row["probability"] * 100
            if probability >= 90:
                probability = self.__terminal.palegreen("%15.3f" % probability)
            elif probability >= 70:
                probability = self.__terminal.ivory("%15.3f" % probability)
            else:
                probability = "%15.3f" % probability
            if identifier in self.__rows:
                line = self.__rows.index(identifier) + 1
            else:
                line = len(self.__rows) + 1
                self.__rows.append(identifier)
                if line >= self.__terminal.height - 1:
                    line -= 1
                    self.__rows.pop(0)
            row = [
                    *[self.__terminal.gray("%15s") % i for i in row[0:index]],
                    self.__labels[inferred],
                    probability
            ]
            print(self.__terminal.move_xy(0, line) + "  ".join(row), end="", flush=True)

    def __del__(self) -> None:
        """
        Destroys the back-end.
        """

        print(self.__terminal.exit_fullscreen(), end="", flush=True)
