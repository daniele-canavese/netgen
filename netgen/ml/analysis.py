"""
Data set analysis functions and classes.
"""
from typing import Any
from typing import Dict
from typing import Tuple

from pandas import DataFrame
from pandas import Series
from pandas import concat


def to_dataframe(data_set: Dict[str, Any]) -> Tuple[DataFrame, Series]:
    """
    Converts a dictionary data set to a data frame data set. This function effectively concatenates all the chunks.

    :param data_set: the dictionary data set to convert
    :return: a tuple where the first value is the data frame data set and the second a series containing the labels
    """

    x = []
    y = []

    for name, sequences in data_set.items():
        for sequence in sequences:
            x.append(sequence)
            y.extend([name] * len(sequence))

    if len(x) > 0:
        x = concat(x).reset_index(drop=True)
    else:
        x = DataFrame()
    if len(y) > 0:
        y = Series(y).reset_index(drop=True)
    else:
        y = Series()

    return x, y
