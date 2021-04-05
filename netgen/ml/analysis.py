"""
Data set analysis functions and classes.
"""

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple

from pandas import DataFrame
from pandas import Series
from pandas import concat
from torch import Tensor


def to_dataframe(data_set: Dict[str, Any], features: Sequence[str]) -> Tuple[DataFrame, DataFrame, Series]:
    """
    Converts a dictionary data set to a data frame and series. This function effectively concatenates all the chunks.

    :param data_set: the dictionary data set to convert
    :param features: the input features to use
    :return: a tuple where the first value is the original data frame, the second value the input data frame and the
             third a series containing the outputs
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

    return x, x[features], y


def to_2d_tensor(data_set: Dict[str, Any], features: Sequence[str]) -> Tuple[DataFrame, Tensor, Series]:
    """
    Converts a dictionary data set to a 2d tensor and a series. This function effectively concatenates all the chunks.

    :param data_set: the dictionary data set to convert
    :param features: the input features to use
    :return: a tuple where the first value is the original data frame, the second value the input tensor and the third a
             series containing the outputs
    """

    original, x, y = to_dataframe(data_set, features)
    x = Tensor(x.to_numpy())

    return original, x, y
