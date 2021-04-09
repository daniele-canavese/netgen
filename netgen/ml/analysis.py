"""
Data set analysis functions and classes.
"""

from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from warnings import catch_warnings
from warnings import simplefilter

from numpy import array
from pandas import DataFrame
from pandas import Series
from pandas import concat
from torch import Tensor
from torch import vstack
from torch import zeros


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


def to_2d_tensors(data_set: Dict[str, Any], features: Sequence[str], max_timesteps: int) -> \
        Tuple[DataFrame, array, Series]:
    """
    Converts a dictionary data set to a list of 2d tensors and a series. This function effectively concatenates all the
    chunks.

    :param data_set: the dictionary data set to convert
    :param features: the input features to use
    :param max_timesteps: the maximum number of timesteps
    :return: a tuple where the first value is the original data frame, the second value the list of input tensors and
             the third a series containing the outputs
    """

    original = []
    x = []
    y = []
    for key, value in data_set.items():
        for j in value:
            original.append(j.iloc[-1].to_dict())
            v = Tensor(j[features].to_numpy())
            if v.shape[0] >= max_timesteps:
                v = v[0:max_timesteps, :]
            else:
                v = vstack((v, zeros(max_timesteps - v.shape[0], v.shape[1])))
            x.append(v)
            y.append(key)
    with catch_warnings():
        simplefilter(action="ignore", category=FutureWarning)
        x = array(x, dtype=object)
        y = Series(y)
        original = DataFrame(original)

    return original, x, y
