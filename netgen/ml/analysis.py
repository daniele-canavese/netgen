"""
Data set analysis functions and classes.
"""

from typing import Optional
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


def to_dataframe(x: Sequence[DataFrame], y: Optional[Sequence[str]], features: Sequence[str]) -> \
        Tuple[DataFrame, DataFrame, Series]:
    """
    Converts a data set to a data frame and series. This function effectively concatenates all the chunks.

    :param x: the list of input data frames
    :param y: the list of classes; sets to None if there are no known classes
    :param features: the input features to use
    :return: a tuple where the first value is the original data frame, the second value is the input data frame and the
             third value is the output series
    """

    labels = []
    if y is not None:
        for index, table in enumerate(x):
            labels.extend([y[index]] * len(table))

    if len(x) > 0:
        x = concat(x).reset_index(drop=True)
    else:
        x = DataFrame()
    if len(labels) > 0:
        y = Series(labels, dtype="category")
    else:
        y = Series()

    return x, x[features], y


def to_2d_tensor(x: Sequence[DataFrame], y: Optional[Sequence[str]], features: Sequence[str]) -> \
        Tuple[DataFrame, Tensor, Series]:
    """
    Converts a data set to a 2d tensor and a series. This function effectively concatenates all the chunks.

    :param x: the list of input data frames
    :param y: the list of classes; sets to None if there are no known classes
    :param features: the input features to use
    :return: a tuple where the first value is the original data frame, the second value the input tensor and the third a
             series containing the outputs
    """

    original, x, y = to_dataframe(x, y, features)
    x = Tensor(x.to_numpy().astype("float32"))

    return original, x, y


def to_2d_tensors(x: Sequence[DataFrame], y: Optional[Sequence[str]], features: Sequence[str], max_timesteps: int) -> \
        Tuple[DataFrame, array, Series]:
    """
    Converts a data set to a list of 2d tensors and a series. This function effectively concatenates all the chunks.

    :param x: the list of input data frames
    :param y: the list of classes; sets to None if there are no known classes
    :param features: the input features to use
    :param max_timesteps: the maximum number of timesteps
    :return: a tuple where the first value is the original data frame, the second value the list of input tensors and
             the third a series containing the outputs
    """

    original = []
    new_x = []
    for index, i in enumerate(x):
        original.append(i.iloc[-1].to_dict())
        v = Tensor(i[features].to_numpy().astype("float32"))
        if v.shape[0] >= max_timesteps:
            v = v[0:max_timesteps, :]
        else:
            v = vstack((v, zeros(max_timesteps - v.shape[0], v.shape[1])))
        new_x.append(v)
    with catch_warnings():
        simplefilter(action="ignore", category=FutureWarning)
        original = DataFrame(original)
        new_x = array(new_x, dtype=object)
        y = Series(y)

    return original, new_x, y


def find_divisors(value: int, max_divisor: int) -> Sequence[int]:
    """
    Finds the divisors of a number.

    :param value: the value to compute the divisors for
    :param max_divisor: the maximum allowed divisor
    :return: the list of divisors
    """

    divisors = []
    for i in range(1, min(max_divisor, value) + 1):
        if value % i == 0:
            divisors.append(i)

    return divisors
