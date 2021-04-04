"""
kNNs stuff.
"""
from typing import Union

from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pandas import Series
from sklearn.neighbors import KNeighborsClassifier


# noinspection DuplicatedCode
def train_knn(trial: Union[Trial, FrozenTrial], x: DataFrame, y: Series) -> KNeighborsClassifier:
    """
    Trains a kNN classifier.

    :param trial: the trial to use
    :param x: the input data
    :param y: the output data

    :return: the trained classifier
    """

    n_neighbors = trial.suggest_int("n_neighbors", 3, min(20, len(x)))
    leaf_size = trial.suggest_int("leaf_size", 5, min(50, len(x)))
    p = trial.suggest_int("p", 2, 10)
    metric = trial.suggest_categorical("metric", ("euclidean", "manhattan", "chebyshev", "minkowski"))
    if isinstance(trial, Trial):
        n_jobs = 1
    else:
        n_jobs = -1

    classifier = KNeighborsClassifier(n_jobs=n_jobs,
                                      n_neighbors=n_neighbors,
                                      leaf_size=leaf_size,
                                      p=p,
                                      metric=metric)
    classifier.fit(x, y)

    return classifier
