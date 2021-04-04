"""
Random forests stuff.
"""
from typing import Union

from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pandas import Series
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC


def train_svm(trial: Union[Trial, FrozenTrial], x: DataFrame, y: Series) -> BaggingClassifier:
    """
    Trains a bagging classifier of SVMs.

    :param trial: the trial to use
    :param x: the input data
    :param y: the output data

    :return: the trained classifier
    """

    c = trial.suggest_float("c", 1e-3, 1)
    gamma = trial.suggest_float("gamma", 1e-3, 1)
    tol = trial.suggest_float("tol", 1e-4, 1e-2)
    cache_size = trial.suggest_int("cache_size", 256, 4096)
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_samples = trial.suggest_float("max_samples", 0.1, 1)
    max_features = trial.suggest_int("max_features", 1, len(x.columns))
    if isinstance(trial, Trial):
        n_jobs = 1
    else:
        n_jobs = -1

    base = SVC(class_weight="balanced",
               C=c, gamma=gamma, tol=tol, cache_size=cache_size)
    classifier = BaggingClassifier(base_estimator=base, bootstrap=False, n_jobs=n_jobs,
                                   max_samples=max_samples,
                                   max_features=max_features,
                                   n_estimators=n_estimators)
    classifier.fit(x, y)

    return classifier
