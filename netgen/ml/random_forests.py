"""
Random forests stuff.
"""
from typing import Union

from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pandas import Series
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(trial: Union[Trial, FrozenTrial], x: DataFrame, y: Series) -> RandomForestClassifier:
    """
    Trains a random forest.

    :param trial: the trial to use
    :param x: the input data
    :param y: the output data

    :return: the trained classifier
    """

    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, min(100, len(x)))
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, min(100, len(x)))
    max_features = trial.suggest_int("max_features", 1, len(x.columns))
    if isinstance(trial, Trial):
        n_jobs = 1
    else:
        n_jobs = -1

    classifier = RandomForestClassifier(class_weight="balanced_subsample", bootstrap=False, n_jobs=n_jobs,
                                        n_estimators=n_estimators,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features)
    classifier.fit(x, y)

    return classifier
