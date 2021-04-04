"""
The amazing NetGen class.
"""
from collections import Sequence
from configparser import ConfigParser
from enum import Enum
from glob import glob
from os import mkdir
from os.path import exists
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from blessed import Terminal
from joblib import load
from numpy import argmax
from numpy import inf
from ops import optimize
from pandas import DataFrame
from pandas import concat
from quill_ml import ClassificationReport
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split

from netgen.ml import to_dataframe
from netgen.ml import train_extra_trees
from netgen.ml import train_random_forest
from netgen.net import TstatAnalyzer


class ClassifierType(Enum):
    """
    The classifier types.
    """

    COMBINATORIAL_TABLE = "combinatorial_table"


class NetGen:
    """
    The amazing NetGen class.
    """

    def __init__(self, configuration: ConfigParser):
        """
        Creates a new NetGen model.
        """

        packets = configuration.getint("data_set", "packets")

        self.__configuration = configuration
        self.__analyzer = TstatAnalyzer(self.__configuration, packets)
        self.__terminal = Terminal()

    def __get_features(self, features: List[str]):
        """
        Gets the list of input features.

        :param features: the initial list of features
        :return: the list of input features
        """

        id_fields = self.__configuration.get("data_set", "id_fields").split()
        for i in id_fields:
            if i in features:
                features.remove(i)

        excluded_fields = self.__configuration.get("data_set", "excluded_fields").split()
        for i in excluded_fields:
            if i in features:
                features.remove(i)

        return features

    @staticmethod
    def get_classes(model_name: str) -> Sequence[str]:
        """
        Retrieves the classes of a classifier.

        :param model_name: the file name of the model
        :return: the classes of the classifier
        """

        model = load(model_name)

        # noinspection PyUnresolvedReferences
        return model["classifier"].classes_

    @staticmethod
    def __infer(classifier: ClassifierMixin, x: DataFrame, y: Optional[DataFrame]) -> DataFrame:
        """
        Infers the classes of some samples.

        :param classifier: the classifier to use
        :param x: the input data
        :param y: the output data; sets to None if the output data is unknown
        :return: a data frame where the first column are the inferred classes, the second the probabilities and the
                 remaining columns the class probabilities
        """

        # noinspection PyUnresolvedReferences
        probabilities = classifier.predict_proba(x)
        indexes = argmax(probabilities, axis=1)
        # noinspection PyUnresolvedReferences
        classes = classifier.classes_

        data = DataFrame()
        if y is not None:
            data["target"] = y
        data["inferred"] = [classes[i] for i in indexes]
        data["probability"] = [probabilities[i, j] for i, j in enumerate(indexes)]
        for i in range(len(classes)):
            data[classes[i]] = probabilities[:, i]

        return data

    def infer(self, model_name: str, target: str) -> DataFrame:
        """
        Performs the classification of some traffic.

        :param model_name: the file name of the model
        :param target: the pcap file to analyze or the interface to sniff
        :return: the classification of the data
        """

        id_fields = self.__configuration.get("data_set", "id_fields").split()

        model = load(model_name)
        classifier = model["classifier"]
        classifier_type = model["type"]

        if exists(target):
            data_set = self.__analyzer.analyze(target)
        else:
            data_set = self.__analyzer.sniff(target)

        x = None
        if classifier_type == ClassifierType.COMBINATORIAL_TABLE:
            x, _ = to_dataframe({"?": data_set})

        if len(x) > 0:
            results = self.__infer(classifier, x[self.__get_features(x.columns.to_list())], None)
            results = concat((x[id_fields], results), axis=1)
        else:
            results = DataFrame()

        return results

    def test(self, model: Dict[str, Any], train_x: DataFrame, test_x: DataFrame, train_y: DataFrame,
             test_y: DataFrame, folder: str, verbose: bool = True) -> None:
        """
        Tests a model.

        :param model: the model to test
        :param train_x: the training set inputs
        :param test_x: the test set inputs
        :param train_y: the training set outputs
        :param test_y: the test set outputs
        :param folder: the test folder
        :param verbose: toggles the verbosity
        """

        if verbose:
            print(self.__terminal.red("TESTING..."))

        classifier = model["classifier"]

        if verbose:
            print(self.__terminal.darkorange("analyzing the training set..."))
        train = self.__infer(classifier, train_x, train_y)

        if verbose:
            print(self.__terminal.darkorange("analyzing the test set..."))
        test = self.__infer(classifier, test_x, test_y)

        if verbose:
            print(self.__terminal.darkorange("generating the report..."))
        report = ClassificationReport("NetGen report")
        report.set_classification_data("IDS", "training set", train)
        report.set_classification_data("IDS", "test set", test)
        if not exists(folder):
            mkdir(folder)
        report.render(folder)

    def train(self, verbose: bool = True) -> Tuple[Dict[str, Any], DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Generates a new IDS.

        :param verbose: toggles the verbosity
        :return: the trained model and four dataframes corresponding to the training and test set inputs and the
                 training and test set outputs
        """

        random_forests = self.__configuration.getboolean("models", "random_forests")
        extra_trees = self.__configuration.getboolean("models", "extra_trees")
        timeout = self.__configuration.getint("models", "timeout")
        test_fraction = self.__configuration.getfloat("data_set", "test_fraction")

        data_set = {}

        if verbose:
            print(self.__terminal.red("TRAINING..."))
        for name in self.__configuration["classes"]:
            if verbose:
                print(self.__terminal.darkorange("analyzing the captures for the class \"%s\"..." % name))
            data = []
            for pcap in sorted(glob(self.__configuration.get("classes", name), recursive=True)):
                if verbose:
                    print("%30s:" % pcap, end="")
                t = self.__analyzer.analyze(pcap)
                if verbose:
                    print(" %6d sequences, %7d timesteps" % (len(t), sum([len(i) for i in t])))
                data.extend(t)
            if verbose:
                print("%30s: %6d sequences, %7d timesteps" % ("total", len(data), sum([len(i) for i in data])))
            data_set[name] = data

        if random_forests or extra_trees:
            if verbose:
                print(self.__terminal.darkorange("creating the tables for the combinatorial models..."))

            x, y = to_dataframe(data_set)
            features = self.__get_features(x.columns.to_list())

            train_x, test_x, train_y, test_y = train_test_split(x[features], y, train_size=1 - test_fraction,
                                                                stratify=y)
            if verbose:
                print("training: %7d samples" % len(train_x))
                print("    test: %7d samples" % len(test_x))
                print("   total: %7d samples" % len(x))

            model = {}
            best = -inf
            if random_forests:
                if verbose:
                    print(self.__terminal.darkorange("optimizing random forests..."))
                classifier, study = optimize("random forests study", train_x, train_y, train_random_forest,
                                             timeout=timeout, verbose=verbose)
                model["classifier"] = classifier
                model["type"] = ClassifierType.COMBINATORIAL_TABLE
                best = study.best_value
                print("new best classifier: random forest")
            if extra_trees:
                if verbose:
                    print(self.__terminal.darkorange("optimizing extra-trees..."))
                classifier, study = optimize("extra trees study", train_x, train_y, train_extra_trees,
                                             timeout=timeout, verbose=verbose)
                if study.best_value > best:
                    model["classifier"] = classifier
                    model["type"] = ClassifierType.COMBINATORIAL_TABLE
                    best = study.best_value
                    print("new best classifier: extra-trees")

            return model, train_x, test_x, train_y, test_y
