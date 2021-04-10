"""
The amazing NetGen class.
"""
from collections import Iterable
from collections import Sequence
from configparser import ConfigParser
from enum import Enum
from glob import glob
from os import mkdir
from os.path import basename
from os.path import dirname
from os.path import exists
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from warnings import catch_warnings
from warnings import simplefilter

from blessed import Terminal
from joblib import load
from numpy import argmax
from numpy import array
from numpy import inf
from ops import optimize
from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pandas import Series
from pandas import concat
from quill_ml import ClassificationReport
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from yaml import CBaseLoader
from yaml import load as yaml_load

from netgen.ml import to_2d_tensor
from netgen.ml import to_2d_tensors
from netgen.ml import to_dataframe
from netgen.ml import train_extra_trees
from netgen.ml import train_fully_connected
from netgen.ml import train_knn
from netgen.ml import train_lstm
from netgen.ml import train_random_forest
from netgen.ml import train_svm
from netgen.ml import train_transformer
from netgen.net import TstatAnalyzer


def infer(classifier: Any, x: Any) -> Any:
    """
    Infers a classification.

    This method tries to use a method called infer in the classifier and if it does not exists it uses the predict
    method. If the obtained results are tuples, then only the first element of the tuple is returned.

    :param classifier: the classifier to use
    :param x: the input data
    :return: the inferred classes
    """

    return classifier.predict(x)


class ClassifierType(Enum):
    """
    The classifier types.
    """

    COMBINATORIAL_TABLE = "combinatorial_table"
    COMBINATORIAL_TENSOR = "combinatorial_tensor"
    SEQUENTIAL_TENSOR = "sequential_tensor"


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
    def __infer(classifier: ClassifierMixin, x: DataFrame, y: Optional[Series]) -> DataFrame:
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

    def __optimize(self, name: str, scale: bool, x: Any, y: Any,
                   train: Callable[[Union[Trial, FrozenTrial], Any, Any], Any],
                   timeout: int, kind: ClassifierType,
                   model: Dict[str, Any], best: float, verbose: bool) -> Tuple[Dict[str, Any], float]:
        """
        Optimizes a new classifier.

        :param name: the name of the model type
        :param scale: enables the input scaling
        :param x: the input values to use
        :param y: the output values to use
        :param train: the training function; it receives in input a trial object, the input and output training values
                      and returns the classifier itself
        :param timeout: the timeout in seconds
        :param kind: the classifier type
        :param model: the best model so far
        :param best: the best score so far
        :param verbose: toggles the verbosity
        :return: a tuple where the first element is the best model and the second element is the best score
        """

        if scale:
            print(self.__terminal.gold("scaling the input data..."))
            scaler = StandardScaler()
            if isinstance(x, (DataFrame, Tensor)):
                scaler.fit(x)
            else:
                for i in x:
                    scaler.partial_fit(i)
            x = self.__scale(scaler, x)
        else:
            scaler = None

        if verbose:
            print(self.__terminal.gold("optimizing a %s..." % name))

        classifier, study = optimize("%s study" % name, x, y, train, infer=infer, timeout=timeout, verbose=verbose)

        if study.best_value > best:
            best = study.best_value
            model = {
                    "scaler":     scaler,
                    "classifier": classifier,
                    "type":       kind
            }
            print("the new best classifier is a %s" % name)

        return model, best

    @staticmethod
    def __scale(scaler: Optional[StandardScaler], x: Any) -> Any:
        """
        Scales some data.

        :param scaler: the scaler to use or None not to scale anything
        :param x: the data to scale
        :return: the (optionally) scaled data
        """

        if scaler is None:
            xx = x
        elif isinstance(x, DataFrame):
            xx = scaler.transform(x)
            xx = DataFrame(data=xx, columns=x.columns, index=x.index)
        elif isinstance(x, Series):
            xx = scaler.transform(x)
            xx = Series(data=xx, index=x.index)
        elif isinstance(x, Tensor):
            xx = scaler.transform(x)
            xx = Tensor(xx)
        elif isinstance(x, Iterable):
            xx = []
            for i in x:
                xx.append(NetGen.__scale(scaler, i))
            with catch_warnings():
                simplefilter(action="ignore", category=FutureWarning)
                xx = array(xx, dtype=object)
        else:
            xx = x

        return xx

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

    def infer(self, model: Union[Dict[str, Any], str], target: Union[str, Any],
              classes: Optional[Sequence[str]] = None) -> DataFrame:
        """
        Performs the classification of some traffic.

        :param model: the file name of the model of the model itself
        :param target: the pcap file to analyze, the interface to sniff or the data itself
        :param classes: the expected list of classes; sets to None if there is no such list
        :return: the classification of the data
        """

        id_fields = self.__configuration.get("data_set", "id_fields").split()
        max_timesteps = self.__configuration.getint("models", "max_timesteps")

        if isinstance(model, str):
            model = load(model)
        scaler = model["scaler"]
        classifier = model["classifier"]
        classifier_type = model["type"]

        if isinstance(target, str):
            if exists(target):
                data_set = self.__analyzer.analyze(target)
            else:
                data_set = self.__analyzer.sniff(target)
        else:
            data_set = target

        if len(data_set) > 0:
            features = self.__get_features(data_set[0].columns.to_list())
        else:
            features = []
        if classifier_type == ClassifierType.COMBINATORIAL_TABLE:
            original, x, y = to_dataframe(data_set, classes, features)
        elif classifier_type == ClassifierType.COMBINATORIAL_TENSOR:
            original, x, y = to_2d_tensor(data_set, classes, features)
        elif classifier_type == ClassifierType.SEQUENTIAL_TENSOR:
            original, x, y = to_2d_tensors(data_set, classes, features, max_timesteps)
        else:
            raise RuntimeError("Unsupported classifier type")

        x = self.__scale(scaler, x)

        if len(original) > 0:
            results = self.__infer(classifier, x, y)
            results = concat((original[id_fields], results), axis=1)
        else:
            results = DataFrame()

        return results

    def test(self, model: Dict[str, Any], train_x: Any, test_x: Any, train_y: Sequence[str], test_y: Sequence[str],
             folder: str, verbose: bool = True) -> None:
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

        id_fields = self.__configuration.get("data_set", "id_fields").split()

        if verbose:
            print(self.__terminal.tomato("TESTING..."))

        if verbose:
            print(self.__terminal.gold("analyzing the training set..."))
        train = self.infer(model, train_x, train_y)
        train = train.iloc[:, len(id_fields):]

        if verbose:
            print(self.__terminal.gold("analyzing the test set..."))
        test = self.infer(model, test_x, test_y)
        test = test.iloc[:, len(id_fields):]

        if verbose:
            print(self.__terminal.gold("generating the report..."))
        with catch_warnings():
            simplefilter("ignore")
            report = ClassificationReport("NetGen report")
            report.set_classification_data("analyzer", "training set", train)
            report.set_classification_data("analyzer", "test set", test)
            if not exists(folder):
                mkdir(folder)
            report.render(folder)

    # noinspection DuplicatedCode
    def train(self, data_file: str, verbose: bool = True) -> Tuple[Dict[str, Any], Any, Any, Any, Any]:
        """
        Generates a new traffic analyzer.

        :param data_file: the name of the data file
        :param verbose: toggles the verbosity
        :return: the trained model and four dataframes corresponding to the training and test set inputs and the
                 training and test set outputs
        """

        with open(data_file) as f:
            data = yaml_load(f, Loader=CBaseLoader)

        random_forest = self.__configuration.get("models", "random_forest")
        extra_trees = self.__configuration.get("models", "extra_trees")
        svm = self.__configuration.get("models", "svm")
        knn = self.__configuration.get("models", "knn")
        fully_connected = self.__configuration.get("models", "fully_connected")
        lstm = self.__configuration.get("models", "lstm")
        transformer = self.__configuration.get("models", "transformer")
        timeout = self.__configuration.getint("models", "timeout")
        max_timesteps = self.__configuration.getint("models", "max_timesteps")
        test_fraction = self.__configuration.getfloat("data_set", "test_fraction")
        id_fields = self.__configuration.get("data_set", "id_fields").split()

        data_set_x = []
        data_set_y = []

        if verbose:
            print(self.__terminal.tomato("TRAINING..."))
        folder = dirname(data_file)
        if folder == "":
            folder = "."
        for name, captures in data.items():
            class_data_set = []
            if verbose:
                print(self.__terminal.gold("analyzing the captures for the class \"%s\"..." % name))
            for entry, rules in captures.items():
                rules = set(rules)  # For a faster search later.
                for capture in sorted(glob("%s/%s" % (folder, entry), recursive=True)):
                    if verbose:
                        print("%30s:" % basename(capture), end="")
                    t = self.__analyzer.analyze(capture)
                    valid = []
                    if len(rules) > 0:
                        for i in t:
                            if " ".join(i.loc[0, id_fields].astype(str)) in rules:
                                valid.append(i)
                    else:
                        valid = t
                    class_data_set.extend(valid)
                    data_set_x.extend(class_data_set)
                    data_set_y.extend([name] * len(class_data_set))
                    if verbose:
                        print(" %6d sequences, %7d timesteps" % (len(valid), sum([len(i) for i in valid])))
            if verbose and len(captures) > 1:
                print("%30s: %6d sequences, %7d timesteps" %
                      ("total", len(class_data_set), sum([len(i) for i in class_data_set])))

        if verbose:
            print(self.__terminal.gold("splitting into training and test sets..."))
        train_x, test_x, train_y, test_y = train_test_split(data_set_x, data_set_y, train_size=1 - test_fraction,
                                                            stratify=data_set_y)
        if verbose:
            print("training: %7d sequences" % len(train_x))
            print("    test: %7d sequences" % len(test_x))
            print("   total: %7d sequences" % len(data_set_x))

        train_sequences = len(train_x)
        train_timesteps = sum([len(i) for i in train_x])

        random_forest = train_timesteps <= 1000000 if random_forest == "auto" else random_forest == "true"
        extra_trees = train_timesteps > 1000000 if extra_trees == "auto" else extra_trees == "true"
        svm = train_timesteps <= 1000 if svm == "auto" else svm == "true"
        knn = train_timesteps <= 1000 if knn == "auto" else knn == "true"
        fully_connected = (10000 <= train_timesteps <= 1000000
                           if fully_connected == "auto" else fully_connected == "true")
        lstm = (10000 <= train_sequences <= 1000000 if lstm == "auto" else lstm == "true")
        transformer = (10000 <= train_sequences <= 1000000 if transformer == "auto" else transformer == "true")

        features = self.__get_features(train_x[0].columns.to_list())
        model = {}
        best = -inf

        if random_forest or extra_trees or svm or knn:
            if verbose:
                print(self.__terminal.gold("creating the tables for the combinatorial models..."))
            _, train_x2, train_y2 = to_dataframe(train_x, train_y, features)
            _, test_x2, test_y2 = to_dataframe(test_x, test_y, features)

            with catch_warnings():
                simplefilter("ignore")
                if random_forest:
                    model, best = self.__optimize("random forest", False, train_x2, train_y2, train_random_forest,
                                                  timeout, ClassifierType.COMBINATORIAL_TABLE, model, best, verbose)
                if extra_trees:
                    model, best = self.__optimize("extra-trees", False, train_x2, train_y2, train_extra_trees,
                                                  timeout, ClassifierType.COMBINATORIAL_TABLE, model, best, verbose)
                if svm:
                    model, best = self.__optimize("bagging classifier of SVMs", True, train_x2, train_y2, train_svm,
                                                  timeout, ClassifierType.COMBINATORIAL_TABLE, model, best, verbose)
                if knn:
                    model, best = self.__optimize("kNN", True, train_x2, train_y2, train_knn, timeout,
                                                  ClassifierType.COMBINATORIAL_TABLE, model, best, verbose)

        if fully_connected:
            if verbose:
                print(self.__terminal.gold("creating the 2D tensors for the combinatorial models..."))
            _, train_x2, train_y2 = to_2d_tensor(train_x, train_y, features)
            _, test_x2, test_y2 = to_2d_tensor(test_x, test_y, features)

            with catch_warnings():
                simplefilter("ignore")
                if fully_connected:
                    model, best = self.__optimize("fully connected neural network", True, train_x2, train_y2,
                                                  train_fully_connected, timeout, ClassifierType.COMBINATORIAL_TENSOR,
                                                  model, best, verbose)

        if lstm or transformer:
            if verbose:
                print(self.__terminal.gold("creating the 2D tensors for the sequential models..."))
            _, train_x2, train_y2 = to_2d_tensors(train_x, train_y, features, max_timesteps)
            _, test_x2, test_y2 = to_2d_tensors(test_x, test_y, features, max_timesteps)

            with catch_warnings():
                simplefilter("ignore")
                if lstm:
                    model, best = self.__optimize("LSTM neural network", True, train_x2, train_y2,
                                                  train_lstm, timeout, ClassifierType.SEQUENTIAL_TENSOR, model, best,
                                                  verbose)
                if transformer:
                    model, best = self.__optimize("transformer neural network", True, train_x2, train_y2,
                                                  train_transformer, timeout, ClassifierType.SEQUENTIAL_TENSOR, model,
                                                  best, verbose)

        return model, train_x, test_x, train_y, test_y
