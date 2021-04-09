"""
Neural network stuff.
"""
from typing import Any
from typing import Dict

from numpy import argmax
from numpy import array
from skorch import NeuralNetClassifier
from torch import Tensor


def infer_neural_network(classifier: Any, x: Any) -> Any:
    """
    Infers a classification.

    This method tries to use a method called infer in the classifier and if it does not exists it uses the predict
    method. If the obtained results are tuples, then only the first element of the tuple is returned.

    :param classifier: the classifier to use
    :param x: the input data
    :return: the inferred classes
    """

    return classifier.predict(x)


class NeuralNetworkClassifier(NeuralNetClassifier):
    """
    A LSTM neural network classifier.
    """

    def infer(self, x: Tensor, **kwargs: Dict[Any, Any]) -> array:
        """
        Predicts a data set.
        :param x: the data set to analyze
        :param kwargs: additional parameters
        :return: the classes
        """

        yy = super(NeuralNetClassifier, self).infer(x, **kwargs)
        return yy

    def predict(self, x: Tensor) -> array:
        """
        Predicts a data set.
        :param x: the data set to analyze
        :return: the classes
        """

        probabilities = self.predict_proba(x)
        indexes = argmax(probabilities, axis=1)
        yy = array([self.classes_[i] for i in indexes])

        return yy
