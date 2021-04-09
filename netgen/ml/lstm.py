"""
LSTM stuff.
"""

from typing import Union

from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from skorch import NeuralNetClassifier
from torch import LongTensor
from torch import Tensor
from torch.cuda import is_available
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Softmax
from torch.optim import Adam

from netgen.ml.nn import NeuralNetworkClassifier


class LSTMModule(Module):
    """
    A fully connected neural network module.
    """

    def __init__(self, inputs: int, outputs: int, layers: int, neurons_per_layer: int, p: float):
        """
        Creates the module.
        :param inputs: the number of input neurons
        :param outputs: the number of output neurons
        :param layers: the number of layers
        :param neurons_per_layer: the number of neurons in the hidden layers
        :param p: the dropout probability
        """

        super(LSTMModule, self).__init__()

        modules = []
        self.__lstm = LSTM(inputs, neurons_per_layer, layers, dropout=p)
        self.__linear = Linear(neurons_per_layer, outputs)
        self.__softmax = Softmax(dim=-1)

        self.__modules = Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass.
        :param x: the input tensor to process
        :return: the resulting tensor
        """

        y, _ = self.__lstm(x)
        y = y[:, -1, :]
        y = self.__linear(y)
        y = self.__softmax(y)

        return y


def train_lstm(trial: Union[Trial, FrozenTrial], x: Tensor, y: Series) -> NeuralNetClassifier:
    """
    Trains a LSTM neural network.

    :param trial: the trial to use
    :param x: the input data
    :param y: the output data

    :return: the trained classifier
    """

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("categorical", (32, 64, 128))
    layers = trial.suggest_int("layers", 1, 4)
    neurons_per_layer = trial.suggest_categorical("neurons_per_layer", (10, 100))

    classes = sorted(y.unique().tolist())
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    y = LongTensor(LabelEncoder().fit_transform(y.to_numpy()))
    classifier = NeuralNetworkClassifier(module=LSTMModule,
                                         module__inputs=x[0].shape[1], module__outputs=len(classes),
                                         classes=classes,
                                         train_split=None,
                                         optimizer=Adam,
                                         iterator_train__shuffle=True,
                                         criterion__weight=Tensor(class_weights),
                                         device="cuda" if is_available() else "cpu",
                                         verbose=0,
                                         lr=lr,
                                         max_epochs=10,
                                         batch_size=batch_size,
                                         module__layers=layers,
                                         module__neurons_per_layer=neurons_per_layer,
                                         module__p=0.1)

    classifier.fit(x, y)

    return classifier
