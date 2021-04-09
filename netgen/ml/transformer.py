"""
Transformer stuff.
"""
from math import log
from typing import Union

from optuna import Trial
from optuna.trial import FrozenTrial
from pandas import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from skorch import NeuralNetClassifier
from torch import LongTensor
from torch import Tensor
from torch import arange
from torch import cos
from torch import exp
from torch import float as torch_float
from torch import sin
from torch import zeros
from torch.cuda import is_available
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Softmax
from torch.nn import TransformerEncoderLayer
from torch.optim import Adam

from netgen.ml.nn import NeuralNetworkClassifier


class PositionalEncoder(Module):
    """
    The positional encoder.
    """

    def __init__(self, inputs: int, max_timesteps: int):
        """
        Creates the encode.

        :param inputs: the number of inputs
        :param max_timesteps: the maximum number of timesteps
        """

        super(PositionalEncoder, self).__init__()

        pe = zeros(max_timesteps, inputs)
        position = arange(0, max_timesteps, dtype=torch_float).unsqueeze(1)
        factor = exp(arange(0, inputs, 2).float() * (-log(10000.0) / inputs))
        pe[:, 0::2] = sin(position * factor)
        pe[:, 1::2] = cos(position * factor)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass.
        :param x: the input tensor to process
        :return: the resulting tensor
        """

        return x + self.pe[:x.shape[0], :]


class TransformerModule(Module):
    """
    A transformer neural network module.
    """

    def __init__(self, inputs: int, outputs: int, max_timesteps: int, layers: int, neurons_per_layer: int, p: float):
        """
        Creates the module.
        :param inputs: the number of input neurons
        :param outputs: the number of output neurons
        :param max_timesteps: the maximum number of timesteps
        :param layers: the number of encoding layers
        :param neurons_per_layer: the number of neurons in the hidden layers
        :param p: the dropout probability
        """

        super(TransformerModule, self).__init__()

        self.__positional_encoder = PositionalEncoder(inputs, max_timesteps)
        encoders = []
        for _ in range(layers):
            encoders.append(TransformerEncoderLayer(inputs, 1, neurons_per_layer, p))
        self.__encoders = Sequential(*encoders)
        self.__linear = Linear(inputs, outputs)
        self.__softmax = Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass.
        :param x: the input tensor to process
        :return: the resulting tensor
        """

        y = self.__positional_encoder(x)
        y = self.__encoders(y)
        y = y[:, -1, :]
        y = self.__linear(y)
        y = self.__softmax(y)

        return y


def train_transformer(trial: Union[Trial, FrozenTrial], x: Tensor, y: Series) -> NeuralNetClassifier:
    """
    Trains a LSTM neural network.

    :param trial: the trial to use
    :param x: the input data
    :param y: the output data

    :return: the trained classifier
    """

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("categorical", (32, 64, 128))
    layers = trial.suggest_int("layers", 1, 6)
    neurons_per_layer = trial.suggest_categorical("neurons_per_layer", (10, 100))

    classes = sorted(y.unique().tolist())
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    y = LongTensor(LabelEncoder().fit_transform(y.to_numpy()))
    classifier = NeuralNetworkClassifier(module=TransformerModule,
                                         module__inputs=x[0].shape[1], module__outputs=len(classes),
                                         module__max_timesteps=x[0].shape[0],
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
