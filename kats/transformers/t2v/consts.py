#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, accuracy_score


def T2VParam(
    mode: str = "regression",
    normalizer: Optional[Any] = None,
    training_output_size: int = 1,
    batch_size: int = 16,
    vector_length: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 64,
    hidden: List[int] = None,
    dropout: float = 0.2,
    optimizer: Any = torch.optim.Adam,
):

    """
    A function for storing all parameters of a t2v model.

    :Parameters:
    mode: str
        Should we treat the training as a classification problem
        or regression problem. Default treats the training as a
        unlabeled regression problem.
    normalizer: Any
        Normalization function imported from utils.
        Currently, only Normalize and Standardize are supported.
        Default value is None.
    training_output_size: int
        How many outputs do we have. This serves as the multi-output
        function under training in regression mode.
        Default value is 1.
    batch_size: int
        Data to be contained in each batch for feeding NN.
    vector_length: int
        Length of each embedding vector.
    learning_rate: float
        Learning rate for training embedding NN.
    epochs: int
        Total number of epochs/iterations used for training embedding
        NN.
    hidden: List[int]
        A list contains how many hidden nodes on each hidden lstm/cnn
        cell, total length is the number of hidden lstm/cnn cells.
    dropout: float
        Percentage of neurons randomly dropped out in the network.
    optimizer: Any
        Pytorch optimizer for training NN.
    validator: Any
        Accuracy validators from scikit-learn. Choose corresponding
        validators based on the mode.
    """

    if mode == "regression":
        loss_function = nn.MSELoss
        validator = mean_absolute_error
    elif mode == "classification":
        loss_function = nn.CrossEntropyLoss
        validator = accuracy_score
    else:
        raise ValueError(f"Mode {mode} cannot be recognized")

    if not hidden:
        hidden = [100]

    T2VParam = NamedTuple(
        "T2VParam",
        [
            ("mode", str),
            ("normalizer", Optional[Any]),
            ("training_output_size", int),
            ("batch_size", int),
            ("vector_length", int),
            ("loss_function", Any),
            ("learning_rate", float),
            ("epochs", int),
            ("hidden", List[int]),
            ("dropout", float),
            ("optimizer", Any),
            ("validator", Any),
        ],
    )

    T2VParam.mode = mode
    T2VParam.normalizer = normalizer
    T2VParam.training_output_size = training_output_size
    T2VParam.batch_size = batch_size
    T2VParam.vector_length = vector_length
    T2VParam.epochs = epochs
    T2VParam.learning_rate = learning_rate
    T2VParam.loss_function = loss_function
    T2VParam.hidden = hidden
    T2VParam.dropout = dropout
    T2VParam.optimizer = optimizer
    T2VParam.validator = validator

    return T2VParam


# NN Cells
# TSV layer
class TSV(nn.Module):
    def __init__(self, vector_length):
        super().__init__()
        self.output_length = vector_length

        W = torch.Tensor(vector_length - 1, 1)
        self.W = nn.Parameter(W)
        nn.init.xavier_uniform_(W)

        P = torch.Tensor(vector_length - 1, 1)
        self.P = nn.Parameter(P)
        nn.init.xavier_uniform_(P)

        w = torch.Tensor(1, 1)
        self.w = nn.Parameter(w)
        nn.init.xavier_uniform_(w)

        p = torch.Tensor(1, 1)
        self.p = nn.Parameter(p)
        nn.init.xavier_uniform_(p)

    def forward(self, x):
        original = self.w * x + self.p

        dotted = torch.matmul(x, self.W.t())
        added = torch.add(dotted, self.P.t())
        sin_trans = torch.sin(added)

        catted = torch.cat([sin_trans, original], dim=-1)
        return catted.view(x.shape[0], x.shape[1], self.output_length)


# LSTM
class LSTM(nn.Module):
    def __init__(self, vector_length, hidden_layer, output_size, window, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.tsv = TSV(vector_length)
        self.reduced_embedding = nn.Linear(window * vector_length, vector_length)

        hidden_layer = [vector_length] + hidden_layer
        self.hidden_layers = 0
        self.lstms = nn.ModuleList()
        for i in range(len(hidden_layer) - 1):
            self.lstms.append(
                nn.LSTM(hidden_layer[i], hidden_layer[i + 1], batch_first=True)
            )
            self.hidden_layers += 1
        self.linear = nn.Linear(hidden_layer[i + 1], output_size)

    def forward(self, input_seq):
        c_out = self.tsv(input_seq)
        embedding = self.reduced_embedding(c_out.view([c_out.shape[0], -1]))
        for lstm in self.lstms:
            c_out, (ht, ct) = lstm(c_out)

        predictions = self.linear(F.dropout(ht[-1], self.dropout))
        return predictions, embedding
