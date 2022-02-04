# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for testing and evaluation.
"""

from typing import Callable, Dict, Union

import numpy as np

try:
    from plotly.graph_objs import Figure
except ImportError:
    Figure = object


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return np.sqrt(np.mean((pred - truth) ** 2))


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return ((np.abs(truth - pred)) ** 2).mean()


def mape(pred: np.ndarray, truth: np.ndarray) -> float:
    return np.mean(np.abs((truth - pred) / truth))


def smape(pred: np.ndarray, truth: np.ndarray) -> float:
    return ((abs(truth - pred) / (truth + pred)).sum()) * (2.0 / truth.size)


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return (np.abs(truth - pred)).mean()


def mase(training_inputs: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> float:
    naive_error = np.abs(np.diff(training_inputs)).sum() / (
        training_inputs.shape[0] - 1
    )
    return ((np.abs(truth - pred)).mean()) / naive_error


ErrorFunc = Union[
    Callable[[np.ndarray, np.ndarray], float],
    Callable[[np.ndarray, np.ndarray, np.ndarray], float],
]

error_funcs: Dict[
    str, ErrorFunc
] = {  # Maps error name to function that calculates error
    "mape": mape,
    "smape": smape,
    "mae": mae,
    "mase": mase,
    "mse": mse,
    "rmse": rmse,
}


class PlotlyAdapter:
    def __init__(self, fig: Figure) -> None:
        self.fig = fig

    def save_fig(self, path: str) -> None:
        # pyre-ignore[16]: `plotly.graph_objs.graph_objs.Figure` has no attribute `write_image`.
        self.fig.write_image(path)


__all__ = [
    "PlotlyAdapter",
    "error_funcs",
]
