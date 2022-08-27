# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
from kats.consts import _log_error, Params, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.models.model import Model

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


@dataclass
class EvaluationObject:
    input_data: Optional[Union[ArrayLike, TimeSeriesData, pd.DataFrame]]
    # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
    model: Optional[Union[Model, DetectorModel]]
    preds: Optional[ArrayLike]
    labels: Optional[ArrayLike]
    results: Optional[pd.DataFrame]


class Evaluator(ABC):
    def __init__(self) -> None:
        self.runs: Dict[str, EvaluationObject] = {}

    def create_evaluation_run(self, run_name: str) -> None:
        self._check_if_valid_run_name(run_name=run_name, should_exist=False)
        self.runs[run_name] = EvaluationObject(None, None, None, None, None)

    def delete_evaluation_run(self, run_name: str) -> None:
        self._check_if_valid_run_name(run_name=run_name)
        del self.runs[run_name]

    def _check_if_valid_run_name(
        self, run_name: str, should_exist: bool = True
    ) -> None:
        if not isinstance(run_name, str):  # Check if name is a string
            msg = f"Name of evaluation run must be a string, but is of type {type(run_name)}."
            raise _log_error(msg)
        # Handle case depending on if run_name should exist or not
        if should_exist != (run_name in self.runs):
            msg = f"Run name {run_name} {'already exists' if run_name in self.runs else 'does not exist'}. Please choose a valid run name."
            raise _log_error(msg)

    def get_evaluation_run(self, run_name: str) -> EvaluationObject:
        self._check_if_valid_run_name(run_name=run_name)  # Check if valid run
        return self.runs[run_name]

    @abstractmethod
    def generate_predictions(
        self,
        run_name: str,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model: Type,
        model_params: Optional[Union[Params, Dict[str, float]]],
        tune_params: bool = False,
    ) -> ArrayLike:
        pass

    def evaluate(
        self,
        run_name: str,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        metric_to_func: Dict[str, Callable],
        labels: ArrayLike,
    ) -> pd.DataFrame:
        # Check preconditions
        self._check_if_valid_run_name(run_name=run_name)  # Check if valid run
        evaluator = self.runs[run_name]
        if np.ndim(evaluator.preds) < 1:  # Check if predictions are a valid type
            msg = f"Invalid type: {type(labels)} for predictions. Must be an iterable object."
            raise _log_error(msg)
        if np.ndim(labels) < 1:  # Check if labels are a valid type
            msg = (
                f"Invalid type: {type(labels)} for labels. Must be an iterable object."
            )
            raise _log_error(msg)
        evaluator.labels = labels
        # pyre-fixme[6]: Incompatible parameter type...
        if len(evaluator.preds) != len(labels):  # Check lengths of preds and labels
            msg = "Predictions and labels have unequal lengths."
            raise _log_error(msg)

        # Run evaluation
        metric_to_result: Dict[str, ArrayLike] = {}
        for metric, func in metric_to_func.items():
            try:
                metric_vals = func(evaluator.preds, evaluator.labels)
            except ValueError as e:
                msg = (
                    f"Error running evaluation for metric {metric}. Full message:\n {e}"
                )
                raise _log_error(msg)
            metric_to_result[metric] = [metric_vals]

        # Save and return evaluation results
        aggregated_results = pd.DataFrame(metric_to_result, copy=False)
        evaluator.results = aggregated_results
        return evaluator.results
