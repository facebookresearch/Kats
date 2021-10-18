# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Type

import kats.utils.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from kats.consts import SearchMethodEnum
from kats.detectors import changepoint_evaluator
from kats.detectors.detector import DetectorModel
from dataclasses import dataclass


def check_metric_is_supported(metric: str):
    """Check if the metric specified by the user is supported by our evaluator."""
    if metric not in ["f_score", "precision", "recall", "delay"]:
        raise Exception(
            "Supported metrics for evaluating detector are: f_score, precision, recall, or delay"
        )


def decompose_params(params: Dict):
    """Decompose params into model_params, threshold_low, and threshold_high"""
    params_model = {k: v for k, v in params.items() if "threshold" not in k}
    # To do: Maybe put this threshold_low and threshold_high_value inside a config file to have the same everywhere in the code?
    threshold_low = params.get("threshold_low", 0.0)
    threshold_high = params.get("threshold_high", 1.0)
    return params_model, threshold_low, threshold_high


def evaluate_parameters_detector_model(params, detector, data):
    turing_model = changepoint_evaluator.TuringEvaluator(
        is_detector_model=True, detector=detector
    )
    params_model, threshold_low, threshold_high = decompose_params(params)
    results = turing_model.evaluate(
        data=data,
        model_params=params_model,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )
    return results, turing_model

@dataclass
class DetectorModelSet:
    def __init__(
        self,
        model_name: str,
        model: Type[DetectorModel],
    ):
        self.model_name = model_name
        self.model = model


class PredefinedModel(DetectorModelSet):
    def __init__(self, model_name: str, model: Type[DetectorModel], parameters: Dict):
        super().__init__(model_name, model)
        self.parameters = parameters

    def get_params(self):
        return self.parameters


class ModelOptimizer(DetectorModelSet):
    def __init__(
        self,
        model_name: str,
        model: Type[DetectorModel],
        parameters_space: List[Dict],
        data_df: pd.DataFrame,
        optimization_metric: str,
        optimize_for_min: bool,
        search_method=SearchMethodEnum.GRID_SEARCH,
        arm_count: int = 4,
    ):
        super().__init__(model_name, model)
        check_metric_is_supported(optimization_metric)
        self.parameters_space = parameters_space
        self.data_df = data_df
        self.optimization_metric = optimization_metric
        self.optimize_for_min = optimize_for_min
        if search_method == SearchMethodEnum.GRID_SEARCH:
            self.arm_count = -1
        else:
            self.arm_count = arm_count
        self.parameter_tuner_grid = tpt.SearchMethodFactory.create_search_method(
            objective_name=optimization_metric,
            parameters=parameters_space,
            selected_search_method=search_method,
        )
        self.parameter_tuning_results_grid = None
        self.best_params = {}

    def get_params(self):
        if not self.best_params:
            self._compute_best_param()
        return self.best_params

    def _evaluation_method(self, params: Dict):
        try:
            results, _ = evaluate_parameters_detector_model(
                params, self.model, self.data_df
            )
            return np.mean(results[self.optimization_metric])
        except Exception:
            logging.warning(f"{params} cannot be evaluated: returning inf.")
            if self.optimize_for_min:
                return float("inf")
            else:
                return float("-inf")

    def _evaluate(self):
        self.parameter_tuner_grid.generate_evaluate_new_parameter_values(
            evaluation_function=self._evaluation_method,
            arm_count=self.arm_count,
        )
        # Retrieve parameter tuning results
        self.parameter_tuning_results_grid = (
            self.parameter_tuner_grid.list_parameter_value_scores()
        )
        if (max(self.parameter_tuning_results_grid["mean"]) == float("-inf")) or (
            min(self.parameter_tuning_results_grid["mean"]) == float("inf")
        ):
            raise Exception("All parameters raised an error")

    def _compute_best_param(self):
        if self.parameter_tuning_results_grid is None:
            self._evaluate()
        if self.optimize_for_min:
            idx_best = np.argmin(self.parameter_tuning_results_grid["mean"])
        else:
            idx_best = np.argmax(self.parameter_tuning_results_grid["mean"])
        self.best_params = self.parameter_tuning_results_grid.parameters.iloc[idx_best]
        return self.best_params


class ModelBenchmark:
    def __init__(
        self,
        data_df: pd.DataFrame,
        models: List[DetectorModelSet],
        is_detector_model: bool = True,
    ):
        self.data_df = data_df
        self.models = models
        self.is_detector_model = is_detector_model
        self.results = {}
        self.best_model = {}
        self.results_tmp = {}
        self.evaluators_tmp = {}
        self.model_metric = []

    def evaluate(self):
        """Evaluate each model of models_dict on data and store output in self.results_tmp."""
        for model in self.models:
            result, turing_evaluator = evaluate_parameters_detector_model(
                model.get_params(), model.model, self.data_df
            )
            self.results_tmp[model.model_name] = result
            self.evaluators_tmp[model.model_name] = turing_evaluator

    def compare_algos_on(self, metric: str):
        """Re-arrange the results to compare all models for each metrics."""
        check_metric_is_supported(metric)
        if metric not in self.results.keys():
            self.model_metric = []
            for model_name, model_result in self.results_tmp.items():
                model_df = model_result[["dataset_name", metric]]
                self.model_metric.append(
                    model_df.rename(columns={metric: model_name}).set_index(
                        "dataset_name"
                    )
                )
            self.results[metric] = pd.concat(self.model_metric, axis=1).reset_index()
            # Compute the best algorithm according to the average of the dataset on the given metric.
            self.best_model[metric] = self.results[metric].mean(axis=0).idxmax(axis=1)
        return self.results[metric]
