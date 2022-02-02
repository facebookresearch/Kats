# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import kats.utils.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from kats.consts import SearchMethodEnum
from kats.detectors.changepoint_evaluator import TuringEvaluator
from kats.detectors.detector import DetectorModel
from kats.utils.time_series_parameter_tuning import TimeSeriesParameterTuning

SUPPORTED_METRICS = {"f_score", "precision", "recall", "delay"}


def check_metric_is_supported(metric: str) -> None:
    """Check if the metric specified by the user is supported by our evaluator."""
    if metric not in SUPPORTED_METRICS:
        raise Exception(
            f"Supported metrics for evaluating detector are: {SUPPORTED_METRICS}"
        )


def decompose_params(params: Dict[str, float]) -> Tuple[Dict[str, float], float, float]:
    """Decompose params into model_params, threshold_low, and threshold_high"""
    params_model = {k: v for k, v in params.items() if "threshold" not in k}
    # To do: Maybe put this threshold_low and threshold_high_value inside a config file to have the same everywhere in the code?
    threshold_low = params.get("threshold_low", 0.0)
    threshold_high = params.get("threshold_high", 1.0)
    return params_model, threshold_low, threshold_high


@dataclass
class DetectorModelSet:
    def __init__(
        self,
        model_name: str,
        model: Type[DetectorModel],
        margin: int = 5,
        alert_style_cp: bool = True,
    ) -> None:
        self.model_name: str = model_name
        self.model: Type[DetectorModel] = model
        self.margin: int = margin
        self.alert_style_cp: bool = alert_style_cp
        self.result: Optional[pd.DataFrame] = None
        self.parameters: Dict[str, float] = {}

    def update_benchmark_results(
        self, benchmark_results: Dict[str, Optional[pd.DataFrame]]
    ) -> None:
        benchmark_results[self.model_name] = self.result

    def evaluate(
        self, data_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], TuringEvaluator]:
        self.result, turing_evaluator = self.evaluate_parameters_detector_model(
            self.get_params(), self.model, data_df
        )

        return {self.model_name: self.result}, turing_evaluator

    def evaluate_parameters_detector_model(
        self,
        params: Dict[str, float],
        detector: Type[DetectorModel],
        data: pd.DataFrame,
        ignore_list: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, TuringEvaluator]:
        turing_model = TuringEvaluator(is_detector_model=True, detector=detector)
        params_model, threshold_low, threshold_high = decompose_params(params)
        results = turing_model.evaluate(
            data=data,
            model_params=params_model,
            ignore_list=ignore_list,
            alert_style_cp=self.alert_style_cp,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            margin=self.margin,
        )
        return results, turing_model

    def get_params(self) -> Dict[str, float]:
        return self.parameters


class PredefinedModel(DetectorModelSet):
    def __init__(
        self, model_name: str, model: Type[DetectorModel], parameters: Dict[str, float]
    ) -> None:
        super().__init__(model_name, model)
        self.parameters: Dict[str, float] = parameters


class ModelOptimizer(DetectorModelSet):
    def __init__(
        self,
        model_name: str,
        model: Type[DetectorModel],
        parameters_space: List[Dict[str, float]],
        data_df: pd.DataFrame,
        optimization_metric: str,
        optimize_for_min: bool,
        search_method: SearchMethodEnum = SearchMethodEnum.GRID_SEARCH,
        arm_count: int = 4,
    ) -> None:
        super().__init__(model_name, model)
        check_metric_is_supported(optimization_metric)
        self.parameters_space = parameters_space
        self.data_df = data_df
        self.optimization_metric = optimization_metric
        self.optimize_for_min = optimize_for_min
        if search_method == SearchMethodEnum.GRID_SEARCH:
            self.arm_count: int = -1
        else:
            search_cardinality = tpt.compute_search_cardinality(self.parameters_space)
            # pyre-fixme[8]: Attribute has type `int`; used as `float`.
            self.arm_count = min(arm_count, search_cardinality)
        self.parameter_tuner_grid: TimeSeriesParameterTuning = (
            tpt.SearchMethodFactory.create_search_method(
                objective_name=optimization_metric,
                parameters=parameters_space,
                selected_search_method=search_method,
            )
        )
        self.parameter_tuning_results_grid: Optional[pd.DataFrame] = None
        self.best_params: Dict[str, float] = {}

    def get_params(self) -> Dict[str, float]:
        if not self.best_params:
            self._compute_best_param()
        return self.best_params

    def _evaluation_method(self, params: Dict[str, float]) -> float:
        try:
            results, _ = self.evaluate_parameters_detector_model(
                params, self.model, self.data_df
            )
            return np.mean(results[self.optimization_metric])
        except Exception:
            logging.warning(f"{params} cannot be evaluated: returning inf.")
            if self.optimize_for_min:
                return float("inf")
            else:
                return float("-inf")

    def _evaluate(self) -> None:
        self.parameter_tuner_grid.generate_evaluate_new_parameter_values(
            evaluation_function=self._evaluation_method,
            arm_count=self.arm_count,
        )
        # Retrieve parameter tuning results
        parameter_tuning_results_grid = (
            self.parameter_tuner_grid.list_parameter_value_scores()
        )
        self.parameter_tuning_results_grid = parameter_tuning_results_grid
        if (max(parameter_tuning_results_grid["mean"]) == float("-inf")) or (
            min(parameter_tuning_results_grid["mean"]) == float("inf")
        ):
            raise Exception("All parameters raised an error")

    def _compute_best_param(self) -> Dict[str, float]:
        if self.parameter_tuning_results_grid is None:
            self._evaluate()

        assert (
            self.parameter_tuning_results_grid is not None
        ), "ModelOptimizer parameter_tuning_results_grid is missing."

        parameter_tuning_results_grid = self.parameter_tuning_results_grid
        if self.optimize_for_min:
            idx_best = np.argmin(parameter_tuning_results_grid["mean"])
        else:
            idx_best = np.argmax(parameter_tuning_results_grid["mean"])
        self.best_params = parameter_tuning_results_grid.parameters.iloc[idx_best]
        return self.best_params


class ModelBenchmark:
    def __init__(
        self,
        data_df: pd.DataFrame,
        models: List[DetectorModelSet],
        is_detector_model: bool = True,
    ) -> None:
        self.data_df: pd.DataFrame = data_df
        self.models: List[DetectorModelSet] = models
        self.is_detector_model: bool = is_detector_model
        self.results: Dict[str, pd.DataFrame] = {}
        self.best_model: Dict[str, pd.DataFrame] = {}
        self.results_tmp: Dict[str, pd.DataFrame] = {}
        self.evaluators_tmp: Dict[str, TuringEvaluator] = {}
        self.model_metric: List[pd.DataFrame] = []

    def evaluate(self) -> None:
        """Evaluate each model of models_dict on data and store output in self.results_tmp."""
        for model in self.models:
            result, turing_evaluator = model.evaluate(self.data_df)
            self.results_tmp.update(result)
            self.evaluators_tmp[model.model_name] = turing_evaluator

    def compare_algos_on(self, metric: str) -> pd.DataFrame:
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
