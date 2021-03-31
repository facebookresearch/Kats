#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import ast
import logging
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, Tuple

import infrastrategy.kats.models.model as m
import infrastrategy.kats.parameter_tuning.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import Params, SearchMethodEnum, TimeSeriesData
from infrastrategy.kats.models import arima, holtwinters, prophet, sarima, stlf, theta
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures


candidate_models = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "prophet": prophet.ProphetModel,
    "theta": theta.ThetaModel,
    "stlf": stlf.STLFModel,
    "sarima": sarima.SARIMAModel,
}

candidate_params = {
    "arima": arima.ARIMAParams,
    "holtwinters": holtwinters.HoltWintersParams,
    "prophet": prophet.ProphetParams,
    "theta": theta.ThetaParams,
    "stlf": stlf.STLFParams,
    "sarima": sarima.SARIMAParams,
}

# Constant to indicate error types supported
ALLOWED_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


class GetMetaData:
    """
    Getting meta data is the first step of meta-learning framework on time series analysis.

    It contains three components:

    - Time series feature vector, such as entropy, seasonal features, ACF/PACF based features.

    - Best hyper-parameters for each candidate models and their corresponding errors for a given time series data.

    - Best model for a given time series data.

    Meta-learning on model selection algorithm is using features as input, and best model as output, while meta-learning on hyper-parameters tuning algorithm is using best hyper-parameters as output.

    :Parameters:
    data: TimeSeriesData
        Time series data.
    all_models: Dict[str, m.Model]
        A dictionary that includes all candidate models.
    all_params: Dict[str, Params]
        A dictionary that includes all candidate hyper-params corresponding to all candidate models.
    min_length: int
        Minimal length of time series. Time series data which is too short will be excluded.
    scale: bool
        It indicates whether to scale TS in order to get a more comparable feature vector.
        If it's true, each value of TS will be divided by the max value of TS.
    method: SearchMethodEnum
        Search method for hyper-parameters tuning.
    executor: Any
        A callable parallel executor. Tune individual model in candidate models parallel.
        The default executor is native implementation with Python's multiprocessing.
    error_method: str
        Type of error metric. Only support mape, smape, mae, mase, mse, rmse.
    num_trials: int
        Number of trials in RandomSearch with default number=5.
    num_arms: int
        Number of arms in RandomSearch with default number=4.

    :Example:
    >>> import pandas as pd
    >>> from infrastrategy.kats.consts import TimeSeriesData
    >>> from infrastrategy.kats.models.metalearner.get_metadata import GetMetaData
    >>> # read data and rename the two columns required by TimeSeriesData structure
    >>> data = pd.read_csv("../fbsource/fbcode/infrastrategy/kats/data/air_passengers.csv")
    >>> data.columns = ["time", "y"]
    >>> TSdata = TimeSeriesData(data)
    >>> # create an object MD of class GetMetaData
    >>> MD = GetMetaData(data=TSdata)
    >>> # get best hyper-params for each candidate model and their corresponding errors
    >>> hpt_res = MD.tune_executor()
    >>> # get meta data, as well as search method and type of error metric
    >>> my_meta_data = MD.get_meta_data()

    :Methods:
    """

    def __init__(
        self,
        data: TimeSeriesData,
        all_models: Dict[str, m.Model] = candidate_models,
        all_params: Dict[str, Params] = candidate_params,
        min_length: int = 30,
        scale: bool = True,
        method: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        executor: Any = None,
        error_method: str = "mae",
        num_trials: int = 5,
        num_arms: int = 4,
        **kwargs,
    ) -> None:
        if not isinstance(data, TimeSeriesData):
            msg = "Input data should be TimeSeriesData"
            raise ValueError(msg)
        self.data = TimeSeriesData(data.to_dataframe().copy())
        self.all_models = all_models
        self.all_params = all_params
        self.min_length = min_length
        self.method = method
        self.error_funcs = {
            "mape": self._calc_mape,
            "smape": self._calc_smape,
            "mae": self._calc_mae,
            "mase": self._calc_mase,
            "mse": self._calc_mse,
            "rmse": self._calc_rmse,
        }
        self.error_method = error_method
        self.num_trials = num_trials
        self.num_arms = num_arms

        # check if the data type is TimeSeriesData
        if not isinstance(self.data, TimeSeriesData):
            msg = "Only support TimeSeriesData, but get {type}.".format(
                type=type(self.data)
            )
            logging.error(msg)
            raise ValueError(msg)

        # check if the time series is univariate
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        # check if use customized tune Executor.
        try:
            self.tune_executor = executor["tune_executor"]
            msg = "Using customized tune_executor from given parameters!"
        except Exception:
            msg = "We are using the default tune_executor for hyper-param tuning!"
        logging.info(msg)

        self._validate_data()
        self._validate_others()
        self._validate_models()

        # scale data if scale is True
        if scale:
            self._scale()

        # split to training set and test set
        split_idx = int(len(self.data) * 0.8)
        self.train_series = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[:split_idx],
                    "value": self.data.value[:split_idx],
                }
            )
        )
        self.test_series = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[split_idx:],
                    "value": self.data.value[split_idx:],
                }
            )
        )

    def _validate_data(self) -> None:
        # check if the time series length greater or equal to minimal length
        if len(self.data) < self.min_length:
            raise ValueError("Time series is too short!")

        # check if the time frequency is constant
        if pd.infer_freq(self.data.time) is None:
            raise ValueError("Only constant frequency is supported for time!")

        # check if the time series is constant
        if self.data.value.nunique() <= 1:
            raise ValueError("It's constant time series!")

        # check if the time series contains NAN, inf or -inf
        if self.data.value.replace([np.inf, -np.inf], np.nan).isna().any():
            raise ValueError("Time series contains NAN or infinity value(s)!")

        msg = "Valid time series data!"
        logging.info(msg)

    # check if candidate models and candidate hyper-params are matched
    def _validate_models(self) -> None:
        if list(self.all_models.keys()) != list(self.all_params.keys()):
            raise ValueError("Unmatched model dict and parameters dict!")

    # check if given serach method and error method are valid
    def _validate_others(self) -> None:
        if self.error_method not in ALLOWED_ERRORS:
            logging.error("Invalid error type passed")
            logging.error("error name: {0}".format(self.error_method))
            raise ValueError("Unsupported error type")

    def _scale(self) -> None:
        # scale time series to make ts features more stable
        self.data.value /= self.data.value.max()
        msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
        logging.info(msg)

    def _tune_single(
        self, single_model: Callable, single_params: Callable
    ) -> Tuple[Dict, float]:

        # define an evaluation_function
        def _evaluation_function(params):
            if "seasonal_order" in params and isinstance(params["seasonal_order"], str):
                params["seasonal_order"] = ast.literal_eval(params["seasonal_order"])

            # use ** operator to unpack the dictionary params
            local_params = single_params(**params)
            local_model = single_model(self.train_series, local_params)
            try:
                local_model.fit()
                model_pred = local_model.predict(steps=self.test_series.time.shape[0])

                training_inputs = np.asarray(self.train_series.value)
                predictions = np.asarray(model_pred.fcst)
                truth = np.asarray(self.test_series.value)
                diffs = abs(truth - predictions)
                # error metric:
                error = self.error_funcs[self.error_method](
                    training_inputs, predictions, truth, diffs
                )
            except Exception as e:
                logging.info(f"Exception in tuning hyper-parameters: {e}.")
                return np.inf
            return error

        # Create search method object
        parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=single_model.get_parameter_search_space(),
            selected_search_method=self.method,
        )
        if self.method == SearchMethodEnum.GRID_SEARCH:
            parameter_tuner.generate_evaluate_new_parameter_values(
                evaluation_function=_evaluation_function, arm_count=-1
            )
        else:
            # Random Search / Bayesian Optimal Search / Others
            for _ in range(self.num_trials):
                parameter_tuner.generate_evaluate_new_parameter_values(
                    evaluation_function=_evaluation_function, arm_count=self.num_arms
                )
        scores = parameter_tuner.list_parameter_value_scores()
        # exclude error = nan, -inf, inf
        scores = scores[~scores["mean"].isin([np.nan, np.inf, -np.inf])]
        if len(scores) == 0:
            return {}, np.inf
        # only need error 'mean' and 'parameters' columns
        scores = scores.nsmallest(1, "mean").iloc[0]

        # return best HP for given model, and corresponding error metric
        return scores["parameters"], scores["mean"]

    def tune_executor(self) -> Dict[str, Any]:
        """Get best hyper-params for each candidate model and their corresponding errors for a given time series data."""
        # Fit individual model with given data
        num_process = min(len(self.all_models), (cpu_count() - 1) // 2)
        pool = ThreadPool(processes=num_process)
        tuned_models = {}
        for single_model in self.all_models:
            # single_model is a model name: str
            tuned_models[single_model] = pool.apply_async(
                self._tune_single,
                args=(self.all_models[single_model], self.all_params[single_model]),
            )
        pool.close()
        pool.join()
        tuned_res = {model: res.get() for model, res in tuned_models.items()}

        # return {model_name: (best_hpt, error_metric), ...}
        return tuned_res

    def get_meta_data(self) -> Dict[str, Any]:
        """Get meta data, as well as search method and type of error metric

        Meta data includes time series features, best hyper-params for each candidate models, and best model.
        """
        features_dict = TsFeatures().transform(self.data)
        feature_vec = list(features_dict.values())

        # feature contains nan, pass
        if np.isnan(feature_vec).any():
            msg = "Feature vector contains NAN."
            logging.error(msg)
            raise ValueError(msg)

        HPT_res = self.tune_executor()

        # sorted by HPT_res[z][-1], i.e., error metric value.
        label = sorted(HPT_res.keys(), key=lambda z: HPT_res[z][-1])[0]

        if self.method == SearchMethodEnum.GRID_SEARCH:
            local_method = "GridSearch"
        elif (
            self.method == SearchMethodEnum.RANDOM_SEARCH_UNIFORM
            or self.method == SearchMethodEnum.RANDOM_SEARCH_SOBOL
        ):
            local_method = "RandomSearch"
        elif self.method == SearchMethodEnum.BAYES_OPT:
            local_method = "BayesOptimalSearch"
        else:
            local_method = "Others"

        return {
            "hpt_res": HPT_res,
            "features": features_dict,
            "best_model": label,
            "search_method": local_method,
            "error_method": self.error_method,
        }

    def _calc_mape(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MAPE")
        return np.mean(np.abs((truth - predictions) / truth))

    def _calc_smape(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating SMAPE")
        return ((abs(truth - predictions) / (truth + predictions)).sum()) * (
            2.0 / truth.size
        )

    def _calc_mae(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MAE")
        return diffs.mean()

    # MASE
    # mean(|actual - forecast| / naiveError), where
    # naiveError = 1/ (n-1) sigma^n_[i=2](|actual_[i] - actual_[i-1]|)
    def _calc_mase(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MASE")
        naive_error = np.abs(np.diff(training_inputs)).sum() / (
            training_inputs.shape[0] - 1
        )
        return diffs.mean() / naive_error

    def _calc_mse(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MSE")
        return ((diffs) ** 2).mean()

    def _calc_rmse(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating RMSE")
        return np.sqrt(self._calc_mse(training_inputs, predictions, truth, diffs))

    def __str__(self):
        return "GetMetaData"
