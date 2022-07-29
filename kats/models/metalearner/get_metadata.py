# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A module for computing the meta-data of time series.

This module contains the class for computing the meta-data of time series. The meta-data of a time series is consists of three parts: 1) time series features;
2) the best hyper-parameters of each candidate models and their corresponding errors; and 3) the best model.
"""

import ast
import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, List, Tuple, Union

import kats.utils.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from kats.consts import SearchMethodEnum, TimeSeriesData
from kats.metrics.metrics import core_metric
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.models.stlf import STLFModel, STLFParams
from kats.models.theta import ThetaModel, ThetaParams
from kats.tsfeatures.tsfeatures import TsFeatures


candidate_models = {
    "arima": ARIMAModel,
    "holtwinters": HoltWintersModel,
    "prophet": ProphetModel,
    "theta": ThetaModel,
    "stlf": STLFModel,
    "sarima": SARIMAModel,
}

candidate_params = {
    "arima": ARIMAParams,
    "holtwinters": HoltWintersParams,
    "prophet": ProphetParams,
    "theta": ThetaParams,
    "stlf": STLFParams,
    "sarima": SARIMAParams,
}

# Constant to indicate error types supported
ALLOWED_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


@dataclass
class GetMetaDataVal:
    hpt_res: Dict[str, Any]
    features: Union[Dict[str, float], List[Dict[str, float]]]
    best_model: str
    search_method: str
    error_method: str


class GetMetaData:
    """A class for generate meta-data of time series.

    The meta-data of time series contains three components:
        1. Time series feature vector, such as entropy, seasonal features, ACF/PACF based features.
        2. Best hyper-parameters for each candidate models and their corresponding errors for a time series data.
        3. Best model for a time series data.
    This class provides tune_executor and get_meta_dat.

    Attributes:
        data: :class:`kats.consts.TimeSeriesData` object representing the input time series data.
        all_models: Optional; A dictionary of candidate model classes. Default dictionary includes models of ARIMA, SARIMA, HoltWinters, Prophet, Theta, and STLF.
        all_params: Optional; A dictionary of the corresponding candidate model parameter classes. Default includes model parameter classes of ARIMA, SARIMA, HoltWinters, Prophet, Theta, and STLF.
        min_length: Optional; An integer for the minimal length of a time series. Time series data whose length is shorter than min_length will be excluded. Default is 30.
        scale: Optional; A boolean to specify whether or not to rescale the time series data by its maximum values. Default is True. Default is True.
        method: Optional; A SearchMethodEnum object defining the search method for hyper-parameters tuning. Default is random search in the default parameter space.
        executor: Optional; A callable parallel executor for tuning individual candidate models in parallel. Default is the native implementation with Python's multiprocessing.
        error_method: Optional; A string for error metric used for model evaluation. Can be 'mape', 'smape', 'mae', 'mase', 'mse', or 'rmse'. Default is 'mae'.
        num_trials: Optional; An integer for the number of trials in hyper-parameter search. Default is 5.
        num_arms: Optional; An integer for the number of arms in hyper-parameter search. Default is 4.

    Sample Usage:
        >>> TSdata = TimeSeriesData(data)
        >>> MD = GetMetaData(data=TSdata)
        >>> hpt_res = MD.tune_executor()
        >>> my_meta_data = MD.get_meta_data() # Get meta-data, hyper-parameter searching method and error metric.
    """

    def __init__(
        self,
        data: TimeSeriesData,
        all_models: Dict[str, Any] = candidate_models,
        all_params: Dict[str, Any] = candidate_params,
        min_length: int = 30,
        scale: bool = True,
        method: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        executor: Any = None,
        error_method: str = "mae",
        num_trials: int = 5,
        num_arms: int = 4,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        if not isinstance(data, TimeSeriesData):
            msg = "Input data should be TimeSeriesData"
            raise ValueError(msg)
        self.data = TimeSeriesData(pd.DataFrame(data.to_dataframe().copy()))
        self.all_models = all_models
        self.all_params = all_params
        self.min_length = min_length
        self.method = method
        self.error_method = error_method
        self.num_trials = num_trials
        self.num_arms = num_arms

        # Check if the data type is TimeSeriesData
        if not isinstance(self.data, TimeSeriesData):
            msg = "Only support TimeSeriesData, but get {type}.".format(
                type=type(self.data)
            )
            logging.error(msg)
            raise ValueError(msg)

        # Check if the time series is univariate
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        # Check if use customized tune Executor.
        try:
            self.tune_executor = executor["tune_executor"]
            msg = "Using customized tune_executor from given parameters!"
        except Exception:
            msg = "We are using the default tune_executor for hyper-param tuning!"
        logging.info(msg)

        self._validate_data()
        self._validate_others()
        self._validate_models()

        # Scale data if scale is True
        if scale:
            self._scale()

        # Split to training set and test set
        split_idx = int(len(self.data) * 0.8)
        self.train_series = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[:split_idx],
                    "value": self.data.value[:split_idx],
                },
                copy=False,
            )
        )
        self.test_series = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[split_idx:],
                    "value": self.data.value[split_idx:],
                },
                copy=False,
            )
        )

    def _validate_data(self) -> None:
        """Validation function for checking input time seris."""

        # check if the time series length greater or equal to minimal length
        if len(self.data) < self.min_length:
            raise ValueError("Time series is too short!")

        # check if the time series is constant
        if self.data.value.nunique() <= 1:
            raise ValueError("It's constant time series!")

        # check if the time series contains NAN, inf or -inf
        if np.any(np.isinf(self.data.value.values)) or np.any(
            np.isnan(self.data.value.values)
        ):
            raise ValueError("Time series contains NAN or infinity value(s)!")

        msg = "Valid time series data!"
        logging.info(msg)

    def _validate_models(self) -> None:
        """Check if candidate models and candidate hyper-params are matched"""

        if set(self.all_models.keys()) != set(self.all_params.keys()):
            raise ValueError(
                f"Unmatched model dict ({set(self.all_models.keys())}) and parameters dict ({set(self.all_params.keys())})!"
            )

    def _validate_others(self) -> None:
        """Check if given serach method and error method are valid"""

        if self.error_method not in ALLOWED_ERRORS:
            logging.error("Invalid error type passed")
            logging.error("error name: {0}".format(self.error_method))
            raise ValueError("Unsupported error type")

    def _scale(self) -> None:
        """Rescale time series."""

        self.data.value /= self.data.value.max()
        msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
        logging.info(msg)

    def _tune_single(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        single_model: Callable,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        single_params: Callable
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
    ) -> Tuple[Dict, float]:
        """Fit and evaluate a single candidate model."""

        # define an evaluation_function
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _evaluation_function(params):
            if "seasonal_order" in params and isinstance(params["seasonal_order"], str):
                params["seasonal_order"] = ast.literal_eval(params["seasonal_order"])

            try:
                # use ** operator to unpack the dictionary params
                local_params = single_params(**params)
                local_model = single_model(self.train_series, local_params)
                local_model.fit()
                model_pred = local_model.predict(steps=self.test_series.time.shape[0])

                training_inputs = np.asarray(self.train_series.value)
                predictions = np.asarray(model_pred.fcst)
                truth = np.asarray(self.test_series.value)
                diffs = abs(truth - predictions)
                error = core_metric(self.error_method)(truth, predictions)
            except Exception as e:
                logging.info(f"Exception in tuning hyper-parameters: {e}.")
                return np.inf
            return error

        # Create search method object
        parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            # pyre-fixme[16]: Anonymous callable has no attribute
            #  `get_parameter_search_space`.
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
        # Exclude error = nan, -inf, inf
        scores = scores[~scores["mean"].isin([np.nan, np.inf, -np.inf])]
        if len(scores) == 0:
            return {}, np.inf
        scores = scores.nsmallest(1, "mean").iloc[0]
        return scores["parameters"], scores["mean"]

    def tune_executor(self) -> Dict[str, Any]:
        """Get the best hyper parameters for each candidate model and their corresponding errors for the time series data.

        Returns:
            A dictionary storing the best hyper-parameters and the errors for each candidate model.
        """
        num_process = min(len(self.all_models), (cpu_count() - 1) // 2)
        if num_process < 1:
            num_process = 1
        pool = ThreadPool(processes=num_process)
        tuned_models = {}
        for single_model in self.all_models:
            tuned_models[single_model] = pool.apply_async(
                self._tune_single,
                args=(self.all_models[single_model], self.all_params[single_model]),
            )
        pool.close()
        pool.join()
        tuned_res = {model: res.get() for model, res in tuned_models.items()}
        return tuned_res

    def get_meta_data(
        self,
        **kwargs: Any,
    ) -> GetMetaDataVal:
        """Get meta data, as well as search method and type of error metric

        Meta data includes time series features, best hyper-params for each candidate models, and best model.

        Returns:
            A dictionary storing the best hyper-parameters and the errors for each candidate model, the features of the time series data, the hyper-parameter searching method,
            the error metric used for model evaluation and the corresponding best model.
        """

        features_dict = TsFeatures(**kwargs).transform(self.data)

        # feature contains nan, pass
        # pyre-fixme[16]: `List` has no attribute `values`.
        if np.isnan(list(features_dict.values())).any():
            msg = f"Feature vector contains NAN, features are {features_dict}."
            logging.warning(msg)

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

        return GetMetaDataVal(
            hpt_res=HPT_res,
            features=features_dict,
            best_model=label,
            search_method=local_method,
            error_method=self.error_method,
        )

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return "GetMetaData"
