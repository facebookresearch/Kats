#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Kats ensemble model

Implementation of the Kats ensemble model. It starts from seasonality detection, if seasonality detected, it
continues to perform STL decomposition, then fit forecasting models on de-seasonalized components and aggregate;
otherwise it simiply leverage individual forecasting models and ensembling. We provided two ensembling methods,
weighted average and median ensembling.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from copy import copy
import logging
import sys
import math
import pandas as pd
import numpy as np
import multiprocessing
from typing import Dict, Tuple, Callable, Any
from multiprocessing import cpu_count

import kats.models.model as mm
from kats.consts import TimeSeriesData, Params
from kats.models.ensemble.ensemble import EnsembleParams
from kats.models.model import Model
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)
from kats.utils.backtesters import BackTesterSimple

# Seasonality detector
from kats.detectors.seasonality import ACFDetector

# STL decomposition
from kats.utils.decomposition import TimeSeriesDecomposition

# models that can fit de_seasonal component
MODELS = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "sarima": sarima.SARIMAModel,
    "prophet": prophet.ProphetModel,
    "linear": linear_model.LinearModel,
    "quadratic": quadratic_model.QuadraticModel,
    "theta": theta.ThetaModel,
}

# models that can fit seasonal time series data
SMODELS = {
    "prophet": prophet.ProphetModel,
    "theta": theta.ThetaModel,
    # "sarima": sarima.SARIMAModel,
}

class KatsEnsemble:
    """Decomposition based ensemble model in Kats
    This is the holistic ensembling class based on decomposition when seasonality presents
    We specifically provide following methods:
        seasonality_detector : detect seasonalities with ACF detector in Kats
        deseasonalize        : perform STL decomposition of seasonality presents
        fitExecutor          : callable function to fit forecast models in parallel
                            services who call KatsEnsemble need to write their own
                            executor to get better performance
        forecastExecutor     : callable function to fit and predict in parallel
                            services who call KatsEnsemble need to write their own
                            executor to get better performance
        fit                  : fit individual models by calling fitExecutor
        predict              : predict the future time series values by a given steps
        forecast             : combination of fit and predict methods
        plot                 : plot the historical and predicted values

    Attributes:
        data  : TimeSeriesData
        params: Dict with following keys
            models: EnsembleParams contains individual model params
                    i.e., [BaseModelParams, ...]
            aggregation: we support median ('median) and
                         weighted average ('weightedavg')
            fitExecutor: callable executor to fit individual models
            forecastExecutor: callable executor to fit and predict individual models
            seasonality_length : the length of seasonality -> TODO: auto determine
            decomposition_method : type of decomposition,
                            we support "additive", and "multiplicative"
    """

    def __init__(
        self,
        data: TimeSeriesData,
        params: Dict[str, Any],
    ) -> None:
        self.data = data
        self.freq = pd.infer_freq(data.time)
        self.params = params
        self.validate_params()

    def validate_params(self):
        # validate aggregation method
        if self.params["aggregation"] not in ("median", "weightedavg"):
            msg = "Only support `median` or `weightedavg` ensemble,\
            but get {method}.".format(
                method=self.params["aggregation"]
            )
            logging.error(msg)
            raise ValueError(msg)

        # validate decomposition method
        if self.params["decomposition_method"] in ("additive", "multiplicative"):
            self.decomposition_method = self.params["decomposition_method"]
        else:
            logging.info("Invalid decomposition method setting specified")
            logging.info("Defaulting to Additive Decomposition")
            self.decomposition_method = "additive"

        # validate m
        if (self.params["seasonality_length"] is not None) and\
    (self.params["seasonality_length"] > int(len(self.data.time) // 2)):
            msg = "seasonality_length value cannot be larger than"
            "1/2 of the length of give time series"
            logging.error(msg)
            raise ValueError(msg)

        # check customized forecastExecutor
        if ("forecastExecutor" in self.params.keys()) and\
    (self.params["forecastExecutor"] is not None):
            msg = "Using customized forecastExecutor from given parameters"
            logging.info(msg)
            self.forecastExecutor = self.params["forecastExecutor"]

        # check customized fitExecutor
        if ("fitExecutor" in self.params.keys()) and\
    (self.params["fitExecutor"] is not None):
            msg = "Using customized fitExecutor from given parameters"
            logging.info(msg)
            self.fitExecutor = self.params["fitExecutor"]

    @staticmethod
    def seasonality_detector(data) -> bool:
        """Detect seasonalities from given TimeSeriesData

        Args:
            data: the input `TimeSeriesData`

        Returns:
            Flag for the presence of seasonality
        """

        detector = ACFDetector(data)
        detector.detector()
        # pyre-fixme[16]: `ACFDetector` has no attribute `seasonality_detected`.
        seasonality = detector.seasonality_detected
        return seasonality

    @staticmethod
    def deseasonalize(data: TimeSeriesData, decomposition_method: str) -> Tuple:
        """STL decomposition to given TimeSeriesData

        Static method to perform decomposition on the input data

        Args:
            data: input time series data
            decomposition_method: the specific method for decomposition

        Returns:
            Tuple of seasonal data and de-seasonalized data
        """
        # create decomposer for time series decomposition
        decomposer = TimeSeriesDecomposition(data, decomposition_method)
        decomp = decomposer.decomposer()

        sea_data = copy(decomp["seasonal"])
        desea_data = copy(data)

        if decomposition_method == "additive":
            desea_data.value = desea_data.value\
                - decomp["seasonal"].value
        else:
            desea_data.value = desea_data.value\
                / decomp["seasonal"].value
        return sea_data, desea_data

    @staticmethod
    def reseasonalize(
        sea_data: TimeSeriesData,
        desea_predict: Dict[str, pd.DataFrame],
        decomposition_method: str,
        seasonality_length: int,
        steps: int,
    ) -> Dict[str, pd.DataFrame]:
        """Re-seasonalize the time series data

        Static method to re-seasonalize the input data

        Args:
            sea_data: the seasonal data from deseasonalize method
            desea_predict: dict of forecasted results for the deseasonalized
                data for each individual forecasting method
            decomposition_method: the specific method for decomposition
            seasonality_lenth: the length of seasonality
            steps: the length of forecasting horizon

        Returns:
            Dict of re-seasonalized data for each individual forecasting model
        """

        rep = math.trunc(1 + steps / seasonality_length)
        seasonality_unit = sea_data.value[-seasonality_length:]

        predicted = {}
        for model_name, desea_pred in desea_predict.items():
            if decomposition_method == "additive":
                if (
                    "fcst_lower" in desea_pred.columns
                    and "fcst_upper" in desea_pred.columns
                ):
                    # check consistency of time being index
                    if "time" in desea_pred.columns:
                        msg = "Setting time column as index"
                        logging.info(msg)
                        desea_pred.set_index("time", inplace=True)

                    # native C.I calculated from individual model
                    predicted[model_name] = desea_pred + \
                        np.tile(
                            np.tile(seasonality_unit, rep)[:steps], [3, 1]
                    ).transpose()
                else:
                    # no C.I from individual model
                    tmp_fcst = desea_pred.fcst\
                        + np.tile(seasonality_unit, rep)[:steps]
                    predicted[model_name] = pd.DataFrame({
                        "time": desea_pred.index,
                        "fcst": tmp_fcst,
                        "fcst_lower": np.nan,
                        "fcst_upper": np.nan,
                    }).set_index("time")

            else:
                # multiplicative, element-wise multiply
                if (
                    "fcst_lower" in desea_pred.columns
                    and "fcst_upper" in desea_pred.columns
                ):
                    # check consistency of time being index
                    if "time" in desea_pred.columns:
                        msg = "Setting time column as index"
                        logging.info(msg)
                        desea_pred.set_index("time", inplace=True)

                    # native C.I calculated from individual model
                    predicted[model_name] = desea_pred * \
                        np.tile(
                            np.tile(seasonality_unit, rep)[:steps], [3, 1]
                    ).transpose()
                else:
                    # no C.I from individual model
                    tmp_fcst = desea_pred.fcst * \
                        np.tile(seasonality_unit, rep)[:steps]
                    predicted[model_name] = pd.DataFrame({
                        "time": desea_pred.index,
                        "fcst": tmp_fcst,
                        "fcst_lower": 0,
                        "fcst_upper": 0,
                    }).set_index("time")

        return predicted

    def fitExecutor(
            self,
            data : TimeSeriesData,
            models : EnsembleParams,
            should_auto_backtest: bool = False,
    # pyre-fixme[31]: Expression `Dict[(str, float)])` is not a valid type.
    ) -> (Dict[str, Model], Dict[str, float]):
        """callable forecast executor

        This is native implementation with Python's multiprocessing
        fit individual model in `models` with given `data`. Services
        who use KatsEnsemble need to implement their own executor for better
        performance, if no executor function is given, the native version will be
        used.

        Attributes:
            data: given TimeSeriesData, could be original or de-seasonalized
            models: EnsembleParams object containing model params
                in BaseModelParams
            should_auto_backtest: boolean flag for additional back testing runs

        Returns:
            Tuple of fitted individual model and weights
        """

        # Fit individual model with given data
        num_process = min(len(MODELS), (cpu_count() - 1) // 2)
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(processes=(num_process), maxtasksperchild=1000)

        fitted_models = {}
        for model in models.models:
            fitted_models[model.model_name] = pool.apply_async(
                self._fit_single,
                args=(
                    data,
                    MODELS[model.model_name.split("_")[0].lower()],
                    model.model_params),
            )
        pool.close()
        pool.join()
        fitted = {model: res.get() for model, res in fitted_models.items()}

        # if auto back testing
        weights = self.backTestExecutor() if should_auto_backtest else None
        return fitted, weights

    def fit(self) -> None:
        """Fit individual forecasting models via calling fitExecutor

        This is the fit methdo to fit individual forecasting model

        Args:
            None

        Returns:
            None
        """

        # pyre-fixme[16]: `KatsEnsemble` has no attribute `seasonality`.
        self.seasonality = KatsEnsemble.seasonality_detector(self.data)

        # check if self.params["seasonality_length"] is given
        if (self.seasonality) and (self.params["seasonality_length"] is None):
            msg = "The given time series contains seasonality,\
            a `seasonality_length` must be given in params"
            logging.error(msg)
            raise ValueError(msg)

        # set up auto backtesting flag
        auto_backtesting = False if self.params["aggregation"] == "median" else True

        # check fitExecutor
        if "fitExecutor" not in self.params.keys():
            fitExecutor = self.fitExecutor

        if self.seasonality:
            # STL decomposition
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `sea_data`.
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `desea_data`.
            self.sea_data, self.desea_data = KatsEnsemble.deseasonalize(
                self.data,
                # pyre-fixme[16]: `KatsEnsemble` has no attribute
                #  `decomposition_method`.
                self.decomposition_method
            )

            # we created extra models
            given_models = copy(self.params["models"].models)
            for m in self.params["models"].models:
                if m.model_name.lower() in SMODELS.keys():
                    tmp = copy(m)
                    tmp.model_name = m.model_name + "_smodel"
                    given_models.append(tmp)

            # pyre-fixme[16]: `KatsEnsemble` has no attribute `model_params`.
            # pyre-fixme[16]: Module `kats` has no attribute `models`.
            self.model_params = EnsembleParams(given_models)
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `fitted`.
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `weights`.
            self.fitted, self.weights = fitExecutor(
                data=self.desea_data,
                models=self.model_params,
                should_auto_backtest=auto_backtesting,
            )
        else:
            # fit models on the original data
            self.model_params = EnsembleParams(self.params["models"].models)
            self.fitted, self.weights = fitExecutor(
                data=self.data,
                models=self.model_params,
                should_auto_backtest=auto_backtesting,
            )
        # pyre-fixme[7]: Expected `None` but got `KatsEnsemble`.
        return self

    def predict(self, steps: int) -> None:
        """Predit future for each individual model

        Args:
            steps : number of steps ahead to forecast

        Returns:
            None
        """

        # pyre-fixme[16]: `KatsEnsemble` has no attribute `steps`.
        self.steps = steps
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `seasonality`.
        if self.seasonality:
            # we should pred two types of model
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `fitted`.
            desea_fitted = {k: v for k, v in self.fitted.items() if "_smodel" not in k}
            desea_predict = {
                k: v.predict(self.steps).set_index("time")
                for k, v in desea_fitted.items()
            }

            # re-seasonalize
            predicted = KatsEnsemble.reseasonalize(
                # pyre-fixme[16]: `KatsEnsemble` has no attribute `sea_data`.
                sea_data=self.sea_data,
                desea_predict=desea_predict,
                # pyre-fixme[16]: `KatsEnsemble` has no attribute
                #  `decomposition_method`.
                decomposition_method=self.decomposition_method,
                seasonality_length=self.params["seasonality_length"],
                steps=self.steps,
            )

            # add extra model prediction results from smodels
            fitted_smodel = {k: v for k, v in self.fitted.items() if "_smodel" in k}
            extra_predict = {
                k: v.predict(self.steps).set_index("time")
                for k, v in fitted_smodel.items()
            }

            predicted.update(extra_predict)
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `predicted`.
            self.predicted = predicted
        else:
            predicted = {
                k: v.predict(self.steps).set_index("time")
                for k, v in self.fitted.items()
            }

            # add dummy C.I if the model doesn't have native C.I
            # this is a hack for median ensemble; everyone model needs to have
            # its native C.I if user choose weighted average ensemble.
            for k, v in predicted.items():
                # if predicted df doesn't have fcst_lower and fcst_upper
                if "fcst_lower" not in v.columns or "fcst_upper" not in v.columns:
                    # add dummy C.I
                    tmp_v = copy(v)
                    tmp_v["fcst_lower"] = np.nan
                    tmp_v["fcst_upper"] = np.nan
                    predicted[k] = tmp_v
            self.predicted = predicted
        # pyre-fixme[7]: Expected `None` but got `KatsEnsemble`.
        return self

    def forecast(self, steps: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
        """Holistic forecast method in Kats ensemble

        combine fit and predict methods to produce forecasted results
        this is especially useful for services which prefer to produce
        final forecasts without saving the fitted model

        Args:
            steps: the length of forecasting horizon

        Returns:
            Tuple of predicted values and weights
        """
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `steps`.
        self.steps = steps
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `seasonality`.
        self.seasonality = KatsEnsemble.seasonality_detector(self.data)

        # check if self.params["seasonality_length"] is given
        if (self.seasonality) and (self.params["seasonality_length"] is None):
            msg = "The given time series contains seasonality,\
            a `seasonality_length` must be given in params"
            logging.error(msg)
            raise ValueError(msg)

        # set up auto backtesting flag
        auto_backtesting = False if self.params["aggregation"] == "median" else True

        if self.seasonality:
            # call forecastExecutor and move to next steps
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `sea_data`.
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `desea_data`.
            self.sea_data, self.desea_data = KatsEnsemble.deseasonalize(
                self.data,
                # pyre-fixme[16]: `KatsEnsemble` has no attribute
                #  `decomposition_method`.
                self.decomposition_method
            )

            # call forecasterExecutor with self.desea_data
            desea_predict, desea_err = self.forecastExecutor(
                data=self.desea_data,
                models=self.params["models"],
                steps=steps,
                should_auto_backtest=auto_backtesting,
            )
            # update the desea_predict with adding seasonality component
            # re-seasonalize
            predicted = KatsEnsemble.reseasonalize(
                sea_data=self.sea_data,
                desea_predict=desea_predict,
                decomposition_method=self.decomposition_method,
                seasonality_length=self.params["seasonality_length"],
                steps=self.steps,
            )

            # call forecasterExecutor with self.data
            # create new models
            # we created extra models
            extra_models = []
            for m in self.params["models"].models:
                if m.model_name.lower() in SMODELS.keys():
                    tmp = copy(m)
                    tmp.model_name = m.model_name + "_smodel"
                    extra_models.append(tmp)

            model_params = EnsembleParams(extra_models)
            extra_predict, extra_error = self.forecastExecutor(
                data=self.data,
                models=model_params,
                steps=self.steps,
                should_auto_backtest=auto_backtesting,
            )

            # combine with predict
            predicted.update(extra_predict)
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `predicted`.
            self.predicted = predicted

            if self.params["aggregation"] == "weightedavg":
                desea_err.update(extra_error)
                # pyre-fixme[16]: `KatsEnsemble` has no attribute `err`.
                self.err = desea_err
        else:
            # no seasonality detected
            predicted, self.err = self.forecastExecutor(
                data=self.data,
                models=self.params["models"],
                steps=self.steps,
                should_auto_backtest=auto_backtesting,
            )

            # same as in predict method above
            # add dummy C.I if the model doesn't have native C.I
            # this is a hack for median ensemble; everyone model needs to have
            # its native C.I if user choose weighted average ensemble.
            for k, v in predicted.items():
                # if predicted df doesn't have fcst_lower and fcst_upper
                if "fcst_lower" not in v.columns or "fcst_upper" not in v.columns:
                    # add dummy C.I
                    tmp_v = copy(v)
                    tmp_v["fcst_lower"] = np.nan
                    tmp_v["fcst_upper"] = np.nan
                    predicted[k] = tmp_v

            self.predicted = predicted

        # we need to transform err to weights if it's weighted avg
        if self.params["aggregation"] == "weightedavg":
            original_weights = {
                model: 1 / (err + sys.float_info.epsilon)
                for model, err in self.err.items()
            }
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `weights`.
            self.weights = {
                model: err / sum(original_weights.values())
                for model, err in original_weights.items()
            }
        else:
            self.weights = None
        return self.predicted, self.weights

    def forecastExecutor(self,
                         data : TimeSeriesData,
                         models : EnsembleParams,
                         steps: int,
                         should_auto_backtest: bool = False,
                         # pyre-fixme[31]: Expression `Dict[(str, float)])` is not a
                         #  valid type.
                         ) -> (Dict[str, pd.DataFrame], Dict[str, float]):
        """Forecast Executor

        This is a callable execution function to
        (1). fit model
        (2). predict with a given steps
        (3). back testing (optional)

        Args:
            data: the input time series data as in `TimeSeriesData`
            models: the ensemble parameters as in `EnsembleParams`
            steps: the length of forecasting horizon
            should_auto_backtest: flag to automatically perform back test, default as False

        Returns:
            The predicted values from each individual model and weights
        """

        # Fit individual model with given data
        num_process = min(len(MODELS), (cpu_count() - 1) // 2)
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(processes=(num_process), maxtasksperchild=1000)

        fitted_models = {}
        for model in models.models:
            fitted_models[model.model_name] = pool.apply_async(
                self._fit_single,
                args=(
                    data,
                    MODELS[model.model_name.split("_")[0].lower()],
                    model.model_params),
            )
        pool.close()
        pool.join()
        fitted = {model: res.get() for model, res in fitted_models.items()}

        # simply predict with given steps
        predicted = {}
        for model_name, model_fitted in fitted.items():
            predicted[model_name] = model_fitted.predict(steps).set_index("time")

        # if auto back testing
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `model_params`.
        self.model_params = models  # used by _backtester_all
        if should_auto_backtest:
            weights, errors = self._backtester_all()
        else:
            errors = None

        return predicted, errors

    def aggregate(self) -> pd.DataFrame:
        """Aggregate the results from predict method

        Args:
            None

        Returns:
            final results in pd.DataFrame
        """

        # create future dates
        last_date = self.data.time.max()
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `steps`.
        dates = pd.date_range(start=last_date, periods=self.steps + 1, freq=self.freq)
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `dates`.
        self.dates = dates[dates != last_date]
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `fcst_dates`.
        self.fcst_dates = self.dates.to_pydatetime()

        # collect the fcst, fcst_lower, and fcst_upper into dataframes
        fcsts = {}
        for col in ["fcst", "fcst_lower", "fcst_upper"]:
            fcsts[col] = pd.concat(
                # pyre-fixme[16]: `KatsEnsemble` has no attribute `predicted`.
                [x[col].reset_index(drop=True) for x in self.predicted.values()], axis=1
            )
            fcsts[col].columns = self.predicted.keys()

        if self.params["aggregation"].lower() == "median":
            # clean up dataframes with C.I as np.nan or zero
            fcsts = self.clean_dummy_CI(fcsts, use_zero=False)
            # pyre-fixme[16]: `KatsEnsemble` has no attribute `fcst_df`.
            self.fcst_df = pd.DataFrame({
                "time": self.dates,
                # pyre-fixme[29]: `Series` is not a function.
                "fcst": fcsts["fcst"].median(axis=1),
                # pyre-fixme[29]: `Series` is not a function.
                "fcst_lower": fcsts["fcst_lower"].median(axis=1),
                # pyre-fixme[29]: `Series` is not a function.
                "fcst_upper": fcsts["fcst_upper"].median(axis=1),
            })
        else:
            if fcsts["fcst_lower"].isnull().values.any() or\
               fcsts["fcst_upper"].isnull().values.any():
                msg = "Conf. interval contains NaN, please check individual model"
                logging.error(msg)
                raise ValueError(msg)
            self.fcst_df = pd.DataFrame({
                "time": self.dates,
                # pyre-fixme[16]: `KatsEnsemble` has no attribute `weights`.
                "fcst": fcsts["fcst"].dot(np.array(list(self.weights.values()))),
                "fcst_lower": fcsts["fcst_lower"]
                .dot(np.array(list(self.weights.values()))),
                "fcst_upper": fcsts["fcst_upper"]
                .dot(np.array(list(self.weights.values()))),
            })

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    @staticmethod
    def clean_dummy_CI(
        fcsts: Dict[str, pd.DataFrame],
        use_zero: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Helper method to clean dummy prediction interval

        Args:
            fcsts: the dict of forecasting results from individual models
            use_zero: flag to use zero to fill nan, default as True

        Returns:
            the cleaned results in a dict
        """

        if use_zero:
            fcsts["fcst_lower"] = fcsts["fcst_lower"].fillna(0)
            fcsts["fcst_upper"] = fcsts["fcst_upper"].fillna(0)
        else:
            fcsts["fcst_lower"] = fcsts["fcst_lower"].replace(0, np.nan)
            fcsts["fcst_upper"] = fcsts["fcst_upper"].replace(0, np.nan)
        return fcsts

    def backTestExecutor(self) -> Dict[str, float]:
        """wrapper for back test executor

        services which use KatsEnsemble need to write their own backtest wrapper

        Args:
            None

        Returns:
            The dict of backtesting results
        """

        weights, errors = self._backtester_all()
        # pyre-fixme[7]: Expected `Dict[str, float]` but got `str`.
        return weights

    def _fit_single(self,
                    data: TimeSeriesData,
                    model_func: Callable,
                    model_param: Params
                    ) -> Model:
        """Private method to fit individual model

        Args:
            data: the input time series data
            model_func: the callable func to fit models
            model_param: the corresponding model parameter class

        Returns:
            Fitted Kats model
        """

        # get the model function call
        m = model_func(params=model_param, data=data)
        m.fit()
        return m

    def _backtester_single(
        self,
        params: Params,
        model_class,
        train_percentage: int = 80,
        test_percentage: int = 20,
        err_method: str = "mape",
    ) -> float:
        """Private method to run single back testing process

        Args:
            params: Kats model parameters
            model_class: Untyped. Defines type of model
            train_percentage: float. Percentage of data used for training
            test_percentage: float. Percentage of data used for testing
            error_method: list of strings indicating which errors to calculate
                we currently support "mape", "smape", "mae", "mase", "mse", "rmse"

        Returns:
            float, the backtesting error
        """

        bt = BackTesterSimple(
            [err_method],
            self.data,
            params,
            train_percentage,
            test_percentage,
            model_class
        )
        bt.run_backtest()
        return bt.get_error_value(err_method)

    def _backtester_all(
        self,
        err_method: str = "mape",
    ) -> Dict[str, float]:
        """Private method to run all backtesting process

        Args:
            error_method: list of strings indicating which errors to calculate
                we currently support "mape", "smape", "mae", "mase", "mse", "rmse"

        Returns:
            Dict of errors from each model
        """

        num_process = min(len(MODELS.keys()), (cpu_count() - 1) // 2)
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(processes=(num_process), maxtasksperchild=1000)
        backtesters = {}
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `model_params`.
        for model in self.model_params.models:
            backtesters[model.model_name] = pool.apply_async(
                self._backtester_single,
                args=(
                    model.model_params,
                    MODELS[model.model_name.split("_")[0].lower()]
                ),
                kwds={"err_method": err_method},
            )
        pool.close()
        pool.join()
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `errors`.
        self.errors = {model: res.get() for model, res in backtesters.items()}
        original_weights = {
            model: 1 / (err + sys.float_info.epsilon)
            for model, err in self.errors.items()
        }
        weights = {
            model: err / sum(original_weights.values())
            for model, err in original_weights.items()
        }
        return weights, self.errors

    def plot(self) -> None:
        """plot forecast results

        Args:
            None

        Returns:
            None
        """
        logging.info("Generating chart for forecast result from Ensemble model.")
        # pyre-fixme[16]: Module `kats` has no attribute `models`.
        # pyre-fixme[16]: `KatsEnsemble` has no attribute `fcst_df`.
        mm.Model.plot(self.data, self.fcst_df)
