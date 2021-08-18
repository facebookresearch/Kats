# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Kats ensemble model

Implementation of the Kats ensemble model. It starts from seasonality detection, if seasonality detected, it
continues to perform STL decomposition, then fit forecasting models on de-seasonalized components and aggregate;
otherwise it simiply leverage individual forecasting models and ensembling. We provided two ensembling methods,
weighted average and median ensembling.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import sys
from copy import copy
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import kats.models.model as mm
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData, Params

# Seasonality detector
from kats.detectors.seasonality import ACFDetector
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)
from kats.models.ensemble.ensemble import EnsembleParams
from kats.models.model import Model
from kats.utils.backtesters import BackTesterSimple

# STL decomposition
from kats.utils.decomposition import TimeSeriesDecomposition

# from numpy.typing import ArrayLike
ArrayLike = Union[Sequence[float], np.ndarray]

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


def _logged_error(msg: str) -> ValueError:
    """Log and raise an error."""
    logging.error(msg)
    return ValueError(msg)


class KatsEnsemble:
    """Decomposition based ensemble model in Kats
    This is the holistic ensembling class based on decomposition when seasonality presents
    """

    seasonality: bool = False
    sea_data: Optional[TimeSeriesData] = None
    desea_data: Optional[TimeSeriesData] = None
    steps: int = -1
    decomposition_method: str = ""
    model_params: Optional[EnsembleParams] = None
    fitted: Optional[Dict[Any, Any]] = None
    weights: Optional[Dict[str, float]] = None
    predicted: Optional[Dict[str, pd.DataFrame]] = None
    err: Optional[Dict[str, float]] = None
    dates: Optional[pd.DatetimeIndex] = None
    fcst_dates: Optional[ArrayLike] = None
    fcst_df: Optional[pd.DataFrame] = None
    errors: Optional[Dict[Any, Any]] = None

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
            method = self.params["aggregation"]
            msg = f"Only support `median` or `weightedavg` ensemble, but got {method}."
            raise _logged_error(msg)

        # validate decomposition method
        if self.params["decomposition_method"] in ("additive", "multiplicative"):
            self.decomposition_method = self.params["decomposition_method"]
        else:
            logging.info("Invalid decomposition method setting specified")
            logging.info("Defaulting to Additive Decomposition")
            self.decomposition_method = "additive"

        # validate m
        if (self.params["seasonality_length"] is not None) and (
            self.params["seasonality_length"] > int(len(self.data.time) // 2)
        ):
            msg = "seasonality_length value cannot be larger than"
            "1/2 of the length of give time series"
            raise _logged_error(msg)

        # check customized forecastExecutor
        if ("forecastExecutor" in self.params.keys()) and (
            self.params["forecastExecutor"] is not None
        ):
            msg = "Using customized forecastExecutor from given parameters"
            logging.info(msg)
            self.forecastExecutor = self.params["forecastExecutor"]

        # check customized fitExecutor
        if ("fitExecutor" in self.params.keys()) and (
            self.params["fitExecutor"] is not None
        ):
            msg = "Using customized fitExecutor from given parameters"
            logging.info(msg)
            self.fitExecutor = self.params["fitExecutor"]

    @staticmethod
    def seasonality_detector(data) -> bool:
        """Detect seasonalities from given TimeSeriesData

        Args:
            data: :class:`kats.consts.TimeSeriesData`, the input `TimeSeriesData`

        Returns:
            Flag for the presence of seasonality
        """

        detector = ACFDetector(data)
        detector.detector()
        seasonality = detector.seasonality_detected
        return seasonality

    @staticmethod
    def deseasonalize(
        data: TimeSeriesData, decomposition_method: str
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        """STL decomposition to given TimeSeriesData

        Static method to perform decomposition on the input data

        Args:
            data: :class:`kats.consts.TimeSeriesData`, input time series data
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
            desea_data.value = desea_data.value - decomp["seasonal"].value
        else:
            desea_data.value = desea_data.value / decomp["seasonal"].value
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
            sea_data: :class:`kats.consts.TimeSeriesData`, the seasonal data from deseasonalize method
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
                    predicted[model_name] = (
                        desea_pred
                        + np.tile(
                            np.tile(seasonality_unit, rep)[:steps], [3, 1]
                        ).transpose()
                    )
                else:
                    # no C.I from individual model
                    tmp_fcst = desea_pred.fcst + np.tile(seasonality_unit, rep)[:steps]
                    predicted[model_name] = pd.DataFrame(
                        {
                            "time": desea_pred.index,
                            "fcst": tmp_fcst,
                            "fcst_lower": np.nan,
                            "fcst_upper": np.nan,
                        }
                    ).set_index("time")

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
                    predicted[model_name] = (
                        desea_pred
                        * np.tile(
                            np.tile(seasonality_unit, rep)[:steps], [3, 1]
                        ).transpose()
                    )
                else:
                    # no C.I from individual model
                    tmp_fcst = desea_pred.fcst * np.tile(seasonality_unit, rep)[:steps]
                    predicted[model_name] = pd.DataFrame(
                        {
                            "time": desea_pred.index,
                            "fcst": tmp_fcst,
                            "fcst_lower": 0,
                            "fcst_upper": 0,
                        }
                    ).set_index("time")

        return predicted

    def fitExecutor(
        self,
        data: TimeSeriesData,
        models: EnsembleParams,
        should_auto_backtest: bool = False,
    ) -> Tuple[Dict[Any, Any], Optional[Dict[str, float]]]:
        """callable forecast executor

        This is native implementation with Python's multiprocessing
        fit individual model in `models` with given `data`. Services
        who use KatsEnsemble need to implement their own executor for better
        performance, if no executor function is given, the native version will be
        used.

        Attributes:
            data: :class:`kats.consts.TimeSeriesData`, given TimeSeriesData, could be original or de-seasonalized
            models: EnsembleParams object containing model params
                in BaseModelParams
            should_auto_backtest: boolean flag for additional back testing runs

        Returns:
            Tuple of fitted individual model and weights
        """

        # Fit individual model with given data
        num_process = min(len(MODELS), (cpu_count() - 1) // 2)
        if num_process < 1:
            num_process = 1
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(
            processes=(num_process), maxtasksperchild=1000
        )

        fitted_models = {}
        for model in models.models:
            fitted_models[model.model_name] = pool.apply_async(
                self._fit_single,
                args=(
                    data,
                    MODELS[model.model_name.split("_")[0].lower()],
                    model.model_params,
                ),
            )
        pool.close()
        pool.join()
        fitted = {model: res.get() for model, res in fitted_models.items()}

        # if auto back testing
        weights = self.backTestExecutor() if should_auto_backtest else None
        return fitted, weights

    def fit(self) -> KatsEnsemble:
        """Fit individual forecasting models via calling fitExecutor

        This is the fit methdo to fit individual forecasting model
        """

        self.seasonality = KatsEnsemble.seasonality_detector(self.data)

        # check if self.params["seasonality_length"] is given
        if self.seasonality and self.params["seasonality_length"] is None:
            msg = "The given time series contains seasonality,\
            a `seasonality_length` must be given in params."
            raise _logged_error(msg)

        # set up auto backtesting flag
        auto_backtesting = False if self.params["aggregation"] == "median" else True

        # check fitExecutor
        if "fitExecutor" not in self.params.keys():
            fitExecutor = self.fitExecutor

        if self.seasonality:
            # STL decomposition
            sea_data, desea_data = KatsEnsemble.deseasonalize(
                self.data, self.decomposition_method
            )
            self.sea_data = sea_data
            self.desea_data = desea_data

            # we created extra models
            given_models = copy(self.params["models"].models)
            for m in self.params["models"].models:
                if m.model_name.lower() in SMODELS.keys():
                    tmp = copy(m)
                    tmp.model_name = m.model_name + "_smodel"
                    given_models.append(tmp)

            self.model_params = model_params = EnsembleParams(given_models)
            # pyre-fixme[61]: `fitExecutor` may not be initialized here.
            self.fitted, self.weights = fitExecutor(
                data=desea_data,
                models=model_params,
                should_auto_backtest=auto_backtesting,
            )
        else:
            # fit models on the original data
            self.model_params = model_params = EnsembleParams(
                self.params["models"].models
            )
            # pyre-fixme[61]: `fitExecutor` may not be initialized here.
            self.fitted, self.weights = fitExecutor(
                data=self.data,
                models=model_params,
                should_auto_backtest=auto_backtesting,
            )
        return self

    def predict(self, steps: int) -> KatsEnsemble:
        """Predit future for each individual model

        Args:
            steps : number of steps ahead to forecast

        Returns:
            None
        """

        fitted = self.fitted
        if fitted is None:
            raise _logged_error("fit must be called before predict.")

        self.steps = steps
        if self.seasonality:
            sea_data = self.sea_data
            assert sea_data is not None
            # we should pred two types of model
            desea_fitted = {k: v for k, v in fitted.items() if "_smodel" not in k}
            desea_predict = {
                k: v.predict(self.steps).set_index("time")
                for k, v in desea_fitted.items()
            }

            # re-seasonalize
            predicted = KatsEnsemble.reseasonalize(
                sea_data=sea_data,
                desea_predict=desea_predict,
                decomposition_method=self.decomposition_method,
                seasonality_length=self.params["seasonality_length"],
                steps=self.steps,
            )

            # add extra model prediction results from smodels
            fitted_smodel = {k: v for k, v in fitted.items() if "_smodel" in k}
            extra_predict = {
                k: v.predict(self.steps).set_index("time")
                for k, v in fitted_smodel.items()
            }

            predicted.update(extra_predict)
            self.predicted = predicted
        else:
            predicted = {
                k: v.predict(self.steps).set_index("time") for k, v in fitted.items()
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
        return self

    def forecast(
        self, steps: int
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, float]]]:
        """Holistic forecast method in Kats ensemble

        combine fit and predict methods to produce forecasted results
        this is especially useful for services which prefer to produce
        final forecasts without saving the fitted model

        Args:
            steps: the length of forecasting horizon

        Returns:
            Tuple of predicted values and weights
        """
        self.steps = steps
        self.seasonality = KatsEnsemble.seasonality_detector(self.data)

        # check if self.params["seasonality_length"] is given
        if (self.seasonality) and (self.params["seasonality_length"] is None):
            msg = "The given time series contains seasonality,\
            a `seasonality_length` must be given in params."
            raise _logged_error(msg)

        # set up auto backtesting flag
        auto_backtesting = False if self.params["aggregation"] == "median" else True

        if self.seasonality:
            # call forecastExecutor and move to next steps
            sea_data, desea_data = KatsEnsemble.deseasonalize(
                self.data, self.decomposition_method
            )
            self.sea_data = sea_data
            self.desea_data = desea_data

            # call forecasterExecutor with self.desea_data
            desea_predict, desea_err = self.forecastExecutor(
                data=desea_data,
                models=self.params["models"],
                steps=steps,
                should_auto_backtest=auto_backtesting,
            )
            # update the desea_predict with adding seasonality component
            # re-seasonalize
            predicted = KatsEnsemble.reseasonalize(
                sea_data=sea_data,
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
            self.predicted = predicted

            if self.params["aggregation"] == "weightedavg":
                if desea_err is None:
                    desea_err = extra_error
                elif extra_error is not None:
                    desea_err.update(extra_error)
                self.err = forecast_error = desea_err
        else:
            # no seasonality detected
            predicted, forecast_error = self.forecastExecutor(
                data=self.data,
                models=self.params["models"],
                steps=self.steps,
                should_auto_backtest=auto_backtesting,
            )
            self.err = forecast_error

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
            # pyre-fixme[61]: `forecast_error` may not be initialized here.
            assert forecast_error is not None
            original_weights = {
                model: 1 / (err + sys.float_info.epsilon)
                for model, err in forecast_error.items()
            }
            self.weights = {
                model: err / sum(original_weights.values())
                for model, err in original_weights.items()
            }
        else:
            self.weights = None
        return predicted, self.weights

    def forecastExecutor(
        self,
        data: TimeSeriesData,
        models: EnsembleParams,
        steps: int,
        should_auto_backtest: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, float]]]:
        """Forecast Executor

        This is a callable execution function to
        (1). fit model
        (2). predict with a given steps
        (3). back testing (optional)

        Args:
            data: :class:`kats.consts.TimeSeriesData`, the input time series data as in :class:`kats.consts.TimeSeriesData`
            models: the ensemble parameters as in `EnsembleParams`
            steps: the length of forecasting horizon
            should_auto_backtest: flag to automatically perform back test, default as False

        Returns:
            The predicted values from each individual model and weights
        """

        # Fit individual model with given data
        num_process = min(len(MODELS), (cpu_count() - 1) // 2)
        if num_process < 1:
            num_process = 1
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(
            processes=(num_process), maxtasksperchild=1000
        )

        fitted_models = {}
        for model in models.models:
            fitted_models[model.model_name] = pool.apply_async(
                self._fit_single,
                args=(
                    data,
                    MODELS[model.model_name.split("_")[0].lower()],
                    model.model_params,
                ),
            )
        pool.close()
        pool.join()
        fitted = {model: res.get() for model, res in fitted_models.items()}

        # simply predict with given steps
        predicted = {}
        for model_name, model_fitted in fitted.items():
            predicted[model_name] = model_fitted.predict(steps).set_index("time")

        # if auto back testing
        self.model_params = models  # used by _backtester_all
        if should_auto_backtest:
            _, errors = self._backtester_all()
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
        predicted = self.predicted
        if predicted is None:
            raise _logged_error("predict must be called before aggregate.")

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=self.steps + 1, freq=self.freq)
        self.dates = dates = dates[dates != last_date]
        self.fcst_dates = dates.to_pydatetime()

        # collect the fcst, fcst_lower, and fcst_upper into dataframes
        fcsts = {}
        for col in ["fcst", "fcst_lower", "fcst_upper"]:
            fcsts[col] = pd.concat(
                [x[col].reset_index(drop=True) for x in predicted.values()], axis=1
            )
            fcsts[col].columns = predicted.keys()

        if self.params["aggregation"].lower() == "median":
            # clean up dataframes with C.I as np.nan or zero
            fcsts = self.clean_dummy_CI(fcsts, use_zero=False)
            self.fcst_df = fcst_df = pd.DataFrame(
                {
                    "time": dates,
                    "fcst": fcsts["fcst"].median(axis=1),
                    "fcst_lower": fcsts["fcst_lower"].median(axis=1),
                    "fcst_upper": fcsts["fcst_upper"].median(axis=1),
                }
            )
        else:
            if (
                fcsts["fcst_lower"].isnull().values.any()
                or fcsts["fcst_upper"].isnull().values.any()
            ):
                msg = "Conf. interval contains NaN, please check individual model."
                raise _logged_error(msg)
            weights = self.weights
            assert weights is not None
            weights = np.array(list(weights.values()))
            self.fcst_df = fcst_df = pd.DataFrame(
                {
                    "time": dates,
                    "fcst": fcsts["fcst"].dot(weights),
                    "fcst_lower": fcsts["fcst_lower"].dot(weights),
                    "fcst_upper": fcsts["fcst_upper"].dot(weights),
                }
            )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=fcst_df))
        return fcst_df

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

        weights, _ = self._backtester_all()
        return weights

    def _fit_single(
        self, data: TimeSeriesData, model_func: Callable, model_param: Params
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
            model_class,
        )
        bt.run_backtest()
        return bt.get_error_value(err_method)

    def _backtester_all(
        self,
        err_method: str = "mape",
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Private method to run all backtesting process

        Args:
            error_method: list of strings indicating which errors to calculate
                we currently support "mape", "smape", "mae", "mase", "mse", "rmse"

        Returns:
            Dict of errors from each model
        """
        model_params = self.model_params
        if model_params is None:
            raise _logged_error("fit must be called before backtesting.")

        num_process = min(len(MODELS.keys()), (cpu_count() - 1) // 2)
        if num_process < 1:
            num_process = 1
        # pyre-fixme[16]: `SyncManager` has no attribute `Pool`.
        pool = multiprocessing.Manager().Pool(
            processes=(num_process), maxtasksperchild=1000
        )
        backtesters = {}
        for model in model_params.models:
            backtesters[model.model_name] = pool.apply_async(
                self._backtester_single,
                args=(
                    model.model_params,
                    MODELS[model.model_name.split("_")[0].lower()],
                ),
                kwds={"err_method": err_method},
            )
        pool.close()
        pool.join()
        self.errors = errors = {model: res.get() for model, res in backtesters.items()}
        original_weights = {
            model: 1 / (err + sys.float_info.epsilon) for model, err in errors.items()
        }
        weights = {
            model: err / sum(original_weights.values())
            for model, err in original_weights.items()
        }
        return weights, errors

    def plot(self) -> None:
        """plot forecast results"""
        fcst_df = self.fcst_df
        if fcst_df is None:
            raise _logged_error("forecast must be called before plot.")

        logging.info("Generating chart for forecast result from Ensemble model.")
        mm.Model.plot(self.data, fcst_df)
