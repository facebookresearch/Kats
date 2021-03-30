#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from multiprocessing import Pool, cpu_count
from typing import List

import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
)


BASE_MODELS = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "sarima": sarima.SARIMAModel,
    "prophet": prophet.ProphetModel,
    "linear": linear_model.LinearModel,
    "quadratic": quadratic_model.QuadraticModel,
}


class BaseModelParams:
    """Ensemble parameter class

    This class contains three attributes:

    - model: model names (str)
    - model_params: model_param is defined in base models
    """

    def __init__(self, model_name: str, model_params: object, **kwargs) -> None:
        self.model_name = model_name
        self.model_params = model_params
        logging.debug(
            "Initialized Base Model parameters: "
            "Model name:{model_name},"
            "model_params:{model_params}".format(
                model_name=model_name, model_params=model_params
            )
        )

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class EnsembleParams:
    __slots__ = ["models"]

    def __init__(self, models: List[BaseModelParams]) -> None:
        self.models = models


class BaseEnsemble:
    """Base ensemble class

    Implement parent class for ensemble method

    - data: TimeSeriesData object
    - params: EnsembleParams object

    methods:
    - fit: fit base models
    - _predict_all: make predictions with multiple base models
    - plot: visualize the ensemble results
    - get_report: performance report for base and ensemble models
    """

    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params

        for model in params.models:
            if model.model_name not in BASE_MODELS:
                msg = "Model {model_name} is not supported.\
                    Only support{models}.".format(
                    model_name=model.model_name, models=BASE_MODELS.keys()
                )
                logging.error(msg)
                raise ValueError(msg)

        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self,):
        # fit individual models and create a new dict
        # {'m1': fitted_m1_obj, 'm2': fitted_m2_obj}
        num_process = min(len(BASE_MODELS.keys()), (cpu_count() - 1) // 2)
        pool = Pool(processes=(num_process), maxtasksperchild=1000)

        fitted_models = {}
        for model in self.params.models:
            fitted_models[model.model_name] = pool.apply_async(
                self._fit_single,
                args=(BASE_MODELS[model.model_name.lower()], model.model_params),
            )
        pool.close()
        pool.join()
        self.fitted = {model: res.get() for model, res in fitted_models.items()}

    def _fit_single(self, model_func, model_param):
        # get the model function call
        m = model_func(params=model_param, data=self.data)
        m.fit()
        return m

    def _predict_all(self, steps, **kwargs):
        predicted = {}
        for model_name, model_fitted in self.fitted.items():
            predicted[model_name] = model_fitted.predict(steps, **kwargs)
        return predicted

    def plot(self):
        pass

    def __str__(self):
        return "Ensemble"
