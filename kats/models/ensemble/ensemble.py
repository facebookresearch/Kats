# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ensemble techniques for forecasting

This implements a set of ensemble techniques including weighted averaging, median ensemble
and STL-based ensembling method. This is the parent class for all ensemble models.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from multiprocessing import Pool, cpu_count
from typing import List

import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models import (
    model,
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

    Attributes:
        model: model names (str)
        model_params: model_param is defined in base models
    """

    def __init__(self, model_name: str, model_params: model.Model, **kwargs) -> None:
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

    Implement parent class for ensemble.
    """

    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params

        for m in params.models:
            if m.model_name not in BASE_MODELS:
                msg = "Model {model_name} is not supported.\
                    Only support{models}.".format(
                    model_name=m.model_name, models=BASE_MODELS.keys()
                )
                logging.error(msg)
                raise ValueError(msg)

        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(
        self,
    ):
        """Fit method for ensemble model

        This method fits each individual model for ensembling
        and create a dict of model and fitted obj, such as
        {'m1': fitted_m1_obj, 'm2': fitted_m2_obj}
        """

        num_process = min(len(BASE_MODELS.keys()), (cpu_count() - 1) // 2)
        if num_process < 1:
            num_process = 1
        pool = Pool(processes=(num_process), maxtasksperchild=1000)

        fitted_models = {}
        for m in self.params.models:
            fitted_models[m.model_name] = pool.apply_async(
                self._fit_single,
                args=(BASE_MODELS[m.model_name.lower()], m.model_params),
            )
        pool.close()
        pool.join()
        self.fitted = {model: res.get() for model, res in fitted_models.items()}

    def _fit_single(self, model_func: model.Model, model_param: Params):
        """Private method to fit individual model

        Args:
            model_func: the callable model function
            model_param: the Kats model parameter class

        Returns:
            None
        """

        # get the model function call
        # pyre-fixme[29]: `Model` is not a function.
        m = model_func(params=model_param, data=self.data)
        m.fit()
        return m

    def _predict_all(self, steps: int, **kwargs):
        """Private method to fit all individual models

        Args:
            steps: the length of forecasting horizon

        Returns:
            None
        """

        predicted = {}
        # pyre-fixme[16]: `BaseEnsemble` has no attribute `fitted`.
        for model_name, model_fitted in self.fitted.items():
            predicted[model_name] = model_fitted.predict(steps, **kwargs)
        return predicted

    def plot(self):
        """Plot method for ensemble model (not implemented yet)"""

        pass

    def __str__(self):
        """Get the class name as a string

        Args:
            None

        Returns:
            Model name as a string
        """

        return "Ensemble"
