# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Ensemble techniques for forecasting

This implements a set of ensemble techniques including weighted averaging, median ensemble
and STL-based ensembling method. This is the parent class for all ensemble models.
"""

import logging
from multiprocessing import cpu_count, Pool
from typing import Dict, List, Optional, Type

import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
)
from kats.models.model import Model


BASE_MODELS = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "sarima": sarima.SARIMAModel,
    "prophet": prophet.ProphetModel,
    "linear": linear_model.LinearModel,
    "quadratic": quadratic_model.QuadraticModel,
}


class BaseModelParams(Params):
    """Ensemble parameter class

    This class contains three attributes:

    Attributes:
        model: model names (str)
        model_params: model_param is defined in base models
    """

    def __init__(self, model_name: str, model_params: Params) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_params = model_params
        logging.debug(
            "Initialized Base Model parameters: "
            "Model name:{model_name},"
            "model_params:{model_params}".format(
                model_name=model_name, model_params=model_params
            )
        )

    def validate_params(self) -> None:
        logging.info("Method validate_params() is not implemented.")
        pass


class EnsembleParams(Params):
    __slots__ = ["models"]

    def __init__(self, models: List[BaseModelParams]) -> None:
        self.models: List[BaseModelParams] = models


# pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
class BaseEnsemble(Model):
    """Base ensemble class

    Implement parent class for ensemble.
    """

    # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
    fitted: Optional[Dict[str, Model]] = None

    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        super().__init__(data, params)

        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        for m in params.models:
            if m.model_name not in BASE_MODELS:
                msg = "Model {model_name} is not supported.\
                    Only support{models}.".format(
                    model_name=m.model_name, models=BASE_MODELS.keys()
                )
                logging.error(msg)
                raise ValueError(msg)

    def fit(self) -> None:
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

    # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
    def _fit_single(self, model_func: Type[Model], model_param: Params) -> Model:
        """Private method to fit individual model

        Args:
            model_func: the callable model function
            model_param: the Kats model parameter class

        Returns:
            None
        """

        # get the model function call
        m = model_func(params=model_param, data=self.data)
        m.fit()
        return m

    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
    def _predict_all(self, steps: int, **kwargs) -> Dict[str, Model]:
        """Private method to fit all individual models

        Args:
            steps: the length of forecasting horizon

        Returns:
            None
        """
        fitted = self.fitted
        if fitted is None:
            raise ValueError("Call fit() before predict().")

        predicted = {}
        for model_name, model_fitted in fitted.items():
            predicted[model_name] = model_fitted.predict(steps, **kwargs)
        return predicted

    def __str__(self) -> str:
        """Get the class name as a string

        Args:
            None

        Returns:
            Model name as a string
        """

        return "Ensemble"
