# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""STLF forecasting model

This model starts from decomposing the time series data with STL decomposition
then it fits individual foreasting model on the de-seasonalized components
it re-seasonalizes the forecasted results with seasonal data to produce the final
forecasting results.
"""

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import logging
import math
from copy import copy
from typing import Any, List, Dict, Optional, Union

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models import (
    linear_model,
    prophet,
    quadratic_model,
    theta,
)
from kats.models.linear_model import LinearModel
from kats.models.model import Model
from kats.models.prophet import ProphetModel
from kats.models.quadratic_model import QuadraticModel
from kats.models.theta import ThetaModel
from kats.utils.decomposition import TimeSeriesDecomposition
from kats.utils.parameter_tuning_utils import get_default_stlf_parameter_search_space

MODELS = ["prophet", "linear", "quadratic", "theta"]


class STLFParams(Params):
    """Parameter class for Prophet model

    This is the parameter class for STLF model, stands for STL-decomposition based
    forecasting model.

    Attributes:
        method: str, the forecasting model to fit on the de-seasonalized component
            it currently supports prophet, linear, quadratic, and theta method.
        m: int, the length of one seasonal cycle
    """

    def __init__(self, method: str, m: int) -> None:
        super().__init__()
        if method not in MODELS:
            msg = "Only support prophet, linear, quadratic and theta method, but get {name}.".format(
                name=method
            )
            logging.error(msg)
            raise ValueError(msg)
        self.method = method
        self.m = m
        logging.debug("Initialized STFLParams instance.")

    def validate_params(self) -> None:
        """Validate the parameters for STLF model"""

        logging.info("Method validate_params() is not implemented.")
        pass


class STLFModel(Model[STLFParams]):
    """Model class for STLF

    This class provides fit, predict, and plot methods for STLF model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `STLFParams`
    """

    decomp: Optional[Dict[str, TimeSeriesData]] = None
    sea_data: Optional[TimeSeriesData] = None
    desea_data: Optional[TimeSeriesData] = None
    model: Optional[Union[LinearModel, ProphetModel, QuadraticModel, ThetaModel]] = None
    freq: Optional[str] = None
    alpha: Optional[float] = None
    y_fcst: Optional[np.ndarray] = None
    fcst_lower: Optional[np.ndarray] = None
    fcst_upper: Optional[np.ndarray] = None
    dates: Optional[pd.DatetimeIndex] = None
    fcst_df: Optional[pd.DataFrame] = None

    def __init__(self, data: TimeSeriesData, params: STLFParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.n: int = self.data.value.shape[0]

        if self.params.m > self.n:
            msg = "The seasonality length m must be smaller than the length of time series"
            logging.error(msg)
            raise ValueError(msg)

    def deseasonalize(self) -> STLFModel:
        """De-seasonalize the time series data

        Args:
            None

        Returns:
            This object, with `decomp`, `sea_data`, and `desea_data` attributes
            set to the decomposition results.
        """

        # create decomposer for time series decomposition
        # pyre-fixme[6]: For 1st param expected `TimeSeriesData` but got
        #  `Optional[TimeSeriesData]`.
        decomposer = TimeSeriesDecomposition(self.data, "multiplicative")
        self.decomp = decomp = decomposer.decomposer()

        self.sea_data = copy(decomp["seasonal"])
        self.desea_data = desea_data = copy(self.data)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        desea_data.value = desea_data.value / decomp["seasonal"].value
        return self

    # pyre-fixme[15]: `fit` overrides method defined in `Model` inconsistently.
    def fit(self, **kwargs: Any) -> STLFModel:
        """Fit STLF model

        Args:
            None

        Returns:
            The fitted STLF model object
        """

        logging.debug(
            "Call fit() with parameters. " "kwargs:{kwargs}".format(kwargs=kwargs)
        )
        self.deseasonalize()
        data = self.desea_data
        assert data is not None
        if self.params.method == "prophet":
            params = prophet.ProphetParams()
            model = prophet.ProphetModel(
                data=data,
                params=params,
            )
            model.fit()
        elif self.params.method == "theta":
            params = theta.ThetaParams(m=1)
            model = theta.ThetaModel(data=data, params=params)
            model.fit()
        elif self.params.method == "linear":
            params = linear_model.LinearModelParams()
            model = linear_model.LinearModel(data=data, params=params)
            model.fit()
        else:
            assert self.params.method == "quadratic"
            params = quadratic_model.QuadraticModelParams()
            model = quadratic_model.QuadraticModel(data=data, params=params)
            model.fit()
        self.model = model
        return self

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, *args: Any, include_history: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """predict with the fitted STLF model

        Args:
            steps: the steps or length of prediction horizon
            include_history: if include the historical data, default as False

        Returns:
            The predicted dataframe with following columns:
            `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")
        decomp = self.decomp
        assert decomp is not None

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.alpha = kwargs.get("alpha", 0.05)

        # trend forecast
        fcst = model.predict(steps=steps, include_history=include_history)

        # re-seasonalize
        m = self.params.m
        rep = math.trunc(1 + fcst.shape[0] / m)

        seasonality = decomp["seasonal"].value[-m:]

        self.y_fcst = fcst.fcst * np.tile(seasonality, rep)[: fcst.shape[0]]
        if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
            self.fcst_lower = (
                fcst.fcst_lower * np.tile(seasonality, rep)[: fcst.shape[0]]
            )
            self.fcst_upper = (
                fcst.fcst_upper * np.tile(seasonality, rep)[: fcst.shape[0]]
            )
        logging.info("Generated forecast data from STLF model.")
        logging.debug("Forecast data: {fcst}".format(fcst=self.y_fcst))

        # TODO: create empirical uncertainty interval
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))

        self.fcst_df = fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.fcst_lower,
                "fcst_upper": self.fcst_upper,
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=fcst_df))
        return fcst_df

    def __str__(self) -> str:
        """AR net moddel as a string

        Args:
            None

        Returns:
            String representation of the model name
        """

        return "STLF"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Provide a parameter space for STLF model

        Move the implementation of get_parameter_search_space() out of stlf
        to keep HPT implementation tighter, and avoid the dependency conflict issue.

        Args:
            None

        Returns:
            List of dicts contains parameter search space
        """

        return get_default_stlf_parameter_search_space()
