# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""STLF forecasting model

This model starts from decomposing the time series data with STL decomposition
then it fits individual foreasting model on the de-seasonalized components
it re-seasonalizes the forecasted results with seasonal data to produce the final
forecasting results.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from copy import copy
from typing import List, Dict

import kats.models.model as m
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models import (
    linear_model,
    prophet,
    quadratic_model,
    theta,
)
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

    def validate_params(self):
        """Validate the parameters for STLF model"""

        logging.info("Method validate_params() is not implemented.")
        pass


class STLFModel(m.Model):
    """Model class for STLF

    This class provides fit, predict, and plot methods for STLF model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `STLFParams`
    """

    def __init__(self, data: TimeSeriesData, params: STLFParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.n = self.data.value.shape[0]

        if self.params.m > self.n:
            msg = "The seasonality length m must be smaller than the length of time series"
            logging.error(msg)
            raise ValueError(msg)

    def deseasonalize(self) -> TimeSeriesData:
        """De-seasonalize the time series data

        Args:
            None

        Returns:
            The seasonal and de-seasonalized data
        """

        # create decomposer for time series decomposition
        decomposer = TimeSeriesDecomposition(self.data, "multiplicative")
        # pyre-fixme[16]: `STLFModel` has no attribute `decomp`.
        self.decomp = decomposer.decomposer()

        # pyre-fixme[16]: `STLFModel` has no attribute `sea_data`.
        self.sea_data = copy(self.decomp["seasonal"])
        # pyre-fixme[16]: `STLFModel` has no attribute `desea_data`.
        self.desea_data = copy(self.data)
        self.desea_data.value = self.desea_data.value / self.decomp["seasonal"].value
        # pyre-fixme[7]: Expected `TimeSeriesData` but got `STLFModel`.
        return self

    def fit(self, **kwargs) -> None:
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
        if self.params.method == "prophet":
            params = prophet.ProphetParams()
            model = prophet.ProphetModel(
                # pyre-fixme[16]: `STLFModel` has no attribute `desea_data`.
                data=self.desea_data,
                params=params,
            )
            model.fit()

        if self.params.method == "theta":
            params = theta.ThetaParams(m=1)
            model = theta.ThetaModel(data=self.desea_data, params=params)
            model.fit()

        if self.params.method == "linear":
            params = linear_model.LinearModelParams()
            model = linear_model.LinearModel(data=self.desea_data, params=params)
            model.fit()

        if self.params.method == "quadratic":
            params = quadratic_model.QuadraticModelParams()
            model = quadratic_model.QuadraticModel(data=self.desea_data, params=params)
            model.fit()
        # pyre-fixme[16]: `STLFModel` has no attribute `model`.
        # pyre-fixme[61]: `model` may not be initialized here.
        self.model = model
        # pyre-fixme[7]: Expected `None` but got `Union[linear_model.LinearModel,
        #  prophet.ProphetModel, quadratic_model.QuadraticModel, theta.ThetaModel]`.
        # pyre-fixme[61]: `model` may not be initialized here.
        return model

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(self, steps: int, include_history=False, **kwargs) -> pd.DataFrame:
        """predict with the fitted STLF model

        Args:
            steps: the steps or length of prediction horizon
            include_history: if include the historical data, default as False

        Returns:
            The predicted dataframe with following columns:
            `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `STLFModel` has no attribute `include_history`.
        self.include_history = include_history
        # pyre-fixme[16]: `STLFModel` has no attribute `freq`.
        # pyre-fixme[16]: `STLFModel` has no attribute `data`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        # pyre-fixme[16]: `STLFModel` has no attribute `alpha`.
        self.alpha = kwargs.get("alpha", 0.05)

        # trend forecast
        # pyre-fixme[16]: `STLFModel` has no attribute `model`.
        fcst = self.model.predict(steps=steps, include_history=include_history)

        # re-seasonalize
        m = self.params.m
        rep = math.trunc(1 + fcst.shape[0] / m)

        # pyre-fixme[16]: `STLFModel` has no attribute `decomp`.
        seasonality = self.decomp["seasonal"].value[-m:]

        # pyre-fixme[16]: `STLFModel` has no attribute `y_fcst`.
        self.y_fcst = fcst.fcst * np.tile(seasonality, rep)[: fcst.shape[0]]
        if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
            # pyre-fixme[16]: `STLFModel` has no attribute `fcst_lower`.
            self.fcst_lower = (
                fcst.fcst_lower * np.tile(seasonality, rep)[: fcst.shape[0]]
            )
            # pyre-fixme[16]: `STLFModel` has no attribute `fcst_upper`.
            self.fcst_upper = (
                fcst.fcst_upper * np.tile(seasonality, rep)[: fcst.shape[0]]
            )
        logging.info("Generated forecast data from STLF model.")
        logging.debug("Forecast data: {fcst}".format(fcst=self.y_fcst))

        # TODO: create empirical uncertainty interval
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        # pyre-fixme[16]: `STLFModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))

        # pyre-fixme[16]: `STLFModel` has no attribute `fcst_df`.
        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.fcst_lower,
                "fcst_upper": self.fcst_upper,
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """plot forecasted results from Prophet model"""

        logging.info("Generating chart for forecast result from STLF model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        """AR net moddel as a string

        Args:
            None

        Returns:
            String representation of the model name
        """

        return "STLF"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """Provide a parameter space for STLF model

        Move the implementation of get_parameter_search_space() out of stlf
        to keep HPT implementation tighter, and avoid the dependency conflict issue.

        Args:
            None

        Returns:
            List of dicts contains parameter search space
        """

        # pyre-fixme[7]: Expected `List[Dict[str, object]]` but got `List[Dict[str,
        #  typing.Union[List[typing.Any], bool, str]]]`.
        return get_default_stlf_parameter_search_space()
