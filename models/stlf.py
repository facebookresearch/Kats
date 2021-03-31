#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pandas as pd
import math
from copy import copy
import infrastrategy.kats.models.model as m
from infrastrategy.kats.consts import Params, TimeSeriesData
from typing import List, Dict
from infrastrategy.kats.utils.decomposition import TimeSeriesDecomposition
from infrastrategy.kats.models import (
    linear_model,
    prophet,
    quadratic_model,
    theta,
)
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_stlf_parameter_search_space
)
MODELS = ['prophet', "linear", "quadratic", "theta"]


class STLFParams(Params):
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
        logging.info("Method validate_params() is not implemented.")
        pass


class STLFModel(m.Model):
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
        # create decomposer for time series decomposition
        decomposer = TimeSeriesDecomposition(self.data, "multiplicative")
        self.decomp = decomposer.decomposer()

        self.sea_data = copy(self.decomp["seasonal"])
        self.desea_data = copy(self.data)
        self.desea_data.value = self.desea_data.value / self.decomp["seasonal"].value
        return self

    def fit(self, **kwargs) -> None:
        logging.debug("Call fit() with parameters. "
        "kwargs:{kwargs}".format(
            kwargs=kwargs
        ))
        self.deseasonalize()
        if self.params.method == "prophet":
            params = prophet.ProphetParams()
            model = prophet.ProphetModel(
                data=self.desea_data,
                params=params)
            model.fit()

        if self.params.method == "theta":
            params = theta.ThetaParams(m=1)
            model = theta.ThetaModel(
                data=self.desea_data,
                params=params)
            model.fit()

        if self.params.method == "linear":
            params = linear_model.LinearModelParams()
            model = linear_model.LinearModel(
                data=self.desea_data,
                params=params)
            model.fit()

        if self.params.method == "quadratic":
            params = quadratic_model.QuadraticModelParams()
            model = quadratic_model.QuadraticModel(
                data=self.desea_data,
                params=params)
            model.fit()
        self.model = model
        return model

    def predict(self, steps: int, include_history=False, **kwargs) -> pd.DataFrame:
        logging.debug("Call predict() with parameters. "
        "steps:{steps}, kwargs:{kwargs}".format(
            steps=steps, kwargs=kwargs
        ))
        self.include_history = include_history
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.alpha = kwargs.get("alpha", 0.05)

        # trend forecast
        fcst = self.model.predict(steps=steps, include_history=include_history)

        # re-seasonalize
        m = self.params.m
        rep = math.trunc(1 + fcst.shape[0] / m)

        seasonality = self.decomp["seasonal"].value[-m:]

        self.y_fcst = fcst.fcst * np.tile(seasonality, rep)[:fcst.shape[0]]
        if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
            self.fcst_lower = fcst.fcst_lower * np.tile(seasonality, rep)[:fcst.shape[0]]
            self.fcst_upper = fcst.fcst_upper * np.tile(seasonality, rep)[:fcst.shape[0]]
        logging.info("Generated forecast data from STLF model.")
        logging.debug("Forecast data: {fcst}".format(fcst=self.y_fcst))

        # TODO: create empirical uncertainty interval
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))

        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.fcst_lower,
                "fcst_upper": self.fcst_upper
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        logging.info("Generating chart for forecast result from STLF model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)


    def __str__(self):
        return "STLF"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of stlf
        to keep HPT implementation tighter, and avoid the dependency conflict issue.
        """
        return get_default_stlf_parameter_search_space()
