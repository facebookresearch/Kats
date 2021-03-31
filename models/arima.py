#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import infrastrategy.kats.models.model as m
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import Params, TimeSeriesData
from statsmodels.tsa.arima_model import ARIMA
from typing import List, Dict
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_arima_parameter_search_space
)


class ARIMAParams(Params):
    """Model parameter class for ARIMA model

    :Parameters:
    p: int
        the order of AR terms
    d: int
        the number of differencing to make the time series stationary
    q: int
        the order of MA terms
    exog: optional
        exogenous variables
    dates: optional
        pandas-compatible datetime object
    freq: optional
        frequency of a given time series
    """
    __slots__ = ["p", "d", "q"]

    def __init__(self, p: int, d: int, q: int, **kwargs) -> None:
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.exog = kwargs.get("exog", None)
        self.dates = kwargs.get("dates", None)
        self.freq = kwargs.get("freq", None)
        logging.debug(
            "Initialized ARIMAParams with parameters. "
            "p:{p}, d:{d}, q:{q}, kwargs:{kwargs}".format(p=p, d=d, q=q, kwargs=kwargs)
        )

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class ARIMAModel(m.Model):
    """
    ARIMA model (stand for Auto Regressive Integrated Moving Average) is a classical statistical model for time series data.
    It contains three main components from its name

    - AR, Auto Regressive, means the variable of interest (time series) is regressed on its own lagged values

    - MA, Moving Average, means the regression error is a linear combination of error terms whose values occurred contemporaneously and at various times in the past

    - I, Integrated, means data values have been replaced with the difference between their values and the previous value

    We use the implementation in `statsmodels <https://www.statsmodels.org/stable/index.html>`_ and re-write the API to adapt Kats development style.

    :Parameters:
    data: TimeSeriesData
        The input historical time series data
    params: ARIMAParams
        Parameters to config ARIMA model

    :Example:
    >>> import pandas as pd
    >>> from infrastrategy.kats.consts import TimeSeriesData
    >>> from infrastrategy.kats.models.arima import ARIMAModel, ARIMAParams
    >>> # read data and rename the two columns required by TimeSeriesData structure
    >>> data = pd.read_csv("../data/example_air_passengers.csv")
    >>> data.columns = ["time", "y"]
    >>> TSdata = TimeSeriesData(data)
    >>> # create ARIMAParam with specifying initial param values
    >>> params = ARIMAParams(p=1, d=1, q=1)
    >>> # create ARIMAModel with given data and params
    >>> m = ARIMAModel(data=TSdata, params=params)
    >>> # call fit method to fit model
    >>> m.fit()
    >>> # call predict method to predict the next 30 steps
    >>> pred = m.predict(steps=30, freq="MS")
    """
    def __init__(self, data: TimeSeriesData, params: ARIMAParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self,
        start_params=None,
        transparams=True,
        method="css-mle",
        trend="c",
        solver="lbfgs",
        maxiter=500,
        full_output=1,
        disp=False,
        callback=None,
        start_ar_lags=None,
        **kwargs,
            ) -> None:
        """Fit ARIMA model with given parameters

        Please refer to `statsmodels' document on each params <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.fit.html#statsmodels.tsa.arima_model.ARIMA.fit>`_ in fit method.
        """
        logging.debug("Call fit() method")
        self.start_params = start_params
        self.transparams = transparams
        self.method = method
        self.trend = trend
        self.solver = solver
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.start_ar_lags = start_ar_lags

        arima = ARIMA(
            self.data.value,
            order=(self.params.p, self.params.d, self.params.q),
            exog=self.params.exog,
            dates=self.data.time,
            freq=self.params.freq,
        )
        logging.info("Created arima model.")
        self.model = arima.fit(
            start_params=self.start_params,
            transparams=self.transparams,
            method=self.method,
            trend=self.trend,
            solver=self.solver,
            maxiter=self.maxiter,
            full_output=self.full_output,
            disp=self.disp,
            callback=self.callback,
            start_ar_lags=self.start_ar_lags,
        )
        logging.info("Fitted arima.")
        return self.model

    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
        """Predict future values of time series by given number of steps
        """
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.exog = kwargs.get("exog", None)
        self.alpha = kwargs.get("alpha", 0.05)
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        fcst = self.model.forecast(steps, exog=self.exog, alpha=self.alpha)
        logging.info("Generated forecast data from arima model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        self.y_fcst = fcst[0]
        self.y_fcst_lower = np.array([x[0] for x in fcst[2]])
        self.y_fcst_upper = np.array([x[1] for x in fcst[2]])

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)

        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """Visualize the results of historical values and forecasted values and confidence intervals (if any)
        """
        m.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "ARIMA"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of arima
        to avoid the massive dependencies of arima and huge build size.
        Check https://fburl.com/kg04hx5y for detail.
        """
        return get_default_arima_parameter_search_space()
