# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The ARIMA model (stand for Auto Regressive Integrated Moving Average) is a classical statistical model for time series data

It contains three main components from its name
- AR, Auto Regressive, means the variable of interest (time series) is regressed on its own lagged values
- MA, Moving Average, means the regression error is a linear combination of error terms whose values occurred contemporaneously and at various times in the past
- I, Integrated, means data values have been replaced with the difference between their values and the previous value
We use the implementation in statsmodels <https://www.statsmodels.org/stable/index.html> and re-write the API to adapt Kats development style.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Dict, Optional, Callable, Any

import kats.models.model as m
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.parameter_tuning_utils import get_default_arima_parameter_search_space
from statsmodels.tsa.arima_model import ARIMA


class ARIMAParams(Params):
    """Parameter class for ARIMA model

    This is the parameter class for ARIMA model, it contains all necessary parameters from the following ARIMA implementation:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html

    Attributes:
        p: An integer for trend autoregressive (AR) order
        d: An integer for trend difference order
        q: An integer for trend moving average (MA) order
        exog: Optional; An array of exogenous regressors
        dates: Optional; pandas-compatible datetime object
        freq: Optional; frequency of a given time series
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
    """Model class for ARIMA model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input historical time series data from TimeSeriesData
        params: The ARIMA model parameters from ARIMAParams
    """

    def __init__(self, data: TimeSeriesData, params: ARIMAParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.exog = None
        self.alpha = 0.05
        self.freq = None
        self.model = None
        self.include_history = False
        self.alpha = 0.05
        self.fcst_df = None
        self.freq = None
        self.y_fcst = None
        self.y_fcst_lower = None
        self.y_fcst_upper = None

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        transparams: bool = True,
        method: str = "css-mle",
        trend: str = "c",
        solver: str = "lbfgs",
        maxiter: int = 500,
        full_output: bool = True,
        disp: int = 5,
        callback: Optional[Callable] = None,
        start_ar_lags: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Fit ARIMA model with given parameters

        For more details on each parameter please refer to the following doc:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.fit.html#statsmodels.tsa.arima_model.ARIMA.fit

        Args:
            start_params: Optional; An array_like object for the initial guess of the solution for the loglikelihood maximization
            transparams: Optional; A boolean to specify whether or not to transform the parameters to ensure stationarity. Default is True
            method: A string that specifies the loglikelihood to maximize. Can be 'css-mle', 'mle' and 'css'. Default is 'css-mle'
            trend: A string that specifies the whether to include a constant in the trend or not. Can be 'c' and 'nc'. Default is 'c'
            solver: Optional; A string that specifies specifies the solver to be used. Can be 'bfgs', 'newton', 'cg', 'ncg' and 'powell'. Default is 'bfgs'
            maxiter: Optional; A integer for the maximum number of function iterations. Default is 500
            tol: Optional; The convergence tolerance for the fitting. Default is 1e-08
            full_output: Optional; A boolean to specify whether to show all output from the solver in the results. Default is True
            disp: Optional; A integer to control the frequency of the output during the iterations. Default is 5
            callback: Optional; A callable object to be called after each iteration. Default is None
            start_ar_lags Optional; An integer to specify the AR lag parameter to fit the start_params. Default is None

        Returns:
            None
        """

        logging.debug("Call fit() method")
        # pyre-fixme[16]: `ARIMAModel` has no attribute `start_params`.
        self.start_params = start_params
        # pyre-fixme[16]: `ARIMAModel` has no attribute `transparams`.
        self.transparams = transparams
        # pyre-fixme[16]: `ARIMAModel` has no attribute `method`.
        self.method = method
        # pyre-fixme[16]: `ARIMAModel` has no attribute `trend`.
        self.trend = trend
        # pyre-fixme[16]: `ARIMAModel` has no attribute `solver`.
        self.solver = solver
        # pyre-fixme[16]: `ARIMAModel` has no attribute `maxiter`.
        self.maxiter = maxiter
        # pyre-fixme[16]: `ARIMAModel` has no attribute `full_output`.
        self.full_output = full_output
        # pyre-fixme[16]: `ARIMAModel` has no attribute `disp`.
        self.disp = disp
        # pyre-fixme[16]: `ARIMAModel` has no attribute `callback`.
        self.callback = callback
        # pyre-fixme[16]: `ARIMAModel` has no attribute `start_ar_lags`.
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

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs
    ) -> pd.DataFrame:
        """Predict with fitted ARIMA model

        Args:
            steps: An integer for forecast steps
            include_history: Optional; A boolearn to specify whether to include historical data. Default is False.

        Returns:
            A pd.DataFrame that contains the forecast and confidence interval
        """

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
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

        # pyre-fixme[16]: `ARIMAModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            }
        )
        if self.include_history:
            try:
                hist_fcst = (
                    self.model.predict(self.params.d, len(self.data))
                    .reset_index()
                    .rename(columns={"index": "time", 0: "fcst"})
                )
                self.fcst_df = pd.concat([hist_fcst, self.fcst_df])
            except Exception as e:
                msg = f"Fail to generate in-sample forecasts for historical data with error message {e}."
                logging.error(msg)
                raise ValueError(msg)
        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """Plot forecast results from the ARIMA model"""

        m.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "ARIMA"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default ARIMA parameter search space.

        Args:
            None

        Returns:
            A dictionary with the default ARIMA parameter search space
        """

        return get_default_arima_parameter_search_space()
