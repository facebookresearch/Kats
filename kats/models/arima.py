# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""ARIMA (Auto Regressive Integrated Moving Average) for time series data.

ARIMA is a classical statistical model with three main components from its name:
- AR, Auto Regressive, means the variable of interest (time series) is regressed
  on its own lagged values
- I, Integrated, means data values have been replaced with the difference between
  their values and the previous value
- MA, Moving Average, means the regression error is a linear combination of error
  terms whose values occurred contemporaneously and at various times in the past

We use the implementation in statsmodels
<https://www.statsmodels.org/stable/index.html> and re-write the API to fit
Kats development style.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import get_default_arima_parameter_search_space
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults


class ARIMAParams(Params):
    """Parameter class for ARIMA model

    This is the parameter class for ARIMA model, it contains all necessary
    parameters from the following ARIMA implementation:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html

    Attributes:
        p: An integer for trend autoregressive (AR) order
        d: An integer for trend difference order
        q: An integer for trend moving average (MA) order
        exog: Optional; An array of exogenous regressors
        dates: Optional; pandas-compatible datetime object
        freq: Optional; frequency of a given time series
    """

    p: int
    d: int
    q: int
    exog: Optional[np.ndarray] = None
    dates: Optional[pd.DatetimeIndex] = None
    freq: Optional[str] = None

    def __init__(self, p: int, d: int, q: int, **kwargs: Any) -> None:
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.exog = kwargs.get("exog", None)
        self.dates = kwargs.get("dates", None)
        self.freq = kwargs.get("freq", None)
        logging.debug(
            "Initialized ARIMAParams with parameters. "
            f"p:{p}, d:{d}, q:{q}, kwargs:{kwargs}"
        )

    def validate_params(self) -> None:
        logging.info("Method validate_params() is not implemented.")
        pass


class ARIMAModel(Model[ARIMAParams]):
    """Model class for ARIMA model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input historical time
            series data from TimeSeriesData
        params: The ARIMA model parameters from ARIMAParams
    """

    exog: Optional[int] = None
    alpha: float = 0.05
    freq: Optional[str] = None
    model: Optional[ARIMAResults] = None
    include_history: bool = False
    fcst_df: Optional[pd.DataFrame] = None
    y_fcst: Optional[np.ndarray] = None
    y_fcst_lower: Optional[np.ndarray] = None
    y_fcst_upper: Optional[np.ndarray] = None
    start_params: Optional[np.ndarray] = None
    transparams: bool = False
    method: Optional[str] = None
    trend: Optional[str] = None
    solver: Optional[str] = None
    maxiter: Optional[int] = None
    full_output: bool = False
    disp: Optional[int] = None
    callback: Optional[Callable[[np.ndarray], None]] = None
    start_ar_lags: Optional[int] = None
    dates: Optional[pd.DatetimeIndex] = None

    def __init__(self, data: TimeSeriesData, params: ARIMAParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = (
                "Only support univariate time series, but got "
                f"{type(self.data.value)}."
            )
            logging.error(msg)
            raise ValueError(msg)

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
        callback: Optional[Callable[[np.ndarray], None]] = None,
        start_ar_lags: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Fit ARIMA model with given parameters

        For more details on each parameter please refer to the following doc:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.fit.html#statsmodels.tsa.arima_model.ARIMA.fit

        Args:
            start_params: Optional; An array_like object for the initial guess
                of the solution for the loglikelihood maximization
            transparams: Optional; A boolean to specify whether or not to
                transform the parameters to ensure stationarity.
            method: A string that specifies the loglikelihood to maximize. Can
                be 'css-mle', 'mle' and 'css'.
            trend: A string that specifies the whether to include a constant in
                the trend or not. Can be 'c' and 'nc'.
            solver: Optional; A string that specifies specifies the solver to be
                used. Can be 'bfgs', 'newton', 'cg', 'ncg' and 'powell'.
            maxiter: Optional; A integer for the maximum number of function
                iterations.
            tol: Optional; The convergence tolerance for the fitting.
            full_output: Optional; A boolean to specify whether to show all
                output from the solver in the results.
            disp: Optional; A integer to control the frequency of the output
                during the iterations.
            callback: Optional; A callable object to be called after each iteration.
            start_ar_lags Optional; An integer to specify the AR lag parameter
                to fit the start_params.

        Returns:
            None
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
            # pyre-fixme[16]: `Optional` has no attribute `value`.
            self.data.value,
            order=(self.params.p, self.params.d, self.params.q),
            exog=self.params.exog,
            # pyre-fixme[16]: `Optional` has no attribute `time`.
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
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """Predict with fitted ARIMA model

        Args:
            steps: An integer for forecast steps
            include_history: Optional; A boolean to specify whether to include
                historical data.

        Returns:
            A pd.DataFrame that contains the forecast and confidence interval
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")

        logging.debug(f"predict(steps:{steps}, kwargs:{kwargs})")
        self.include_history = include_history
        self.exog = kwargs.get("exog", None)
        self.alpha = kwargs.get("alpha", 0.05)
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        fcst = model.forecast(steps, exog=self.exog, alpha=self.alpha)
        logging.info("Generated forecast data from arima model.")
        logging.debug(f"Forecast data: {fcst}")

        self.y_fcst = fcst[0]
        lower, upper = fcst[2].transpose()
        self.y_fcst_lower = lower
        self.y_fcst_upper = upper

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)

        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.fcst_df = fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": lower,
                "fcst_upper": upper,
            },
            copy=False,
        )
        if self.include_history:
            try:
                hist_fcst = (
                    # pyre-fixme[6]: For 1st param expected `Sized` but got
                    #  `Optional[TimeSeriesData]`.
                    model.predict(self.params.d, len(self.data))
                    .reset_index()
                    .rename(columns={"index": "time", 0: "fcst"})
                )
                self.fcst_df = fcst_df = pd.concat([hist_fcst, fcst_df], copy=False)
            except Exception as e:
                msg = (
                    "Fail to generate in-sample forecasts for historical data "
                    f"with error message {e}."
                )
                logging.error(msg)
                raise ValueError(msg)
        logging.debug(f"Return forecast data: {fcst_df}")
        return fcst_df

    def __str__(self) -> str:
        return "ARIMA"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default ARIMA parameter search space.

        Args:
            None

        Returns:
            A dictionary with the default ARIMA parameter search space
        """

        return get_default_arima_parameter_search_space()
