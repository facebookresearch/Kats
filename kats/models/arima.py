# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


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

import numpy.typing as npt
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import get_default_arima_parameter_search_space
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def _log_deprecation_warnings(**kwargs: Any) -> None:
    for arg_name, arg_val in kwargs.items():
        if arg_val is not None:
            logging.warning(
                f"ARIMA arg {arg_name} was deprecated in statsmodels 0.12.0 and no direct replacement exists, argument will be ignored and will throw an error in the next release."
            )


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
    exog: Optional[npt.NDArray] = None
    dates: Optional[pd.DatetimeIndex] = None
    freq: Optional[str] = None

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        exog: Optional[npt.NDArray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        freq: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        # pyre-fixme[8]: Incompatible attribute type, Attribute `exog` declared in class `ARIMAParams` has type `Optional[ndarray[typing.Any, dtype[typing.Any]]]` but is used as type `Union[None, _SupportsArray[dtype[typing.Any]], _NestedSequence[_SupportsArray[dtype[typing.Any]]],...
        self.exog = exog
        self.dates = dates
        self.freq = freq

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

    # fit args
    trend: Optional[str] = None
    transformed: bool = True
    includes_fixed: bool = False
    method: Optional[str] = None
    method_kwargs: Optional[Dict[str, Any]] = None
    gls: Optional[bool] = None
    gls_kwargs: Optional[Dict[str, Any]] = None
    cov_type: Optional[str] = None
    cov_kwds: Optional[Dict[str, Any]] = None

    # results placeholder
    model: Optional[ARIMAResults] = None

    # predict args
    include_history: bool = False
    fcst_exog: Optional[npt.NDArray[Any]] = None
    alpha: float = 0.05
    freq: Optional[str] = None
    fcst_df: Optional[pd.DataFrame] = None
    y_fcst: Optional[npt.NDArray] = None
    y_fcst_lower: Optional[npt.NDArray] = None
    y_fcst_upper: Optional[npt.NDArray] = None
    start_params: Optional[npt.NDArray] = None
    transparams: bool = False
    method: Optional[str] = None
    trend: Optional[str] = None
    solver: Optional[str] = None
    maxiter: Optional[int] = None
    full_output: bool = False
    disp: Optional[int] = None
    callback: Optional[Callable[[npt.NDArray], None]] = None
    start_ar_lags: Optional[int] = None
    dates: Optional[pd.DatetimeIndex] = None

    def __init__(self, data: TimeSeriesData, params: ARIMAParams) -> None:
        super().__init__(data, params)
        self.data = data
        if not isinstance(self.data.value, pd.Series):
            msg = (
                f"Only support univariate time series, but got {type(self.data.value)}."
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(
        self,
        trend: str = "n",
        start_params: Optional[npt.NDArray] = None,
        transformed: bool = True,
        includes_fixed: bool = False,
        method: Optional[str] = None,
        method_kwargs: Optional[Dict[str, Any]] = None,
        gls: Optional[bool] = None,
        gls_kwargs: Optional[Dict[str, Any]] = None,
        cov_type: Optional[str] = None,
        cov_kwds: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Fit ARIMA model with given parameters

        For more details on each parameter please refer to the following doc:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

        Args:
            trend: Optional; A string representing the trend of time series, can be 'c', 't', or 'ct' (constant, linear trend, or both). Consult statsmodels documentation for further details.
            start_params: Optional; Initial guess of the solution for the loglikelihood maximization.
            transformed: Optional; Whether start_params are already transformed (default True)
            includes_fixed: Optional; If params were previously fixed fwith the fix_params method (default False)
            method: Optional; The parameter estimation method to use ('statespace', 'innovations_mle', 'hannan_rissanen','burg','innovations', or 'yule_walker')
            method_kwargs: Optional; Additional keyword arguments to pass to the method.
            gls: Optional; Whether to use generalized least squares (default depends on method).
            gls_kwargs: Optional; Additional keyword arguments to pass to the GLS fit method.
            cov_type: Optional; sets the method for calculating the covariance matrix of parameter estimates ('opg', 'oim', 'approx', 'robust', 'robust_approx', or 'none').
            cov_kwds: Optional; Additional keyword arguments to pass to the covariance matrix method.
            transparams: DEPRECATED;
            solver: DEPRECATED;
            maxiter: DEPRECATED;
            full_output: DEPRECATED;
            disp: DEPRECATED;
            callback: DEPRECATED;
            start_ar_lags: DEPRECATED;

        Returns:
            None
        """

        self.trend = trend
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.method = method
        self.method_kwargs = method_kwargs
        self.gls = gls
        self.gls_kwargs = gls_kwargs
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds

        _log_deprecation_warnings(
            transparams=kwargs.get("transparams"),
            solver=kwargs.get("solver"),
            maxiter=kwargs.get("maxiter"),
            full_output=kwargs.get("full_output"),
            disp=kwargs.get("disp"),
            callback=kwargs.get("callback"),
            start_ar_lags=kwargs.get("start_ar_lags"),
        )

        if self.trend not in ("n", "c", "t", "ct"):
            raise ValueError("Trend must be one of 'n', 'c', 't', or 'ct'")

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
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            method=self.method,
            method_kwargs=self.method_kwargs,
            gls=self.gls,
            gls_kwargs=self.gls_kwargs,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            return_params=False,
            low_memory=False,
        )

    # TODO: the Model base class should be converted to an ABC because otherwise these method overrides throw Pyre errors
    # pyre-fixme
    def predict(
        self,
        steps: int,
        include_history: bool = False,
        exog: Optional[npt.NDArray] = None,
        alpha: float = 0.05,
        **kwargs: Any,
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

        self.fcst_exog = exog
        self.alpha = alpha
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))

        fcst = model.get_forecast(
            steps,
            exog=self.fcst_exog,
            signal_only=False,
        ).summary_frame()
        logging.info("Generated forecast data from arima model.")

        self.y_fcst = fcst["mean"].ravel()
        lower = fcst["mean_ci_lower"].ravel()
        upper = fcst["mean_ci_upper"].ravel()
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

        self.include_history = include_history
        if self.include_history:
            try:
                t_pred = self.data.time[self.params.d :]
                hist_fcst = (
                    model.predict(start=self.params.d, end=len(self.data))
                    .rename("fcst")
                    .to_frame()
                    .assign(time=t_pred)
                    .dropna(subset=["time"])
                )
                self.fcst_df = fcst_df = pd.concat(
                    [hist_fcst, fcst_df], axis=0, ignore_index=True
                )
            except Exception as e:
                msg = (
                    "Fail to generate in-sample forecasts for historical data "
                    f"with error message {e}."
                )
                logging.error(msg)
                raise ValueError(msg)
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
