# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Dict, Optional, Tuple, Callable, Union, Any

import kats.models.model as m
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData, Params
from kats.utils.parameter_tuning_utils import (
    get_default_sarima_parameter_search_space,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

ArrayLike = np.ndarray


class SARIMAParams(Params):
    """Parameter class for SARIMA model

    This is the parameter class for SARIMA model, it contains all necessary parameters as defined in SARIMA model implementation:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html.

    Attributes:
        p: An integer for trend autoregressive (AR) order.
        d: An integer for trend difference order.
        q: An integer for trend moving average (MA) order.
        exog: Optional; An array of exogenous regressors.
        seasonal_order: Optional; A tuple for (P,D,Q,s) order of the seasonal component for AR order, difference order, MA order, and periodicity. Default is (0,0,0,0).
        trend: Optional; A string or an iterable for deterministic trend. Can be 'c' (constant), 't' (linear trend with time), 'ct' (both constant and linear trend), or an iterable of integers defining the non-zero polynomial exponents to include. Default is None (not to include trend).
        measurement_error: Optional; A boolean to specify whether or not to assume the observed time series were measured with error. Default is False.
        time_varying_regression: Optional; A boolean to specify whether or not coefficients on the regressors (if provided) are allowed to vary over time. Default is False.
        mle_regression: Optional; A boolean to specify whether or not to estimate coefficients of regressors as part of maximum likelihood estimation or through Kalman filter.
                        If time_varying_regression is True, this must be set to False. Default is True.
        simple_differencing: Optional; A boolean to specify whether or not to use partially conditional maximum likelihood estimation.
                            See https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html for more details. Default is False.
        enforce_stationarity: Optional; A boolean to specify whether or not to transform the AR parameters to enforce stationarity in AR component. Default is True.
        enforce_invertibility: Optional; A boolean to specify whether or not to transform the MA parameters to enforce invertibility in MA component. Default is True.
        hamilton_representation: Optional; A boolean to specify whether or not to use the Hamilton representation or the Harvey representation (if False). Default is False.
        concentrate_scale: Optional; A boolean to specify whether or not to concentrate the scale (variance of the error term) out of the likelihood. Default is False.
        trend_offset: Optional; An integer for the offset at which to start time trend value. Default is 1.
        use_exact_diffuse: Optional; A boolean to specify whether or not to use exact diffuse initialization for non-stationary states. Default is False.
    """

    __slots__ = ["p", "d", "q"]

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        exog=None,
        seasonal_order: Tuple = (0, 0, 0, 0),
        trend=None,
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        simple_differencing: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        hamilton_representation: bool = False,
        concentrate_scale: bool = False,
        trend_offset: int = 1,
        use_exact_diffuse: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.exog = exog
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.use_exact_diffuse = use_exact_diffuse
        logging.debug(
            "Initialized SARIMAParams with parameters. "
            "p:{p}, d:{d}, q:{q},seasonal_order:{seasonal_order}".format(
                p=p, d=d, q=q, seasonal_order=seasonal_order
            )
        )

    def validate_params(self):
        """Not implemented."""

        logging.info("Method validate_params() is not implemented.")
        pass


class SARIMAModel(m.Model):
    """Model class for SARIMA.

    This class provides fit, predict and plot methods for SARIMA model.

    Attributes:
        data: :class:`kats.consts.TimeSeriesData` object for input time series.
        params: :class:`SARIMAParams` for model parameters.
    """

    def __init__(
        self,
        data: TimeSeriesData,
        params: SARIMAParams,
    ) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.start_params = None
        self.transformed = None
        self.includes_fixed = None
        self.cov_type = None
        self.cov_kwds = None
        self.method = None
        self.maxiter = None
        self.full_output = None
        self.disp = None
        self.callback = None
        self.return_params = None
        self.optim_score = None
        self.optim_complex_step = None
        self.optim_hessian = None
        self.low_memory = None
        self.model = None
        self.include_history = False
        self.alpha = 0.05
        self.fcst_df = None
        self.freq = None

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        transformed: bool = True,
        includes_fixed: bool = False,
        cov_type: Optional[str] = None,
        cov_kwds: Optional[Dict] = None,
        method: str = "lbfgs",
        maxiter: int = 50,
        full_output: bool = True,
        disp: bool = False,
        callback: Optional[Callable] = None,
        return_params: bool = False,
        optim_score: Optional[str] = None,
        optim_complex_step: bool = True,
        optim_hessian: Optional[str] = None,
        low_memory: bool = False,
    ) -> None:
        """Fit SARIMA model by maximum likelihood via Kalman filter.

        See reference https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html#statsmodels.tsa.statespace.sarimax.SARIMAX.fit for more details.

        Args:
            start_params: Optional; An array_like object for the initial guess of the solution for the loglikelihood maximization.
            transformed: Optional; A boolean to specify whether or not start_params is already transformed. Default is True.
            includes_fixed: Optional; A boolean to specify whether or not start_params includes the fixed parameters in addition to the free parameters. Default is False.
            cov_type: Optional; A string for the method for calculating the covariance matrix of parameter estimates. Can be 'opg' (outer product of gradient estimator), 'oim' (observed information matrix estimato),
                      'approx' (observed information matrix estimator), 'robust' (approximate (quasi-maximum likelihood) covariance matrix), or 'robust_approx'. Default is 'opg' when memory conservation is not used, otherwise default is ‘approx’.
            cov_kwds: Optional; A dictionary of arguments for covariance matrix computation. See reference for more details.
            method: Optional; A string for solver from scipy.optimize to be used. Can be 'newton', 'nm', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg' or 'basinhopping'. Default is 'lbfgs'.
            maxiter: Optional; An integer for the maximum number of iterations to perform. Default is 50.
            full_output: Optional; A boolean to specify whether or not to have all available output in the Results object’s mle_retvals attribute. Default is True.
            disp: Optional; A boolean to specify whether or not to print convergence messages. Default is False.
            callback: Optional; A callable object to be called after each iteration. Default is None.
            return_params: Optional; A boolean to specify whether or not to return only the array of maximizing parameters. Default is False.
            optim_score: Optional; A string for the method by which the score vector is calculated. Can be 'harvey', 'approx' or None. Default is None.
            optim_complex_step: Optional; A boolean to specify whether or not to use complex step differentiation when approximating the score. Default is True.
            optim_hessian: Optional; A string for the method by which the Hessian is numerically approximated. Can be 'opg', 'oim', 'approx'. Default is None.
            low_memory: Optional; A boolean to specify whether or not to reduce memory usage. If True, some features of the results object will not be available. Default is False.

        Returns:
            None.
        """

        logging.debug("Call fit() method")
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.low_memory = low_memory

        logging.info("Created SARIMA model.")
        sarima = SARIMAX(
            self.data.value,
            order=(self.params.p, self.params.d, self.params.q),
            exog=self.params.exog,
            seasonal_order=self.params.seasonal_order,
            trend=self.params.trend,
            measurement_error=self.params.measurement_error,
            time_varying_regression=self.params.time_varying_regression,
            mle_regression=self.params.mle_regression,
            simple_differencing=self.params.simple_differencing,
            enforce_stationarity=self.params.enforce_stationarity,
            enforce_invertibility=self.params.enforce_invertibility,
            hamilton_representation=self.params.hamilton_representation,
            concentrate_scale=self.params.concentrate_scale,
            trend_offset=self.params.trend_offset,
            use_exact_diffuse=self.params.use_exact_diffuse,
        )
        self.model = sarima.fit(
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            method=self.method,
            maxiter=self.maxiter,
            full_output=self.full_output,
            disp=self.disp,
            callback=self.callback,
            return_params=self.return_params,
            optim_score=self.optim_score,
            optim_complex_step=self.optim_complex_step,
            optim_hessian=self.optim_hessian,
            low_memory=self.low_memory,
        )
        logging.info("Fitted SARIMA.")

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self,
        steps: int,
        exog: Optional[ArrayLike] = None,
        include_history: bool = False,
        alpha: float = 0.05,
        **kwargs
    ) -> pd.DataFrame:
        """Predict with fitted SARIMA model.

        Args:
            steps: An integer for forecast steps.
            include_history: Optional; A boolearn to specify whether to include historical data. Default is False.
            alpha: A float for confidence level. Default is 0.05.
            exog: A numpy array of exogenous values to be passed to forecast

        Returns:
            A :class:`pandas.DataFrame` of forecasts and confidence intervals.
        """
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
        self.freq = kwargs.get("freq", self.data.infer_freq_robust())
        self.alpha = alpha

        if (self.params.exog is not None) and (exog is None):
            msg = "SARIMA model was initialized with exogenous variables. Exogenous variables must be used to predict. use `exog=`"
            logging.error(msg)
            raise ValueError(msg)

        fcst = self.model.get_forecast(steps, exog=exog)

        logging.info("Generated forecast data from SARIMA model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        if fcst.predicted_mean.isna().sum() == steps:
            msg = "SARIMA model fails to generate forecasts, i.e., all forecasts are NaNs."
            logging.error(msg)
            raise ValueError(msg)

        # pyre-fixme[16]: `SARIMAModel` has no attribute `y_fcst`.
        self.y_fcst = fcst.predicted_mean
        pred_interval = fcst.conf_int(alpha=self.alpha)

        if pred_interval.iloc[0, 0] < pred_interval.iloc[0, 1]:
            # pyre-fixme[16]: `SARIMAModel` has no attribute `y_fcst_lower`.
            self.y_fcst_lower = np.array(pred_interval.iloc[:, 0])
            # pyre-fixme[16]: `SARIMAModel` has no attribute `y_fcst_upper`.
            self.y_fcst_upper = np.array(pred_interval.iloc[:, 1])
        else:
            self.y_fcst_lower = np.array(pred_interval.iloc[:, 1])
            self.y_fcst_upper = np.array(pred_interval.iloc[:, 0])

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)

        # pyre-fixme[16]: `SARIMAModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            # generate historical fit
            history_fcst = self.model.get_prediction(0)
            history_ci = history_fcst.conf_int()
            if ("lower" in history_ci.columns[0]) and (
                "upper" in history_ci.columns[1]
            ):
                ci_lower_name, ci_upper_name = (
                    history_ci.columns[0],
                    history_ci.columns[1],
                )
            else:
                msg = (
                    "Error when getting prediction interval from statsmodels SARIMA API"
                )
                logging.error(msg)
                raise ValueError(msg)
            self.fcst_df = pd.DataFrame(
                {
                    "time": np.concatenate(
                        (pd.to_datetime(self.data.time), self.dates)
                    ),
                    "fcst": np.concatenate((history_fcst.predicted_mean, self.y_fcst)),
                    "fcst_lower": np.concatenate(
                        (history_ci[ci_lower_name], self.y_fcst_lower)
                    ),
                    "fcst_upper": np.concatenate(
                        (history_ci[ci_upper_name], self.y_fcst_upper)
                    ),
                }
            )

            # the first k elements of the fcst and lower/upper are not legitmate
            # thus we need to assign np.nan to avoid confusion
            # k = max(p, d, q) + max(P, D, Q) * seasonal_order + 1
            k = (
                max(self.params.p, self.params.d, self.params.q)
                + max(self.params.seasonal_order[0:3]) * self.params.seasonal_order[3]
                + 1
            )

            self.fcst_df.loc[0:k, ["fcst", "fcst_lower", "fcst_upper"]] = np.nan
        else:
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
        """Plot forecasted results from SARIMA model."""
        logging.info("Generating chart for forecast result from SARIMA model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "SARIMA"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Union[List[Any], bool, str]]]:
        """Get default SARIMA parameter search space.

        Returns:
            A dictionary representing the default SARIMA parameter search space.
        """
        return get_default_sarima_parameter_search_space()
