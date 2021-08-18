# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of theta model which is basically a simple
exponential smoothing model with drift.

For more details refer to: https://robjhyndman.com/papers/Theta.pdf
"""

from __future__ import annotations

import logging
import math
from copy import copy
from typing import Any, Dict, List, Optional

import kats.models.model as m
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.decomposition import TimeSeriesDecomposition
from kats.utils.parameter_tuning_utils import get_default_theta_parameter_search_space
from scipy.stats import norm  # @manual
from statsmodels.tsa.holtwinters import HoltWintersResults, SimpleExpSmoothing
from statsmodels.tsa.stattools import acf


class ThetaParams(Params):
    """Parameter class for Theta model

    This is the parameter class for Theta forecasting model.

    Attributes:
        m: Number of observations before the seasonal pattern repeats
           For ex, m=12 for montly data with yearly seasonality
    """

    def __init__(self, m: int = 1) -> None:
        super().__init__()
        self.m = m
        logging.debug("Initialized ThetaParams instance.")

    def validate_params(self) -> None:
        """Validate the parameters for theta model"""

        logging.info("Method validate_params() is not implemented.")
        pass


class ThetaModel(m.Model):
    """Model class for Theta

    This class provides fit, predict, and plot methods for STLF model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `ThetaParams`
    """

    seasonal: bool = False
    decomp: Optional[Dict[str, TimeSeriesData]] = None
    ses_model: Optional[HoltWintersResults] = None
    drift: Optional[float] = None
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    fcst_df: Optional[pd.DataFrame] = None
    dates: Optional[pd.DatetimeIndex] = None
    y_fcst: Optional[np.ndarray] = None
    y_fcst_lower: Optional[np.ndarray] = None
    y_fcst_upper: Optional[np.ndarray] = None
    freq: Optional[str] = None
    alpha: Optional[float] = None
    include_history: bool = False

    def __init__(
        self,
        data: Optional[TimeSeriesData] = None,
        params: Optional[ThetaParams] = None,
    ) -> None:
        super().__init__(data, params)
        self.n = None
        self.__subtype__ = "theta"
        if self.data is not None:
            if not isinstance(self.data.value, pd.Series):
                msg = "Only support univariate time series, but get {type}.".format(
                    type=type(self.data.value)
                )
                logging.error(msg)
                raise ValueError(msg)
            self.n = self.data.value.shape[0]

    def check_seasonality(self) -> None:
        """Determine if the metirc to be forecasted is seasonal or not"""

        y = self.data.value
        m = self.params.m
        if (m > 1) and (y.nunique() > 1) and (self.n > 2 * m):
            r = acf(y, nlags=m)
            stat = np.sqrt((1 + 2 * np.sum(np.delete(r, [0, m]) ** 2)) / self.n)
            self.seasonal = np.abs(r[m] / stat) > 1.644
        else:
            self.seasonal = False

    def deseasonalize(self) -> TimeSeriesData:
        """Returns the deseasonalized input time series"""

        deseas_data = copy(self.data)
        decomp = None
        if self.seasonal:
            decomp = TimeSeriesDecomposition(deseas_data, "multiplicative").decomposer()
            if (abs(decomp["seasonal"].value) < 10 ** -10).sum():
                logging.info(
                    "Seasonal indexes equal to zero. Using non-seasonal Theta method"
                )
            else:
                deseas_data.value = deseas_data.value / decomp["seasonal"].value
        self.decomp = decomp
        return deseas_data

    def fit(self, **kwargs) -> ThetaModel:
        """Fit Theta model"""
        if self.n is None:
            self.n = self.data.value.shape[0]
        self.check_seasonality()
        deseas_data = self.deseasonalize()
        self.ses_model = ses_model = SimpleExpSmoothing(deseas_data.value).fit()
        # creating x and intercept variables to fit a straight line
        regr = np.vstack([np.arange(self.n), np.ones(self.n)]).T
        slope, _ = np.linalg.lstsq(regr, deseas_data.value.values)[0]
        self.drift = slope / 2

        if self.seasonal:
            decomp = self.decomp
            if decomp is None:
                raise ValueError("seasonal data must be deseasonalized before fit.")

            self.fitted_values = ses_model.fittedvalues * decomp["seasonal"].value
        else:
            self.fitted_values = ses_model.fittedvalues
        self.residuals = self.data.value - self.fitted_values

        return self

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self,
        steps: int,
        include_history: bool = False,
        freq: Optional[str] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Predict with the fitted Theta model

        Args:
            steps: Number of time steps to forecast
            include_history: If True, fitted values for training data are also returned
            freq: optional, frequency of timeseries data. If None, infer from
                time index.
            alpha: optional, significance level of confidence interval.

        Returns:
            DataFrame of predicted results with following columns:
            `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        ses_model = self.ses_model
        if ses_model is None:
            raise ValueError("fit must be called before predict.")

        logging.debug(
            f"Call predict(steps={steps}, include_history={include_history},"
            f"freq={freq}, alpha={alpha})"
        )
        if freq is None:
            freq = pd.infer_freq(self.data.time)
        self.freq = freq
        self.alpha = alpha
        self.include_history = include_history
        # exp forecast
        fcst_ses = ses_model.forecast(steps)
        smoothing_level = max(1e-10, ses_model.params["smoothing_level"])
        # combine forecasts
        const = (1 - (1 - smoothing_level) ** self.n) / smoothing_level
        drift = self.drift
        assert drift is not None
        fcst = fcst_ses + drift * (np.arange(steps) + const)
        # reseasonalize
        if self.seasonal:
            decomp = self.decomp
            assert decomp is not None
            m = self.params.m
            rep = math.trunc(1 + steps / m)
            seasonality = decomp["seasonal"].value[-m:]
            self.y_fcst = y_fcst = fcst * np.tile(seasonality, rep)[:steps]
        else:
            self.y_fcst = y_fcst = fcst
        logging.info("Generated forecast data from theta model.")
        logging.debug("Forecast data: {fcst}".format(fcst=self.y_fcst))

        # prediction intervals
        # (Note: "this formula for se does not include the variation due
        # to estimation error and will therefore give intervals
        # which are too narrow", as stated in Hyndman et. al)
        p = 2  # 2 params: slope and level
        # pyre-ignore[16]: `HoltWintersResults` has no attribute `sse`.
        sigma2 = np.sqrt(ses_model.sse / (self.n - p))
        se = sigma2 * np.sqrt(np.arange(steps) * smoothing_level ** 2 + 1)
        zt = -norm.ppf(alpha / 2)
        self.y_fcst_lower = y_fcst - zt * se
        self.y_fcst_upper = y_fcst + zt * se

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)

        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            fitted_values = self.fitted_values
            assert fitted_values is not None
            # generate historical fit
            fcst_df = pd.DataFrame(
                {
                    "time": np.concatenate(
                        (pd.to_datetime(self.data.time), self.dates)
                    ),
                    "fcst": np.concatenate((fitted_values, self.y_fcst)),
                    "fcst_lower": np.concatenate(
                        (fitted_values - zt * sigma2, self.y_fcst_lower)
                    ),
                    "fcst_upper": np.concatenate(
                        (fitted_values + zt * sigma2, self.y_fcst_upper)
                    ),
                }
            )
        else:
            fcst_df = pd.DataFrame(
                {
                    "time": self.dates,
                    "fcst": self.y_fcst,
                    "fcst_lower": self.y_fcst_lower,
                    "fcst_upper": self.y_fcst_upper,
                }
            )
        self.fcst_df = fcst_df
        logging.debug(f"Return forecast data: {fcst_df}")
        return fcst_df

    def plot(self):
        """Plot forecasted results from Theta model"""

        logging.info("Generating chart for forecast result from theta model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self) -> str:
        """Theta model as a string

        Returns:
            String representation of the model name
        """

        return "Theta"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Provide a parameter space for Theta model

        Move the implementation of get_parameter_search_space() out of arima
        to avoid the massive dependencies of arima and huge build size.
        """
        return get_default_theta_parameter_search_space()
