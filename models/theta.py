#!/usr/bin/env python3

'''
Implementation of theta model which is basically a simple
exponential smoothing model with drift.

For more details refer to: https://robjhyndman.com/papers/Theta.pdf
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import infrastrategy.kats.models.model as m
import numpy as np
import pandas as pd
import math
from copy import copy
from scipy.stats import norm  # @manual
from infrastrategy.kats.consts import Params, TimeSeriesData
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from infrastrategy.kats.utils.decomposition import TimeSeriesDecomposition
from typing import List, Dict
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_theta_parameter_search_space
)


class ThetaParams(Params):
    def __init__(self, m : int = 1) -> None:
        """
        Arguments:
        m: Number of observations before the seasonal pattern repeats
           For ex, m=12 for montly data with yearly seasonality
        """
        super().__init__()
        self.m = m
        logging.debug("Initialized ThetaParams instance.")

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class ThetaModel(m.Model):
    def __init__(self, data: TimeSeriesData = None, params: ThetaParams = None) -> None:
        super().__init__(data, params)
        self.n = None
        self.__subtype__ = 'theta'
        if self.data is not None:
            if not isinstance(self.data.value, pd.Series):
                msg = "Only support univariate time series, but get {type}.".format(
                    type=type(self.data.value)
                )
                logging.error(msg)
                raise ValueError(msg)
            self.n = self.data.value.shape[0]

    def check_seasonality(self) -> None:
        y = self.data.value
        m = self.params.m
        if (m > 1) and (y.nunique() > 1) and (self.n > 2 * m):
            r = acf(y, nlags=m)
            stat = np.sqrt((1 + 2 * np.sum(np.delete(r, [0 , m])**2)) / self.n)
            self.seasonal = (np.abs(r[m] / stat) > 1.644)
        else:
            self.seasonal = False

    def deseasonalize(self) -> TimeSeriesData:
        deseas_data = copy(self.data)
        decomp = None
        if (self.seasonal):
            decomp = TimeSeriesDecomposition(deseas_data, 'multiplicative').decomposer()
            if (abs(decomp['seasonal'].value) < 10**-10).sum():
                logging.info("Seasonal indexes equal to zero. Using non-seasonal Theta method")
            else:
                deseas_data.value = deseas_data.value / decomp['seasonal'].value
        self.decomp = decomp
        return deseas_data

    def fit(self, **kwargs) -> None:
        if self.n is None:
            self.n = self.data.value.shape[0]
        self.check_seasonality()
        deseas_data = self.deseasonalize()
        self.ses_model = SimpleExpSmoothing(deseas_data.value).fit()
        regr = np.vstack([np.arange(self.n), np.ones(self.n)]).T
        slope, _ = np.linalg.lstsq(regr, deseas_data.value.values)[0]
        self.drift = slope / 2

        if self.seasonal:
            self.fitted_values = (self.ses_model.fittedvalues
                                  * self.decomp['seasonal'].value)
        else:
            self.fitted_values = self.ses_model.fittedvalues
        self.residuals = self.data.value - self.fitted_values

        return self

    def predict(self, steps: int, include_history=False, **kwargs) -> pd.DataFrame:
        logging.debug("Call predict() with parameters. "
        "steps:{steps}, kwargs:{kwargs}".format(
            steps=steps, kwargs=kwargs
        ))
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.alpha = kwargs.get("alpha", 0.05)
        self.include_history = include_history
        # exp forecast
        fcst_ses = self.ses_model.forecast(steps)
        smoothing_level = max(1e-10, self.ses_model.params['smoothing_level'])
        # combine forecasts
        const = (1 - (1 - smoothing_level)**self.n) / smoothing_level
        fcst = fcst_ses + self.drift * (np.arange(steps) + const)
        # reseasonalize
        if self.seasonal:
            m = self.params.m
            rep = math.trunc(1 + steps / m)
            seasonality = self.decomp['seasonal'].value[-m:]
            self.y_fcst = fcst * np.tile(seasonality, rep)[:steps]
        else:
            self.y_fcst = fcst
        logging.info("Generated forecast data from theta model.")
        logging.debug("Forecast data: {fcst}".format(fcst=self.y_fcst))

        # prediction intervals
        # (Note: "this formula for se does not include the variation due
        # to estimation error and will therefore give intervals
        # which are too narrow", as stated in Hyndman et. al)
        p = 2  # 2 params: slope and level
        sigma2 = np.sqrt(self.ses_model.sse / (self.n - p))
        se = sigma2 * np.sqrt(np.arange(steps) * smoothing_level**2 + 1)
        zt = -norm.ppf(self.alpha / 2)
        self.y_fcst_lower = self.y_fcst - zt * se
        self.y_fcst_upper = self.y_fcst + zt * se

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)

        self.dates = dates[dates != last_date]  # Return correct number of periods

        if include_history:
            # generate historical fit
            self.fcst_df = pd.DataFrame(
                {
                    "time": np.concatenate((pd.to_datetime(self.data.time), self.dates)),
                    "fcst": np.concatenate((self.fitted_values, self.y_fcst)),
                    "fcst_lower": np.concatenate((self.fitted_values-zt*sigma2, self.y_fcst_lower)),
                    "fcst_upper": np.concatenate((self.fitted_values+zt*sigma2, self.y_fcst_upper)),
                }
            )
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
        logging.info("Generating chart for forecast result from theta model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "Theta"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of arima
        to avoid the massive dependencies of arima and huge build size.
        Check https://fburl.com/kg04hx5y for detail.
        """
        return get_default_theta_parameter_search_space()
