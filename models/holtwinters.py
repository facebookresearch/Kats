#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List

import infrastrategy.kats.models.model as m
import pandas as pd
import numpy as np
from infrastrategy.kats.consts import Params, TimeSeriesData
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWinters
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_holtwinters_parameter_search_space
)


class HoltWintersParams(Params):
    __slots__ = ["trend", "damped", "seasonal", "seasonal_periods"]

    def __init__(
        self,
        trend: str = "add",
        damped: bool = False,
        seasonal: str = None,
        seasonal_periods: int = None,
    ) -> None:
        super().__init__()
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.validate_params()
        logging.debug(
            "Initialized HoltWintersParams with parameters. "
            "trend:{trend},\
            damped:{damped},\
            seasonal:{seasonal},\
            seasonal_periods{seasonal_periods}".format(
                trend=trend,
                damped=damped,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        )

    def validate_params(self):
        if self.trend not in ["add", "mul", "additive", "multiplicative", None]:
            msg = "trend parameter is not valid!\
                         use 'add' or 'mul' instead!"
            logging.error(msg)
            raise ValueError(msg)

        if self.seasonal not in ["add", "mul", "additive", "multiplicative", None]:
            msg = "seasonal parameter is not valid!\
                         use 'add' or 'mul' instead!"
            logging.error(msg)
            raise ValueError(msg)


class HoltWintersModel(m.Model):
    def __init__(self, data: TimeSeriesData, params: HoltWintersParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs) -> None:
        logging.debug("Call fit() with parameters:{kwargs}".format(kwargs=kwargs))
        holtwinters = HoltWinters(
            self.data.value,
            trend=self.params.trend,
            damped=self.params.damped,
            seasonal=self.params.seasonal,
            seasonal_periods=self.params.seasonal_periods,
        )
        self.model = holtwinters.fit()
        return self.model

    def predict(self, steps: int, include_history=False, **kwargs) -> pd.DataFrame:
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
        if "freq" not in kwargs:
            self.freq = pd.infer_freq(self.data.time)
        else:
            self.freq = kwargs["freq"]
        fcst = self.model.forecast(steps)
        logging.info("Generated forecast data from Holt-Winters model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods
        self.y_fcst = fcst

        if include_history:
            history_fcst = self.model.predict(start=0, end=len(self.data.time) + steps - 1)
            self.fcst_df = pd.DataFrame(
                {
                    "time": np.concatenate((pd.to_datetime(self.data.time), self.dates)),
                    "fcst": history_fcst,
                }
            )
        else:
            self.fcst_df = pd.DataFrame({"time": self.dates, "fcst": fcst})

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        logging.info("Generating chart for forecast result from arima model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "HoltWinters"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of holtwinters
        to avoid the massive dependencies of holtwinters and huge build size.
        Check https://fburl.com/kg04hx5y for detail.
        """
        return get_default_holtwinters_parameter_search_space()
