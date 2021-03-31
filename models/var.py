#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List

import infrastrategy.kats.models.model as m
import pandas as pd
from infrastrategy.kats.consts import Params, TimeSeriesData
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_var_parameter_search_space
)


class VARParams(Params):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.maxlags = kwargs.get("maxlags", None)
        self.method = kwargs.get("method", "ols")
        self.ic = kwargs.get("ic", None)
        self.verbose = kwargs.get("verbose", False)
        self.trend = kwargs.get("trend", "c")
        logging.debug("Initialized VARParam instance.:{kwargs}".format(kwargs=kwargs))

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class VARModel(m.Model):
    def __init__(self, data: TimeSeriesData, params: VARParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.DataFrame):
            msg = "Only support multivariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs) -> None:
        logging.debug("Call fit()")
        if self.params.maxlags is None:
            self.params.maxlags = int(12 * (len(self.data.time) / 100.0) ** (1.0 / 4))

        # create VAR model
        var = VAR(self.data.value)
        logging.info("Created VAR model.")

        # fit VAR model
        self.model = var.fit(
            maxlags=self.params.maxlags,
            method=self.params.method,
            ic=self.params.ic,
            verbose=self.params.verbose,
            trend=self.params.trend,
        )
        logging.info("Fitted VAR model.")

        self.k_ar = self.model.k_ar
        self.sigma_u = self.model.sigma_u
        self.resid = self.model.resid

    def predict(self, steps: int, **kwargs) -> Dict[str, TimeSeriesData]:
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.freq = kwargs.get("freq", "D")
        self.alpha = kwargs.get("alpha", 0.05)

        fcst = self.model.forecast_interval(
            y=self.model.y, steps=steps, alpha=self.alpha
        )
        logging.info("Generated forecast data from VAR model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.fcst_dict = {}
        ts_names = list(self.data.value.columns)

        for i, name in enumerate(ts_names):
            fcst_df = pd.DataFrame(
                {
                    "time": self.dates,
                    "fcst": fcst[0][:, i],
                    "fcst_lower": fcst[1][:, i],
                    "fcst_upper": fcst[2][:, i],
                }
            )
            self.fcst_dict[name] = fcst_df

        logging.debug(
            "Return forecast data: {fcst_dict}".format(fcst_dict=self.fcst_dict)
        )
        ret = {k: TimeSeriesData(v) for k, v in self.fcst_dict.items()}
        return ret

    def plot(self) -> None:
        logging.info("Generating chart for forecast result from VAR model.")

        fig, axes = plt.subplots(ncols=2, dpi=120, figsize=(10, 6))
        for i, ax in enumerate(axes.flatten()):
            ts_name = list(self.fcst_dict.keys())[i]
            data = self.fcst_dict[ts_name]
            ax.plot(pd.to_datetime(self.data.time), self.data.value[ts_name], "k")
            fcst_dates = self.dates.to_pydatetime()
            ax.plot(fcst_dates, data.fcst, ls="-", c="#4267B2")

            ax.fill_between(
                fcst_dates, data.fcst_lower, data.fcst_upper, color="#4267B2", alpha=0.2
            )

            ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
            ax.set_xlabel(xlabel="time")
            ax.set_ylabel(ylabel=ts_name)

        plt.tight_layout()

    def __str__(self):
        return "VAR"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of var
        to avoid the massive dependencies of var and huge build size.
        Check https://fburl.com/kg04hx5y for detail.
        """
        return get_default_var_parameter_search_space()
