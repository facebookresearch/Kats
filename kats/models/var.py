# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""VAR forecasting Model

VAR model is a multivariate extension of the univariate autoregressive (AR) model.
It captures the linear interdependencies between multiple variables using a system of equations.
Each variable depends not only on its own lagged values but also on the lagged values of other variables.
We use the implementation in statsmodels and re-write the API to adapt Kats development style.

    Typical usage example:

    params = VARParams()
    m = VARModel(data=TSData_multi, params=params)
    m.fit()
    res = m.predict(steps=30)
    m.plot()
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List

import kats.models.model as m
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.parameter_tuning_utils import get_default_var_parameter_search_space
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR


class VARParams(Params):
    """Parameter class for VAR model

    This is the parameter class for VAR forecasting model which stands
    for Vector Autoregressive Model.

    Attributes:
        maxlags: Maximum number of lags to check for order selection,
                 Defaults to 12 * (nobs/100.)**(1./4)
        method: Estimation method to use
                Defaults to OLS
        ic: Information criterion to use for VAR order selection
            Defaults to None
        trend: “c” - add constant (Default),
            “ct” - constant and trend,
            “ctt” - constant, linear and quadratic trend,
            “n”/“nc” - no constant, no trend
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.maxlags = kwargs.get("maxlags", None)
        self.method = kwargs.get("method", "ols")
        self.ic = kwargs.get("ic", None)
        self.verbose = kwargs.get("verbose", False)
        self.trend = kwargs.get("trend", "c")
        logging.debug("Initialized VARParam instance.:{kwargs}".format(kwargs=kwargs))

    def validate_params(self):
        """Validate the parameters for VAR model"""

        logging.info("Method validate_params() is not implemented.")
        pass


class VARModel(m.Model):
    """Model class for VAR

    This class provides fit, predict, and plot methods for VAR model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `VARParams`
    """

    def __init__(self, data: TimeSeriesData, params: VARParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.DataFrame):
            msg = "Only support multivariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.include_history = False

    def fit(self, **kwargs) -> None:
        """Fit VAR model"""

        logging.debug("Call fit()")
        if self.params.maxlags is None:
            self.params.maxlags = int(12 * (len(self.data.time) / 100.0) ** (1.0 / 4))

        # create VAR model
        var = VAR(self.data.value)
        logging.info("Created VAR model.")

        # fit VAR model
        # pyre-fixme[16]: `VARModel` has no attribute `model`.
        self.model = var.fit(
            maxlags=self.params.maxlags,
            method=self.params.method,
            ic=self.params.ic,
            verbose=self.params.verbose,
            trend=self.params.trend,
        )
        logging.info("Fitted VAR model.")

        # pyre-fixme[16]: `VARModel` has no attribute `k_ar`.
        self.k_ar = self.model.k_ar
        # pyre-fixme[16]: `VARModel` has no attribute `sigma_u`.
        self.sigma_u = self.model.sigma_u
        # pyre-fixme[16]: `VARModel` has no attribute `resid`.
        self.resid = self.model.resid

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs
    ) -> Dict[str, TimeSeriesData]:
        """Predict with the fitted VAR model

        Args:
            steps: Number of time steps to forecast
            include_history: optional, A boolearn to specify whether to include historical data. Default is False.
            freq: optional, frequency of timeseries data.
                Defaults to automatically inferring from time index.
            alpha: optional, significance level of confidence interval.
                Defaults to 0.05

        Returns:
            Disctionary of predicted results for each metric. Each metric result
            has following columns: `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
        # pyre-fixme[16]: `VARModel` has no attribute `freq`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        # pyre-fixme[16]: `VARModel` has no attribute `alpha`.
        self.alpha = kwargs.get("alpha", 0.05)

        # pyre-fixme[16]: `VARModel` has no attribute `model`.
        fcst = self.model.forecast_interval(
            y=self.model.y, steps=steps, alpha=self.alpha
        )
        logging.info("Generated forecast data from VAR model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        dates = dates[1:]  # Return correct number of periods
        # pyre-fixme[16]: `VARModel` has no attribute `dates`.
        self.dates = dates

        # pyre-fixme[16]: `VARModel` has no attribute `fcst_dict`.
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

        if self.include_history:
            try:
                hist_fcst = self.model.fittedvalues.values
                hist_dates = self.data.time.iloc[-len(hist_fcst) :]
                for i, name in enumerate(ts_names):
                    print(pd.DataFrame({"time": hist_dates, "fcst": hist_fcst[:, i]}))
                    fcst_df = pd.concat(
                        [
                            pd.DataFrame({"time": hist_dates, "fcst": hist_fcst[:, i]}),
                            self.fcst_dict[name],
                        ]
                    )
                    self.fcst_dict[name] = fcst_df

            except Exception as e:
                msg = f"Fail to generate in-sample forecasts for historical data with error message {e}."
                logging.error(msg)
                raise ValueError(msg)

        logging.debug(
            "Return forecast data: {fcst_dict}".format(fcst_dict=self.fcst_dict)
        )
        ret = {k: TimeSeriesData(v) for k, v in self.fcst_dict.items()}
        return ret

    # pyre-fixme[14]: `plot` overrides method defined in `Model` inconsistently.
    # pyre-fixme[40]: Non-static method `plot` cannot override a static method
    #  defined in `m.Model`.
    def plot(self) -> None:
        """Plot forecasted results from VAR model"""
        logging.info("Generating chart for forecast result from VAR model.")

        fig, axes = plt.subplots(ncols=2, dpi=120, figsize=(10, 6))
        for i, ax in enumerate(axes.flatten()):
            # pyre-fixme[16]: `VARModel` has no attribute `fcst_dict`.
            ts_name = list(self.fcst_dict.keys())[i]
            data = self.fcst_dict[ts_name]
            ax.plot(pd.to_datetime(self.data.time), self.data.value[ts_name], "k")
            # pyre-fixme[16]: `VARModel` has no attribute `dates`.
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
        """VAR model as a string

        Returns:
            String representation of the model name
        """

        return "VAR"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Provide a parameter space for VAR model

        Move the implementation of get_parameter_search_space() out of var
        to avoid the massive dependencies of var and huge build size.
        """

        return get_default_var_parameter_search_space()
