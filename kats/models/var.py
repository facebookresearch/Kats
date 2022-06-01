# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import get_default_var_parameter_search_space
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults


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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.maxlags: int = kwargs.get("maxlags", None)
        self.method: str = kwargs.get("method", "ols")
        self.ic: str = kwargs.get("ic", None)
        self.verbose: bool = kwargs.get("verbose", False)
        self.trend: str = kwargs.get("trend", "c")
        logging.debug("Initialized VARParam instance.:{kwargs}".format(kwargs=kwargs))

    def validate_params(self) -> None:
        """Validate the parameters for VAR model"""

        logging.info("Method validate_params() is not implemented.")
        pass


class VARModel(Model[VARParams]):
    """Model class for VAR

    This class provides fit, predict, and plot methods for VAR model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `VARParams`
    """

    model: Optional[VARResults] = None
    k_ar: Optional[int] = None
    sigma_u: Optional[np.ndarray] = None
    resid: Optional[np.ndarray] = None
    freq: Optional[str] = None
    alpha: Optional[float] = None
    dates: Optional[pd.DatetimeIndex] = None
    fcst_dict: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(self, data: TimeSeriesData, params: VARParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.DataFrame):
            msg = "Only support multivariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs: Any) -> None:
        """Fit VAR model"""

        logging.debug("Call fit()")
        if self.params.maxlags is None:
            # pyre-fixme[16]: `Optional` has no attribute `time`.
            self.params.maxlags = int(12 * (len(self.data.time) / 100.0) ** (1.0 / 4))

        # create VAR model
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        var = VAR(self.data.value)
        logging.info("Created VAR model.")

        # fit VAR model
        self.model = model = var.fit(
            maxlags=self.params.maxlags,
            method=self.params.method,
            ic=self.params.ic,
            verbose=self.params.verbose,
            trend=self.params.trend,
        )
        logging.info("Fitted VAR model.")

        self.k_ar = model.k_ar
        self.sigma_u = model.sigma_u
        self.resid = model.resid

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs: Any
    ) -> Dict[str, TimeSeriesData]:
        """Predict with the fitted VAR model

        Args:
            steps: Number of time steps to forecast
            include_history: optional, A boolearn to specify whether to include
                historical data. Default is False.
            freq: optional, frequency of timeseries data.
                Defaults to automatically inferring from time index.
            alpha: optional, significance level of confidence interval.
                Defaults to 0.05

        Returns:
            Disctionary of predicted results for each metric. Each metric result
            has following columns: `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.include_history = include_history
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.alpha = alpha = kwargs.get("alpha", 0.05)

        fcst = model.forecast_interval(y=model.y, steps=steps, alpha=alpha)
        logging.info("Generated forecast data from VAR model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        dates = dates[1:]  # Return correct number of periods
        self.dates = dates

        self.fcst_dict = fcst_dict = {}
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        ts_names = list(self.data.value.columns)

        for i, name in enumerate(ts_names):
            fcst_df = pd.DataFrame(
                {
                    "time": dates,
                    "fcst": fcst[0][:, i],
                    "fcst_lower": fcst[1][:, i],
                    "fcst_upper": fcst[2][:, i],
                },
                copy=False,
            )
            fcst_dict[name] = fcst_df

        if self.include_history:
            try:
                hist_fcst = model.fittedvalues.values
                hist_dates = self.data.time.iloc[-len(hist_fcst) :]
                for i, name in enumerate(ts_names):
                    print(
                        pd.DataFrame(
                            {"time": hist_dates, "fcst": hist_fcst[:, i]}, copy=False
                        )
                    )
                    fcst_df = pd.concat(
                        [
                            pd.DataFrame(
                                {"time": hist_dates, "fcst": hist_fcst[:, i]},
                                copy=False,
                            ),
                            fcst_dict[name],
                        ],
                        copy=False,
                    )
                    fcst_dict[name] = fcst_df

            except Exception as e:
                msg = (
                    "Failed to generate in-sample forecasts for historical data "
                    f"with error message {e}."
                )
                logging.error(msg)
                raise ValueError(msg)

        logging.debug(
            "Return forecast data: {fcst_dict}".format(fcst_dict=self.fcst_dict)
        )
        ret = {k: TimeSeriesData(v) for k, v in fcst_dict.items()}
        return ret

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 120,
        forecast_color: str = "#4267B2",
        history_color: str = "k",
        grid: bool = True,
        xlabel: str = "time",
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot forecasted results from VAR model"""
        fcst_dict = self.fcst_dict
        if fcst_dict is None:
            raise ValueError("Call predict() before plot().")
        if ax is not None:
            raise ValueError("VARModel does not support the ax parameter.")
        dates = self.dates
        assert dates is not None
        fcst_dates = dates.to_pydatetime()
        logging.info("Generating chart for forecast result from VAR model.")

        if figsize is None:
            figsize = (10, 6)
        fig, axes = plt.subplots(ncols=2, dpi=dpi, figsize=figsize)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        for ts_name, ax in zip(self.data.value.columns, axes.flat):
            # pyre-fixme[16]: `Optional` has no attribute `time`.
            ax.plot(self.data.time, self.data.value[ts_name], history_color)

            data = fcst_dict[ts_name]
            ax.plot(fcst_dates, data["fcst"], ls="-", c=forecast_color)
            ax.fill_between(
                fcst_dates,
                data["fcst_lower"],
                data["fcst_upper"],
                color=forecast_color,
                alpha=0.2,
            )

            if grid:
                ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
            ax.set_xlabel(xlabel=xlabel)
            ax.set_ylabel(ylabel=ts_name)

        fig.set_tight_layout(True)
        return axes

    def __str__(self) -> str:
        """VAR model as a string

        Returns:
            String representation of the model name
        """

        return "VAR"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Provide a parameter space for VAR model

        Move the implementation of get_parameter_search_space() out of var
        to avoid the massive dependencies of var and huge build size.
        """

        return get_default_var_parameter_search_space()
