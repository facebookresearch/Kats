#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from statsmodels.tsa.seasonal import STL, seasonal_decompose


class TimeSeriesDecomposition:
    """Model class for Time Series Decomposition.

    This class provides utilities to decompose an input time series

    Attributes:
        data: the input time series data as `TimeSeriesData`
        decomposition: `additive` or `multiplicative` decomposition
        method: `STL decompostion` or `seasonal_decompose`
    Specific arguments to seasonal_decompose and STL functions can be passed via kwargs
    """

    def __init__(
        self, data: TimeSeriesData, decomposition="additive", method="STL", **kwargs
    ) -> None:
        self.data = data
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        if decomposition in ("additive", "multiplicative"):
            self.decomposition = decomposition
        else:
            logging.info("Invalid decomposition setting specified")
            logging.info("Defaulting to Additive Decomposition")
            self.decomposition = "additive"
        if method in ("STL", "seasonal_decompose"):
            self.method = method
        else:
            logging.info("Invalid decomposition setting specified")
            logging.info("Possible Values: STL, seasonal_decompose")
            logging.info("Defaulting to STL")
            self.method = "STL"

        ## The following are params for the STL Module
        self.period = kwargs.get("period", None)
        self.seasonal = kwargs.get("seasonal", 7)
        self.trend = kwargs.get("trend", None)
        self.low_pass = kwargs.get("low_pass", None)
        self.seasonal_deg = kwargs.get("seasonal_deg", 1)
        self.trend_deg = kwargs.get("trend_deg", 1)
        self.low_pass_deg = kwargs.get("low_pass_deg", 1)
        self.robust = kwargs.get("robust", False)
        self.seasonal_jump = kwargs.get("seasonal_jump", 1)
        self.trend_jump = kwargs.get("trend_jump", 1)
        self.low_pass_jump = kwargs.get("low_pass_jump", 1)

    def __clean_ts(self):
        """Internal function to clean the time series.

        Internal function to interpolate time series and infer frequency of time series required for decomposition
        """

        original = pd.DataFrame(
            list(self.data.value), index=self.data.time, columns=["y"]
        )

        original.columns = ["y"]

        original.index = pd.to_datetime(original.index)

        if pd.infer_freq(original.index) is None:
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")

        self.freq = pd.infer_freq(original.index)

        original = original.interpolate(
            method="polynomial", limit_direction="both", order=3
        )

        ## This is a hack since polynomial interpolation is not working here
        if sum((np.isnan(x) for x in original["y"])):
            original = original.interpolate(method="linear", limit_direction="both")

        return original

    def __decompose_seasonal(self, original):
        """Internal function to call seasonal_decompose to do the decomposition."""
        if self.period is not None:
            result = seasonal_decompose(
                original, model=self.decomposition, period=self.period
            )
        else:
            if "T" in self.freq:
                result = seasonal_decompose(
                    original, model=self.decomposition, period=2
                )
                logging.warning(
                    "Seasonal Decompose cannot handle sub day level granularity"
                )
                logging.warning(
                    "Please consider setting period yourself based on the input data"
                )
                logging.warning("Defaulting to a period of 2")
            else:
                result = seasonal_decompose(original, model=self.decomposition)

        output = {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }

        return output

    def __decompose_STL(self, original):
        """Internal function to call STL to do the decomposition.

        The arguments to STL can be passed in the class via kwargs
        """
        if "T" in self.freq and self.period is None:
            logging.warning("STL cannot handle sub day level granularity")
            logging.warning(
                "Please consider setting period yourself based on the input data"
            )
            logging.warning("Defaulting to a period of 2")
            self.period = 2
        if self.decomposition == "additive":
            result = STL(
                original,
                period=self.period,
                seasonal=self.seasonal,
                trend=self.trend,
                low_pass=self.low_pass,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                low_pass_deg=self.low_pass_deg,
                robust=self.robust,
                seasonal_jump=self.seasonal_jump,
                trend_jump=self.trend_jump,
                low_pass_jump=self.low_pass_jump,
            ).fit()
            output = {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
            }
        else:
            if np.any(original <= 0):
                logging.error(
                    "Multiplicative seasonality is not appropriate "
                    "for zero and negative values"
                )
            original_transformed = np.log(original)
            result = STL(
                original_transformed,
                period=self.period,
                seasonal=self.seasonal,
                trend=self.trend,
                low_pass=self.low_pass,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                low_pass_deg=self.low_pass_deg,
                robust=self.robust,
                seasonal_jump=self.seasonal_jump,
                trend_jump=self.trend_jump,
                low_pass_jump=self.low_pass_jump,
            ).fit()
            output = {
                "trend": np.exp(result.trend),
                "seasonal": np.exp(result.seasonal),
                "resid": np.exp(result.resid),
            }

        return output

    def __decompose(self, original):

        if self.method == "STL":
            output = self.__decompose_STL(original)
        else:
            output = self.__decompose_seasonal(original)

        return {
            "trend": TimeSeriesData(
                output["trend"].reset_index(), time_col_name=self.data.time_col_name
            ),
            "seasonal": TimeSeriesData(
                output["seasonal"].reset_index(), time_col_name=self.data.time_col_name
            ),
            "rem": TimeSeriesData(
                output["resid"].reset_index(), time_col_name=self.data.time_col_name
            ),
        }

    def decomposer(self):
        """Decompose the time series.

        Args:
            None.

        Returns:
            A dictionary with three time series for the three components:
            `trend` : Trend
            `seasonal` : Seasonality, and
            `rem` : Residual
        """
        original = self.__clean_ts()
        self.results = self.__decompose(original)

        return self.results

    def plot(self):
        """Plot the original time series and the three decomposed components."""

        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 10), sharex=True)

        ax[0].plot(
            self.data.time.values,
            self.data.value.values,
            linewidth=3,
        )
        ax[0].set_title("Original Time Series")
        ax[1].plot(
            self.results["trend"].time.values,
            self.results["trend"].value.values,
            linewidth=3,
        )
        ax[1].set_title("Trend")

        ax[2].plot(
            self.results["seasonal"].time.values,
            self.results["seasonal"].value.values,
            linewidth=3,
        )
        ax[2].set_title("Seasonality")

        ax[3].plot(
            self.results["rem"].time.values,
            self.results["rem"].value.values,
            linewidth=3,
        )
        ax[3].set_title("Residual")
        ax[3].set_xlabel("Time")
        plt.subplots_adjust(hspace=0.2)
        return fig, ax
