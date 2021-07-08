# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from statsmodels.tsa.seasonal import STL, seasonal_decompose

# from numpy.typing import ArrayLike
ArrayLike = Union[np.ndarray, Sequence[float]]
Figsize = Tuple[int, int]


def _identity(x: ArrayLike) -> ArrayLike:
    return x


class TimeSeriesDecomposition:
    """Model class for Time Series Decomposition.

    This class provides utilities to decompose an input time series

    Pass specific arguments to seasonal_decompose and STL functions via kwargs.

    Attributes:
        data: the input time series data as `TimeSeriesData`
        decomposition: `additive` or `multiplicative` decomposition
        method: `STL decompostion` or `seasonal_decompose`
    """

    freq: Optional[str] = None
    results: Optional[Dict[str, TimeSeriesData]] = None

    def __init__(
        self,
        data: TimeSeriesData,
        decomposition: str = "additive",
        method: str = "STL",
        **kwargs,
    ) -> None:
        if not isinstance(data.value, pd.Series):
            msg = f"Only support univariate time series, but got {type(data.value)}."
            logging.error(msg)
            raise ValueError(msg)
        self.data = data
        if decomposition in ("additive", "multiplicative"):
            self.decomposition = decomposition
        else:
            logging.info(
                "Invalid decomposition setting specified; "
                "defaulting to Additive Decomposition."
            )
            self.decomposition = "additive"
        if method == "seasonal_decompose":
            self.method = self.__decompose_seasonal
        else:
            if method != "STL":
                logging.info(
                    f"""Invalid decomposition setting {method} specified.
                    Possible Values: STL, seasonal_decompose.
                    Defaulting to STL."""
                )
            self.method = self.__decompose_STL

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

    def __clean_ts(self) -> pd.DataFrame:
        """Internal function to clean the time series.

        Internal function to interpolate time series and infer frequency of
        time series required for decomposition.
        """

        original = pd.DataFrame(
            list(self.data.value), index=pd.to_datetime(self.data.time), columns=["y"]
        )

        if pd.infer_freq(original.index) is None:
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")

        self.freq = pd.infer_freq(original.index)

        original.interpolate(
            method="polynomial", limit_direction="both", order=3, inplace=True
        )

        ## This is a hack since polynomial interpolation is not working here
        if any(original["y"].isna()):
            original.interpolate(method="linear", limit_direction="both", inplace=True)

        # pyre-ignore[7]: Expected `DataFrame` but got
        #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
        return original

    def _get_period(self) -> Optional[int]:
        period = self.period
        freq = self.freq
        if period is None:
            if freq is not None and "T" in freq:
                logging.warning(
                    """Seasonal Decompose cannot handle sub day level granularity.
                    Please consider setting period yourself based on the input data.
                    Defaulting to a period of 2."""
                )
                period = 2
        return period

    def __decompose_seasonal(self, original: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Internal function to call seasonal_decompose to do the decomposition."""
        period = self._get_period()
        result = seasonal_decompose(original, model=self.decomposition, period=period)

        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "rem": result.resid,
        }

    def __decompose_STL(self, original: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Internal function to call STL to do the decomposition.

        The arguments to STL can be passed in the class via kwargs
        """
        self.period = period = self._get_period()

        if self.decomposition == "additive":
            data = original
            post_transform = _identity
        else:
            if np.any(original <= 0):
                logging.error(
                    "Multiplicative seasonality is not appropriate "
                    "for zero and negative values."
                )
            data = np.log(original)
            post_transform = np.exp

        result = STL(
            data,
            period=period,
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

        return {
            "trend": post_transform(result.trend),
            "seasonal": post_transform(result.seasonal),
            "rem": post_transform(result.resid),
        }

    def __decompose(self, original: pd.DataFrame) -> Dict[str, TimeSeriesData]:
        output = self.method(original)
        return {
            name: TimeSeriesData(
                ts.reset_index(), time_col_name=self.data.time_col_name
            )
            for name, ts in output.items()
        }

    def decomposer(self) -> Dict[str, TimeSeriesData]:
        """Decompose the time series.

        Returns:
            A dictionary with three time series for the three components:
            `trend` : Trend
            `seasonal` : Seasonality, and
            `rem` : Residual
        """
        original = self.__clean_ts()
        self.results = result = self.__decompose(original)
        return result

    def plot(
        self,
        figsize: Optional[Figsize] = None,
        linewidth: int = 3,
        xlabel: str = "Time",
        original_title: str = "Original Time Series",
        trend_title="Trend",
        seasonality_title="Seasonality",
        residual_title="Residual",
        subplot_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
        """Plot the original time series and the three decomposed components."""
        results = self.results
        if results is None:
            raise ValueError("Call decomposer() before plot().")

        if figsize is None:
            figsize = (20, 10)
        if subplot_kwargs is None:
            subplot_kwargs = {"hspace": 0.2}

        sharex = kwargs.pop("sharex", True)
        fig, axs = plt.subplots(
            nrows=4, ncols=1, figsize=figsize, sharex=sharex, **kwargs
        )
        titles = [trend_title, seasonality_title, residual_title]
        parts = ["trend", "seasonal", "rem"]

        axs[0].plot(
            self.data.time.values,
            self.data.value.values,
            linewidth=linewidth,
        )
        axs[0].set_title(original_title)

        for part, ax, title in zip(parts, axs, titles):
            ts: TimeSeriesData = results[part]
            ax.plot(ts.time.values, ts.value.values, linewidth=linewidth)
            ax.set_title(title)

        axs[3].set_xlabel(xlabel)
        plt.subplots_adjust(**subplot_kwargs)
        return (axs[0], axs[1], axs[2], axs[3])
