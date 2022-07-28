# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generic, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
from kats.consts import TimeSeriesData

ParamsType = TypeVar("ParamsType")


class Model(Generic[ParamsType]):
    __slots__ = ["data"]

    """Base forecasting model

    This is the parent class for all forecasting models in Kats

    Attributes:
        data: `TimeSeriesData` object
        params: model parameters
        validate_frequency: validate the frequency of time series
        validate_dimension: validate the dimension of time series
    """
    data: Optional[TimeSeriesData]
    fcst_df: Optional[pd.DataFrame] = None
    include_history: bool = False

    def __init__(
        self,
        data: Optional[TimeSeriesData],
        params: ParamsType,
        validate_frequency: bool = False,
        validate_dimension: bool = False,
    ) -> None:
        self.data = data
        self.params = params
        self.__type__ = "model"
        if data is not None:
            data.validate_data(validate_frequency, validate_dimension)

    def setup_data(self) -> None:
        """abstract method to set up dataset

        This is a declaration for setup data method
        """
        pass

    def validate_inputs(self) -> None:
        """abstract method to validate the inputs

        This is a declaration for validate_inputs method
        """
        pass

    def fit(self) -> None:
        """abstract method to fit model

        This is a declaration for model fitting
        """
        pass

    def predict(self, *_args: Any, **_kwargs: Any) -> None:
        """abstract method to predict

        This is a declaration for predict method
        """
        pass

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot method for forecasting models

        This method provides base plotting functionality for all forecasting
        models.

        Args:
            ax: optional Matplotlib Axes to use.
        Returns:
            The matplotlib Axes object.
        """
        fcst_df = self.fcst_df
        if fcst_df is None:
            raise ValueError("predict() must be called before plot().")
        data = self.data
        assert data is not None
        include_history = self.include_history

        logging.info("Generating chart for forecast result.")
        if ax is None:
            if figsize is None:
                figsize = (10, 6)
            fig, ax = plt.subplots(facecolor="w", figsize=figsize)
        else:
            fig = plt.gcf()

        # Allow subclasses to specify different kwargs by fetching them
        # here instead of in the method signature.
        intervals = kwargs.get("intervals", True)
        ls = kwargs.get("ls", "-")
        history_color = kwargs.get("history_color", "k")
        forecast_color = kwargs.get("forecast_color", "#4267B2")
        grid = kwargs.get("grid", True)
        xlabel = kwargs.get("xlabel", "time")
        ylabel = kwargs.get("ylabel", "y")

        if grid:
            ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)

        if include_history:
            ax.plot(data.time, data.value, history_color)

        ax.plot(fcst_df["time"], fcst_df["fcst"], ls=ls, c=forecast_color)

        if intervals and {"fcst_lower", "fcst_upper"}.issubset(fcst_df.columns):
            ax.fill_between(
                fcst_df["time"],
                fcst_df["fcst_lower"],
                fcst_df["fcst_upper"],
                color=forecast_color,
                alpha=0.2,
            )

        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        fig.set_tight_layout(True)
        return ax

    @staticmethod
    def get_parameter_search_space() -> None:
        """method to query default parameter search space

        abstract method to be implemented by downstream forecasting models
        """
        pass
