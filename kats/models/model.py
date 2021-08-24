# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Generic, Optional, TypeVar

import pandas as pd
from kats.consts import TimeSeriesData
from matplotlib import pyplot as plt

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
            # pyre-fixme[16]: `Optional` has no attribute `validate_data`.
            self.data.validate_data(validate_frequency, validate_dimension)

    def setup_data(self):
        """abstract method to set up dataset

        This is a declaration for setup data method
        """
        pass

    def validate_inputs(self):
        """abstract method to validate the inputs

        This is a declaration for validate_inputs method
        """
        pass

    def fit(self):
        """abstract method to fit model

        This is a declaration for model fitting
        """
        pass

    def predict(self, *_args, **_kwargs):
        """abstract method to predict

        This is a declaration for predict method
        """
        pass

    @staticmethod
    def plot(
        data: TimeSeriesData,
        fcst: pd.DataFrame,
        include_history=False,
    ) -> None:
        """plot method for forecasting models

        This method provides the plotting functionality for all forecasting
        models.

        Args:
            data: `TimeSeriesData`, the historical time series data set
            fcst: forecasted results from forecasting models
            include_history: if True, include the historical data when plotting.
        """
        logging.info("Generating chart for forecast result.")
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(pd.to_datetime(data.time), data.value, "k")

        last_date = data.time.max()
        steps = fcst.shape[0]
        freq = pd.infer_freq(data.time)
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)

        dates_to_plot = dates[dates != last_date]  # Return correct number of periods

        fcst_dates = dates_to_plot.to_pydatetime()

        if include_history:
            ax.plot(fcst.time, fcst.fcst, ls="-", c="#4267B2")

            if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
                ax.fill_between(
                    fcst.time,
                    fcst.fcst_lower,
                    fcst.fcst_upper,
                    color="#4267B2",
                    alpha=0.2,
                )
        else:
            ax.plot(fcst_dates, fcst.fcst, ls="-", c="#4267B2")

            if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
                ax.fill_between(
                    fcst_dates,
                    fcst.fcst_lower,
                    fcst.fcst_upper,
                    color="#4267B2",
                    alpha=0.2,
                )

        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel(xlabel="time")
        ax.set_ylabel(ylabel="y")
        fig.tight_layout()

    @staticmethod
    def get_parameter_search_space():
        """method to query default parameter search space

        abstract method to be implemented by downstream forecasting models
        """
        pass
