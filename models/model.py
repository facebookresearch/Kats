#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from infrastrategy.kats.consts import TimeSeriesData, Params
import logging
from matplotlib import pyplot as plt
import pandas as pd


class Model:
    __slots__ = [
        "data"
    ]

    """Base forecasting model

    Parameters
    ---------
    data: TimeSeriesData object
    params: model parameters
    """
    def __init__(
        self,
        data: TimeSeriesData,
        params: Params,
        validate_frequency: bool = False,
        validate_dimension: bool = False,
    ) -> None:
        self.data = data
        self.params = params
        self.__type__ = 'model'
        if data is not None:
            self.data.validate_data(validate_frequency, validate_dimension)

    def setup_data(self):
        pass

    def validate_inputs(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    @staticmethod
    def plot(data: TimeSeriesData, fcst: pd.DataFrame, include_history=False) -> None:
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
                    fcst.time, fcst.fcst_lower, fcst.fcst_upper, color="#4267B2", alpha=0.2
                )
        else:
            ax.plot(fcst_dates, fcst.fcst, ls="-", c="#4267B2")

            if ("fcst_lower" in fcst.columns) and ("fcst_upper" in fcst.columns):
                ax.fill_between(
                    fcst_dates, fcst.fcst_lower, fcst.fcst_upper, color="#4267B2", alpha=0.2
                )

        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel(xlabel="time")
        ax.set_ylabel(ylabel="y")
        fig.tight_layout()

    @staticmethod
    def get_parameter_search_space():
        pass
