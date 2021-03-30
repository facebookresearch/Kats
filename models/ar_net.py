#!/usr/bin/env python3

# Forecasting with AR_Net Model
# For more information, see https://arxiv.org/pdf/1911.12436.pdf
#
#  NOTE: This model is still under active development in collaboration
#        with Stanford and subsequent updates will be made as progress
#        is made.
#

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List

import data_ai.ar_net.data_loader as data_loader
import data_ai.ar_net.training as training
import infrastrategy.kats.models.model as m
import numpy as np
import pandas as pd
import torch
from infrastrategy.kats.consts import Params, TimeSeriesData
from matplotlib import pyplot as plt
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_arnet_parameter_search_space
)


# Example usage
#
# m = ar_net.ARNet(data=TSData, params=params)
# m.fit()
class ARNetParams(Params):
    """
        Sepcifes parameters for the model.
        Sample usage:
            params = ar_net.ARNetParams(input_size=7,
                                        output_size=7,
                                        batch_size=10)
        Parameters:
            input_size: how many time-stamps to use
                        as an input to predict  the
                        future.
            output_size: each time we make a prediction
                         this is how many timestamps
                         we will predict at a time.
            batch_size: during model training, this
                        is how many examples from the
                        past we will use on each
                        iteration.
    """

    def __init__(
        self, input_size: int = 7, output_size: int = 7, batch_size: int = 10, **kwargs
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class ARNet(m.Model):
    """
    Trains and predicts using the ARNet model.

    Sample usage:
        params = ar_net.ARNetParams(input_size=7,
                                    output_size=7,
                                    batch_size=10)
        m = ar_net.ARNet(data=TSData, params=params)
        m.fit()
    Parameters:
        data: Timeseries data to train from.
        params: The model parameters such
                (see above for details).
    """

    def __init__(self, data: TimeSeriesData, params: ARNetParams) -> None:
        super().__init__(data, params)

    def fit(self, **kwargs) -> None:
        series = pd.DataFrame(self.data.value)
        self.n_metric = series.shape[1]
        dimensions = self.params.input_size * self.n_metric

        (
            dataset_train,
            dataset_test,
            series_train,
            series_test,
        ) = data_loader.create_dataset(
            series.values,
            test=0.2,
            sample_inp_size=self.params.input_size,
            sample_out_size=self.params.output_size,
            verbose=False,
        )

        (
            self.model,
            predicted,
            actual,
            losses,
            weights,
            test_mse,
            avg_losses,
            val_loss,
        ) = training.train_ar_net(
            dataset_train,
            dataset_test,
            dimensions,
            self.n_metric * self.params.output_size,
            self.params.batch_size,
            self.params.input_size,
        )

    def predict(self, steps, **kwargs):
        series_pred = (self.data.value.values[-(self.params.input_size) :]).reshape(
            -1, self.n_metric
        )
        for _ in range(steps):
            x = torch.from_numpy(
                series_pred[-(self.params.input_size) :].reshape(1, -1)
            ).type(torch.FloatTensor)
            y = self.model.forward(x)
            series_pred = np.vstack(
                (series_pred, y.detach().numpy().reshape(-1, self.n_metric))
            )
        # Drop the initial input.
        series_pred = series_pred[self.params.input_size :]
        self._X_future = list(range(len(series_pred)))
        self.y_fcst = series_pred

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=len(self.y_fcst) + 1)
        dates = dates[dates != last_date]
        self.fcst_dates = dates.to_pydatetime()

        self.fcst_dict = {}
        ts_names = (
            ["Metric"]
            if len(self.data.value.shape) == 1 or self.data.value.shape[1] <= 1
            else list(self.data.value.columns)
        )

        for i, name in enumerate(ts_names):
            series_pred_local = (
                series_pred[:, 0]
                if len(self.data.value.shape) == 1 or self.data.value.shape[1] <= 1
                else series_pred[:, i]
            )
            fcst_df = pd.DataFrame(
                {
                    "time": self.fcst_dates,
                    "fcst": series_pred_local,
                    # TODO: Add learned variance from model.
                    "fcst_lower": series_pred_local * 0.95,
                    "fcst_upper": series_pred_local * 1.05,
                }
            )
            self.fcst_dict[name] = fcst_df

        logging.debug(f"Return forecast data: {self.fcst_dict}")
        return self.fcst_dict

    def plot(self):
        logging.info("Generating chart for forecast result from ARNet model.")
        print(len(self.fcst_dict.keys()))
        fig, axes = plt.subplots(
            ncols=len(self.fcst_dict.keys()), dpi=120, figsize=(10, 6)
        )
        for i, ax in (
            enumerate(axes.flatten())
            if len(self.fcst_dict.keys()) > 1
            else enumerate([axes])
        ):
            ts_name = list(self.fcst_dict.keys())[i]
            data = self.fcst_dict[ts_name]
            ax.plot(
                pd.to_datetime(self.data.time),
                self.data.value
                if len(self.data.value.shape) == 1 or self.data.value.shape[1] <= 1
                else self.data.value[ts_name],
                "k",
            )
            ax.plot(self.fcst_dates, data.fcst, ls="-", c="#4267B2")

            ax.fill_between(
                self.fcst_dates,
                data.fcst_lower,
                data.fcst_upper,
                color="#4267B2",
                alpha=0.2,
            )

            ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
            ax.set_xlabel(xlabel="time")
            ax.set_ylabel(ylabel=ts_name)

        plt.tight_layout()

    def __str__(self):
        return "ARNet"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of ar_net
        to avoid the massive dependencies of ar_net and huge build size.
        Check https://fburl.com/kg04hx5y for detail.
        """
        return get_default_arnet_parameter_search_space()
