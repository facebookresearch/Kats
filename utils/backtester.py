#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData


class BackTester:
    def __init__(
        self,
        data: TimeSeriesData,
        params: object,
        alpha: float,
        train_percentage: float,
        test_percentage: float,
        model_class,
        err_method: str,
        **kwargs
    ):
        self.data = data
        self.params = params
        self.size = len(data.time)
        assert self.size > 0, "Passing an empty time series"
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.alpha = alpha
        self.model_class = model_class
        self.err_method = err_method
        self.args = kwargs

    def _get_percent_size(self, size, percent):
        return int(np.floor(size * percent / 100))

    def _create_train_test(self):
        self.train_size = self._get_percent_size(self.size, self.train_percentage)
        self.test_size = self._get_percent_size(self.size, self.test_percentage)

        # infer the freq
        self.freq = pd.infer_freq(self.data.time)
        self.data_train = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[: self.train_size],
                    "y": self.data.value[: self.train_size],
                }
            )
        )
        self.data_test = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[
                        self.train_size : (self.train_size + self.test_size)
                    ],
                    "y": self.data.value[
                        self.train_size : (self.train_size + self.test_size)
                    ],
                }
            )
        )

    def _create_model(self, **kwargs):
        self.model_train = self.model_class(data=self.data_train, params=self.params)
        self.model_train.fit()
        self.fcst = self.model_train.predict(steps=self.test_size, freq=self.freq)

    def run_backtest(self):
        self._create_train_test()
        self._create_model()

    def get_error(self):
        func = {
            "mse": self.get_mse(),
            "mape": self.get_mape(),
            "rmse": self.get_rmse(),
            "mad": self.get_mad(),
        }
        return func[self.err_method]

    def get_mse(self):
        return (
            np.sum((self.data_test.value.values - self.fcst.fcst.values) ** 2)
            / self.test_size
        )

    def get_mape(self):
        return np.mean(
            np.abs(
                (self.data_test.value.values - self.fcst.fcst.values)
                / self.data_test.value.values
            )
        )

    def get_rmse(self):
        return np.sqrt(self.get_mse())

    def get_mad(self):
        return (
            np.sum(np.abs((self.data_test.value.values - self.fcst.fcst.values)))
            / self.test_size
        )
