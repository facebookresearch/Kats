# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import unittest
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.lstm import LSTMModel, LSTMParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class LSTMModelTest(TestCase):
    def setUp(self):
        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

    def test_fit_forecast(self) -> None:
        # use smaller time window and epochs for testing to reduce testing time
        params = LSTMParams(hidden_size=10, time_window=4, num_epochs=5)
        m = LSTMModel(data=self.TSData, params=params)
        m.fit()
        m.predict(steps=15)
        m.plot()

        m_daily = LSTMModel(data=self.TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30)
        m_daily.plot()


if __name__ == "__main__":
    unittest.main()
