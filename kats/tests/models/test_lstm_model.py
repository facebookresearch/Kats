# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

from kats.consts import TimeSeriesData
from kats.data.utils import load_data, load_air_passengers
from kats.models.lstm import LSTMModel, LSTMParams


class LSTMModelTest(TestCase):
    def setUp(self):
        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        self.TSData = load_air_passengers()
        self.params = LSTMParams(hidden_size=10, time_window=3, num_epochs=4)

    def test_fit_forecast(self) -> None:
        # use smaller time window and epochs for testing to reduce testing time
        m = LSTMModel(data=self.TSData, params=self.params)
        m.fit()
        m.predict(steps=15)
        m.plot()

    def test_fit_forecast_daily(self) -> None:
        # use smaller time window and epochs for testing to reduce testing time
        m_daily = LSTMModel(data=self.TSData_daily, params=self.params)
        m_daily.fit()
        m_daily.predict(steps=30)
        m_daily.plot()


if __name__ == "__main__":
    unittest.main()
