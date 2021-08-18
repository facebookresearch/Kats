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
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class HoltWintersModelTest(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_fit_forecast(self) -> None:
        params = HoltWintersParams(
            # pyre-fixme[6]: Expected `str` for 1st param but got `None`.
            trend=None,
            damped=False,
            # pyre-fixme[6]: Expected `str` for 3rd param but got `None`.
            seasonal=None,
            # pyre-fixme[6]: Expected `int` for 4th param but got `None`.
            seasonal_periods=None,
        )
        m = HoltWintersModel(data=self.TSData, params=params)
        m.fit()
        m.predict(steps=30)
        m.predict(steps=30, include_history=True, alpha=0.05)
        m.plot()

        m_daily = HoltWintersModel(data=self.TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30)
        m.predict(
            steps=30,
            include_history=True,
            alpha=0.05,
            error_metrics=["mape"],
            train_percentage=80,
            test_percentage=10,
            sliding_steps=3,
        )
        m.plot()

    def test_others(self) -> None:
        # test param validation
        self.assertRaises(
            ValueError,
            HoltWintersParams,
            trend="random_trend",
        )

        self.assertRaises(
            ValueError,
            HoltWintersParams,
            seasonal="random_seasonal",
        )

        params = HoltWintersParams()
        self.assertRaises(
            ValueError,
            HoltWintersModel,
            self.TSData_multi,
            params,
        )

        m = HoltWintersModel(self.TSData, params)

        # test __str__ method
        self.assertEqual(m.__str__(), "HoltWinters")

        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "trend",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
                {
                    "name": "damped",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "seasonal",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
                {
                    "name": "seasonal_periods",
                    "type": "choice",
                    # The number of periods in this seasonality
                    # (e.g. 7 periods for daily data would be used for weekly seasonality)
                    "values": [4, 7, 10, 14, 24, 30],
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
