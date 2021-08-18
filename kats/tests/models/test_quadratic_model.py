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
from kats.models.quadratic_model import (
    QuadraticModel,
    QuadraticModelParams,
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class QuadraticModelTest(TestCase):
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
        params = QuadraticModelParams()
        m = QuadraticModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        m.predict(steps=30, freq="MS", include_history=True)
        m.plot()

        m_daily = QuadraticModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()

        m_daily.predict(steps=30, freq="D", include_history=True)
        m.plot()

    def test_others(self) -> None:
        params = QuadraticModelParams()
        params.validate_params()
        self.assertRaises(
            ValueError,
            QuadraticModel,
            self.TSData_multi,
            params,
        )

        m = QuadraticModel(self.TSData, params)

        # test __str__ method
        self.assertEqual(m.__str__(), "Quadratic")

        # test search space
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "alpha",
                    "type": "choice",
                    "value_type": "float",
                    "values": [0.01, 0.05, 0.1, 0.25],
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
