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
from kats.models.stlf import STLFModel, STLFParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class testSTLFModel(TestCase):
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
        for method in ["theta", "prophet", "linear", "quadratic"]:
            params = STLFParams(m=12, method=method)
            m = STLFModel(self.TSData, params)
            m.fit()
            m.predict(steps=30)
            m.predict(steps=30, include_history=True)

        params = STLFParams(m=7, method="theta")
        m_daily = STLFModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30)
        m.plot()

        m_daily.predict(steps=30, include_history=True)
        m.plot()

        # test when m > the length of time series
        params = STLFParams(m=10000, method="theta")
        self.assertRaises(
            ValueError,
            STLFModel,
            self.TSData_daily,
            params,
        )

    def test_others(self) -> None:
        # test param value error
        self.assertRaises(
            ValueError,
            STLFParams,
            method="random_model",
            m=12,
        )

        params = STLFParams(m=12, method="theta")
        params.validate_params()

        # test model param
        self.assertRaises(
            ValueError,
            STLFModel,
            self.TSData_multi,
            params,
        )

        # test __str__ method
        m = STLFModel(self.TSData, params)
        self.assertEqual(m.__str__(), "STLF")


if __name__ == "__main__":
    unittest.main()
