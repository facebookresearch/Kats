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
from kats.models.theta import ThetaModel, ThetaParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class ThetaModelTest(TestCase):
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
        params = ThetaParams(m=12)
        m = ThetaModel(self.TSData, params)
        m.fit()
        m.predict(steps=15, alpha=0.05)
        m.plot()

        params = ThetaParams()
        m_daily = ThetaModel(data=self.TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30)
        m.plot()

        params = ThetaParams(m=12)
        m = ThetaModel(self.TSData, params)
        m.fit()
        m.predict(steps=15, alpha=0.05, include_history=True)
        m.plot()

        params = ThetaParams()
        m_daily = ThetaModel(data=self.TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30, include_history=True)
        m.plot()

    def test_others(self) -> None:
        params = ThetaParams(m=12)
        params.validate_params()

        self.assertRaises(
            ValueError,
            ThetaModel,
            self.TSData_multi,
            params,
        )

        m = ThetaModel(self.TSData, params)

        # test __str__ method
        self.assertEqual(m.__str__(), "Theta")


if __name__ == "__main__":
    unittest.main()
