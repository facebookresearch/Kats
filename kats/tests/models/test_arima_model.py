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
from kats.models.arima import ARIMAModel, ARIMAParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class ARIMAModelTest(TestCase):
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
        params = ARIMAParams(p=1, d=1, q=1)
        m = ARIMAModel(data=self.TSData, params=params)
        m.fit(
            start_params=None,
            transparams=True,
            method="css-mle",
            trend="c",
            solver="lbfgs",
            maxiter=500,
            # pyre-fixme[6]: Expected `bool` for 7th param but got `int`.
            full_output=1,
            disp=False,
            callback=None,
            start_ar_lags=None,
        )
        m.predict(steps=30)
        m.plot()

        m_daily = ARIMAModel(data=self.TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30, include_history=True)
        m.plot()

    def test_others(self) -> None:
        params = ARIMAParams(p=1, d=1, q=1)
        params.validate_params()
        m = ARIMAModel(data=self.TSData, params=params)

        # test __str__ method
        self.assertEqual(m.__str__(), "ARIMA")

        # test input error
        self.assertRaises(
            ValueError,
            ARIMAModel,
            self.TSData_multi,
            params,
        )

        # test search space
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "p",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "d",
                    "type": "choice",
                    "values": list(range(1, 3)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "q",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
