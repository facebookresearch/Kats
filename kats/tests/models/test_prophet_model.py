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
from kats.models.prophet import ProphetModel, ProphetParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class ProphetModelTest(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

    def test_fit_forecast(self) -> None:
        params = ProphetParams()
        m = ProphetModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        # adding cap and floor
        params = ProphetParams(cap=1000, floor=10, growth="logistic")
        m = ProphetModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        m_daily = ProphetModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()

        # add historical fit
        params = ProphetParams()
        m = ProphetModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS", include_history=True)
        m.plot()

        m_daily = ProphetModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D", include_history=True)
        m.plot()

        # test logistic growth with cap
        params = ProphetParams(growth="logistic", cap=1000)
        m = ProphetModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        params = ProphetParams(growth="logistic", cap=20)
        m_daily = ProphetModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()

        # Testing custom seasonalities.
        params = ProphetParams(
            custom_seasonalities=[
                {
                    "name": "monthly",
                    "period": 30.5,
                    "fourier_order": 5,
                },
            ],
        )
        params.validate_params()  # Validate params and ensure no errors raised.
        m = ProphetModel(self.TSData, params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        params = ProphetParams(
            custom_seasonalities=[
                {
                    "name": "semi_annually",
                    "period": 365.25 / 2,
                    "fourier_order": 5,
                },
            ],
        )
        params.validate_params()  # Validate params and ensure no errors raised.
        m_daily = ProphetModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()


if __name__ == "__main__":
    unittest.main()
