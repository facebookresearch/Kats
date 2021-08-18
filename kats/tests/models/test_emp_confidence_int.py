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
from kats.utils.emp_confidence_int import EmpConfidenceInt


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class testEmpConfidenceInt(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)
        params = ProphetParams(seasonality_mode="multiplicative")
        self.params = params
        self.unfit_ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            ProphetModel,
            confidence_level=0.9,
        )
        self.ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            ProphetModel,
            confidence_level=0.9,
        )

    def test_empConfInt_Prophet(self) -> None:
        result = self.ci.get_eci(steps=10, freq="MS")
        expected = pd.DataFrame(
            data={
                "time": pd.date_range("1961-01-01", "1961-10-01", freq="MS"),
                "fcst": [
                    452.077721,
                    433.529496,
                    492.499917,
                    495.895518,
                    504.532772,
                    580.506512,
                    654.849614,
                    650.944635,
                    554.067652,
                    490.207818,
                ],
                "fcst_lower": [
                    428.329060,
                    408.808464,
                    466.806514,
                    469.229744,
                    476.894627,
                    551.895995,
                    625.266726,
                    620.389377,
                    522.540022,
                    457.707818,
                ],
                "fcst_upper": [
                    475.826382,
                    458.250528,
                    518.193320,
                    522.561292,
                    532.170918,
                    609.117028,
                    684.432501,
                    681.499894,
                    585.595281,
                    522.707819,
                ],
            }
        )
        pd.testing.assert_frame_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
