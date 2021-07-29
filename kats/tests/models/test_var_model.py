# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest import TestCase
import pkgutil
import io

import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.var import VARModel, VARParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class testVARModel(TestCase):
    def setUp(self):
        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_fit_forecast(self) -> None:
        params = VARParams()
        m = VARModel(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True)


if __name__ == "__main__":
    unittest.main()
