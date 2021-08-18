# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.harmonic_regression import (
    HarmonicRegressionModel,
    HarmonicRegressionParams,
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class testHarmonicRegression(TestCase):
    def setUp(self):
        times = pd.to_datetime(
            np.arange(start=1576195200, stop=1577836801, step=60 * 60), unit="s"
        )
        self.series_times = pd.Series(times)
        harms = HarmonicRegressionModel.fourier_series(self.series_times, 24, 3)
        self.harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        self.data = TimeSeriesData(
            pd.DataFrame({"time": self.series_times, "values": self.harms_sum})
        )

        self.params = HarmonicRegressionParams(24, 3)

    def test_fit_and_predict(self) -> None:
        hrm = HarmonicRegressionModel(self.data, self.params)
        hrm.fit()
        self.assertIsNotNone(hrm.params)
        # pyre-fixme[16]: `HarmonicRegressionModel` has no attribute `harms`.
        self.assertIsNotNone(hrm.harms)

        # pyre-fixme[6]: Expected `Series` for 1st param but got
        #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
        preds = hrm.predict(self.series_times.head(1))
        self.assertAlmostEqual(preds["fcst"][0], self.harms_sum[0], delta=0.0001)


if __name__ == "__main__":
    unittest.main()
