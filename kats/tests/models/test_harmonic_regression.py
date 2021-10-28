# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.harmonic_regression import (
    HarmonicRegressionModel,
    HarmonicRegressionParams,
)


class testHarmonicRegression(TestCase):
    def setUp(self) -> None:
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
        self.assertIsNotNone(hrm.harms)

        preds = hrm.predict(cast(pd.Series, self.series_times.head(1)))
        self.assertAlmostEqual(preds["fcst"][0], self.harms_sum[0], delta=0.0001)


if __name__ == "__main__":
    unittest.main()
