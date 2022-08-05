# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models.harmonic_regression import (
    HarmonicRegressionModel,
    HarmonicRegressionParams,
)
from parameterized import parameterized


TEST_DATA = {
    "default": {
        "ts": pd.Series(
            pd.to_datetime(
                np.arange(start=1576195200, stop=1577836801, step=60 * 60),
                unit="s",
            )
        ),
        "params": HarmonicRegressionParams(24, 3),
    },
    "multivariate": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv"))
    },
}


class testHarmonicRegression(TestCase):
    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "default",
                TEST_DATA["default"]["ts"],
                TEST_DATA["default"]["params"],
            ],
        ]
    )
    def test_forecast(
        self,
        testcase_name: str,
        series_times: pd.Series,
        params: HarmonicRegressionParams,
    ) -> None:
        harms = HarmonicRegressionModel.fourier_series(series_times, 24, 3)
        harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        data = TimeSeriesData(pd.DataFrame({"time": series_times, "values": harms_sum}))

        hrm = HarmonicRegressionModel(data, params)
        hrm.fit()
        self.assertIsNotNone(hrm.params)
        self.assertIsNotNone(hrm.harms)

        preds = hrm.predict(series_times.iloc[[0]])
        # pyre-fixme[6]: For 3rd param expected `None` but got `float`.
        self.assertAlmostEqual(preds["fcst"][0], harms_sum[0], delta=0.0001)

    def test_plot(self) -> None:
        series_times = pd.Series(TEST_DATA["default"]["ts"])
        harms = HarmonicRegressionModel.fourier_series(series_times, 24, 3)
        harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        data = TimeSeriesData(pd.DataFrame({"time": series_times, "values": harms_sum}))

        hrm = HarmonicRegressionModel(data, HarmonicRegressionParams(24, 3))
        hrm.fit()
        hrm.plot()

    def test_data_setup_and_validation(self) -> None:
        series_times = pd.Series(TEST_DATA["default"]["ts"])
        harms = HarmonicRegressionModel.fourier_series(series_times, 24, 3)
        harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        data = TimeSeriesData(pd.DataFrame({"time": series_times, "values": harms_sum}))

        hrm = HarmonicRegressionModel(data, HarmonicRegressionParams(24, 3))
        hrm.setup_data()
        hrm.validate_inputs()

    def test_model_raise(self) -> None:
        # Fourier can't <= 0
        self.assertRaises(
            ValueError,
            HarmonicRegressionParams,
            24,
            -1,
        )

        # Period can't <= 0
        self.assertRaises(
            ValueError,
            HarmonicRegressionParams,
            -1,
            3,
        )

        # Does support multivariate
        self.assertRaises(
            ValueError,
            HarmonicRegressionModel,
            TEST_DATA["multivariate"]["ts"],
            TEST_DATA["default"]["params"],
        )

        harms = HarmonicRegressionModel.fourier_series(
            # pyre-fixme[6]: Expected `pd.core.series.Series` for 1st positional only parameter to call `HarmonicRegressionModel.fourier_series`
            TEST_DATA["default"]["ts"],
            24,
            3,
        )
        harms_sum = np.sum([1, 1, 1, 1, 1, 1] * harms, axis=1)
        data = TimeSeriesData(
            pd.DataFrame({"time": TEST_DATA["default"]["ts"], "values": harms_sum})
        )
        # pyre-fixme[6]: Expected `HarmonicRegressionParams` for 2nd positional only parameter to call `HarmonicRegressionModel.__init__`
        hrm = HarmonicRegressionModel(data, TEST_DATA["default"]["params"])

        self.assertRaises(
            ValueError,
            hrm.predict,
            # pyre-fixme[16]: Undefined attribute [16]: `TimeSeriesData` has no attribute `head`
            TEST_DATA["default"]["ts"].head(1),
        )
        self.assertRaises(ValueError, hrm.plot)


if __name__ == "__main__":
    unittest.main()
