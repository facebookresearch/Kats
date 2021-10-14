# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data, load_air_passengers
from kats.models.theta import ThetaModel, ThetaParams
from pandas.util.testing import assert_series_equal
from parameterized import parameterized

np.random.seed(0)

TEST_DATA = {
    "short": {
        "ts": TimeSeriesData(
            pd.DataFrame(
                {
                    "time": [
                        pd.Timestamp("1961-01-01 00:00:00"),
                        pd.Timestamp("1961-02-01 00:00:00"),
                    ],
                    "y": [1.0, 2.0],
                }
            )
        ),
        "params": ThetaParams(m=2),
    },
    "constant": {
        "ts": TimeSeriesData(
            pd.DataFrame(
                {"time": pd.date_range("1960-12-01", "1963-01-01", freq="m"), "y": 10.0}
            )
        ),
        "params": ThetaParams(m=2),
    },
    "nonseasonal": {
        "ts": TimeSeriesData(
            pd.DataFrame(
                {
                    "time": pd.date_range("1960-12-01", "1963-01-01", freq="m"),
                    "y": np.random.randn(25),
                }
            )
        ),
        "params": ThetaParams(m=4),
    },
    "daily": {
        "ts": TimeSeriesData(
            load_data("peyton_manning.csv").set_axis(["time", "y"], axis=1)
        ),
        "params": ThetaParams(),
        "params_negative": ThetaParams(m=-5),
    },
    "monthly": {
        "ts": load_air_passengers(),
        "params": ThetaParams(m=12),
    },
    "multivariate": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv"))
    },
}


class ThetaModelTest(TestCase):
    def test_params(self) -> None:
        # Test default value
        params = ThetaParams()
        params.validate_params()
        self.assertEqual(params.m, 1)

        params = ThetaParams(m=12)
        self.assertEqual(params.m, 12)

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                15,
                0.05,
                False,
                None,
                None,
            ],
            [
                "monthly, include history",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                15,
                0.05,
                True,
                None,
                None,
            ],
            [
                "daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
                30,
                0.05,
                False,
                None,
                None,
            ],
            [
                "daily, include history",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_negative"],
                30,
                0.05,
                True,
                None,
                None,
            ],
        ]
    )
    def test_forecast(
        self,
        testcase_name: str,
        ts: TimeSeriesData,
        params: ThetaParams,
        steps: int,
        alpha: float,
        include_history: bool,
        freq: Optional[str],
        truth: TimeSeriesData,
    ) -> None:
        m = ThetaModel(data=ts, params=params)
        m.fit()
        m.predict(steps=steps, alpha=alpha, include_history=include_history, freq=freq)
        # TODO: validate results

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "m less than 1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_negative"],
                False,
            ],
            [
                "data too short",
                TEST_DATA["short"]["ts"],
                TEST_DATA["short"]["params"],
                False,
            ],
            [
                "constant data",
                TEST_DATA["constant"]["ts"],
                TEST_DATA["constant"]["params"],
                False,
            ],
            [
                "seasonal",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                True,
            ],
        ]
    )
    def test_check_seasonality(
        self,
        testcase_name: str,
        ts: TimeSeriesData,
        params: ThetaParams,
        is_seasonal: bool,
    ) -> None:
        m = ThetaModel(ts, params)
        m.check_seasonality()
        self.assertEqual(m.seasonal, is_seasonal)

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "nonseasonal",
                False,
                TEST_DATA["nonseasonal"]["ts"],
                TEST_DATA["nonseasonal"]["params"],
                False,
                True,
            ],
            [
                "seasonal",
                True,
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                True,
                False,
            ],
        ]
    )
    def test_deseasonalize(
        self,
        testcase_name: str,
        seasonal: bool,
        ts: TimeSeriesData,
        params: ThetaParams,
        seasonality_removed: bool,
        decomp_is_none: bool,
    ) -> None:
        m = ThetaModel(ts, params)
        m.seasonal = seasonal
        deseas_data = m.deseasonalize()

        if seasonality_removed:
            self.assertFalse(ts.value.equals(deseas_data.value))
        else:
            assert_series_equal(deseas_data.value, ts.value)

        self.assertEqual(decomp_is_none, m.decomp is None)

    def test_multivar(self) -> None:
        # Theta model does not support multivariate time data
        self.assertRaises(
            ValueError,
            ThetaModel,
            TEST_DATA["multivariate"]["ts"],
            ThetaParams(),
        )

    def test_exec_plot(self):
        m = ThetaModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        m.fit()
        m.predict(steps=15, alpha=0.05)
        m.plot()

    def test_name(self):
        m = ThetaModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        self.assertEqual(m.__str__(), "Theta")

    def test_search_space(self):
        self.assertEqual(
            ThetaModel.get_parameter_search_space(),
            [
                {
                    "name": "m",
                    "type": "choice",
                    "values": list(range(1, 31)),
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
