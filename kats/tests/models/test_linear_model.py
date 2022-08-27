# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import cast, Dict, Union
from unittest import TestCase

import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models.linear_model import LinearModel, LinearModelParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_LINEAR_95,
    AIR_FCST_LINEAR_99,
    PEYTON_FCST_LINEAR_95,
    PEYTON_FCST_LINEAR_99,
    PEYTON_FCST_LINEAR_INVALID_NEG_ONE,
    PEYTON_FCST_LINEAR_INVALID_ZERO,
    PEYTON_FCST_LINEAR_NAN,
    PEYTON_INPUT_NAN,
)
from parameterized.parameterized import parameterized

TEST_DATA: Dict[
    str,
    Union[
        Dict[str, Union[TimeSeriesData, int, pd.DataFrame, str]],
        Dict[str, Union[TimeSeriesData, pd.DataFrame, str]],
        Dict[str, TimeSeriesData],
    ],
] = {
    "daily": {
        "ts": TimeSeriesData(load_data("peyton_manning.csv", reset_columns=True)),
        "ts_nan": TimeSeriesData(df=PEYTON_INPUT_NAN),
        "freq": "D",
        "res_95": PEYTON_FCST_LINEAR_95,
        "res_99": PEYTON_FCST_LINEAR_99,
        "res_invalid": PEYTON_FCST_LINEAR_NAN,
        "res_invalid_param_1": PEYTON_FCST_LINEAR_INVALID_ZERO,
        "res_invalid_param_2": PEYTON_FCST_LINEAR_INVALID_NEG_ONE,
        "invalid_param_1": 0,
        "invalid_param_2": -1,
    },
    "monthly": {
        "ts": TimeSeriesData(load_data("air_passengers.csv", reset_columns=True)),
        "freq": "MS",
        "res_95": AIR_FCST_LINEAR_95,
        "res_99": AIR_FCST_LINEAR_99,
    },
    "multi": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv")),
    },
}


class LinearModelTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["freq"],
                TEST_DATA["monthly"]["res_95"],
                TEST_DATA["monthly"]["res_99"],
            ],
            [
                "daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["freq"],
                TEST_DATA["daily"]["res_95"],
                TEST_DATA["daily"]["res_99"],
            ],
            [
                "invalid",
                TEST_DATA["daily"]["ts_nan"],
                TEST_DATA["daily"]["freq"],
                TEST_DATA["daily"]["res_invalid"],
                TEST_DATA["daily"]["res_invalid"],
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        freq: str,
        truth_95: pd.DataFrame,
        truth_99: pd.DataFrame,
    ) -> None:
        # Set up params
        params_95 = LinearModelParams(alpha=0.05)
        params_99 = LinearModelParams(alpha=0.01)
        params_95.validate_params()
        params_99.validate_params()
        # Fit forecast
        m_95 = LinearModel(ts, params_95)
        m_99 = LinearModel(ts, params_99)
        m_95.fit()
        m_99.fit()
        res_95 = m_95.predict(steps=30, freq=freq)
        res_99 = m_99.predict(steps=30, freq=freq)
        # Test result
        assert_frame_equal(truth_95, res_95)
        assert_frame_equal(truth_99, res_99)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["invalid_param_1"],
                TEST_DATA["daily"]["freq"],
                TEST_DATA["daily"]["res_invalid_param_1"],
            ],
            [
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["invalid_param_2"],
                TEST_DATA["daily"]["freq"],
                TEST_DATA["daily"]["res_invalid_param_2"],
            ],
        ]
    )
    def test_invalid_params(
        self, ts: TimeSeriesData, invalid_param: float, freq: str, truth: pd.DataFrame
    ) -> None:
        # Set up params
        params = LinearModelParams(alpha=invalid_param)
        # Fit forecast
        m = LinearModel(ts, params)
        m.fit()
        res = m.predict(steps=30, freq=freq, include_history=True)
        # Test result
        assert_frame_equal(truth, res)

    def test_multivar(self) -> None:
        self.assertRaises(ValueError, LinearModel, TEST_DATA["multi"]["ts"], None)

    def test_exec_plot(self) -> None:
        # Set up params
        params = LinearModelParams(alpha=0.05)
        # Fit forecast
        m = LinearModel(cast(TimeSeriesData, TEST_DATA["daily"]["ts"]), params)
        m.fit()
        _ = m.predict(steps=2, freq=TEST_DATA["daily"]["freq"])
        # Test plotting
        m.plot()

    def test_name(self) -> None:
        m = LinearModel(
            cast(TimeSeriesData, TEST_DATA["daily"]["ts"]), LinearModelParams()
        )
        self.assertEqual(m.__str__(), "Linear Model")

    def test_search_space(self) -> None:
        m = LinearModel(
            cast(TimeSeriesData, TEST_DATA["daily"]["ts"]), LinearModelParams()
        )
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "alpha",
                    "type": "choice",
                    "value_type": "float",
                    "values": [0.01, 0.05, 0.1, 0.25],
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
