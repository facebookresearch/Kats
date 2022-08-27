# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, Dict, Optional, Union
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from kats.compat import statsmodels
from kats.compat.pandas import assert_frame_equal, assert_series_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.theta import ThetaModel, ThetaParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_15_THETA_INCL_HIST_SM_11,
    AIR_FCST_15_THETA_INCL_HIST_SM_12,
    AIR_FCST_15_THETA_SM_11,
    AIR_FCST_15_THETA_SM_12,
    NONSEASONAL_INPUT,
    PEYTON_FCST_30_THETA_INCL_HIST_SM_11,
    PEYTON_FCST_30_THETA_INCL_HIST_SM_12,
    PEYTON_FCST_30_THETA_SM_11,
    PEYTON_FCST_30_THETA_SM_12,
)
from parameterized.parameterized import parameterized


TEST_DATA: Dict[str, Dict[str, Union[ThetaParams, TimeSeriesData, pd.DataFrame]]] = {
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
        "ts": TimeSeriesData(NONSEASONAL_INPUT),
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

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
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
                (
                    AIR_FCST_15_THETA_SM_11
                    if statsmodels.version < "0.12"
                    else AIR_FCST_15_THETA_SM_12
                ),
            ],
            [
                "monthly, include history",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                15,
                0.05,
                True,
                None,
                (
                    AIR_FCST_15_THETA_INCL_HIST_SM_11
                    if statsmodels.version < "0.12"
                    else AIR_FCST_15_THETA_INCL_HIST_SM_12
                ),
            ],
            [
                "daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
                30,
                0.05,
                False,
                None,
                (
                    PEYTON_FCST_30_THETA_SM_11
                    if statsmodels.version < "0.12"
                    else PEYTON_FCST_30_THETA_SM_12
                ),
            ],
            [
                "daily, include history",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_negative"],
                30,
                0.05,
                True,
                None,
                (
                    PEYTON_FCST_30_THETA_INCL_HIST_SM_11
                    if statsmodels.version < "0.12"
                    else PEYTON_FCST_30_THETA_INCL_HIST_SM_12
                ),
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
        truth: pd.DataFrame,
    ) -> None:
        np.random.seed(0)
        m = ThetaModel(data=ts, params=params)
        m.fit()
        forecast_df = m.predict(
            steps=steps, alpha=alpha, include_history=include_history, freq=freq
        )
        assert_frame_equal(truth, forecast_df)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
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

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
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
            assert_series_equal(
                cast(pd.Series, ts.value), cast(pd.Series, deseas_data.value)
            )

        self.assertEqual(decomp_is_none, m.decomp is None)

    def test_multivar(self) -> None:
        # Theta model does not support multivariate time data
        self.assertRaises(
            ValueError,
            ThetaModel,
            TEST_DATA["multivariate"]["ts"],
            ThetaParams(),
        )

    def test_exec_plot(self) -> None:
        m = ThetaModel(
            cast(TimeSeriesData, TEST_DATA["daily"]["ts"]),
            cast(ThetaParams, TEST_DATA["daily"]["params"]),
        )
        m.fit()
        m.predict(steps=15, alpha=0.05)
        m.plot()

    def test_name(self) -> None:
        m = ThetaModel(
            cast(TimeSeriesData, TEST_DATA["daily"]["ts"]),
            cast(ThetaParams, TEST_DATA["daily"]["params"]),
        )
        self.assertEqual(m.__str__(), "Theta")

    def test_search_space(self) -> None:
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

    def test_others(self) -> None:
        m = ThetaModel(
            cast(TimeSeriesData, TEST_DATA["daily"]["ts"]),
            cast(ThetaParams, TEST_DATA["daily"]["params"]),
        )
        # fit must be called before predict
        self.assertRaises(ValueError, m.predict, 30)

        # seasonal data must be deseasonalized before fit
        with patch.object(
            m, "deseasonalize", (lambda self: self.data).__get__(m)
        ), patch.object(m, "check_seasonality"):
            m.seasonal = True
            m.decomp = None
            self.assertRaises(ValueError, m.fit)

        with patch(
            "kats.utils.decomposition.TimeSeriesDecomposition.decomposer",
            return_value={
                "seasonal": cast(TimeSeriesData, TEST_DATA["daily"]["ts"]) * 0
            },
        ):
            # Don't deseasonalize if any seasonal index = 0
            deseas_data = m.deseasonalize()
            expected = cast(
                pd.Series, cast(TimeSeriesData, TEST_DATA["daily"]["ts"]).value
            )
            assert_series_equal(expected, cast(pd.Series, deseas_data.value))


if __name__ == "__main__":
    unittest.main()
