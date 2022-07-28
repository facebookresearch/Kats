# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from typing import Any, cast, Dict
from unittest import TestCase

import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_HW_1,  # first param combination results
    AIR_FCST_HW_2,
)
from parameterized.parameterized import parameterized

pd_ver = float(re.findall("([0-9]+\\.[0-9]+)\\..*", pd.__version__)[0])

ADD_PARAMS = HoltWintersParams(
    trend="add",
    seasonal="add",
    seasonal_periods=7,
)
MUL_PARAMS = HoltWintersParams(
    trend="mul",
    seasonal="mul",
    seasonal_periods=7,
)
TS: TimeSeriesData = cast(TimeSeriesData, load_air_passengers())


class HoltWintersModelTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "monthly_addiitive",
                TS,
                "MS",
                ADD_PARAMS,
                {"freq": "MS", "include_history": True},
                AIR_FCST_HW_1,
            ],
            [
                "monthly_multiplicative",
                TS,
                "MS",
                MUL_PARAMS,
                {"alpha": 0.9},
                AIR_FCST_HW_2,
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        freq: str,
        params: HoltWintersParams,
        predict_args: Dict[str, Any],
        truth: pd.DataFrame,
    ) -> None:
        params.validate_params()

        # Fit forecast
        m = HoltWintersModel(ts, params)
        m.fit()
        res = m.predict(steps=30, **predict_args)

        # Test result
        assert_frame_equal(truth, res, check_less_precise=0, rtol=1)

    def test_invalid_params(self) -> None:
        self.assertRaises(
            ValueError,
            HoltWintersParams,
            trend="random_trend",
        )

        self.assertRaises(
            ValueError,
            HoltWintersParams,
            seasonal="random_seasonal",
        )

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        TSData_multi = TimeSeriesData(DATA_multi)

        params = HoltWintersParams()

        self.assertRaises(
            ValueError,
            HoltWintersModel,
            data=TSData_multi,
            params=params,
        )

    def test_exec_plot(self) -> None:
        # Fit forecast
        m = HoltWintersModel(TS, ADD_PARAMS)
        m.fit()
        _ = m.predict(steps=2, freq="MS")

        # Test plotting
        m.plot()

    def test_name(self) -> None:
        m = HoltWintersModel(TS, ADD_PARAMS)
        self.assertEqual(m.__str__(), "HoltWinters")

    def test_search_space(self) -> None:
        m = HoltWintersModel(TS, ADD_PARAMS)
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "trend",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
                {
                    "name": "damped",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "seasonal",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
                {
                    "name": "seasonal_periods",
                    "type": "choice",
                    "values": [4, 7, 10, 14, 24, 30],
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
