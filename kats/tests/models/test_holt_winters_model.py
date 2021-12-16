# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import re
import unittest
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_HW_1,  # first param combination results
    AIR_FCST_HW_2,
)
from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

TEST_DATA = {
    "monthly": {
        "ts": load_air_passengers(),
        "freq": "MS",
        "res_1": AIR_FCST_HW_1,
        "res_2": AIR_FCST_HW_2,
    },
}

pd_ver = float(re.findall("([0-9]+\\.[0-9]+)\\..*", pd.__version__)[0])


class HoltWintersModelTest(TestCase):
    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["freq"],
                TEST_DATA["monthly"]["res_1"],
                TEST_DATA["monthly"]["res_2"],
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        freq: str,
        truth_1: pd.DataFrame,
        truth_2: pd.DataFrame,
    ):
        # Set up params
        params_1 = HoltWintersParams(
            trend="add",
            seasonal="add",
            seasonal_periods=7,
        )
        params_2 = HoltWintersParams(
            trend="mul",
            seasonal="mul",
            seasonal_periods=7,
        )

        params_1.validate_params()
        params_2.validate_params()
        # Fit forecast
        m_1 = HoltWintersModel(ts, params_1)
        m_2 = HoltWintersModel(ts, params_2)
        m_1.fit()
        m_2.fit()
        res_1 = m_1.predict(steps=30, freq=freq, include_history=True)
        res_2 = m_2.predict(steps=30, alpha=0.9)

        # Test result
        if pd_ver < 1.1:
            # pyre-fixme
            assert_frame_equal(truth_1, res_1, check_less_precise=1)
            # pyre-fixme
            assert_frame_equal(truth_2, res_2, check_less_precise=1)
        else:
            # pyre-fixme
            assert_frame_equal(truth_1, res_1, rtol=1)
            # pyre-fixme
            assert_frame_equal(truth_2, res_2, rtol=1)

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

    def test_exec_plot(self):
        # Set up params
        params = HoltWintersParams(
            trend="add",
            seasonal="add",
            seasonal_periods=7,
        )
        # Fit forecast
        m = HoltWintersModel(TEST_DATA["monthly"]["ts"], params)
        m.fit()
        _ = m.predict(steps=2, freq=TEST_DATA["monthly"]["freq"])

        # Test plotting
        m.plot()

    def test_name(self):
        m = HoltWintersModel(TEST_DATA["monthly"]["ts"], None)
        self.assertEqual(m.__str__(), "HoltWinters")

    def test_search_space(self):
        m = HoltWintersModel(TEST_DATA["monthly"]["ts"], None)
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
