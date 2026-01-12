# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict
from unittest import TestCase

import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_data  # @manual
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.tests.models.test_models_dummy_data import (
    PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1,
    PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_1,
    PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1,
    PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_1,
)
from parameterized.parameterized import parameterized


RTOL: float = 0.005
STEPS_1: int = 15
STEPS_2: int = 30
TS: TimeSeriesData = TimeSeriesData(load_data("peyton_manning.csv"))

TEST_DATA: Dict[str, Any] = {
    "daily": {
        "ts": TS,
        "invalid_ts": TimeSeriesData(
            load_data("multivariate_anomaly_simulated_data.csv")
        ),
        "freq": "D",
        "p1": ARIMAParams(p=1, d=1, q=1),
        "p2": ARIMAParams(p=1, d=1, q=3),
        "m": {
            "trend": "n",
        },
        "invalid_m": {
            "trend": "x",
        },
        "steps_1": STEPS_1,
        "steps_2": STEPS_2,
        "no_incl_hist": False,
        "incl_hist": True,
        "truth_p1_m1_15": PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_15": PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_1,
        "truth_p1_m1_30": PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_30": PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_1,
        "truth_p1_m1_15_incl_hist": pd.concat(
            [
                pd.DataFrame({"time": TS.time[1:], "fcst": TS.value[1:]}),
                PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1,
            ],
            axis=0,
            ignore_index=True,
        ),
        "truth_p1_m1_30_incl_hist": pd.concat(
            [
                pd.DataFrame({"time": TS.time[1:], "fcst": TS.value[1:]}),
                PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1,
            ],
            axis=0,
            ignore_index=True,
        ),
    },
}


class ARIMAModelTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "daily_p1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p1_m1_15"],
                TEST_DATA["daily"]["truth_p1_m1_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p2",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p2"],
                TEST_DATA["daily"]["m"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p2_m1_15"],
                TEST_DATA["daily"]["truth_p2_m1_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p1_incl_hist",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p1_m1_15_incl_hist"],
                TEST_DATA["daily"]["truth_p1_m1_30_incl_hist"],
                TEST_DATA["daily"]["incl_hist"],
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        params: ARIMAParams,
        model_params: Dict[str, Any],
        steps_1: int,
        steps_2: int,
        truth_1: pd.DataFrame,
        truth_2: pd.DataFrame,
        include_history: bool = False,
    ) -> None:
        m = ARIMAModel(data=ts, params=params)
        m.fit(**model_params)
        res_1 = m.predict(steps=steps_1, include_history=include_history)
        res_2 = m.predict(steps=steps_2, include_history=include_history)

        res_1.to_csv(f"/tmp/{name}_res_1.csv")
        res_2.to_csv(f"/tmp/{name}_res_2.csv")
        if include_history:
            res_1, res_2 = res_1.reset_index(drop=True), res_2.reset_index(drop=True)
            truth_1, truth_2 = (
                truth_1.reset_index(drop=True),
                truth_2.reset_index(drop=True),
            )

            assert res_1 is not None
            assert res_2 is not None
            assert truth_1 is not None
            assert truth_2 is not None

            assert_frame_equal(
                res_1,
                truth_1[res_1.columns],
                check_exact=False,
                check_less_precise=True,
                rtol=0.2,  # this doesnt need to be equality, just need to check if past approx is reasonably close to true so peak 20% err is fine
            )
            assert_frame_equal(
                res_2,
                truth_2[res_2.columns],
                check_exact=False,
                check_less_precise=True,
                rtol=0.2,
            )
        else:
            assert_frame_equal(
                res_1,
                truth_1,
                check_less_precise=True,
                check_exact=False,
                rtol=RTOL,
            )
            assert_frame_equal(
                res_2,
                truth_2,
                check_less_precise=True,
                check_exact=False,
                rtol=RTOL,
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "invalid_m",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["invalid_m"],
            ],
            [
                "invalid_ts",
                TEST_DATA["daily"]["invalid_ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m"],
            ],
        ]
    )
    def test_invalid_params(
        self,
        name: str,
        ts: TimeSeriesData,
        params: ARIMAParams,
        model_params: Dict[str, Any],
    ) -> None:
        with self.assertRaises(ValueError):
            ARIMAModel(data=ts, params=params).fit(**model_params)

    def test_exec_plot(self) -> None:
        m = ARIMAModel(data=TEST_DATA["daily"]["ts"], params=TEST_DATA["daily"]["p1"])
        m.fit(**TEST_DATA["daily"]["m"])
        _ = m.predict(steps=STEPS_1)
        m.plot()

    def test_name(self) -> None:
        m = ARIMAModel(data=TEST_DATA["daily"]["ts"], params=TEST_DATA["daily"]["p1"])
        self.assertEqual(m.__str__(), "ARIMA")

    def test_search_space(self) -> None:
        m = ARIMAModel(data=TEST_DATA["daily"]["ts"], params=TEST_DATA["daily"]["p1"])
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "p",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "d",
                    "type": "choice",
                    "values": list(range(1, 3)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "q",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )

    # Filler test
    def test_validate_params(self) -> None:
        params = TEST_DATA["daily"]["p1"]
        params.validate_params()


if __name__ == "__main__":
    unittest.main()
