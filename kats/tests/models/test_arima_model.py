# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, Optional, Union
from unittest import TestCase

import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_data  # @manual
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.tests.models.test_models_dummy_data import (
    PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1,
    PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1_INCL_HIST,
    PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_2,
    PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_1,
    PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_2,
    PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1,
    PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1_INCL_HIST,
    PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_2,
    PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_1,
    PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_2,
)
from parameterized.parameterized import parameterized

STEPS_1 = 15
STEPS_2 = 30
# pyre-fixme[5]: Global expression must be annotated.
TEST_DATA = {
    "daily": {
        "ts": TimeSeriesData(load_data("peyton_manning.csv")),
        "invalid_ts": TimeSeriesData(
            load_data("multivariate_anomaly_simulated_data.csv")
        ),
        # "ts_nan": TimeSeriesData(df=PEYTON_INPUT_NAN),
        "freq": "D",
        "p1": ARIMAParams(p=1, d=1, q=1),
        "p2": ARIMAParams(p=1, d=1, q=3),
        "invalid_p": ARIMAParams(p=1, d=2, q=1),
        "m1": {
            "method": "css-mle",
            "trend": "c",
            "solver": "lbfgs",
            "maxiter": 500,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": None,
        },
        "m2": {
            "method": "css",
            "trend": "nc",
            "solver": "newton",
            "maxiter": 2,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": 10,
        },
        "invalid_m1": {
            "method": "css-mle",
            "trend": "c",
            "solver": "lbfgs",
            "maxiter": 500,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": 2,
        },
        "invalid_m2": {
            "method": "css-mle",
            "trend": "c",
            "solver": "invalid",
            "maxiter": 500,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": None,
        },
        "invalid_m3": {
            "method": "css-mle",
            "trend": "invalid",
            "solver": "lbfgs",
            "maxiter": 500,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": None,
        },
        "invalid_m4": {
            "method": "invalid",
            "trend": "c",
            "solver": "lbfgs",
            "maxiter": 500,
            "full_output": False,
            "disp": 0,
            "start_ar_lags": None,
        },
        "steps_1": STEPS_1,
        "steps_2": STEPS_2,
        "no_incl_hist": False,
        "incl_hist": True,
        "truth_p1_m1_15": PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_15": PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_1,
        "truth_p1_m2_15": PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_2,
        "truth_p2_m2_15": PEYTON_FCST_15_ARIMA_PARAM_2_MODEL_2,
        "truth_p1_m1_30": PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_30": PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_1,
        "truth_p1_m2_30": PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_2,
        "truth_p2_m2_30": PEYTON_FCST_30_ARIMA_PARAM_2_MODEL_2,
        "truth_p1_m1_15_incl_hist": PEYTON_FCST_15_ARIMA_PARAM_1_MODEL_1_INCL_HIST,
        "truth_p1_m1_30_incl_hist": PEYTON_FCST_30_ARIMA_PARAM_1_MODEL_1_INCL_HIST,
    },
}


class ARIMAModelTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "daily_p1_m1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m1"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p1_m1_15"],
                TEST_DATA["daily"]["truth_p1_m1_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p2_m1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p2"],
                TEST_DATA["daily"]["m1"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p2_m1_15"],
                TEST_DATA["daily"]["truth_p2_m1_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p1_m2",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m2"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p1_m2_15"],
                TEST_DATA["daily"]["truth_p1_m2_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p2_m2",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p2"],
                TEST_DATA["daily"]["m2"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p2_m2_15"],
                TEST_DATA["daily"]["truth_p2_m2_30"],
                TEST_DATA["daily"]["no_incl_hist"],
            ],
            [
                "daily_p1_m1_incl_hist",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m1"],
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
        model_params: Dict[str, Optional[Union[str, int, bool]]],
        steps_1: int,
        steps_2: int,
        truth_1: pd.DataFrame,
        truth_2: pd.DataFrame,
        include_history: bool = False,
    ) -> None:
        m = ARIMAModel(data=ts, params=params)
        # pyre-fixme[6]: Incompatible parameter type...
        m.fit(**model_params)
        res_1 = m.predict(steps=steps_1, include_history=include_history)
        res_2 = m.predict(steps=steps_2, include_history=include_history)
        if include_history:
            assert_frame_equal(
                res_1.reset_index(drop=True),
                truth_1.reset_index(drop=True),
                check_exact=False,
                check_less_precise=True,
                rtol=0.001,
            )
            assert_frame_equal(
                res_2.reset_index(drop=True),
                truth_2.reset_index(drop=True),
                check_exact=False,
                check_less_precise=True,
                rtol=0.001,
            )
        else:
            assert_frame_equal(
                res_1,
                truth_1,
                check_less_precise=True,
                check_exact=False,
                rtol=0.001,
            )
            assert_frame_equal(
                res_2,
                truth_2,
                check_less_precise=True,
                check_exact=False,
                rtol=0.001,
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "invalid_p",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["invalid_p"],
                TEST_DATA["daily"]["m1"],
            ],
            [
                "invalid_m1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["invalid_m1"],
            ],
            [
                "invalid_m2",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["invalid_m2"],
            ],
            [
                "invalid_m3",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["invalid_m3"],
            ],
            [
                "invalid_m4",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["invalid_m4"],
            ],
            [
                "invalid_ts",
                TEST_DATA["daily"]["invalid_ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m1"],
            ],
        ]
    )
    def test_invalid_params(
        self,
        name: str,
        ts: TimeSeriesData,
        params: ARIMAParams,
        model_params: Dict[str, Optional[Union[str, int, bool]]],
    ) -> None:
        with self.assertRaises(ValueError):
            m = ARIMAModel(data=ts, params=params)
            # pyre-fixme[6]: Incompatible parameter type...
            m.fit(**model_params)

    def test_exec_plot(self) -> None:
        m = ARIMAModel(data=TEST_DATA["daily"]["ts"], params=TEST_DATA["daily"]["p1"])
        m.fit(**TEST_DATA["daily"]["m1"])
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
