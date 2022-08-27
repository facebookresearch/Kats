# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, Optional, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_15_SARIMA_PARAM_1_MODEL_1,
    AIR_FCST_15_SARIMA_PARAM_1_MODEL_1_INCL_HIST,
    AIR_FCST_15_SARIMA_PARAM_1_MODEL_2,
    AIR_FCST_15_SARIMA_PARAM_2_MODEL_1,
    AIR_FCST_15_SARIMA_PARAM_2_MODEL_2,
    AIR_FCST_30_SARIMA_PARAM_1_MODEL_1,
    AIR_FCST_30_SARIMA_PARAM_1_MODEL_1_INCL_HIST,
    AIR_FCST_30_SARIMA_PARAM_1_MODEL_2,
    AIR_FCST_30_SARIMA_PARAM_2_MODEL_1,
    AIR_FCST_30_SARIMA_PARAM_2_MODEL_2,
    EXOG_FCST_15_SARIMA_PARAM_EXOG_MODEL_1,
)
from parameterized.parameterized import parameterized

AIR_TS: pd.DataFrame = load_air_passengers()
MULTI_DF: pd.DataFrame = load_data("multivariate_anomaly_simulated_data.csv")
STEPS_1 = 15
STEPS_2 = 30
TEST_DATA: Dict[str, Dict[str, Any]] = {
    "monthly": {
        "ts": AIR_TS,
        "invalid_ts": TimeSeriesData(
            load_data("multivariate_anomaly_simulated_data.csv")
        ),
        "freq": "MS",
        "p1": SARIMAParams(
            p=1,
            d=1,
            q=1,
        ),
        "p2": SARIMAParams(
            p=2,
            d=1,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
        ),
        "invalid_p": SARIMAParams(
            p=-1,
            d=1,
            q=1,
        ),
        "m1": {
            "cov_type": None,
            "method": "lbfgs",
            "maxiter": 50,
            "full_output": False,
            "optim_score": None,
            "optim_complex_step": False,
            "optim_hessian": None,
        },
        "m2": {
            "cov_type": None,
            "method": "newton",
            "maxiter": 1,
            "full_output": False,
            "optim_score": "harvey",
            "optim_complex_step": True,
            "optim_hessian": "opg",
        },
        "invalid_m1": {
            "start_params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "cov_type": "None",
            "method": "lbfgs",
            "maxiter": 50,
            "full_output": False,
            "optim_score": None,
            "optim_complex_step": False,
            "optim_hessian": None,
        },
        "invalid_m2": {
            "cov_type": None,
            "method": "invalid",
            "maxiter": 50,
            "full_output": False,
            "optim_score": None,
            "optim_complex_step": False,
            "optim_hessian": None,
        },
        "steps_1": STEPS_1,
        "steps_2": STEPS_2,
        "no_incl_hist": False,
        "incl_hist": True,
        "truth_p1_m1_15": AIR_FCST_15_SARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_15": AIR_FCST_15_SARIMA_PARAM_2_MODEL_1,
        "truth_p1_m2_15": AIR_FCST_15_SARIMA_PARAM_1_MODEL_2,
        "truth_p2_m2_15": AIR_FCST_15_SARIMA_PARAM_2_MODEL_2,
        "truth_p1_m1_30": AIR_FCST_30_SARIMA_PARAM_1_MODEL_1,
        "truth_p2_m1_30": AIR_FCST_30_SARIMA_PARAM_2_MODEL_1,
        "truth_p1_m2_30": AIR_FCST_30_SARIMA_PARAM_1_MODEL_2,
        "truth_p2_m2_30": AIR_FCST_30_SARIMA_PARAM_2_MODEL_2,
        "truth_p1_m1_15_incl_hist": AIR_FCST_15_SARIMA_PARAM_1_MODEL_1_INCL_HIST,
        "truth_p1_m1_30_incl_hist": AIR_FCST_30_SARIMA_PARAM_1_MODEL_1_INCL_HIST,
    },
}


class SARIMAModelTest(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "monthly_p1_m1",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["m1"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p1_m1_15"],
                TEST_DATA["monthly"]["truth_p1_m1_30"],
                TEST_DATA["monthly"]["no_incl_hist"],
                TEST_DATA["monthly"]["freq"],
            ],
            # TODO: Figure out why results aren't deterministic: T103684646
            # [
            #     "monthly_p2_m1",
            #     TEST_DATA["monthly"]["ts"],
            #     TEST_DATA["monthly"]["p2"],
            #     TEST_DATA["monthly"]["m1"],
            #     TEST_DATA["monthly"]["steps_1"],
            #     TEST_DATA["monthly"]["steps_2"],
            #     TEST_DATA["monthly"]["truth_p2_m1_15"],
            #     TEST_DATA["monthly"]["truth_p2_m1_30"],
            #     TEST_DATA["monthly"]["no_incl_hist"],
            #     TEST_DATA["monthly"]["freq"],
            # ],
            [
                "monthly_p1_m2",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["m2"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p1_m2_15"],
                TEST_DATA["monthly"]["truth_p1_m2_30"],
                TEST_DATA["monthly"]["no_incl_hist"],
                TEST_DATA["monthly"]["freq"],
            ],
            [
                "monthly_p2_m2",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p2"],
                TEST_DATA["monthly"]["m2"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p2_m2_15"],
                TEST_DATA["monthly"]["truth_p2_m2_30"],
                TEST_DATA["monthly"]["no_incl_hist"],
                TEST_DATA["monthly"]["freq"],
            ],
            [
                "monthly_p1_m1_incl_hist",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["m1"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p1_m1_15_incl_hist"],
                TEST_DATA["monthly"]["truth_p1_m1_30_incl_hist"],
                TEST_DATA["monthly"]["incl_hist"],
                TEST_DATA["monthly"]["freq"],
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        params: SARIMAParams,
        model_params: Dict[str, Optional[Union[str, int, bool]]],
        steps_1: int,
        steps_2: int,
        truth_1: pd.DataFrame,
        truth_2: pd.DataFrame,
        include_history: bool = False,
        freq: str = "MS",
    ) -> None:
        m = SARIMAModel(data=ts, params=params)
        # pyre-fixme[6]: Incompatible parameter type...
        m.fit(**model_params)
        res_1 = m.predict(
            steps=steps_1, include_history=include_history, freq=freq
        ).reset_index(
            drop=True,
        )
        res_2 = m.predict(
            steps=steps_2, include_history=include_history, freq=freq
        ).reset_index(
            drop=True,
        )
        assert_frame_equal(truth_1, res_1, rtol=0.01)
        assert_frame_equal(truth_2, res_2, rtol=0.01)

    def test_exog(self) -> None:
        # Prep data
        steps = STEPS_1
        df = MULTI_DF
        ts_original = TimeSeriesData(df)
        endog = df["0"][:-steps]
        time = ts_original.time_to_index()[:-steps]
        exog = df["1"][:-steps].values
        fcst_exog = df["1"][-steps:].values  # exog to be used for predictions
        ts = TimeSeriesData(value=endog, time=time)
        params = SARIMAParams(
            p=2,
            d=1,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
            exog=exog,
        )

        # Fit/Predict
        m = SARIMAModel(ts, params)
        m.fit(**TEST_DATA["monthly"]["m1"])
        res = m.predict(steps=steps, exog=fcst_exog, freq="D").reset_index(
            drop=True,
        )

        # Compare against truth
        assert_frame_equal(EXOG_FCST_15_SARIMA_PARAM_EXOG_MODEL_1, res, rtol=0.01)

        # Should raise a ValueError if exogenous variables aren't used to predict
        self.assertRaises(ValueError, m.predict, steps, "D")

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "invalid_p",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["invalid_p"],
                TEST_DATA["monthly"]["m1"],
            ],
            [
                "invalid_m1",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["invalid_m1"],
            ],
            [
                "invalid_m2",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["invalid_m2"],
            ],
            [
                "invalid_ts",
                TEST_DATA["monthly"]["invalid_ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["m1"],
            ],
        ]
    )
    def test_invalid_params(
        self,
        name: str,
        ts: TimeSeriesData,
        params: SARIMAParams,
        model_params: Dict[str, Optional[Union[str, int, bool]]],
    ) -> None:
        with self.assertRaises(
            (ValueError, np.linalg.LinAlgError, NotImplementedError)
        ):
            m = SARIMAModel(data=ts, params=params)
            # pyre-fixme[6]: Incompatible parameter type...
            m.fit(**model_params)

    def test_exec_plot(self) -> None:
        m = SARIMAModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p1"]
        )
        m.fit(**TEST_DATA["monthly"]["m1"])
        _ = m.predict(steps=STEPS_1)
        m.plot()

    def test_name(self) -> None:
        m = SARIMAModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p1"]
        )
        self.assertEqual(m.__str__(), "SARIMA")

    def test_search_space(self) -> None:
        m = SARIMAModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p1"]
        )
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
                {
                    "name": "seasonal_order",
                    "type": "choice",
                    "values": [
                        (1, 0, 1, 7),
                        (1, 0, 2, 7),
                        (2, 0, 1, 7),
                        (2, 0, 2, 7),
                        (1, 1, 1, 7),
                        (0, 1, 1, 7),
                    ],
                    # Note: JSON representation must be 'int', 'float', 'bool' or 'str'.
                    # so we use 'str' here instead of 'Tuple'
                    # when doing HPT, we need to convert it back to tuple
                    "value_type": "str",
                },
                {
                    "name": "trend",
                    "type": "choice",
                    "values": ["n", "c", "t", "ct"],
                    "value_type": "str",
                },
            ],
        )

    # Filler test
    def test_validate_params(self) -> None:
        params = TEST_DATA["monthly"]["p1"]
        params.validate_params()


if __name__ == "__main__":
    unittest.main()
