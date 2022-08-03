# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.lstm import LSTMModel, LSTMParams
from kats.tests.models.test_models_dummy_data import (
    AIR_FCST_15_LSTM_PARAM_1_MODEL_1_MONTHLY,
    AIR_FCST_15_LSTM_PARAM_1_MODEL_2_MONTHLY,
    AIR_FCST_15_LSTM_PARAM_2_MODEL_1_MONTHLY,
    AIR_FCST_15_LSTM_PARAM_2_MODEL_2_MONTHLY,
    AIR_FCST_30_LSTM_PARAM_1_MODEL_1_MONTHLY,
    AIR_FCST_30_LSTM_PARAM_1_MODEL_2_MONTHLY,
    AIR_FCST_30_LSTM_PARAM_2_MODEL_1_MONTHLY,
    AIR_FCST_30_LSTM_PARAM_2_MODEL_2_MONTHLY,
    PT_FCST_15_LSTM_PARAM_1_MODEL_1_DAILY,
    PT_FCST_15_LSTM_PARAM_1_MODEL_2_DAILY,
    PT_FCST_15_LSTM_PARAM_2_MODEL_1_DAILY,
    PT_FCST_15_LSTM_PARAM_2_MODEL_2_DAILY,
    PT_FCST_30_LSTM_PARAM_1_MODEL_1_DAILY,
    PT_FCST_30_LSTM_PARAM_1_MODEL_2_DAILY,
    PT_FCST_30_LSTM_PARAM_2_MODEL_1_DAILY,
    PT_FCST_30_LSTM_PARAM_2_MODEL_2_DAILY,
)
from parameterized import parameterized

# pyre-fixme[5]: Global expression must be annotated.
AIR_TS = load_air_passengers()
PT_TS = TimeSeriesData(load_data("peyton_manning.csv"))
MULTI_DF = TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv"))
STEPS_1 = 15
STEPS_2 = 30
# pyre-fixme[5]: Global expression must be annotated.
TEST_DATA = {
    "monthly": {
        "ts": AIR_TS,
        "freq": "MS",
        "p1": LSTMParams(hidden_size=20, time_window=7, num_epochs=10),
        "p2": LSTMParams(hidden_size=10, time_window=3, num_epochs=4),
        "invalid_p": LSTMParams(hidden_size=-10, time_window=3, num_epochs=4),
        "invalid_ts": MULTI_DF,
        "m1": {
            "lr": 0.001,
        },
        "m2": {
            "lr": 0.0001,
        },
        "invalid_m1": {
            "lr": -0.001,
        },
        "steps_1": STEPS_1,
        "steps_2": STEPS_2,
        "truth_p1_m1_15": AIR_FCST_15_LSTM_PARAM_1_MODEL_1_MONTHLY,
        "truth_p2_m1_15": AIR_FCST_15_LSTM_PARAM_2_MODEL_1_MONTHLY,
        "truth_p1_m2_15": AIR_FCST_15_LSTM_PARAM_1_MODEL_2_MONTHLY,
        "truth_p2_m2_15": AIR_FCST_15_LSTM_PARAM_2_MODEL_2_MONTHLY,
        "truth_p1_m1_30": AIR_FCST_30_LSTM_PARAM_1_MODEL_1_MONTHLY,
        "truth_p2_m1_30": AIR_FCST_30_LSTM_PARAM_2_MODEL_1_MONTHLY,
        "truth_p1_m2_30": AIR_FCST_30_LSTM_PARAM_1_MODEL_2_MONTHLY,
        "truth_p2_m2_30": AIR_FCST_30_LSTM_PARAM_2_MODEL_2_MONTHLY,
    },
    "daily": {
        "ts": PT_TS,
        "freq": "D",
        "p1": LSTMParams(hidden_size=20, time_window=7, num_epochs=10),
        "p2": LSTMParams(hidden_size=10, time_window=3, num_epochs=4),
        "invalid_p": LSTMParams(hidden_size=-10, time_window=3, num_epochs=4),
        "m1": {
            "lr": 0.001,
        },
        "m2": {
            "lr": 0.0001,
        },
        "invalid_m1": {
            "lr": -0.001,
        },
        "steps_1": STEPS_1,
        "steps_2": STEPS_2,
        "truth_p1_m1_15": PT_FCST_15_LSTM_PARAM_1_MODEL_1_DAILY,
        "truth_p2_m1_15": PT_FCST_15_LSTM_PARAM_2_MODEL_1_DAILY,
        "truth_p1_m2_15": PT_FCST_15_LSTM_PARAM_1_MODEL_2_DAILY,
        "truth_p2_m2_15": PT_FCST_15_LSTM_PARAM_2_MODEL_2_DAILY,
        "truth_p1_m1_30": PT_FCST_30_LSTM_PARAM_1_MODEL_1_DAILY,
        "truth_p2_m1_30": PT_FCST_30_LSTM_PARAM_2_MODEL_1_DAILY,
        "truth_p1_m2_30": PT_FCST_30_LSTM_PARAM_1_MODEL_2_DAILY,
        "truth_p2_m2_30": PT_FCST_30_LSTM_PARAM_2_MODEL_2_DAILY,
    },
}


class LSTMModelTest(TestCase):
    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
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
                TEST_DATA["monthly"]["freq"],
            ],
            [
                "monthly_p2_m1",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p2"],
                TEST_DATA["monthly"]["m1"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p2_m1_15"],
                TEST_DATA["monthly"]["truth_p2_m1_30"],
                TEST_DATA["monthly"]["freq"],
            ],
            [
                "monthly_p1_m2",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["p1"],
                TEST_DATA["monthly"]["m2"],
                TEST_DATA["monthly"]["steps_1"],
                TEST_DATA["monthly"]["steps_2"],
                TEST_DATA["monthly"]["truth_p1_m2_15"],
                TEST_DATA["monthly"]["truth_p1_m2_30"],
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
                TEST_DATA["monthly"]["freq"],
            ],
            [
                "daily_p1_m1",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["p1"],
                TEST_DATA["daily"]["m1"],
                TEST_DATA["daily"]["steps_1"],
                TEST_DATA["daily"]["steps_2"],
                TEST_DATA["daily"]["truth_p1_m1_15"],
                TEST_DATA["daily"]["truth_p1_m1_30"],
                TEST_DATA["daily"]["freq"],
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
                TEST_DATA["daily"]["freq"],
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
                TEST_DATA["daily"]["freq"],
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
                TEST_DATA["daily"]["freq"],
            ],
        ]
    )
    def test_fcst(
        self,
        name: str,
        ts: TimeSeriesData,
        params: LSTMParams,
        model_params: Dict[str, int],
        steps_1: int,
        steps_2: int,
        truth_1: pd.DataFrame,
        truth_2: pd.DataFrame,
        freq: str,
    ) -> None:
        torch.manual_seed(0)
        m = LSTMModel(data=ts, params=params)
        m.fit(**model_params)
        res_1 = m.predict(
            steps=steps_1,
            freq=freq,
        ).reset_index(drop=True)
        res_2 = m.predict(
            steps=steps_2,
            freq=freq,
        ).reset_index(drop=True)
        assert_frame_equal(res_1, truth_1)
        assert_frame_equal(res_2, truth_2)

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
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
                TEST_DATA["monthly"]["p2"],
                TEST_DATA["monthly"]["invalid_m1"],
            ],
            [
                "invalid_ts",
                TEST_DATA["monthly"]["invalid_ts"],
                TEST_DATA["monthly"]["p2"],
                TEST_DATA["monthly"]["m1"],
            ],
        ]
    )
    def test_invalid_params(
        self,
        name: str,
        ts: TimeSeriesData,
        params: LSTMParams,
        model_params: Dict[str, int],
    ) -> None:
        with self.assertRaises(
            (ValueError, np.linalg.LinAlgError, NotImplementedError)
        ):
            m = LSTMModel(data=ts, params=params)
            m.fit(**model_params)

    def test_exec_plot(self) -> None:
        m = LSTMModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p2"]
        )
        m.fit(**TEST_DATA["monthly"]["m1"])
        _ = m.predict(steps=STEPS_1)
        m.plot()

    def test_name(self) -> None:
        m = LSTMModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p2"]
        )
        self.assertEqual(m.__str__(), "LSTM")

    def test_search_space(self) -> None:
        m = LSTMModel(
            data=TEST_DATA["monthly"]["ts"], params=TEST_DATA["monthly"]["p2"]
        )
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "hidden_size",
                    "type": "choice",
                    "values": [
                        1,
                        11,
                        21,
                        31,
                        41,
                        51,
                        61,
                        71,
                        81,
                        91,
                        101,
                        111,
                        121,
                        131,
                        141,
                        151,
                        161,
                        171,
                        181,
                        191,
                        201,
                        211,
                        221,
                        231,
                        241,
                        251,
                        261,
                        271,
                        281,
                        291,
                        301,
                        311,
                        321,
                        331,
                        341,
                        351,
                        361,
                        371,
                        381,
                        391,
                        401,
                        411,
                        421,
                        431,
                        441,
                        451,
                        461,
                        471,
                        481,
                        491,
                    ],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "time_window",
                    "type": "choice",
                    "values": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                    ],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "num_epochs",
                    "type": "choice",
                    "values": [
                        50,
                        100,
                        150,
                        200,
                        250,
                        300,
                        350,
                        400,
                        450,
                        500,
                        550,
                        600,
                        650,
                        700,
                        750,
                        800,
                        850,
                        900,
                        950,
                        1000,
                        1050,
                        1100,
                        1150,
                        1200,
                        1250,
                        1300,
                        1350,
                        1400,
                        1450,
                        1500,
                        1550,
                        1600,
                        1650,
                        1700,
                        1750,
                        1800,
                        1850,
                        1900,
                        1950,
                    ],
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        )

    # Filler test
    def test_validate_params(self) -> None:
        params = TEST_DATA["monthly"]["p2"]
        params.validate_params()


if __name__ == "__main__":
    unittest.main()
