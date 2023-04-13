# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.ml_ar import MLARModel, MLARParams
from parameterized.parameterized import parameterized

single_univariate_ts = [
    TimeSeriesData(
        pd.DataFrame(
            {
                "time": pd.date_range("2022-05-06", periods=10),
                "value": np.arange(10) + np.sin(np.arange(10) * np.pi),
            }
        )
    )
]
single_univariate_param = MLARParams(
    horizon=3, input_window=2, target_variable=["value"], freq="D"
)


Multi_Data_df_1 = pd.DataFrame(
    {
        "time": pd.date_range("2022-05-06", periods=20),
        "cat_1": ["US"] * 20,
        "cat_2": ["Wed"] * 20,
        "reg_1": np.arange(20),
        "reg_2": np.arange(20) + 15 * np.sin(np.arange(20)),
        "target_1": np.arange(20) + 15 * np.cos(np.arange(20)),
        "target_2": np.arange(20)
        + 15 * np.cos(np.arange(20))
        - 15 * np.sin(np.arange(20)),
        "future_reg_1": np.sin(np.arange(20)),
        "future_reg_2": np.cos(np.arange(20)),
    }
)

Multi_Data_df_1_short: pd.DataFrame = Multi_Data_df_1.copy()
Multi_Data_df_1_short.loc[10:, ["target_1", "target_2"]] = np.nan

shr_st_ts_1 = TimeSeriesData(Multi_Data_df_1[["time", "reg_1", "target_1"]])

shr_st_param = MLARParams(
    horizon=2,
    input_window=5,
    target_variable=["target_1"],
    cov_history_input_windows={"reg_1": 3},
)

mhr_mt_ts_1 = TimeSeriesData(
    Multi_Data_df_1[["time", "reg_1", "reg_2", "target_1", "target_2"]],
)

mhr_mt_param = MLARParams(
    horizon=3,
    input_window=5,
    target_variable=["target_1", "target_2"],
    cov_history_input_windows={"reg_1": 3, "reg_2": 2},
    calendar_features=["day", "year"],
)

mht_mt_sc_ts_1 = TimeSeriesData(
    Multi_Data_df_1[["time", "reg_1", "reg_2", "target_1", "target_2", "cat_1"]],
    categorical_var=["cat_1"],
)

mht_mt_sc_param = MLARParams(
    horizon=5,
    input_window=5,
    target_variable=["target_1", "target_2"],
    cov_history_input_windows={"reg_1": 3, "reg_2": 2},
    categoricals=["cat_1"],
    calendar_features=["weekday"],
    fourier_features_period=[],
    fourier_features_order=[],
    fourier_features_offset=0,
)

mht_mt_sc_nf_param = MLARParams(
    horizon=5,
    input_window=5,
    target_variable=["target_1", "target_2"],
    cov_history_input_windows={"reg_1": 3, "reg_2": 2},
    categoricals=["cat_1"],
    calendar_features=[],
    fourier_features_period=[],
    fourier_features_order=[],
    fourier_features_offset=0,
)


mht_mft_mt_mc_ts_1 = TimeSeriesData(Multi_Data_df_1, categorical_var=["cat_1", "cat_2"])
mht_mft_mt_mc_param = MLARParams(
    horizon=5,
    input_window=5,
    target_variable=["target_1", "target_2"],
    cov_history_input_windows={"reg_1": 3, "reg_2": 2},
    cov_future_input_windows={"future_reg_1": 1, "future_reg_2": 2},
    categoricals=["cat_1", "cat_2"],
)


class TestMLARModel(TestCase):
    # pyre-fixme
    @parameterized.expand(
        [
            (
                "single_univariate_model",
                single_univariate_ts,
                single_univariate_param,
                20,
            ),
            ("uni_target_uni_regressor", [shr_st_ts_1], shr_st_param, 1),
            ("mul_target_mul_regressor_cal_feature", [mhr_mt_ts_1], mhr_mt_param, 2),
            ("uni_target_uni_regressor_mul_ts", [shr_st_ts_1] * 2, shr_st_param, 1),
            (
                "mul_target_mul_regressor_cal_feature_mul_ts",
                [mhr_mt_ts_1] * 2,
                mhr_mt_param,
                2,
            ),
            (
                "mul_target_mul_regressor_uni_cat_mul_ts",
                [mht_mt_sc_ts_1] * 2,
                mht_mt_sc_param,
                5,
            ),
            (
                "mul_target_mul_regressor_mul_cat_single_ts",
                [mht_mft_mt_mc_ts_1],
                mht_mft_mt_mc_param,
                3,
            ),
        ]
    )
    def test_ml_ar_model(
        self, name: str, data: List[TimeSeriesData], params: MLARParams, steps: int
    ) -> None:
        model = MLARModel(params)
        model.train(data)
        fcst = model.predict(steps)
        for t in fcst:
            # pyre-fixme
            self.assertEqual(fcst[t]["forecast"].isna().sum(), 0)
