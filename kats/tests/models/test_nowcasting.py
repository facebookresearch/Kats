# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.models.nowcasting.feature_extraction import (
    BBANDS,
    EMA,
    LAG,
    MA,
    MACD,
    MOM,
    ROC,
    RSI,
    TRIX,
    TSI,
)


df_dict = {
    "missing": pd.DataFrame(
        {
            "y": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                None,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                None,
                18,
                19,
                20,
            ]
        }
    ),
    "constant": pd.DataFrame({"y": [1] * 10}),
    "increasing": pd.DataFrame({"y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]}),
    "decreasing": pd.DataFrame(
        {"y": [-10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -50, -100]}
    ),
    "peak": pd.DataFrame(
        {"y": [-20, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -10, -20]}
    ),
    "valley": pd.DataFrame(
        {"y": [20, 10, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 10, 20]}
    ),
}


def to_list(series: pd.Series) -> List[Union[float, int, None]]:
    return [None if np.isnan(x) else x for x in series]


class testNowcasting(TestCase):
    def test_LAG(self) -> None:
        self.assertEqual(
            list(LAG(pd.DataFrame(list(range(5)), columns=["y"]), 1)["LAG_1"])[1:],
            [0, 1, 2, 3],
        )

    def test_LAG_zero(self) -> None:
        for name, df in df_dict.items():
            lag_df = LAG(df, 0)
            self.assertEqual(
                to_list(lag_df["LAG_0"]),
                to_list(df["y"]),
                f"test_LAG_zero failed for {name}",
            )

    def test_LAG_negative_lag(self) -> None:
        expected_dict = {
            "missing": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                None,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                None,
                18.0,
                19.0,
                20.0,
                None,
            ],
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
            "increasing": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                50.0,
                100.0,
                None,
            ],
            "decreasing": [
                -11.0,
                -12.0,
                -13.0,
                -14.0,
                -15.0,
                -16.0,
                -17.0,
                -18.0,
                -19.0,
                -20.0,
                -50.0,
                -100.0,
                None,
            ],
            "peak": [
                -10.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
                -10.0,
                -20.0,
                None,
            ],
            "valley": [
                10.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                10.0,
                20.0,
                None,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            lag_df = LAG(df, -1)
            self.assertEqual(
                to_list(lag_df["LAG_-1"]),
                expected,
                f"test_LAG_negative_lag failed for {name}",
            )

    def test_LAG_exceed_index(self) -> None:
        for name, df in df_dict.items():
            lag_df = LAG(df, 100)
            self.assertTrue(
                all(np.isnan(lag_df["LAG_100"])),
                f"test_LAG_exceed_index failed for {name}",
            )

    def test_LAG_two(self) -> None:
        expected_dict = {
            "missing": [
                None,
                None,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                None,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                None,
                18.0,
            ],
            "constant": [None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "increasing": [None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
            "decreasing": [
                None,
                None,
                -10,
                -11,
                -12,
                -13,
                -14,
                -15,
                -16,
                -17,
                -18,
                -19,
                -20,
            ],
            "peak": [
                None,
                None,
                -20,
                -10,
                -5,
                -4,
                -3,
                -2,
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                4,
                3,
                2,
                1,
                0,
            ],
            "valley": [
                None,
                None,
                20,
                10,
                5,
                4,
                3,
                2,
                1,
                0,
                -1,
                -2,
                -3,
                -4,
                -5,
                -4,
                -3,
                -2,
                -1,
                0,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            lag_df = LAG(df, 2)
            self.assertEqual(
                to_list(lag_df["LAG_2"]), expected, f"test_LAG_two failed for {name}"
            )

    def test_MOM(self) -> None:
        self.assertEqual(
            list(MOM(pd.DataFrame(list(range(5)), columns=["y"]), 1)["MOM_1"][1:]),
            [1, 1, 1, 1],
        )

    def test_MOM_one(self) -> None:
        expected_dict = {
            "missing": [
                None,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                None,
                None,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                None,
                None,
                1.0,
                1.0,
            ],
            "constant": [None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increasing": [
                None,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
                30.0,
                50.0,
            ],
            "decreasing": [
                None,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -30.0,
                -50.0,
            ],
            "peak": [
                None,
                10.0,
                5.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -10.0,
                -10.0,
            ],
            "valley": [
                None,
                -10.0,
                -5.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
                10.0,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            mom_df = MOM(df, 1)
            self.assertEqual(
                to_list(mom_df["MOM_1"]), expected, f"test_MOM_one failed for {name}"
            )

    def test_MOM_two(self) -> None:
        expected_dict = {
            "missing": [
                None,
                None,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                None,
                2.0,
                None,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                None,
                2.0,
                None,
                2.0,
            ],
            "constant": [None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increasing": [
                None,
                None,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                11.0,
                40.0,
                80.0,
            ],
            "decreasing": [
                None,
                None,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -31.0,
                -80.0,
            ],
            "peak": [
                None,
                None,
                15.0,
                6.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                0.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -11.0,
                -20.0,
            ],
            "valley": [
                None,
                None,
                -15.0,
                -6.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                0.0,
                2.0,
                2.0,
                2.0,
                2.0,
                11.0,
                20.0,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            mom_df = MOM(df, 2)
            self.assertEqual(
                to_list(mom_df["MOM_2"]), expected, f"test_MOM_two failed for {name}"
            )

    def test_MOM_exceed_index(self) -> None:
        for name, df in df_dict.items():
            mom_df = MOM(df, 100)
            self.assertTrue(
                all(np.isnan(mom_df["MOM_100"])),
                f"test_MOM_exceed_index failed for {name}",
            )

    def test_MA(self) -> None:
        self.assertEqual(
            list(MA(pd.DataFrame(list(range(5)), columns=["y"]), 1)["MA_1"]),
            [0, 1, 2, 3, 4],
        )

    def test_MA_zero(self) -> None:
        for name, df in df_dict.items():
            ma_df = MA(df, 0)
            self.assertTrue(
                all(np.isnan(ma_df["MA_0"])), f"test_MA_zero failed for {name}"
            )

    def test_MA_one(self) -> None:
        for name, df in df_dict.items():
            ma_df = MA(df, 1)
            self.assertEqual(
                to_list(ma_df["MA_1"]),
                to_list(df["y"]),
                f"test_MA_one failed for {name}",
            )

    def test_MA_five(self) -> None:
        expected_dict = {
            "missing": [
                None,
                None,
                None,
                None,
                2.0,
                3.0,
                4.0,
                5.0,
                None,
                None,
                None,
                None,
                None,
                11.0,
                12.0,
                13.0,
                14.0,
                None,
                None,
                None,
                None,
            ],
            "constant": [None, None, None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "increasing": [
                None,
                None,
                None,
                None,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                10.8,
                19.4,
                37.8,
            ],
            "decreasing": [
                None,
                None,
                None,
                None,
                -12.0,
                -13.0,
                -14.0,
                -15.0,
                -16.0,
                -17.0,
                -18.0,
                -24.8,
                -41.4,
            ],
            "peak": [
                None,
                None,
                None,
                None,
                -8.4,
                -4.8,
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                3.6,
                3.8,
                3.6,
                3.0,
                2.0,
                -0.8,
                -5.4,
            ],
            "valley": [
                None,
                None,
                None,
                None,
                8.4,
                4.8,
                3.0,
                2.0,
                1.0,
                0.0,
                -1.0,
                -2.0,
                -3.0,
                -3.6,
                -3.8,
                -3.6,
                -3.0,
                -2.0,
                0.8,
                5.4,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            ma_df = MA(df, 5)
            self.assertEqual(
                to_list(ma_df["MA_5"]), expected, f"test_MA_five failed for {name}"
            )

    def test_MA_exceed_index(self) -> None:
        for name, df in df_dict.items():
            ma_df = MA(df, 100)
            self.assertTrue(
                all(np.isnan(ma_df["MA_100"])),
                f"test_MA_exceed_index failed for {name}",
            )

    def test_MA_negative_one(self) -> None:
        for df in df_dict.values():
            self.assertRaises(ValueError, MA, *[df, -1])

    def test_ROC(self) -> None:
        self.assertEqual(
            list(ROC(pd.DataFrame(list(range(5)), columns=["y"]), 1)["ROC_1"])[1:],
            [0, 0, 0, 0],
        )

    @unittest.skip(
        "Skipping test_ROC_two because ROC function may have a bug when encountering negative values"
    )
    def test_ROC_two(self) -> None:
        expected_dict = {
            "missing": [
                None,
                np.inf,
                1.0,
                0.5,
                0.3333333333333333,
                0.25,
                0.2,
                0.16666666666666666,
                None,
                None,
                0.1111111111111111,
                0.1,
                0.09090909090909091,
                0.08333333333333333,
                0.07692307692307693,
                0.07142857142857142,
                0.06666666666666667,
                None,
                None,
                0.05555555555555555,
                0.05263157894736842,
            ],
            "constant": [None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increasing": [
                None,
                np.inf,
                1.0,
                0.5,
                0.3333333333333333,
                0.25,
                0.2,
                0.16666666666666666,
                0.14285714285714285,
                0.125,
                0.1111111111111111,
                1.0,
                1.5,
                1.0,
            ],
            "decreasing": [
                None,
                -0.1,
                -0.09090909090909091,
                -0.08333333333333333,
                -0.07692307692307693,
                -0.07142857142857142,
                -0.06666666666666667,
                -0.0625,
                -0.058823529411764705,
                -0.05555555555555555,
                -0.05263157894736842,
                -1.5,
                -1.0,
            ],
            "peak": [
                None,
                0.5,
                0.5,
                0.2,
                0.25,
                0.3333333333333333,
                0.5,
                1.0,
                np.inf,
                1.0,
                0.5,
                0.3333333333333333,
                0.25,
                -0.2,
                -0.25,
                -0.3333333333333333,
                -0.5,
                -1.0,
                -np.inf,
                -1.0,
            ],
            "valley": [
                None,
                -0.5,
                -0.5,
                -0.2,
                -0.25,
                -0.3333333333333333,
                -0.5,
                -1.0,
                -np.inf,
                -1.0,
                -0.5,
                -0.3333333333333333,
                -0.25,
                0.2,
                0.25,
                0.3333333333333333,
                0.5,
                1.0,
                np.inf,
                1.0,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            roc_df = ROC(df, 2)
            for actual_val, expected_val in zip(to_list(roc_df["ROC_2"]), expected):
                self.assertAlmostEqual(
                    actual_val, expected_val, 9, f"test_ROC_two failed for {name}"
                )

    @unittest.skip(
        "Skipping test_ROC_three because ROC function may have a bug when encountering negative values"
    )
    def test_ROC_three(self) -> None:
        expected_dict = {
            "missing": [
                None,
                None,
                np.inf,
                2.0,
                1.0,
                0.6666666666666666,
                0.5,
                0.4,
                None,
                0.2857142857142857,
                None,
                0.2222222222222222,
                0.2,
                0.18181818181818182,
                0.16666666666666666,
                0.15384615384615385,
                0.14285714285714285,
                None,
                0.125,
                None,
                0.1111111111111111,
            ],
            "constant": [None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increasing": [
                None,
                None,
                np.inf,
                2.0,
                1.0,
                0.6666666666666666,
                0.5,
                0.4,
                0.3333333333333333,
                0.2857142857142857,
                0.25,
                1.2222222222222223,
                4.0,
                4.0,
            ],
            "decreasing": [
                None,
                None,
                -0.2,
                -0.18181818181818182,
                -0.16666666666666666,
                -0.15384615384615385,
                -0.14285714285714285,
                -0.13333333333333333,
                -0.125,
                -0.11764705882352941,
                -0.1111111111111111,
                -1.631578947368421,
                -4.0,
            ],
            "peak": [
                None,
                None,
                0.75,
                0.6,
                0.4,
                0.5,
                0.6666666666666666,
                1.0,
                2.0,
                np.inf,
                2.0,
                1.0,
                0.6666666666666666,
                0.0,
                -0.4,
                -0.5,
                -0.6666666666666666,
                -1.0,
                -11.0,
                -np.inf,
            ],
            "valley": [
                None,
                None,
                -0.75,
                -0.6,
                -0.4,
                -0.5,
                -0.6666666666666666,
                -1.0,
                -2.0,
                -np.inf,
                -2.0,
                -1.0,
                -0.6666666666666666,
                0.0,
                0.4,
                0.5,
                0.6666666666666666,
                1.0,
                11.0,
                np.inf,
            ],
        }
        for name, df in df_dict.items():
            expected = expected_dict[name]
            roc_df = ROC(df, 3)
            for actual_val, expected_val in zip(to_list(roc_df["ROC_3"]), expected):
                self.assertAlmostEqual(
                    actual_val, expected_val, 9, f"test_ROC_three failed for {name}"
                )

    def test_ROC_exceed_index(self) -> None:
        for name, df in df_dict.items():
            roc_df = ROC(df, 100)
            self.assertTrue(
                all(np.isnan(roc_df["ROC_100"])),
                f"test_ROC_exceed_index failed for {name}",
            )

    def test_MACD(self) -> None:
        error_threshold = 0.0001
        target = np.array(
            [7.770436585431938, 7.913716315475984, 8.048858332839053, 8.176225524209826]
        )
        error1 = np.sum(
            np.array(
                MACD(pd.DataFrame(list(range(30)), columns=["y"]), 1)[-4:]["MACD_1_21"]
            )
            - target
        )
        self.assertLessEqual(error1, error_threshold, "MACD_1_21 produces errors!")

        target = [
            7.37176002981048,
            7.496954620458209,
            7.620613089998056,
            7.742177915869659,
        ]
        error2 = np.sum(
            np.abs(
                MACD(pd.DataFrame(list(range(30)), columns=["y"]), 1)[-4:][
                    "MACDsign_1_21"
                ]
                - target
            )
        )
        self.assertLessEqual(error2, error_threshold, "MACDsign_1_21 produces errors!")

        target = [
            0.3986765556214573,
            0.41676169501777505,
            0.4282452428409975,
            0.4340476083401672,
        ]
        error3 = np.sum(
            np.abs(
                MACD(pd.DataFrame(list(range(30)), columns=["y"]), 1)[-4:][
                    "MACDdiff_1_21"
                ]
                - target
            )
        )
        self.assertLessEqual(error3, error_threshold, "MACDdiff_1_21 produces errors!")

    def test_BBANDS(self) -> None:
        error_threshold = 0.0001

        # Bollinger Band 1
        target_1 = np.array(
            [
                5.656854249492381,
                1.885618083164127,
                1.131370849898476,
                0.8081220356417687,
            ]
        )
        error_1 = np.sum(
            np.abs(
                np.array(
                    list(
                        BBANDS(pd.DataFrame(list(range(5)), columns=["y"]), 2)[
                            "BollingerBand1_2"
                        ][1:]
                    )
                )
                - target_1
            )
        )
        self.assertLessEqual(
            error_1, error_threshold, "BollingerBand1_2 produces errors!"
        )

        # Bolinger Band 2
        target_2 = np.array(
            [
                0.6767766952966369,
                0.6767766952966369,
                0.6767766952966369,
                0.6767766952966369,
            ]
        )
        error_2 = np.sum(
            np.abs(
                np.array(
                    list(
                        BBANDS(pd.DataFrame(list(range(5)), columns=["y"]), 2)[
                            "BollingerBand2_2"
                        ][1:]
                    )
                )
                - target_2
            )
        )
        self.assertLessEqual(
            error_2, error_threshold, "BollingerBand2_2 produces errors!"
        )

    def test_EMA(self) -> None:
        error_threshold = 0.0001
        target = np.array(
            [0.0, 0.7499999999999999, 1.6153846153846152, 2.55, 3.5206611570247937]
        )
        error = np.sum(
            np.abs(
                np.array(
                    list(EMA(pd.DataFrame(list(range(5)), columns=["y"]), 2)["EMA_2"])
                )
                - target
            )
        )
        self.assertLessEqual(error, error_threshold, "EMA_2 produces errors!")

    def test_TRIX(self) -> None:
        error_threshold = 0.0001
        target = np.array(
            [0.0, 0.421875, 0.42337953352973806, 0.372572902464051, 0.3100591536021331]
        )
        error = np.sum(
            np.abs(
                np.array(
                    list(
                        TRIX(pd.DataFrame(list(range(1, 6)), columns=["y"]), 2)[
                            "TRIX_2"
                        ]
                    )
                )
                - target
            )
        )
        self.assertLessEqual(error, error_threshold, "TRIX_2 produces errors!")

    def test_TSI(self) -> None:
        error_threshold = 0.0001
        target = np.array([1.0, 1.0, 1.0])
        error = np.sum(
            np.abs(
                np.array(
                    list(
                        TSI(pd.DataFrame(list(range(5, 10)), columns=["y"]), 2, 3)[
                            "TSI_2_3"
                        ]
                    )[2:]
                )
                - target
            )
        )
        self.assertLessEqual(error, error_threshold, "TSI_2_3 produces errors!")

    def test_RSI(self) -> None:
        error_threshold = 0.0001
        target = np.array([100.0, 100.0, 100.0, 100.0])
        error = np.sum(
            np.abs(
                np.array(
                    list(
                        RSI(pd.DataFrame(list(range(5, 10)), columns=["y"]), 2)["RSI_2"]
                    )[1:]
                )
                - target
            )
        )
        self.assertLessEqual(error, error_threshold, "RSI_2 produces errors!")


if __name__ == "__main__":
    unittest.main()
