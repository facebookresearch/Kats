# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
