# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.models.nowcasting.feature_extraction import (
    LAG,
    MOM,
    MA,
    ROC,
    MACD,
    BBANDS,
    TRIX,
    EMA,
    TSI,
    RSI,
)


class testNowcasting(TestCase):
    def test_LAG(self) -> None:
        self.assertEqual(
            list(LAG(pd.DataFrame(list(range(5)), columns=["y"]), 1)["LAG_1"])[1:],
            [0, 1, 2, 3],
        )

    def test_MOM(self) -> None:
        self.assertEqual(
            list(MOM(pd.DataFrame(list(range(5)), columns=["y"]), 1)["MOM_1"][1:]),
            [1, 1, 1, 1],
        )

    def test_MA(self) -> None:
        self.assertEqual(
            list(MA(pd.DataFrame(list(range(5)), columns=["y"]), 1)["MA_1"]),
            [0, 1, 2, 3, 4],
        )

    def test_ROC(self) -> None:
        self.assertEqual(
            list(ROC(pd.DataFrame(list(range(5)), columns=["y"]), 1)["ROC_1"])[1:],
            [0, 0, 0, 0],
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
