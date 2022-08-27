# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.threshold_detector import StaticThresholdModel


class TestStaticThresholdModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(928)
        self.n = 72
        time = pd.date_range("2021-01-01", periods=self.n, freq="H")
        value = np.random.normal(1, 0.1, self.n)
        df = pd.DataFrame(
            {
                "time": time,
                "value": value,
            }
        )
        tsd = TimeSeriesData(
            df=df,
            time=df["time"],
        )
        model = StaticThresholdModel()
        self.anomaly_score = model.fit_predict(data=tsd)

    def test_streaming_length_match(self) -> None:
        self.assertEqual(len(self.anomaly_score.scores), self.n)
