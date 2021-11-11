from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.threshold_detector import (
    StaticThresholdModel,
)


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
        model = StaticThresholdModel(
            upper_threshold=0.5,
            lower_threshold=0.2,
        )
        self.anomaly_score = model.fit_predict(data=tsd)

    def test_streaming_length_match(self) -> None:
        self.assertEqual(len(self.anomaly_score.scores), self.n)
