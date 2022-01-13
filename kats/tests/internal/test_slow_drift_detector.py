# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.slow_drift_detector import (
    SlowDriftDetectorModel,
    time_series_to_data_points,
)


class TestSlowDriftDetector(TestCase):
    def generate_ts_with_slope_and_seasonality(
        self,
        num_points: int = 1000,
        end_time: int = 0,
        level: float = 10000.0,  # Initial Level
        slope: float = 20.0,  # Change per day
        seasonality: float = 0.9,  # Multiplier on weekends
    ) -> TimeSeriesData:
        granularity = 86400
        if end_time <= 0:
            end_time = int(time.time())
        start_time = end_time - num_points * granularity
        times = []
        values = []
        for i in range(num_points):
            timestamp = start_time + i * granularity
            value = level + i * slope
            if i % 7 >= 5:
                value *= seasonality
            times.append(timestamp)
            values.append(value)
        return TimeSeriesData(
            time=pd.Series(times),
            value=pd.Series(values),
            use_unix_time=True,
            unix_time_units="s",
        )

    def test_time_series_data_to_data_points(self) -> None:
        ts = TimeSeriesData(
            df=pd.DataFrame.from_dict(
                {
                    "time": [0, 1],
                    "value": [1.5, 3.0],
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
        )

        result = time_series_to_data_points(ts)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].timestamp, 0)
        self.assertEqual(result[0].value, 1.5)
        self.assertEqual(result[1].timestamp, 1)
        self.assertEqual(result[1].value, 3.0)

    def test_constructor(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=100,
            algorithm_version=2,
            seasonality_period=14,
            seasonality_num_points=12,
        )
        slow_drift_model = model.model
        model_data = slow_drift_model.get_model_data()
        self.assertEquals(model_data.slowDriftWindow, 100)
        self.assertEquals(model_data.algorithmVersion, 2)
        esp = slow_drift_model.get_parameters()
        self.assertEquals(esp.seasonalityPeriod, 14)
        self.assertEquals(esp.seasonalityNumPoints, 12)
        self.assertIsNone(slow_drift_model.get_ongoing_anomaly())

    def test_holt_winters(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=0,
            seasonality_num_points=0,
        )

        ts = self.generate_ts_with_slope_and_seasonality()
        result = model.fit_predict(ts)
        anomaly_scores = result.scores
        # Anomalies may happen near the beginning due to initialization.  Assert that
        # none occur after things have settled down.
        scores_after_t100 = anomaly_scores.value[100:]
        self.assertEquals(len(scores_after_t100[scores_after_t100 > 0]), 0)
        slow_drift_model = model.model
        slow_drift_model.pruneData()
        # By the end of the series, trend should converge and not have much noise
        # This expected value is due to the seasonal adjustment multiplier
        expected_slope = 20.0 * (5 * 1.0 + 2 * 0.9) / 7
        for trend_dp in slow_drift_model.get_trend_series():
            self.assertAlmostEqual(trend_dp.value, expected_slope, delta=0.001)
        # Level should be linear, with the expected slope as above
        level_series = slow_drift_model.get_level_series()
        for i, level_dp in enumerate(level_series):
            expected_level = level_series[0].value + i * expected_slope
            self.assertAlmostEqual(level_dp.value, expected_level, delta=0.001)
        # Seasonality state should be such that the weekday values are all equal
        # and weekend values are 0.9x the weekdays
        seasonality = slow_drift_model._model_data.recentData.seasonalitySeries
        self.assertAlmostEqual(seasonality[0].value, seasonality[1].value, delta=0.001)
        self.assertAlmostEqual(
            seasonality[5].value, 0.9 * seasonality[0].value, delta=0.001
        )

    def test_serialize(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=14,
            seasonality_num_points=12,
        )

        ts = self.generate_ts_with_slope_and_seasonality()
        model.fit(ts)

        serialized_model = model.serialize()
        self.assertIsInstance(serialized_model, bytes)
        loaded_model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=14,
            seasonality_num_points=12,
            serialized_model=serialized_model,
        )
        self.assertEqual(model.model._model_data, loaded_model.model._model_data)

    def test_no_data(self) -> None:
        model = SlowDriftDetectorModel(
            slow_drift_window=28 * 86400,
            algorithm_version=3,
            seasonality_period=0,
            seasonality_num_points=0,
        )

        ts = self.generate_ts_with_slope_and_seasonality(num_points=56)
        result = model.fit_predict(ts)
        self.assertEqual(len(result.scores), 1)
        self.assertEqual(len(result.anomaly_magnitude_ts), 1)
