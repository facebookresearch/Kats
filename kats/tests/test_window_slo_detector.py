
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from unittest import TestCase
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.window_slo_detector import WindowSloDetectorModel


class TestMultipleWindowSloDetector(TestCase):
    def test_multi_detector_data(self) -> None:
        date_start_str = "2020-08-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First prediction should be True and second should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

        window_size = (2, 5)
        error_budget = (1.0 / 5, 2.0 / 5)
        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=True,
        )
        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)
        # All predictions should be False
        self.assertFalse(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

        window_size = (3, 5)
        error_budget = (1.0 / 5, 2.0 / 5)
        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=True,
        )
        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)
        # All predictions should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])

    def test_multi_detector_data_2(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First and Second predictions should be True
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

        window_size = (2, 5)
        error_budget = (1.0 / 2, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First prediction should be False and second should be True
        self.assertFalse(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

    def test_multi_detector_data_Long(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 20

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        violation_ts = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
        violation_ts.extend([0] * 10)
        self.assertEqual(len(violation_ts), 20)
        values = [violation_ts, [1] * len(violation_ts)]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8 : len(violation_ts)]},
                    **{
                        f"value_{i}": values[i][8 : len(violation_ts)]
                        for i in range(num_seq)
                    },
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

        # First and Second predictions should be True
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertTrue(pred.scores.value.iloc[1])

    def test_multi_detector_none_historical_data(self) -> None:
        date_start_str = "2020-08-16"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 20

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0] * num_days, [1] * num_days]

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": values[i] for i in range(num_seq)},
                }
            )
        )

        window_size = (2, 5)
        error_budget = (2.0 / 5, 2.0 / 5)

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget, error_budget)
        self.assertEqual(slo_detect.window_size, window_size)

        pred = slo_detect.fit_predict(data=data)

        # Since there are not enough historical_data, data will be divided
        # Prediction should returns scores of same length or original data
        self.assertEqual(len(pred.scores), len(data))
        self.assertEqual(len(pred.anomaly_magnitude_ts), len(data))

    def test_choose_model_parameters(self) -> None:
        sli_error_budget = 0.01
        error_budget = (1.0 * sli_error_budget, 0.1 * sli_error_budget)

        precision_ttd_tradeoff_factor = 0.8
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (120, 1440)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)

        precision_ttd_tradeoff_factor = 0.5
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (60, 720)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)

        precision_ttd_tradeoff_factor = 0.2
        slo_detect = WindowSloDetectorModel(
            precision_ttd_tradeoff_factor=precision_ttd_tradeoff_factor,
            sli_error_budget=sli_error_budget,
            windows_have_same_end=False,
        )
        window_size = (30, 360)
        self.assertEqual(slo_detect.window_size, window_size)
        self.assertEqual(slo_detect.error_budget, error_budget)


class TestSingleWindowSloDetector(TestCase):
    def test_multi_detector_data(self) -> None:
        date_start_str = "2020-08-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 2
        num_days = 10

        previous_seq = [date_start + timedelta(days=x) for x in range(num_days)]
        values = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1] * 10]

        historical_data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:8]},
                    **{f"value_{i}": values[i][0:8] for i in range(num_seq)},
                }
            )
        )

        data = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[8:10]},
                    **{f"value_{i}": values[i][8:10] for i in range(num_seq)},
                }
            )
        )

        window_size = 5
        error_budget = 2.0 / 5

        slo_detect = WindowSloDetectorModel(
            window_size=window_size,
            error_budget=error_budget,
            windows_have_same_end=False,
        )
        self.assertAlmostEqual(slo_detect.error_budget[0], error_budget)
        self.assertEqual(slo_detect.window_size[0], window_size)

        pred = slo_detect.fit_predict(historical_data=historical_data, data=data)

        # Prediction should returns scores of same length
        self.assertEqual(len(pred.scores), len(data))

        # First prediction should be True and second should be False
        self.assertTrue(pred.scores.value.iloc[0])
        self.assertFalse(pred.scores.value.iloc[1])
