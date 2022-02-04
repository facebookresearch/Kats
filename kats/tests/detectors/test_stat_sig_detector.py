# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest.mock as mock
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.stat_sig_detector import (
    MultiStatSigDetectorModel,
    StatSigDetectorModel,
    SeasonalityHandler,
)
from kats.utils.simulator import Simulator

_SERIALIZED = b'{"n_control": 20, "n_test": 7, "time_unit": "s"}'
_SERIALIZED2 = b'{"n_control": 20, "n_test": 7, "time_unit": "s", "rem_season": false, "seasonal_period": "weekly", "use_corrected_scores": false, "max_split_ts_length": 500}'


class TestStatSigDetector(TestCase):
    def setUp(self) -> None:
        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = np.random.randn(len(previous_seq))
        self.ts_init = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[0:30], "value": values[0:30]})
        )

        self.ts_later = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[30:35], "value": values[30:35]})
        )
        self.ss_detect = StatSigDetectorModel(n_control=20, n_test=7)

    def test_detector(self) -> None:
        np.random.seed(100)
        pred_later = self.ss_detect.fit_predict(
            historical_data=self.ts_init, data=self.ts_later
        )
        self.ss_detect.visualize()

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(self.ts_later))

    def test_pmm_use_case(self) -> None:
        random.seed(100)
        time_unit = 86400
        hist_data_time = [x * time_unit for x in range(0, 28)]
        data_time = [x * time_unit for x in range(28, 35)]

        hist_data_value = [random.normalvariate(100, 10) for _ in range(0, 28)]
        data_value = [random.normalvariate(130, 10) for _ in range(28, 35)]

        hist_ts = TimeSeriesData(
            time=pd.Series(hist_data_time),
            value=pd.Series(hist_data_value),
            use_unix_time=True,
            unix_time_units="s",
        )
        data_ts = TimeSeriesData(
            time=pd.Series(data_time),
            value=pd.Series(data_value),
            use_unix_time=True,
            unix_time_units="s",
        )

        ss_detect2 = StatSigDetectorModel(
            n_control=20 * time_unit, n_test=7 * time_unit, time_unit="S"
        )
        pred_later2 = ss_detect2.fit_predict(historical_data=hist_ts, data=data_ts)
        self.assertEqual(len(pred_later2.scores), len(data_ts))

        self.assertTrue(pred_later2.scores.value.values.max() > 2.0)

        # remove seasonality (with history)
        ss_detect22 = StatSigDetectorModel(
            n_control=20 * time_unit,
            n_test=7 * time_unit,
            time_unit="S",
            rem_season=True,
            seasonal_period="weekly",
        )
        pred_later22 = ss_detect22.fit_predict(historical_data=hist_ts, data=data_ts)
        self.assertEqual(len(pred_later22.scores), len(data_ts))
        self.assertTrue(pred_later22.scores.value.values.max() > 2.0)

        # case with no history
        ss_detect3 = StatSigDetectorModel(
            n_control=10 * time_unit, n_test=10 * time_unit, time_unit="S"
        )
        pred_later3 = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(pred_later3.scores), len(hist_ts))

        # remove seasonality (without history)
        ss_detect33 = StatSigDetectorModel(
            n_control=10 * time_unit,
            n_test=10 * time_unit,
            time_unit="S",
            rem_season=True,
            seasonal_period="weekly",
        )
        pred_later33 = ss_detect33.fit_predict(data=hist_ts)
        self.assertEqual(len(pred_later33.scores), len(hist_ts))

    def test_no_historical_data(self) -> None:
        n = 35
        control_time = pd.date_range(start="2018-01-01", freq="D", periods=n)
        control_val = [random.normalvariate(100, 10) for _ in range(n)]
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))

        n_control = 5
        n_test = 5

        ss_detect3 = StatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(anom.scores), n)

        # for the first n_control + n_test  - 1 values, score is zero,
        # afterwards it is non zero once we reach (n_control + n_test) data points
        for i in range(n_control + n_test - 1):
            self.assertEqual(anom.scores.value.iloc[i], 0.0)

        self.assertNotEqual(anom.scores.value.iloc[n_control + n_test - 1], 0.0)

    def test_not_enough_historical_data(self) -> None:
        n_control = 12
        n_test = 8
        num_control = 8
        num_test = 12

        control_time = pd.date_range(start="2018-01-01", freq="D", periods=num_control)

        test_time = pd.date_range(start="2018-01-09", freq="D", periods=num_test)
        control_val = [random.normalvariate(100, 10) for _ in range(num_control)]
        test_val = [random.normalvariate(120, 10) for _ in range(num_test)]

        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))

        ss_detect = StatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)

        self.assertEqual(len(anom.scores), len(data_ts))
        # until we reach n_control + n_test, we get zeroes
        # non zero afterwards

        for i in range(n_control + n_test - num_control - 1):
            self.assertEqual(anom.scores.value.iloc[i], 0.0)

        self.assertNotEqual(anom.scores.value.iloc[-1], 0.0)

    def test_bigdata_transform(self) -> None:
        # This unit test is confirming that the results are identical when we use the
        # single time series vs. split time series codepaths.

        # use_corrected_scores=True, split data
        n_control = 28
        n_test = 7
        random.seed(0)
        control_time = pd.date_range(
            start="2018-01-06", freq="D", periods=(n_control + n_test - 5)
        )
        test_time = pd.date_range(start="2018-02-05", freq="D", periods=500)
        control_val = [
            random.normalvariate(0, 5) for _ in range(n_control + n_test - 5)
        ]
        test_val = [random.normalvariate(0, 5) for _ in range(500)]
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect1 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=True,
            max_split_ts_length=100,
        )
        anom1 = ss_detect1.fit_predict(data=data_ts, historical_data=hist_ts)

        # use_corrected_scores=True, not split data
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect2 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=True,
            max_split_ts_length=1000,
        )
        anom2 = ss_detect2.fit_predict(data=data_ts, historical_data=hist_ts)

        self.assertEqual(len(test_val), len(anom2.scores.value.values))
        self.assertEqual(len(test_val), len(anom1.scores.value.values))

        for i in range(500):
            self.assertAlmostEqual(
                anom1.scores.value.values[i], anom2.scores.value.values[i], places=10
            )

    def test_bigdata_flag_logic(self) -> None:
        n_control = 28
        n_test = 7
        random.seed(0)
        control_time = pd.date_range(
            start="2018-01-06", freq="D", periods=(n_control + n_test)
        )
        test_time = pd.date_range(start="2018-02-05", freq="D", periods=50)
        control_val = [random.normalvariate(0, 5) for _ in range(n_control + n_test)]
        test_val = [random.normalvariate(0, 5) for _ in range(50)]

        # use_corrected_scores = False and bigdata_trans_flag = False
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect1 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=False,
            max_split_ts_length=100,
        )
        _ = ss_detect1.fit_predict(data=data_ts, historical_data=hist_ts)
        self.assertEqual(ss_detect1.bigdata_trans_flag, False)

        # True and True
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect1 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=True,
            max_split_ts_length=10,
        )
        _ = ss_detect1.fit_predict(data=data_ts, historical_data=hist_ts)
        self.assertEqual(ss_detect1.bigdata_trans_flag, True)

        # True and False 1: not reach threshold
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect1 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=True,
            max_split_ts_length=100,
        )
        _ = ss_detect1.fit_predict(data=data_ts, historical_data=hist_ts)
        self.assertEqual(ss_detect1.bigdata_trans_flag, False)

        # True and False 2: time unit difference
        n_control = 2
        n_test = 2
        random.seed(0)
        # weekly historical data
        control_time = pd.date_range(
            start="2018-01-06", freq="W", periods=(n_control + n_test)
        )
        # daily test data
        test_time = pd.date_range(start="2018-02-05", freq="D", periods=50)
        control_val = [random.normalvariate(0, 5) for _ in range(n_control + n_test)]
        test_val = [random.normalvariate(0, 5) for _ in range(50)]
        hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
        data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
        ss_detect1 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            use_corrected_scores=True,
            max_split_ts_length=10,
        )
        _ = ss_detect1.fit_predict(data=data_ts, historical_data=hist_ts)
        self.assertEqual(ss_detect1.bigdata_trans_flag, False)

    def test_logging(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 3

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        self.assertEqual(self.ss_detect.n_test, 7)
        with self.assertRaises(ValueError):
            self.ss_detect.fit_predict(historical_data=ts_init, data=ts_later)

    def test_load_from_serialized(self) -> None:
        detector = StatSigDetectorModel(serialized_model=_SERIALIZED)
        self.assertEqual(detector.n_control, 20)
        self.assertEqual(detector.n_test, 7)
        self.assertEqual(detector.time_unit, "s")
        self.assertEqual(detector.rem_season, False)
        self.assertEqual(detector.seasonal_period, "weekly")

    def test_serialize(self) -> None:
        detector = StatSigDetectorModel(n_control=20, n_test=7, time_unit="s")
        self.assertEqual(_SERIALIZED2, detector.serialize())

    def test_missing_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = StatSigDetectorModel()

    def test_visualize_unpredicted(self) -> None:
        detector = StatSigDetectorModel(n_control=20, n_test=7)
        with self.assertRaises(ValueError):
            detector.visualize()

    def test_missing_time_unit(self) -> None:
        detector = StatSigDetectorModel(n_control=20, n_test=7)
        with mock.patch.object(detector, "_set_time_unit"):
            with self.assertRaises(ValueError):
                detector.fit_predict(data=self.ts_later, historical_data=self.ts_init)

    def test_no_update(self) -> None:
        detector = StatSigDetectorModel(n_control=20, n_test=7)
        with mock.patch.object(detector, "_should_update") as su:
            su.return_value = False
            result = detector.fit_predict(
                data=self.ts_later, historical_data=self.ts_init
            )
            self.assertEqual(detector.response, result)

    def test_fallback_on_historical_time_unit(self) -> None:
        data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 2, 1),
                    ],
                    "values": [0, 1, 2],
                }
            )
        )
        detector = StatSigDetectorModel(n_control=20, n_test=7)
        detector.fit_predict(data=data, historical_data=self.ts_init)
        self.assertEqual("D", detector.time_unit)

    def test_remove_season(self) -> None:
        sim3 = Simulator(n=120, start="2018-01-01")
        ts3 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.575,
        )
        n_control = 14 * 86400
        n_test = 14 * 86400
        ss_detect5 = StatSigDetectorModel(
            n_control=n_control,
            n_test=n_test,
            time_unit="sec",
            rem_season=True,
            seasonal_period="biweekly",
        )
        anom3 = ss_detect5.fit_predict(data=ts3)
        self.assertEqual(np.min(anom3.scores.value.values) < -5, True)

    def test_season_handler(self) -> None:
        sim3 = Simulator(n=120, start="2018-01-01")
        ts3 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.575,
        )
        with self.assertRaises(ValueError):
            SeasonalityHandler(data=ts3, seasonal_period="weekly", lpj_factor=0.1)
        with self.assertRaises(ValueError):
            SeasonalityHandler(data=ts3, seasonal_period="daily")


class TestMultiStatSigDetector(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        date_start = datetime.strptime("2020-03-01", "%Y-%m-%d")
        num_seq = 3

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.num_seq = num_seq
        self.previous_seq = previous_seq
        self.values = values
        self.ts_init = ts_init
        self.ts_later = ts_later
        self.ss_detect = ss_detect

    def _check_tsdata_nonnull(self, ts: TimeSeriesData) -> None:
        for v in ts.value.values:
            self.assertIsNotNone(v)

    def test_multi_detector(self) -> None:
        self.assertEqual(self.ss_detect.n_test, 7)

    def test_multi_predict_fit(self) -> None:
        pred_later = self.ss_detect.fit_predict(
            historical_data=self.ts_init, data=self.ts_later
        )

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(self.ts_later))
        self._check_tsdata_nonnull(pred_later.scores)

    def test_multi_after_rename(self) -> None:
        # rename the time series and make sure everthing still works as it did above
        ts_init_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": self.previous_seq[0:30]},
                    **{f"ts_{i}": self.values[i][0:30] for i in range(self.num_seq)},
                }
            )
        )

        ts_later_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": self.previous_seq[30:35]},
                    **{f"ts_{i}": self.values[i][30:35] for i in range(self.num_seq)},
                }
            )
        )
        pred_later = self.ss_detect.fit_predict(
            historical_data=ts_init_renamed, data=ts_later_renamed
        )
        self.assertEqual(len(pred_later.scores), len(ts_later_renamed))
        self._check_tsdata_nonnull(pred_later.scores)

    def test_no_historical_data(self) -> None:
        n = 35
        num_seq = 3
        control_time = pd.date_range(start="2018-01-01", freq="D", periods=n)
        control_val = [
            [random.normalvariate(100, 10) for _ in range(n)] for _ in range(num_seq)
        ]

        hist_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": control_time},
                    **{f"ts_{i}": control_val[i] for i in range(num_seq)},
                }
            )
        )

        n_control = 5
        n_test = 5

        ss_detect3 = MultiStatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(anom.scores), n)
        self._check_tsdata_nonnull(anom.scores)

        # for the first n_control + n_test  - 1 values, score is zero,
        # afterwards it is non zero once we reach (n_control + n_test) data points
        for i in range(n_control + n_test - 1):
            self.assertEqual(
                anom.scores.value.iloc[i, :].tolist(), np.zeros(num_seq).tolist()
            )

        for j in range(anom.scores.value.shape[1]):
            self.assertNotEqual(anom.scores.value.iloc[n_control + n_test - 1, j], 0.0)

    def test_not_enough_historical_data(self) -> None:
        n_control = 12
        n_test = 8
        num_control = 8
        num_test = 12
        num_seq = 3

        control_time = pd.date_range(start="2018-01-01", freq="D", periods=num_control)

        test_time = pd.date_range(start="2018-01-09", freq="D", periods=num_test)
        control_val = [
            [random.normalvariate(100, 10) for _ in range(num_control)]
            for _ in range(num_seq)
        ]
        test_val = [
            [random.normalvariate(120, 10) for _ in range(num_test)]
            for _ in range(num_seq)
        ]

        hist_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": control_time},
                    **{f"ts_{i}": control_val[i] for i in range(num_seq)},
                }
            )
        )

        data_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": test_time},
                    **{f"ts_{i}": test_val[i] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=n_control, n_test=n_test)
        anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)

        self.assertEqual(len(anom.scores), len(data_ts))
        self._check_tsdata_nonnull(anom.scores)
        # until we reach n_control + n_test, we get zeroes
        # non zero afterwards

        for i in range(n_control + n_test - num_control - 1):
            self.assertEqual(
                anom.scores.value.iloc[i, :].tolist(), np.zeros(num_seq).tolist()
            )

        for j in range(anom.scores.value.shape[1]):
            self.assertNotEqual(anom.scores.value.iloc[-1, j], 0.0)

    def test_logging(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 1

        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]

        ts_init = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"value_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"value_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        with self.assertRaises(ValueError):
            ss_detect.fit_predict(historical_data=ts_init, data=ts_later)

    def test_multi_no_update(self) -> None:
        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        with mock.patch.object(ss_detect, "_should_update") as su:
            su.return_value = False
            pred_later = ss_detect.fit_predict(
                historical_data=self.ts_init, data=self.ts_later
            )

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), ss_detect.last_N)
        self._check_tsdata_nonnull(pred_later.scores)

    def test_multi_fit(self) -> None:
        self.assertIsNone(self.ss_detect.fit(self.ts_init))

    def test_multi_predict(self) -> None:
        with self.assertRaises(ValueError):
            self.ss_detect.predict(self.ts_later)

    def test_remove_season_multi(self) -> None:
        sim3 = Simulator(n=120, start="2018-01-01")
        ts1 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.51,
        )
        ts2 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.575,
        )

        ts3 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.53,
        )

        ts4 = sim3.level_shift_sim(
            cp_arr=[60],
            level_arr=[1.35, 1.05],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.55,
        )
        data_ts = TimeSeriesData(
            time=ts3.time,
            value=pd.DataFrame(
                {
                    "ts_1": list(ts1.value.values),
                    "ts_2": list(ts2.value.values),
                    "ts_3": list(ts3.value.values),
                    "ts_4": list(ts4.value.values),
                }
            ),
        )
        n_control = 14 * 86400
        n_test = 14 * 86400
        ss_detect5 = MultiStatSigDetectorModel(
            n_control=n_control, n_test=n_test, time_unit="sec"
        )
        anom3 = ss_detect5.fit_predict(data=data_ts, rem_season=True)
        self._check_tsdata_nonnull(anom3.scores)
