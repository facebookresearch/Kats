# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import re
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.stat_sig_detector import (
    MultiStatSigDetectorModel,
    StatSigDetectorModel,
)

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class TestStatSigDetector(TestCase):
    def test_detector(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = np.random.randn(len(previous_seq))
        ts_init = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[0:30], "value": values[0:30]})
        )

        ts_later = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[30:35], "value": values[30:35]})
        )

        ss_detect = StatSigDetectorModel(n_control=20, n_test=7)

        pred_later = ss_detect.fit_predict(historical_data=ts_init, data=ts_later)
        ss_detect.visualize()

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later))

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

        # case with no history
        ss_detect3 = StatSigDetectorModel(
            n_control=10 * time_unit, n_test=10 * time_unit, time_unit="S"
        )
        pred_later3 = ss_detect3.fit_predict(data=hist_ts)
        self.assertEqual(len(pred_later3.scores), len(hist_ts))

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
        import random

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

        ss_detect = StatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        with self.assertRaises(ValueError):
            ss_detect.fit_predict(historical_data=ts_init, data=ts_later)


class TestMultiStatSigDetector(TestCase):
    def test_multi_detector(self) -> None:
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

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        pred_later = ss_detect.fit_predict(historical_data=ts_init, data=ts_later)

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later))

        # rename the time series and make sure everthing still works as it did above
        ts_init_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[0:30]},
                    **{f"ts_{i}": values[i][0:30] for i in range(num_seq)},
                }
            )
        )

        ts_later_renamed = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[30:35]},
                    **{f"ts_{i}": values[i][30:35] for i in range(num_seq)},
                }
            )
        )

        ss_detect = MultiStatSigDetectorModel(n_control=20, n_test=7)
        self.assertEqual(ss_detect.n_test, 7)
        pred_later = ss_detect.fit_predict(
            historical_data=ts_init_renamed, data=ts_later_renamed
        )

        # prediction returns scores of same length
        self.assertEqual(len(pred_later.scores), len(ts_later_renamed))

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
        import random

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
