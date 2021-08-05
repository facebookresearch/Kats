# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd

import statsmodels
from kats.consts import TimeSeriesData

from kats.detectors.detector_consts import (
    AnomalyResponse,
    ChangePointInterval,
    ConfidenceBand,
    PercentageChange,
    SingleSpike,
)

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class SingleSpikeTest(TestCase):
    def test_spike(self) -> None:
        spike_time_str = "2020-03-01"
        spike_time = datetime.strptime(spike_time_str, "%Y-%m-%d")
        spike = SingleSpike(time=spike_time, value=1.0, n_sigma=3.0)
        self.assertEqual(spike.time_str, spike_time_str)


class ChangePointIntervalTest(TestCase):
    def test_changepoint(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        current_length = 10

        current_seq = [
            previous_seq[10] + timedelta(days=x) for x in range(current_length)
        ]
        previous_values = np.random.randn(len(previous_seq))
        current_values = np.random.randn(len(current_seq))

        # add a very large value to detect spikes
        current_values[0] = 100.0

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": previous_values})
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )

        previous_extend = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[9:], "value": previous_values[9:]})
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_end`.
        self.current_end = current_seq[-1] + timedelta(days=1)

        previous_int = ChangePointInterval(self.prev_start, self.prev_end)
        previous_int.data = self.previous

        # tests whether data is clipped property to start and end dates
        np.testing.assert_array_equal(previous_values[0:9], previous_int.data)

        # test extending the data
        # now the data is extended to include the whole sequence
        previous_int.end_time = previous_seq[-1] + timedelta(days=1)
        previous_int.extend_data(previous_extend)

        self.assertEqual(len(previous_int), len(previous_seq))

        current_int = ChangePointInterval(self.current_start, self.current_end)
        current_int.data = self.current
        current_int.previous_interval = previous_int

        # check all the properties
        self.assertEqual(current_int.start_time, self.current_start)
        self.assertEqual(current_int.end_time, self.current_end)
        self.assertEqual(
            current_int.start_time_str,
            datetime.strftime(self.current_start, "%Y-%m-%d"),
        )
        self.assertEqual(
            current_int.end_time_str, datetime.strftime(self.current_end, "%Y-%m-%d")
        )

        self.assertEqual(current_int.mean_val, np.mean(current_values))
        self.assertEqual(current_int.variance_val, np.var(current_values))
        self.assertEqual(len(current_int), current_length)
        self.assertEqual(current_int.previous_interval, previous_int)

        # check spike detection
        spike_list = current_int.spikes
        # pyre-fixme[16]: `List` has no attribute `value`.
        self.assertEqual(spike_list[0].value, 100.0)
        self.assertEqual(
            # pyre-fixme[16]: `List` has no attribute `time_str`.
            spike_list[0].time_str,
            datetime.strftime(self.current_start, "%Y-%m-%d"),
        )

    def test_multichangepoint(self) -> None:
        # test for multivariate time series
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")

        previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        current_length = 10

        current_seq = [
            previous_seq[10] + timedelta(days=x) for x in range(current_length)
        ]

        num_seq = 5
        previous_values = [np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        current_values = [np.random.randn(len(current_seq)) for _ in range(num_seq)]

        # add a very large value to detect spikes
        for i in range(num_seq):
            current_values[i][0] = 100 * (i + 1)

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": previous_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(num_seq)},
                }
            )
        )

        previous_extend = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq[9:]},
                    **{f"value_{i}": previous_values[i][9:] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        #  `current_start`.
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `ChangePointIntervalTest` has no attribute `current_end`.
        self.current_end = current_seq[-1] + timedelta(days=1)

        previous_int = ChangePointInterval(self.prev_start, self.prev_end)
        previous_int.data = self.previous

        # tests whether data is clipped property to start and end dates
        for i in range(num_seq):
            self.assertEqual(
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                previous_int.data[:, i].tolist(),
                previous_values[i][0:9].tolist(),
            )

        # test extending the data
        # now the data is extended to include the whole sequence except the last point
        previous_int.end_time = previous_seq[-1]  # + timedelta(days=1)
        previous_int.extend_data(previous_extend)
        self.assertEqual(len(previous_int) + 1, len(previous_seq))

        # let's repeat this except without truncating the final point
        previous_int2 = ChangePointInterval(self.prev_start, self.prev_end)
        previous_int2.data = self.previous
        previous_int2.end_time = previous_seq[-1] + timedelta(days=1)
        previous_int2.extend_data(previous_extend)
        self.assertEqual(len(previous_int2), len(previous_seq))

        # let's extend the date range so it's longer than the data
        # this should not change the results
        previous_int3 = ChangePointInterval(self.prev_start, self.prev_end)
        previous_int3.data = self.previous
        previous_int3.end_time = previous_seq[-1] + timedelta(days=2)
        previous_int3.extend_data(previous_extend)
        self.assertEqual(len(previous_int3), len(previous_seq))

        # let's construct the current ChangePointInterval
        current_int = ChangePointInterval(self.current_start, self.current_end)
        current_int.data = self.current
        current_int.previous_interval = previous_int

        # check all the properties
        self.assertEqual(current_int.start_time, self.current_start)
        self.assertEqual(current_int.end_time, self.current_end)
        self.assertEqual(current_int.num_series, num_seq)
        self.assertEqual(
            current_int.start_time_str,
            datetime.strftime(self.current_start, "%Y-%m-%d"),
        )
        self.assertEqual(
            current_int.end_time_str, datetime.strftime(self.current_end, "%Y-%m-%d")
        )

        self.assertEqual(
            # pyre-fixme[16]: `float` has no attribute `tolist`.
            current_int.mean_val.tolist(),
            [np.mean(current_values[i]) for i in range(num_seq)],
        )
        self.assertEqual(
            current_int.variance_val.tolist(),
            [np.var(current_values[i]) for i in range(num_seq)],
        )
        self.assertEqual(len(current_int), current_length)
        self.assertEqual(current_int.previous_interval, previous_int)

        # check spike detection
        spike_array = current_int.spikes
        self.assertEqual(len(spike_array), num_seq)

        for i in range(num_seq):
            # pyre-fixme[16]: `SingleSpike` has no attribute `__getitem__`.
            self.assertEqual(spike_array[i][0].value, 100 * (i + 1))
            self.assertEqual(
                spike_array[i][0].time_str,
                datetime.strftime(self.current_start, "%Y-%m-%d"),
            )


class PercentageChangeTest(TestCase):
    def test_perc_change(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        current_length = 31
        # offset one to make the new interval start one day after the previous one ends
        current_seq = [
            previous_seq[-1] + timedelta(days=(x + 1)) for x in range(current_length)
        ]
        previous_values = 1.0 + 0.25 * np.random.randn(len(previous_seq))
        current_values = 10.0 + 0.25 * np.random.randn(len(current_seq))

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": previous_values})
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_end`.
        self.current_end = current_seq[-1]

        previous_int = ChangePointInterval(
            previous_seq[0], (previous_seq[-1] + timedelta(days=1))
        )
        previous_int.data = self.previous

        current_int = ChangePointInterval(
            current_seq[0], (current_seq[-1] + timedelta(days=1))
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        perc_change_1 = PercentageChange(current=current_int, previous=previous_int)

        previous_mean = np.mean(previous_values)
        current_mean = np.mean(current_values)

        # test the ratios
        ratio_val = current_mean / previous_mean
        self.assertEqual(perc_change_1.ratio_estimate, ratio_val)

        ratio_estimate = perc_change_1.ratio_estimate
        assert isinstance(ratio_estimate, float)
        self.assertAlmostEqual(ratio_estimate, 10.0, 0)

        self.assertEqual(perc_change_1.perc_change, (ratio_val - 1) * 100)
        self.assertEqual(perc_change_1.direction, "up")
        self.assertEqual(perc_change_1.stat_sig, True)
        self.assertTrue(perc_change_1.p_value < 0.05)
        self.assertTrue(perc_change_1.score > 1.96)

        # test a detector with false stat sig
        second_values = 10.005 + 0.25 * np.random.randn(len(previous_seq))
        second = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": second_values})
        )

        second_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        second_int.data = second

        perc_change_2 = PercentageChange(current=current_int, previous=second_int)
        self.assertEqual(perc_change_2.stat_sig, False)
        self.assertFalse(perc_change_2.p_value < 0.05)
        self.assertFalse(perc_change_2.score > 1.96)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_2 = ChangePointInterval(current_seq[0], current_seq[1])

        current_int_2.data = self.current

        perc_change_3 = PercentageChange(current=current_int_2, previous=previous_int)
        self.assertTrue(perc_change_3.score > 1.96)

        # TODO delta method tests

    def test_multi_perc_change(self) -> None:
        # test for multivariate time series
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        current_length = 31
        # offset one to make the new interval start one day after the previous one ends
        current_seq = [
            previous_seq[-1] + timedelta(days=(x + 1)) for x in range(current_length)
        ]

        num_seq = 5

        previous_values = np.array(
            [1.0 + 0.0001 * np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        )
        current_values = np.array(
            [10.0 + 0.0001 * np.random.randn(len(current_seq)) for _ in range(num_seq)]
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `previous`.
        self.previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": previous_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current`.
        self.current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(num_seq)},
                }
            )
        )

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_start`.
        self.prev_start = previous_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `prev_end`.
        self.prev_end = previous_seq[9]

        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_start`.
        self.current_start = current_seq[0]
        # pyre-fixme[16]: `PercentageChangeTest` has no attribute `current_end`.
        self.current_end = current_seq[-1]

        previous_int = ChangePointInterval(
            previous_seq[0], previous_seq[-1] + timedelta(days=1)
        )
        previous_int.data = self.previous
        current_int = ChangePointInterval(
            current_seq[0], current_seq[-1] + timedelta(days=1)
        )
        current_int.data = self.current
        current_int.previous_interval = previous_int

        perc_change_1 = PercentageChange(current=current_int, previous=previous_int)

        previous_mean = np.array([np.mean(previous_values[i]) for i in range(num_seq)])
        current_mean = np.array([np.mean(current_values[i]) for i in range(num_seq)])

        # test the ratios
        ratio_val = current_mean / previous_mean
        ratio_estimate = perc_change_1.ratio_estimate
        assert isinstance(ratio_estimate, np.ndarray)
        self.assertEqual(ratio_estimate.tolist(), ratio_val.tolist())

        for r in ratio_estimate:
            self.assertAlmostEqual(r, 10.0, 0)

        perc_change = perc_change_1.perc_change
        assert isinstance(perc_change, np.ndarray)
        self.assertEqual(perc_change.tolist(), ((ratio_val - 1) * 100).tolist())

        direction = perc_change_1.direction
        assert isinstance(direction, np.ndarray)
        self.assertEqual(direction.tolist(), ["up"] * num_seq)

        stat_sig = perc_change_1.stat_sig
        assert isinstance(stat_sig, np.ndarray)
        self.assertEqual(stat_sig.tolist(), [True] * num_seq)

        p_value_list, score_list = perc_change_1.p_value, perc_change_1.score
        assert isinstance(p_value_list, Iterable)
        assert isinstance(score_list, Iterable)
        for p_value, score in zip(p_value_list, score_list):
            self.assertLess(p_value, 0.05)
            self.assertLess(1.96, score)

        # test a detector with false stat sig
        second_values = np.array(
            [10.005 + 0.25 * np.random.randn(len(previous_seq)) for _ in range(num_seq)]
        )

        second = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": second_values[i] for i in range(num_seq)},
                }
            )
        )

        second_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        second_int.data = second

        perc_change_2 = PercentageChange(current=current_int, previous=second_int)

        stat_sig_list, p_value_list, score_list = (
            perc_change_2.stat_sig,
            perc_change_2.p_value,
            perc_change_2.score,
        )
        assert isinstance(stat_sig_list, Iterable)
        assert isinstance(p_value_list, Iterable)
        assert isinstance(score_list, Iterable)

        for stat_sig, p_value, score in zip(stat_sig_list, p_value_list, score_list):
            self.assertFalse(stat_sig)
            self.assertLess(0.05, p_value)
            self.assertLess(score, 1.96)

        # test a detector with a negative spike
        third_values = np.array(
            [
                1000.0 + 0.0001 * np.random.randn(len(previous_seq))
                for _ in range(num_seq)
            ]
        )

        third = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": third_values[i] for i in range(num_seq)},
                }
            )
        )

        third_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        third_int.data = third

        perc_change_3 = PercentageChange(current=current_int, previous=third_int)

        p_value_list, score_list = perc_change_3.p_value, perc_change_3.score
        assert isinstance(p_value_list, Iterable)
        assert isinstance(score_list, Iterable)
        for p_value, score in zip(p_value_list, score_list):
            self.assertLess(p_value, 0.05)
            self.assertLess(score, -1.96)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_single_point = ChangePointInterval(current_seq[0], current_seq[1])

        current_int_single_point.data = self.current

        perc_change_single_point = PercentageChange(
            current=current_int_single_point, previous=previous_int
        )

        p_value_list, score_list = (
            perc_change_single_point.p_value,
            perc_change_single_point.score,
        )
        assert isinstance(p_value_list, Iterable)
        assert isinstance(score_list, Iterable)

        for p_value, score in zip(p_value_list, score_list):
            self.assertLess(p_value, 0.05)
            self.assertLess(1.96, score)


class TestAnomalyResponse(TestCase):
    def test_response(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]
        score_ts = TimeSeriesData(
            pd.DataFrame(
                {"time": previous_seq, "value": np.random.randn(len(previous_seq))}
            )
        )
        upper_values = 1.0 + np.random.randn(len(previous_seq))
        upper_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": upper_values})
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": (upper_values - 0.1)})
        )

        conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        stat_sig_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": np.ones(len(previous_seq))})
        )

        response = AnomalyResponse(
            scores=score_ts,
            confidence_band=conf_band,
            predicted_ts=pred_ts,
            anomaly_magnitude_ts=mag_ts,
            stat_sig_ts=stat_sig_ts,
        )

        #  Ensure that num_series is properly populated - this response object is univariate
        self.assertEqual(response.num_series, 1)

        # test update
        new_date = previous_seq[-1] + timedelta(days=1)
        common_val = 1.23
        response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=(common_val - 0.1),
            pred=common_val,
            anom_mag=common_val,
            stat_sig=0,
        )

        # assert that all the lengths of the time series are preserved
        N = len(previous_seq)
        self.assertEqual(len(response.scores), N)
        self.assertEqual(len(response.confidence_band.upper), N)
        self.assertEqual(len(response.confidence_band.lower), N)
        self.assertEqual(len(response.predicted_ts), N)
        self.assertEqual(len(response.anomaly_magnitude_ts), N)
        self.assertEqual(len(response.stat_sig_ts), N)

        # assert that each time series has moved one point forward
        self.assertEqual(response.scores.value[0], score_ts.value[1])
        self.assertEqual(
            response.confidence_band.upper.value[0], conf_band.upper.value[1]
        )
        self.assertEqual(
            response.confidence_band.lower.value[0], conf_band.lower.value[1]
        )
        self.assertEqual(response.predicted_ts.value[0], pred_ts.value[1])
        self.assertEqual(response.anomaly_magnitude_ts.value[0], mag_ts.value[1])
        self.assertEqual(response.stat_sig_ts.value[0], stat_sig_ts.value[1])

        # assert that a new point has been added to the end
        self.assertEqual(response.scores.value.values[-1], common_val)
        self.assertEqual(response.confidence_band.upper.value.values[-1], common_val)
        self.assertEqual(
            response.confidence_band.lower.value.values[-1], common_val - 0.1
        )
        self.assertEqual(response.predicted_ts.value.values[-1], common_val)
        self.assertEqual(response.anomaly_magnitude_ts.value.values[-1], common_val)
        self.assertEqual(response.stat_sig_ts.value.values[-1], 0.0)

        # assert that we return the last N values
        score_list = response.scores.value.values.tolist()

        n_val = 10
        response_last_n = response.get_last_n(n_val)
        self.assertEqual(len(response_last_n.scores), n_val)
        self.assertEqual(len(response_last_n.confidence_band.upper), n_val)
        self.assertEqual(len(response_last_n.confidence_band.lower), n_val)
        self.assertEqual(len(response_last_n.predicted_ts), n_val)
        self.assertEqual(len(response_last_n.anomaly_magnitude_ts), n_val)
        self.assertEqual(len(response_last_n.stat_sig_ts), n_val)

        self.assertEqual(
            response_last_n.scores.value.values.tolist(), score_list[-n_val:]
        )

    def test_multi_response(self) -> None:
        # test anomaly response for multivariate time series
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        num_seq = 5

        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        score_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        upper_values = [
            1.0 + np.random.randn(len(previous_seq)) for _ in range(num_seq)
        ]

        upper_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": upper_values[i] for i in range(num_seq)},
                }
            )
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": upper_values[i] - 0.1 for i in range(num_seq)},
                }
            )
        )

        conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(num_seq)
                    },
                }
            )
        )

        stat_sig_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.ones(len(previous_seq)) for i in range(num_seq)
                    },
                }
            )
        )

        response = AnomalyResponse(
            scores=score_ts,
            confidence_band=conf_band,
            predicted_ts=pred_ts,
            anomaly_magnitude_ts=mag_ts,
            stat_sig_ts=stat_sig_ts,
        )

        # Ensure that num_series is properly populated
        self.assertEqual(response.num_series, num_seq)

        # test update
        new_date = previous_seq[-1] + timedelta(days=1)
        common_val = 1.23 * np.ones(num_seq)

        response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=common_val - 0.1,
            pred=common_val,
            anom_mag=common_val,
            stat_sig=np.zeros(num_seq),
        )

        N = len(previous_seq)

        # assert that all the lengths of the time series are preserved
        self.assertEqual(len(response.scores), N)
        self.assertEqual(len(response.confidence_band.upper), N)
        self.assertEqual(len(response.confidence_band.lower), N)
        self.assertEqual(len(response.predicted_ts), N)
        self.assertEqual(len(response.anomaly_magnitude_ts), N)
        self.assertEqual(len(response.stat_sig_ts), N)

        # assert that each time series has moved one point forward
        self.assertEqual(
            response.scores.value.iloc[0].tolist(), score_ts.value.iloc[1].tolist()
        )
        self.assertEqual(
            response.confidence_band.upper.value.iloc[0].tolist(),
            conf_band.upper.value.iloc[1].tolist(),
        )
        self.assertEqual(
            response.confidence_band.lower.value.iloc[0].tolist(),
            conf_band.lower.value.iloc[1].tolist(),
        )
        self.assertEqual(
            response.predicted_ts.value.iloc[0].tolist(), pred_ts.value.iloc[1].tolist()
        )
        self.assertEqual(
            response.anomaly_magnitude_ts.value.iloc[0].tolist(),
            mag_ts.value.iloc[1].tolist(),
        )
        self.assertEqual(
            response.stat_sig_ts.value.iloc[0].tolist(),
            stat_sig_ts.value.iloc[1].tolist(),
        )

        # assert that a new point has been added to the end
        assert isinstance(common_val, np.ndarray)
        self.assertEqual(response.scores.value.iloc[-1].tolist(), common_val.tolist())
        self.assertEqual(
            response.confidence_band.upper.value.iloc[-1].tolist(), common_val.tolist()
        )
        self.assertEqual(
            response.confidence_band.lower.value.iloc[-1].tolist(),
            (common_val - 0.1).tolist(),
        )
        self.assertEqual(
            response.predicted_ts.value.iloc[-1].tolist(), common_val.tolist()
        )
        self.assertEqual(
            response.anomaly_magnitude_ts.value.iloc[-1].tolist(), common_val.tolist()
        )
        self.assertEqual(
            response.stat_sig_ts.value.iloc[-1].tolist(), np.zeros(num_seq).tolist()
        )

        # assert that we return the last N values
        n_val = 10

        score_array = response.scores.value.values
        response_last_n = response.get_last_n(n_val)
        self.assertEqual(len(response_last_n.scores), n_val)
        self.assertEqual(len(response_last_n.confidence_band.upper), n_val)
        self.assertEqual(len(response_last_n.confidence_band.lower), n_val)
        self.assertEqual(len(response_last_n.predicted_ts), n_val)
        self.assertEqual(len(response_last_n.anomaly_magnitude_ts), n_val)
        self.assertEqual(len(response_last_n.stat_sig_ts), n_val)

        self.assertEqual(
            response_last_n.scores.value.values.tolist(), score_array[-n_val:].tolist()
        )
