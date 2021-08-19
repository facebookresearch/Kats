# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from operator import attrgetter
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
from parameterized import parameterized

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class SingleSpikeTest(TestCase):
    def test_spike(self) -> None:
        spike_time_str = "2020-03-01"
        spike_time = datetime.strptime(spike_time_str, "%Y-%m-%d")
        spike = SingleSpike(time=spike_time, value=1.0, n_sigma=3.0)
        self.assertEqual(spike.time_str, spike_time_str)


class UnivariateChangePointIntervalTest(TestCase):
    # test for univariate time series
    def setUp(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        self.previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        self.current_length = 10

        current_seq = [
            self.previous_seq[10] + timedelta(days=x)
            for x in range(self.current_length)
        ]
        self.previous_values = np.random.randn(len(self.previous_seq))
        current_values = np.random.randn(len(current_seq))

        # add a very large value to detect spikes
        current_values[0] = 100.0

        previous = TimeSeriesData(
            pd.DataFrame({"time": self.previous_seq, "value": self.previous_values})
        )

        current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )
        self.current_mean = np.mean(current_values)
        self.current_variance = np.var(current_values)

        previous_extend = TimeSeriesData(
            pd.DataFrame(
                {"time": self.previous_seq[9:], "value": self.previous_values[9:]}
            )
        )

        prev_start = self.previous_seq[0]
        prev_end = self.previous_seq[9]

        self.current_start = current_seq[0]
        self.current_start_time_str = datetime.strftime(self.current_start, "%Y-%m-%d")
        self.current_end = current_seq[-1] + timedelta(days=1)
        self.current_end_time_str = datetime.strftime(self.current_end, "%Y-%m-%d")

        self.previous_int = ChangePointInterval(prev_start, prev_end)
        self.previous_int.data = previous

        # construct interval to test whether data is clipped property to start and end dates
        self.previous_int_test = ChangePointInterval(prev_start, prev_end)
        self.previous_int_test.data = previous

        # test extending the data
        # now the data is extended to include the whole sequence
        self.previous_int.end_time = self.previous_seq[-1] + timedelta(days=1)
        self.previous_int.extend_data(previous_extend)

        self.current_int = ChangePointInterval(self.current_start, self.current_end)
        self.current_int.data = current
        self.current_int.previous_interval = self.previous_int

        self.spike_list = self.current_int.spikes

    def test_start_end_date(self) -> None:
        # tests whether data is clipped property to start and end dates
        np.testing.assert_array_equal(
            self.previous_values[0:9], self.previous_int_test.data
        )

    def test_interval_seq_length(self) -> None:
        self.assertEqual(len(self.previous_int), len(self.previous_seq))

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["start_time", "current_start"],
            ["end_time", "current_end"],
            ["start_time_str", "current_start_time_str"],
            ["end_time_str", "current_end_time_str"],
            ["mean_val", "current_mean"],
            ["variance_val", "current_variance"],
            ["previous_interval", "previous_int"],
        ]
    )
    # check all the properties
    def test_properties(self, attribute, initial_object) -> None:
        self.assertEqual(
            attrgetter(attribute)(self.current_int), attrgetter(initial_object)(self)
        )

    def test_length(self) -> None:
        self.assertEqual(len(self.current_int), self.current_length)

    # check spike detection
    def test_spike_start_value(self) -> None:
        self.assertEqual(self.spike_list[0].value, 100.0)

    def test_spike_start_time_str(self) -> None:
        self.assertEqual(
            self.spike_list[0].time_str,
            self.current_start_time_str,
        )


class MultivariateChangePointIntervalTest(TestCase):
    # test for multivariate time series
    def setUp(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")

        self.previous_seq = [date_start + timedelta(days=x) for x in range(15)]

        self.current_length = 10

        current_seq = [
            self.previous_seq[10] + timedelta(days=x)
            for x in range(self.current_length)
        ]

        self.num_seq = 5
        self.previous_values = [
            np.random.randn(len(self.previous_seq)) for _ in range(self.num_seq)
        ]
        current_values = [
            np.random.randn(len(current_seq)) for _ in range(self.num_seq)
        ]
        self.current_values_mean = [
            np.mean(current_values[i]) for i in range(self.num_seq)
        ]
        self.current_values_variance = [
            np.var(current_values[i]) for i in range(self.num_seq)
        ]

        # add a very large value to detect spikes
        for i in range(self.num_seq):
            current_values[i][0] = 100 * (i + 1)

        previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": self.previous_seq},
                    **{
                        f"value_{i}": self.previous_values[i]
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(self.num_seq)},
                }
            )
        )

        previous_extend = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": self.previous_seq[9:]},
                    **{
                        f"value_{i}": self.previous_values[i][9:]
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        prev_start = self.previous_seq[0]
        prev_end = self.previous_seq[9]

        #  `current_start`.
        self.current_start = current_seq[0]
        self.current_start_date_str = datetime.strftime(self.current_start, "%Y-%m-%d")

        self.current_end = current_seq[-1] + timedelta(days=1)
        self.current_end_date_str = datetime.strftime(self.current_end, "%Y-%m-%d")

        self.previous_int = ChangePointInterval(prev_start, prev_end)
        self.previous_int.data = previous

        # now the data is extended to include the whole sequence except the last point
        self.previous_int.end_time = self.previous_seq[-1]  # + timedelta(days=1)
        self.previous_int.extend_data(previous_extend)

        # construct data to test if clipped properly to start and end dates
        self.previous_int1 = ChangePointInterval(prev_start, prev_end)
        self.previous_int1.data = previous

        # let's repeat this except without truncating the final point
        self.previous_int2 = ChangePointInterval(prev_start, prev_end)
        self.previous_int2.data = previous
        self.previous_int2.end_time = self.previous_seq[-1] + timedelta(days=1)
        self.previous_int2.extend_data(previous_extend)

        # let's extend the date range so it's longer than the data
        # this should not change the results
        self.previous_int3 = ChangePointInterval(prev_start, prev_end)
        self.previous_int3.data = previous
        self.previous_int3.end_time = self.previous_seq[-1] + timedelta(days=2)
        self.previous_int3.extend_data(previous_extend)

        # let's construct the current ChangePointInterval
        self.current_int = ChangePointInterval(self.current_start, self.current_end)
        self.current_int.data = current
        self.current_int.previous_interval = self.previous_int
        self.current_int_length = len(self.current_int)

        # spike detection
        self.spike_array = self.current_int.spikes

    def test_start_end_date(self) -> None:
        # tests whether data is clipped properly to start and end dates
        for i in range(self.num_seq):
            self.assertEqual(
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                self.previous_int1.data[:, i].tolist(),
                self.previous_values[i][0:9].tolist(),
            )

    def test_extend_length_except_last_point(self) -> None:
        # test extending the data to include the whole sequence except the last point
        self.assertEqual(len(self.previous_int) + 1, len(self.previous_seq))

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["previous_int2"],
            ["previous_int3"],
        ]
    )
    def test_extend_length(self, attribute) -> None:
        self.assertEqual(len(attrgetter(attribute)(self)), len(self.previous_seq))

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["start_time", "current_start"],
            ["end_time", "current_end"],
            ["num_series", "num_seq"],
            ["start_time_str", "current_start_date_str"],
            ["end_time_str", "current_end_date_str"],
            ["previous_interval", "previous_int"],
        ]
    )
    def check_current_int_properties(self, attribute, initial_object) -> None:
        # check all the properties
        self.assertEqual(
            attrgetter(attribute)(self.current_int), attrgetter(initial_object)(self)
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["mean_val", "current_values_mean"],
            ["variance_val", "current_values_variance"],
        ]
    )
    def check_current_int_mean_var(self, attribute, initial_object) -> None:
        self.assertEqual(
            attrgetter(attribute)(self.current_int).tolist(),
            attrgetter(initial_object)(self),
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [["current_int", "current_length"], ["spike_array", "num_seq"]]
    )
    def check_length(self, attribute, initial_object) -> None:
        self.assertEqual(
            len(attrgetter(attribute)(self)), attrgetter(initial_object)(self)
        )

    def check_spike_array_value(self) -> None:
        for i in range(self.num_seq):
            self.assertEqual(self.spike_array[i][0].value, 100 * (i + 1))

    def check_spike_array_time_str(self) -> None:
        for i in range(self.num_seq):
            self.assertEqual(
                self.spike_array[i][0].time_str,
                self.current_start_date_str,
            )


class UnivariatePercentageChangeTest(TestCase):
    # test for univariate time series
    def setUp(self):
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

        previous = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": previous_values})
        )

        current = TimeSeriesData(
            pd.DataFrame({"time": current_seq, "value": current_values})
        )

        previous_int = ChangePointInterval(
            previous_seq[0], (previous_seq[-1] + timedelta(days=1))
        )
        previous_int.data = previous

        current_int = ChangePointInterval(
            current_seq[0], (current_seq[-1] + timedelta(days=1))
        )
        current_int.data = current
        current_int.previous_interval = previous_int

        self.perc_change_1 = PercentageChange(
            current=current_int, previous=previous_int
        )

        previous_mean = np.mean(previous_values)
        current_mean = np.mean(current_values)

        # test the ratios
        self.ratio_val_1 = current_mean / previous_mean

        # test a detector with false stat sig
        second_values = 10.005 + 0.25 * np.random.randn(len(previous_seq))
        second = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": second_values})
        )

        second_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        second_int.data = second

        self.perc_change_2 = PercentageChange(current=current_int, previous=second_int)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_2 = ChangePointInterval(current_seq[0], current_seq[1])

        current_int_2.data = current

        self.perc_change_3 = PercentageChange(
            current=current_int_2, previous=previous_int
        )

    @parameterized.expand([["perc_change_1", True], ["perc_change_2", False]])
    def test_stat_sig(self, obj, ans):
        self.assertEqual(attrgetter(obj)(self).stat_sig, ans)

    @parameterized.expand([["perc_change_1", True], ["perc_change_2", False]])
    def test_p_value(self, obj, ans):
        self.assertEqual(attrgetter(obj)(self).p_value < 0.05, ans)

    @parameterized.expand(
        [
            ["perc_change_1", True],
            ["perc_change_2", False],
            ["perc_change_3", True],
        ]
    )
    def test_score(self, obj, ans):
        self.assertEqual(attrgetter(obj)(self).score > 1.96, ans)

    def test_ratio_estimate(self):
        self.assertEqual(self.perc_change_1.ratio_estimate, self.ratio_val_1)

    def test_approx_ratio_estimate(self):
        self.assertAlmostEqual(self.perc_change_1.ratio_estimate, 10.0, 0)

    def test_direction(self):
        self.assertEqual(self.perc_change_1.direction, "up")

    def test_perc_change(self):
        self.assertEqual(self.perc_change_1.perc_change, (self.ratio_val_1 - 1) * 100)

    # TODO delta method tests


class MultivariatePercentageChangeTest(TestCase):
    # test for multivariate time series
    def setUp(self) -> None:
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        current_length = 31
        # offset one to make the new interval start one day after the previous one ends
        current_seq = [
            previous_seq[-1] + timedelta(days=(x + 1)) for x in range(current_length)
        ]

        self.num_seq = 5

        previous_values = np.array(
            [
                1.0 + 0.0001 * np.random.randn(len(previous_seq))
                for _ in range(self.num_seq)
            ]
        )
        current_values = np.array(
            [
                10.0 + 0.0001 * np.random.randn(len(current_seq))
                for _ in range(self.num_seq)
            ]
        )

        previous = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": previous_values[i] for i in range(self.num_seq)},
                }
            )
        )

        current = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": current_seq},
                    **{f"value_{i}": current_values[i] for i in range(self.num_seq)},
                }
            )
        )

        previous_int = ChangePointInterval(
            previous_seq[0], previous_seq[-1] + timedelta(days=1)
        )
        previous_int.data = previous
        current_int = ChangePointInterval(
            current_seq[0], current_seq[-1] + timedelta(days=1)
        )
        current_int.data = current
        current_int.previous_interval = previous_int

        self.perc_change_1 = PercentageChange(
            current=current_int, previous=previous_int
        )

        previous_mean = np.array(
            [np.mean(previous_values[i]) for i in range(self.num_seq)]
        )
        current_mean = np.array(
            [np.mean(current_values[i]) for i in range(self.num_seq)]
        )

        # test the ratios
        self.ratio_val_1 = current_mean / previous_mean

        # test a detector with false stat sig
        second_values = np.array(
            [
                10.005 + 0.25 * np.random.randn(len(previous_seq))
                for _ in range(self.num_seq)
            ]
        )

        second = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": second_values[i] for i in range(self.num_seq)},
                }
            )
        )

        second_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        second_int.data = second

        self.perc_change_2 = PercentageChange(current=current_int, previous=second_int)

        # test a detector with a negative spike
        third_values = np.array(
            [
                1000.0 + 0.0001 * np.random.randn(len(previous_seq))
                for _ in range(self.num_seq)
            ]
        )

        third = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": third_values[i] for i in range(self.num_seq)},
                }
            )
        )

        third_int = ChangePointInterval(previous_seq[0], previous_seq[-1])
        third_int.data = third

        self.perc_change_3 = PercentageChange(current=current_int, previous=third_int)

        # test the edge case when one of the intervals
        # contains a single data point
        current_int_single_point = ChangePointInterval(current_seq[0], current_seq[1])

        current_int_single_point.data = current

        self.perc_change_single_point = PercentageChange(
            current=current_int_single_point, previous=previous_int
        )

    @parameterized.expand(
        [
            ["perc_change_1", True],
            ["perc_change_2", False],
            ["perc_change_3", True],
            ["perc_change_single_point", True],
        ]
    )
    def test_p_value(self, obj, ans):
        self.assertListEqual(
            (attrgetter(obj)(self).p_value < 0.05).tolist(), [ans] * self.num_seq
        )

    @parameterized.expand(
        [
            ["perc_change_1", True],
            ["perc_change_2", False],
        ]
    )
    def test_stat_sig(self, obj, ans):
        self.assertListEqual(
            (attrgetter(obj)(self).stat_sig).tolist(), [ans] * self.num_seq
        )

    @parameterized.expand(
        [
            ["perc_change_1", True],
            ["perc_change_2", False],
            ["perc_change_single_point", True],
        ]
    )
    def test_score(self, obj, ans):
        self.assertListEqual(
            (attrgetter(obj)(self).score > 1.96).tolist(), [ans] * self.num_seq
        )

    def test_score_negative(self):
        self.assertListEqual(
            (self.perc_change_3.score < -1.96).tolist(), [True] * self.num_seq
        )

    def test_approx_ratio_estimate(self):
        for r in self.perc_change_1.ratio_estimate:
            self.assertAlmostEqual(r, 10.0, 0)

    def test_direction(self):
        self.assertEqual(self.perc_change_1.direction.tolist(), ["up"] * self.num_seq)

    def test_perc_change(self):
        self.assertListEqual(
            self.perc_change_1.perc_change.tolist(),
            ((self.ratio_val_1 - 1) * 100).tolist(),
        )


class TestUnivariateAnomalyResponse(TestCase):
    # test anomaly response for univariate time series
    def setUp(self):
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(30)]
        self.score_ts = TimeSeriesData(
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

        self.conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        self.pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        self.mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": previous_seq,
                    "value": (10.0 + 0.25 * np.random.randn(len(previous_seq))),
                }
            )
        )

        self.stat_sig_ts = TimeSeriesData(
            pd.DataFrame({"time": previous_seq, "value": np.ones(len(previous_seq))})
        )

        self.response = AnomalyResponse(
            scores=self.score_ts,
            confidence_band=self.conf_band,
            predicted_ts=self.pred_ts,
            anomaly_magnitude_ts=self.mag_ts,
            stat_sig_ts=self.stat_sig_ts,
        )

        new_date = previous_seq[-1] + timedelta(days=1)
        self.N = len(previous_seq)
        common_val = 1.23

        self.response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=(common_val - 0.1),
            pred=common_val,
            anom_mag=common_val,
            stat_sig=0,
        )

    def test_response_univariate(self):
        #  Ensure that num_series is properly populated - this response object is univariate
        self.assertEqual(self.response.num_series, 1)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores"],
            ["confidence_band.upper"],
            ["confidence_band.lower"],
            ["predicted_ts"],
            ["anomaly_magnitude_ts"],
            ["stat_sig_ts"],
        ]
    )
    def test_update_response_preserves_length(self, attribute) -> None:
        # assert that all the lengths of the time series are preserved
        self.assertEqual(len(attrgetter(attribute)(self.response)), self.N)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores"],
            ["confidence_band.upper"],
            ["confidence_band.lower"],
            ["predicted_ts"],
            ["anomaly_magnitude_ts"],
            ["stat_sig_ts"],
        ]
    )
    def test_get_last_n_length(self, attribute) -> None:
        n_val = 10
        response_last_n = self.response.get_last_n(n_val)
        self.assertEqual(len(attrgetter(attribute)(response_last_n)), n_val)

    @parameterized.expand(
        [
            ["scores", "score_ts"],
            ["confidence_band.upper", "conf_band.upper"],
            ["confidence_band.lower", "conf_band.lower"],
            ["predicted_ts", "pred_ts"],
            ["anomaly_magnitude_ts", "mag_ts"],
            ["stat_sig_ts", "stat_sig_ts"],
        ]
    )
    def test_update_one_point_forward(self, attribute, initial_object):
        self.assertEqual(
            attrgetter(attribute)(self.response).value[0],
            attrgetter(initial_object)(self).value[1],
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores", 1.23],  # common_val
            ["confidence_band.upper", 1.23],  # common_val
            ["confidence_band.lower", 1.13],  # common_val-0.1
            ["predicted_ts", 1.23],  # common_val
            ["anomaly_magnitude_ts", 1.23],  # common_val
            ["stat_sig_ts", 0],  # not stat sig
        ]
    )
    def test_last_point(self, attribute, new_value) -> None:
        # assert that a new point has been added to the end
        self.assertEqual(
            attrgetter(attribute)(self.response).value.values[-1], new_value
        )

    def test_get_last_n_values(self) -> None:
        n_val = 10
        response_last_n = self.response.get_last_n(n_val)

        # assert that we return the last N values
        score_list = self.response.scores.value.values.tolist()
        self.assertEqual(
            response_last_n.scores.value.values.tolist(), score_list[-n_val:]
        )


class TestMultivariateAnomalyResponse(TestCase):
    # test anomaly response for multivariate time series

    def setUp(self):
        np.random.seed(100)

        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        self.num_seq = 5

        previous_seq = [date_start + timedelta(days=x) for x in range(30)]

        self.score_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.random.randn(len(previous_seq))
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        upper_values = [
            1.0 + np.random.randn(len(previous_seq)) for _ in range(self.num_seq)
        ]

        upper_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{f"value_{i}": upper_values[i] for i in range(self.num_seq)},
                }
            )
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": upper_values[i] - 0.1 for i in range(self.num_seq)
                    },
                }
            )
        )

        self.conf_band = ConfidenceBand(upper=upper_ts, lower=lower_ts)

        self.pred_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        self.mag_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": 10.0 + 0.25 * np.random.randn(len(previous_seq))
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        self.stat_sig_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": previous_seq},
                    **{
                        f"value_{i}": np.ones(len(previous_seq))
                        for i in range(self.num_seq)
                    },
                }
            )
        )

        self.response = AnomalyResponse(
            scores=self.score_ts,
            confidence_band=self.conf_band,
            predicted_ts=self.pred_ts,
            anomaly_magnitude_ts=self.mag_ts,
            stat_sig_ts=self.stat_sig_ts,
        )

        # test update
        new_date = previous_seq[-1] + timedelta(days=1)
        common_val = 1.23 * np.ones(self.num_seq)

        self.response.update(
            time=new_date,
            score=common_val,
            ci_upper=common_val,
            ci_lower=common_val - 0.1,
            pred=common_val,
            anom_mag=common_val,
            stat_sig=np.zeros(self.num_seq),
        )

        self.N = len(previous_seq)

    def test_response_num_series(self):
        # Ensure that num_series is properly populated
        self.assertEqual(self.response.num_series, self.num_seq)

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores"],
            ["confidence_band.upper"],
            ["confidence_band.lower"],
            ["predicted_ts"],
            ["anomaly_magnitude_ts"],
            ["stat_sig_ts"],
        ]
    )
    def test_update_response_preserves_length(self, attribute) -> None:
        # assert that all the lengths of the time series are preserved
        self.assertEqual(len(attrgetter(attribute)(self.response)), self.N)

    @parameterized.expand(
        [
            ["scores", "score_ts"],
            ["confidence_band.upper", "conf_band.upper"],
            ["confidence_band.lower", "conf_band.lower"],
            ["predicted_ts", "pred_ts"],
            ["anomaly_magnitude_ts", "mag_ts"],
            ["stat_sig_ts", "stat_sig_ts"],
        ]
    )
    def test_update_one_point_forward(self, attribute, initial_object):
        self.assertEqual(
            attrgetter(attribute)(self.response).value.iloc[0].tolist(),
            attrgetter(initial_object)(self).value.iloc[1].tolist(),
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores", 1.23],  # common_val
            ["confidence_band.upper", 1.23],  # common_val
            ["confidence_band.lower", 1.13],  # common_val-0.1
            ["predicted_ts", 1.23],  # common_val
            ["anomaly_magnitude_ts", 1.23],  # common_val
            ["stat_sig_ts", 0],  # not stat sig
        ]
    )
    def test_last_point(self, attribute, new_value) -> None:
        # assert that a new point has been added to the end
        self.assertEqual(
            attrgetter(attribute)(self.response).value.iloc[-1].tolist(),
            (new_value * np.ones(self.num_seq)).tolist(),
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            ["scores"],
            ["confidence_band.upper"],
            ["confidence_band.lower"],
            ["predicted_ts"],
            ["anomaly_magnitude_ts"],
            ["stat_sig_ts"],
        ]
    )
    def test_get_last_n_length(self, attribute) -> None:
        n_val = 10
        response_last_n = self.response.get_last_n(n_val)
        self.assertEqual(len(attrgetter(attribute)(response_last_n)), n_val)

    def test_get_last_n_values(self) -> None:
        n_val = 10
        response_last_n = self.response.get_last_n(n_val)

        # assert that we return the last N values
        score_list = self.response.scores.value.values.tolist()
        self.assertEqual(
            response_last_n.scores.value.values.tolist(), score_list[-n_val:]
        )
