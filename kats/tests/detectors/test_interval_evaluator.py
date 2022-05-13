# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
import pandas as pd
from kats.detectors.interval_evaluator import (
    measure,
    true_positives_delay,
)

from kats.detectors.cusum_model import CUSUMDetectorModel
from kats.detectors.prophet_detector import ProphetDetectorModel

from kats.consts import IntervalAnomaly
from kats.detectors.interval_evaluator import (
    get_cp_index_from_alert_score,
    combine_interval,
    combine_interval_user_anno,
    IntervalAnomalyEvaluator,
)
from kats.consts import TimeSeriesData


def generate_data() -> pd.DataFrame:
    np.random.seed(0)
    val1 = np.random.normal(1.5, 3, 100)
    time1 = pd.date_range(
        start="2018-01-01", freq="s", periods=100
    )
    ts1 = TimeSeriesData(pd.DataFrame({"time":time1, "value":pd.Series(val1)}))
    anno1 = {"1": [
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:00:02"), pd.to_datetime("2018-01-01 00:00:05")),
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:00:16"), pd.to_datetime("2018-01-01 00:00:17"))
        ],
        "2": [
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:01:26"), pd.to_datetime("2018-01-01 00:01:27")),
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:01:33"), pd.to_datetime("2018-01-01 00:01:36"))
        ]}

    np.random.seed(0)
    val2 = np.random.normal(0.5, 3, 100)
    time2 = pd.date_range(
        start="2018-01-01", freq="s", periods=100
    )
    ts2 = TimeSeriesData(pd.DataFrame({"time":time2, "value":pd.Series(val2)}))
    anno2 = {"1": [
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:00:04"), pd.to_datetime("2018-01-01 00:00:05")),
        IntervalAnomaly(pd.to_datetime("2018-01-01 00:01:26"), pd.to_datetime("2018-01-01 00:01:27")),
        ],
        "2": []}

    df = pd.DataFrame(
        [
            {
                "dataset_name": "eg_1",
                "time_series": ts1,
                "annotation": anno1,
            },
            {
                "dataset_name": "eg_2",
                "time_series": ts2,
                "annotation": anno2,
            },
        ]
    )

    return df


class TestIntervalAnomalyEvaluator(TestCase):
    def test_combine_interval(self) -> None:
        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )

        y1 = [IntervalAnomaly(time[20], time[22]), IntervalAnomaly(time[23], time[25]), IntervalAnomaly(time[30], time[42])]
        r1 = combine_interval(
            X=y1,
            ts_len_sec=300,
        )

        y2 = [IntervalAnomaly(time[20], time[22]), IntervalAnomaly(time[23], time[25]), IntervalAnomaly(time[30], time[42])]
        r2 = combine_interval(
            X=y2,
            ts_len_sec=50,
        )

        y3 = [IntervalAnomaly(time[11], time[22]), IntervalAnomaly(time[23], time[25]), IntervalAnomaly(time[30], time[42])]
        r3 = combine_interval(
            X=y3,
            ts_len_sec=100,
        )

        self.assertEquals(len(r1), 2)
        self.assertEquals(len(r2), 3)
        self.assertEquals(len(r3), 2)

        return

    def test_get_cp_index_from_alert_score(self) -> None:
        np.random.seed(0)
        x = np.random.normal(0.5, 3, 100)
        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )
        ts = pd.DataFrame({"time":time, "value":pd.Series(x)})

        res = get_cp_index_from_alert_score(
            score_df=ts,
            threshold_low=2,
            threshold_high=2,
            direction='up',
            detection_window_sec=3,
            fraction=1,
            onlineflag=True,
            eval_start_time_point=0
        )
        self.assertEqual(res[0].start, pd.to_datetime("2018-01-01 00:00:04"))
        self.assertEqual(res[0].end, pd.to_datetime("2018-01-01 00:00:05"))
        self.assertEqual(res[1].start, pd.to_datetime("2018-01-01 00:01:26"))
        self.assertEqual(res[1].end, pd.to_datetime("2018-01-01 00:01:27"))

        res = get_cp_index_from_alert_score(
            score_df=ts,
            threshold_low=-1,
            threshold_high=2,
            direction='down',
            detection_window_sec=3,
            fraction=1,
            onlineflag=True,
            eval_start_time_point=0
        )
        self.assertEqual(res[0].start, pd.to_datetime("2018-01-01 00:00:42"))
        self.assertEqual(res[0].end, pd.to_datetime("2018-01-01 00:00:43"))
        self.assertEqual(res[1].start, pd.to_datetime("2018-01-01 00:01:17"))
        self.assertEqual(res[1].end, pd.to_datetime("2018-01-01 00:01:18"))

        res = get_cp_index_from_alert_score(
            score_df=ts,
            threshold_low=-1,
            threshold_high=2,
            direction='both',
            detection_window_sec=3,
            fraction=1,
            onlineflag=True,
            eval_start_time_point=0
        )

        self.assertEquals(len(res), 6)

        res = get_cp_index_from_alert_score(
            score_df=ts,
            threshold_low=-1,
            threshold_high=2,
            direction='both',
            detection_window_sec=5,
            fraction=0.8,
            onlineflag=True,
            eval_start_time_point=0
        )

        self.assertEquals(len(res), 6)

    def test_combine_interval_user_anno(self) -> None:
        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )

        y1 = [IntervalAnomaly(time[24], time[26]), IntervalAnomaly(time[30], time[42])]
        y2 = [IntervalAnomaly(time[20], time[22]), IntervalAnomaly(time[23], time[25])]
        anno = {"1": y1, "2": y2}
        r1, r2 = combine_interval_user_anno(
            X=anno,
            ts_len_sec=50,
        )
        self.assertEquals(len(r1), 3)
        self.assertEquals(len(r2["1"]), 2)
        self.assertEquals(len(r2["2"]), 2)

        y1 = [IntervalAnomaly(time[24], time[26]), IntervalAnomaly(time[30], time[42])]
        y2 = [IntervalAnomaly(time[10], time[22]), IntervalAnomaly(time[23], time[25])]
        anno = {"1": y1, "2": y2}
        r1, r2 = combine_interval_user_anno(
            X=anno,
            ts_len_sec=50,
        )

        self.assertEquals(len(r1), 2)
        self.assertEquals(len(r2["1"]), 2)
        self.assertEquals(len(r2["2"]), 1)

        self.assertEqual(r1[0].start, pd.to_datetime("2018-01-01 00:00:10"))
        self.assertEqual(r1[0].end, pd.to_datetime("2018-01-01 00:00:26"))
        self.assertEqual(r1[1].start, pd.to_datetime("2018-01-01 00:00:30"))
        self.assertEqual(r1[1].end, pd.to_datetime("2018-01-01 00:00:42"))

    def test_true_positives_delay(self) -> None:
        """
        For example: label = [[10, 15], [20, 26], [30, 40]], prediction = [[12, 23]]. Then TP = 2, FN = 1.
        margin = 2 means we get prediction = [[12, 17], [17, 18], [18, 23]],
        so we get delay_dict = {10: 2, 20: -2} and an additional FP. (repeat_cnt = 2)
        If there is overlap between margin intervals, split down the middle.
        E.g., margin = 5 means we get prediction = [[12, 17.5], [17.5, 23]]
        and delay_dict = {10: 2, 10: -2.5}. (repeat_cnt = 1)
        """

        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )

        label = [
            IntervalAnomaly(time[10], time[15]),
            IntervalAnomaly(time[20], time[26]),
            IntervalAnomaly(time[30], time[40])
        ]

        pred1 = [IntervalAnomaly(time[12], time[23])]
        pred2 = [IntervalAnomaly(time[12], time[50])]
        pred3 = [IntervalAnomaly(time[43], time[50])]
        pred4 = [IntervalAnomaly(time[40], time[50])]
        pred5 = [IntervalAnomaly(time[5], time[10])]
        pred6 = [IntervalAnomaly(time[5], time[10])]

        # test margin
        res1 = true_positives_delay(labels=label, predictions=pred1, margin_sec=5)
        res2 = true_positives_delay(labels=label, predictions=pred1, margin_sec=2)
        res3 = true_positives_delay(labels=label, predictions=pred1, margin_sec=0)

        # test margin-split
        res4 = true_positives_delay(labels=label, predictions=pred2, margin_sec=2)
        res5 = true_positives_delay(labels=label, predictions=pred3, margin_sec=2)
        res6 = true_positives_delay(labels=label, predictions=pred4, margin_sec=0)
        res7 = true_positives_delay(labels=label, predictions=pred5, margin_sec=0)
        res8 = true_positives_delay(labels=label, predictions=pred6, margin_sec=2)

        self.assertEqual(
            res1,
            ({
                pd.to_datetime('2018-01-01 00:00:10'): 2.0,
                pd.to_datetime('2018-01-01 00:00:20'): -2.5
            },
            1)
        )

        self.assertEqual(
            res2,
            ({
                pd.to_datetime('2018-01-01 00:00:10'): 2.0,
                pd.to_datetime('2018-01-01 00:00:20'): -3.0
            },
            2)
        )

        self.assertEqual(
            res3,
            ({
                pd.to_datetime('2018-01-01 00:00:10'): 2.0,
                pd.to_datetime('2018-01-01 00:00:20'): -5.0
            },
            2)
        )

        self.assertEqual(
            res4,
            ({
                pd.to_datetime('2018-01-01 00:00:10'): 2.0,
                pd.to_datetime('2018-01-01 00:00:20'): -3.0,
                pd.to_datetime('2018-01-01 00:00:30'): -2.0
            },
            4)
        )

        self.assertEqual(res5, ({}, 0))

        self.assertEqual(
            res6,
            ({
                pd.to_datetime('2018-01-01 00:00:30'): 10.0
            },
            1)
        )

        self.assertEqual(res7, ({}, 0))

        self.assertEqual(
            res8, ({pd.to_datetime('2018-01-01 00:00:10'): -5.0}, 1)
        )

    def test_measure(self) -> None:
        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )

        label = [
            IntervalAnomaly(time[10], time[15]),
            IntervalAnomaly(time[20], time[26]),
            IntervalAnomaly(time[30], time[40])
        ]

        pred1 = [IntervalAnomaly(time[12], time[23])]
        pred2 = [IntervalAnomaly(time[12], time[50])]
        pred3 = [IntervalAnomaly(time[43], time[50])]
        pred4 = [IntervalAnomaly(time[40], time[50])]
        pred5 = [IntervalAnomaly(time[5], time[10])]
        pred6 = [IntervalAnomaly(time[5], time[10])]

        res1 = measure(anno_all=label, predictions=pred1, margin=5)
        res2 = measure(anno_all=label, predictions=pred1, margin=0)
        res3 = measure(anno_all=label, predictions=pred1, margin=2)
        res4 = measure(anno_all=label, predictions=pred2, margin=2)
        res5 = measure(anno_all=label, predictions=pred3, margin=2)
        res6 = measure(anno_all=label, predictions=pred4, margin=0)
        res7 = measure(anno_all=label, predictions=pred5, margin=0)
        res8 = measure(anno_all=label, predictions=pred6, margin=2)

        self.assertDictEqual(res1.to_dict(),
            {
                'f_score': 0.8,
                'precision': 1.0,
                'recall': 0.66667,
                'delay': 2.0,
                'TP': 2,
                'FP': 0,
                'FN': 1,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res2.to_dict(),
            {
                'f_score': 0.66667,
                'precision': 0.66667,
                'recall': 0.66667,
                'delay': 2.0,
                'TP': 2,
                'FP': 1,
                'FN': 1,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res3.to_dict(),
            {
                'f_score': 0.66667,
                'precision': 0.66667,
                'recall': 0.66667,
                'delay': 2.0,
                'TP': 2,
                'FP': 1,
                'FN': 1,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res4.to_dict(),
            {
                'f_score': 0.75,
                'precision': 0.6,
                'recall': 1.0,
                'delay': 2.0,
                'TP': 3,
                'FP': 2,
                'FN': 0,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res5.to_dict(),
            {
                'f_score': 0,
                'precision': 0.0,
                'recall': 0.0,
                'delay': np.float("Inf"),
                'TP': 0,
                'FP': 1,
                'FN': 3,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res6.to_dict(),
            {
                'f_score': 0.4,
                'precision': 0.5,
                'recall': 0.33333,
                'delay': 10.0,
                'TP': 1,
                'FP': 1,
                'FN': 2,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res7.to_dict(),
            {
                'f_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'delay': np.float("Inf"),
                'TP': 0,
                'FP': 1,
                'FN': 3,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res8.to_dict(),
            {
                'f_score': 0.4,
                'precision': 0.5,
                'recall': 0.33333,
                'delay': 0,
                'TP': 1,
                'FP': 1,
                'FN': 2,
                'dataset_name': 'dataset_name',
            }
        )

        label1 = []
        pred11 = [IntervalAnomaly(time[12], time[23])]
        pred12 = []

        label2 = [IntervalAnomaly(time[12], time[23])]
        pred21 = []

        res11 = measure(anno_all=label1, predictions=pred11, margin=5)
        res12 = measure(anno_all=label1, predictions=pred12, margin=5)
        res13 = measure(anno_all=label2, predictions=pred21, margin=5)

        self.assertDictEqual(res11.to_dict(),
            {
                'f_score': 0,
                'precision': 0,
                'recall': 0,
                'delay': np.float("Inf"),
                'TP': 0,
                'FP': 1,
                'FN': 0,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res12.to_dict(),
            {
                'f_score': 0,
                'precision': 0,
                'recall': 0,
                'delay': np.float("Inf"),
                'TP': 0,
                'FP': 0,
                'FN': 0,
                'dataset_name': 'dataset_name',
            }
        )

        self.assertDictEqual(res13.to_dict(),
            {
                'f_score': 0,
                'precision': 0,
                'recall': 0,
                'delay': np.float("Inf"),
                'TP': 0,
                'FP': 0,
                'FN': 1,
                'dataset_name': 'dataset_name',
            }
        )

    def test_errors(self) -> None:
        time = pd.date_range(
            start="2018-01-01", freq="s", periods=100
        )
        label = [ IntervalAnomaly(time[10], time[15])]
        pred1 = [IntervalAnomaly(time[12], time[23])]

        with self.assertRaises(ValueError):
            _ = true_positives_delay(labels=label, predictions=pred1, margin_sec=-5)

        with self.assertRaises(ValueError):
            _ = IntervalAnomaly(pd.to_datetime("2018-01-01 00:00:04"), pd.to_datetime("2018-01-01 00:00:01"))

    def test_evaluator(self) -> None:
        eg_df = generate_data()

        # Test statsig detector
        itv_evaluator = IntervalAnomalyEvaluator(detector=CUSUMDetectorModel)
        params = {"scan_window": 20, "historical_window": 20.0}
        eval_agg_df = itv_evaluator.evaluate(
            data=eg_df,
            model_params=params,
            ignore_list=None,
            threshold_low=0.0,
            threshold_high=2.0,
            margin=2,
            direction="both",
            detection_window_sec=3,
            fraction=1.0,
            eval_start_time_sec=20,
            training_window_sec=None,
            retrain_freq_sec=None,
            perc_last_itv=0.1,
            perc_total_len=0.01,
        )
        self.assertEqual(eval_agg_df.shape[0], eg_df.shape[0])

        # test prophet model
        itv_evaluator = IntervalAnomalyEvaluator(detector=ProphetDetectorModel)
        params = {"scoring_confidence_interval": 0.9}
        eval_agg_df = itv_evaluator.evaluate(
            data=eg_df,
            model_params=params,
            ignore_list=None,
            threshold_low=0.0,
            threshold_high=2.0,
            margin=2,
            direction="both",
            detection_window_sec=3,
            fraction=1.0,
            eval_start_time_sec=25,
            training_window_sec=20,
            retrain_freq_sec=20,
            perc_last_itv=0.1,
            perc_total_len=0.01,
        )
        self.assertEqual(eval_agg_df.shape[0], eg_df.shape[0])


        # test ignore list
        itv_evaluator = IntervalAnomalyEvaluator(detector=ProphetDetectorModel)
        params = {"scoring_confidence_interval": 0.9}
        eval_agg_df = itv_evaluator.evaluate(
            data=eg_df,
            model_params=params,
            ignore_list=["eg_1"],
            threshold_low=0.0,
            threshold_high=2.0,
            margin=2,
            direction="both",
            detection_window_sec=3,
            fraction=1.0,
            eval_start_time_sec=25,
            training_window_sec=20,
            retrain_freq_sec=20,
            perc_last_itv=0.1,
            perc_total_len=0.01,
        )
        self.assertEqual(eval_agg_df.shape[0], eg_df.shape[0] - 1)
