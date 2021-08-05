# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.detectors.bocpd import BOCPDetector
from kats.detectors.bocpd_model import BocpdDetectorModel
from kats.detectors.changepoint_evaluator import (
    TuringEvaluator,
    Evaluation,
    EvalAggregate,
    measure,
    true_positives,
)
from kats.detectors.cusum_detection import (
    CUSUMDetector,
)
from kats.detectors.cusum_model import (
    CUSUMDetectorModel,
    CusumScoreFunction,
)
from kats.detectors.robust_stat_detection import RobustStatDetector
from kats.detectors.stat_sig_detector import (
    StatSigDetectorModel,
)

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class TestChangepointEvaluator(TestCase):
    def test_eval_agg(self) -> None:
        eval_1 = Evaluation(
            dataset_name="eg_1", precision=0.3, recall=0.5, f_score=0.6, delay=-2
        )

        eval_2 = Evaluation(
            dataset_name="eg_2", precision=0.3, recall=0.5, f_score=0.7, delay=1
        )

        eval_agg_1 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_precision = eval_agg_1.get_avg_precision()
        self.assertAlmostEqual(avg_precision, 0.3, places=4)

        eval_agg_2 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_recall = eval_agg_2.get_avg_recall()
        self.assertAlmostEqual(avg_recall, 0.5, places=4)

        eval_agg_3 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_f_score = eval_agg_3.get_avg_f_score()
        self.assertAlmostEqual(avg_f_score, 0.65, places=4)

        eval_agg_4 = EvalAggregate(eval_list=[eval_1, eval_2])
        avg_delay = eval_agg_4.get_avg_delay()
        self.assertAlmostEqual(avg_delay, -0.5, places=4)

    def test_measure(self) -> None:
        """
        tests the correctness of measure, by comparing results with
        https://arxiv.org/pdf/2003.06222.pdf and TCPDBench github
        project
        """
        brent_spot_anno = {
            "6": [219, 230, 288],
            "8": [227, 381],
            "9": [86, 219, 230, 279, 375],
            "12": [169, 172, 217, 228, 287, 368, 382, 389, 409],
            "13": [170, 180, 219, 229, 246, 271, 286, 379, 409, 444, 483],
        }

        brent_spot_prophet_default_cploc = [259, 279, 299, 319, 339, 359]

        # scores are defined in
        # https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/output/summaries/summary_brent_spot.json

        f_brent_spot = measure(
            annotations=brent_spot_anno, predictions=brent_spot_prophet_default_cploc
        )
        self.assertAlmostEqual(f_brent_spot["f_score"], 0.2485875706214689, places=3)
        self.assertAlmostEqual(f_brent_spot["precision"], 0.2857142857142857, places=3)
        self.assertAlmostEqual(f_brent_spot["recall"], 0.21999999999999997, places=3)

    def test_true_positives(self) -> None:
        """
        tests the correctness of true_positives.
        """
        tp1 = true_positives({1, 10, 20, 23}, {3, 8, 20})
        self.assertDictEqual(tp1, {1: 2, 10: -2, 20: 0})
        tp2 = true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
        self.assertDictEqual(tp2, {1: 0, 10: -5, 20: 0})
        tp3 = true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20}, choose_earliest=False)
        self.assertDictEqual(tp3, {1: 0, 10: -2, 20: 0})

    def test_evaluator(self) -> None:
        date_range = pd.date_range(start="2010-02-01", end="2020-03-31", freq="M")
        date_range_start = [x + timedelta(days=1) for x in date_range]
        y_m_d_str = [datetime.strftime(x, "%Y-%m-%d") for x in date_range_start]
        y_m_str = [datetime.strftime(x, "%Y-%m") for x in date_range_start]
        int_str = [str(x) for x in range(len(date_range_start))]
        int_val = list(range(len(date_range_start)))

        val = np.random.randn(len(date_range_start))

        eg_anno = {"1": [2, 6, 10], "2": [3, 6]}
        y_m_d_dict = {k: v for k, v in zip(y_m_d_str, val)}
        y_m_dict = {k: v for k, v in zip(y_m_str, val)}
        int_dict = {k: v for k, v in zip(int_str, val)}
        int_val_dict = {k: v for k, v in zip(int_val, val)}

        eg_df = pd.DataFrame(
            [
                {
                    "dataset_name": "eg_1",
                    "time_series": str(y_m_d_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_2",
                    "time_series": str(y_m_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_3",
                    "time_series": str(int_dict),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_4",
                    "time_series": str(int_val_dict),
                    "annotation": str(eg_anno),
                },
            ]
        )

        model_params = {"p_value_cutoff": 5e-3, "comparison_window": 2}

        # Test RobustStatDetector
        turing_2 = TuringEvaluator(detector=RobustStatDetector)
        eval_agg_2_df = turing_2.evaluate(data=eg_df, model_params=model_params)
        self.assertEqual(eval_agg_2_df.shape[0], eg_df.shape[0])

        # Test CUSUMDetector
        turing_3 = TuringEvaluator(detector=CUSUMDetector)
        eval_agg_3_df = turing_3.evaluate(data=eg_df)
        self.assertEqual(eval_agg_3_df.shape[0], eg_df.shape[0])

        # Test BOCPDDetector
        turing_4 = TuringEvaluator(detector=BOCPDetector)
        eval_agg_4_df = turing_4.evaluate(data=eg_df)
        self.assertEqual(eval_agg_4_df.shape[0], eg_df.shape[0])

        # test the eval_agg
        eval_agg_4 = turing_4.get_eval_aggregate()
        eval_agg_df = eval_agg_4.get_eval_dataframe()
        self.assertEqual(eval_agg_df.shape[0], eg_df.shape[0])
        avg_precision = eval_agg_4.get_avg_precision()
        avg_recall = eval_agg_4.get_avg_recall()
        avg_f_score = eval_agg_4.get_avg_f_score()
        self.assertTrue(0.0 <= avg_precision <= 1.0)
        self.assertTrue(0.0 <= avg_recall <= 1.0)
        self.assertTrue(0.0 <= avg_f_score <= 1.0)

        # test load data
        turing_5 = TuringEvaluator(detector=RobustStatDetector)
        eval_agg_5_df = turing_5.evaluate(data=None, model_params=model_params)
        self.assertTrue(eval_agg_5_df.shape[0] > 0)

        # test ignore list
        turing_6 = TuringEvaluator(detector=RobustStatDetector)
        eval_agg_6_df = turing_6.evaluate(
            data=eg_df, model_params=model_params, ignore_list=["eg_2"]
        )
        self.assertEqual(eval_agg_6_df.shape[0], eg_df.shape[0] - 1)

        # test the detectormodels
        turing_7 = TuringEvaluator(detector=BocpdDetectorModel, is_detector_model=True)
        eval_agg_7_df = turing_7.evaluate(data=eg_df, model_params=None)
        self.assertEqual(eval_agg_7_df.shape[0], eg_df.shape[0])

        # test Statsig
        num_secs_in_month = 86400 * 30
        statsig_model_params = {
            "n_control": 7 * num_secs_in_month,
            "n_test": 7 * num_secs_in_month,
            "time_unit": "sec",
        }

        turing_8 = TuringEvaluator(
            detector=StatSigDetectorModel,
            is_detector_model=True,
        )
        eval_agg_8_df = turing_8.evaluate(
            data=eg_df,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, float]]` for 2nd
            #  param but got `Dict[str, typing.Union[int, str]]`.
            model_params=statsig_model_params,
            alert_style_cp=False,
            threshold_low=-5.0,
            threshold_high=5.0,
        )

        self.assertEqual(eval_agg_8_df.shape[0], eg_df.shape[0])

        # test CUSUM
        # since CUSUM needs daily data, constructing another eg_df
        eg_start_unix_time = 1613764800
        num_secs_in_day = 3600 * 24

        date_range_daily = pd.date_range(start="2020-03-01", end="2020-03-31", freq="D")
        date_range_start_daily = [x + timedelta(days=1) for x in date_range_daily]
        y_m_d_str_daily = [
            datetime.strftime(x, "%Y-%m-%d") for x in date_range_start_daily
        ]
        int_daily = [
            (eg_start_unix_time + x * num_secs_in_day)
            for x in range(len(date_range_start_daily))
        ]
        int_str_daily = [str(x) for x in int_daily]

        val_daily = np.random.randn(len(date_range_start_daily))

        y_m_d_dict_daily = {k: v for k, v in zip(y_m_d_str_daily, val_daily)}
        int_dict_daily = {k: v for k, v in zip(int_daily, val_daily)}
        int_str_dict_daily = {k: v for k, v in zip(int_str_daily, val_daily)}

        eg_df_daily = pd.DataFrame(
            [
                {
                    "dataset_name": "eg_1",
                    "time_series": str(y_m_d_dict_daily),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_3",
                    "time_series": str(int_dict_daily),
                    "annotation": str(eg_anno),
                },
                {
                    "dataset_name": "eg_4",
                    "time_series": str(int_str_dict_daily),
                    "annotation": str(eg_anno),
                },
            ]
        )

        cusum_model_params = {
            "scan_window": 8 * num_secs_in_day,
            "historical_window": 8 * num_secs_in_day,
            "threshold": 0.01,
            "delta_std_ratio": 1.0,
            "change_directions": ["increase", "decrease"],
            "score_func": CusumScoreFunction.percentage_change,
            "remove_seasonality": False,
        }

        turing_9 = TuringEvaluator(detector=CUSUMDetectorModel, is_detector_model=True)
        eval_agg_9_df = turing_9.evaluate(
            data=eg_df_daily,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, float]]` for 2nd
            #  param but got `Dict[str, typing.Union[typing.List[str],
            #  CusumScoreFunction, float]]`.
            model_params=cusum_model_params,
            alert_style_cp=True,
            threshold_low=-0.1,
            threshold_high=0.1,
        )

        self.assertEqual(eval_agg_9_df.shape[0], eg_df_daily.shape[0])
