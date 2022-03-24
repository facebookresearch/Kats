# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from unittest import TestCase, mock

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.changepoint_evaluator import (
    TuringEvaluator,
    Evaluation,
    EvalAggregate,
    measure,
    true_positives,
)
from kats.detectors.cusum_model import (
    CusumScoreFunction,
)
from kats.detectors.detector import Detector

from typing import Any, Dict, List, Type, Union

from kats.detectors.detector import DetectorModel
from kats.detectors.prophet_detector import (
    ProphetDetectorModel,
    ProphetTrendDetectorModel,
)
from kats.detectors.slow_drift_detector import SlowDriftDetectorModel
from kats.detectors.stat_sig_detector import StatSigDetectorModel
from kats.detectors.trend_mk_model import MKDetectorModel


OUTLIER_SPACE: List[Dict[str, Any]] = [
    {
        "name": "iqr_mult",
        "type": "choice",
        "values": [2.0, 3.0, 4.0],
        "value_type": "float",
        "is_ordered": True,
    },
]

MK_SPACE: List[Dict[str, Any]] = [
    {
        "name": "window_size",
        "type": "choice",
        "values": [5, 10, 15, 20],
        "value_type": "int",
        "is_ordered": True,
    },
]

STATSIG_SPACE: List[Dict[str, Any]] = [
    {
        "name": "n_control",
        "type": "choice",
        "values": [2, 5, 10],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "n_test",
        "type": "choice",
        "values": [2, 5, 10],
        "value_type": "int",
        "is_ordered": True,
    },
]

PROPHET_TREND_SPACE: List[Dict[str, Any]] = [
    {
        "name": "changepoint_prior_scale",
        "type": "choice",
        "values": [0.01, 0.05, 0.1],
        "value_type": "float",
        "is_ordered": True,
    },
]

PROPHET_SPACE: List[Dict[str, Any]] = [
    {
        "name": "scoring_confidence_interval",
        "type": "choice",
        "values": [0.8, 0.9],
        "value_type": "float",
        "is_ordered": True,
    },
]

SLOWDRIFT_SPACE: List[Dict[str, Any]] = [
    {
        "name": "slow_drift_window",
        "type": "choice",
        "values": [1, 2],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "algorithm_version",
        "type": "choice",
        "values": [1, 2],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "seasonality_period",
        "type": "choice",
        "values": [7],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "seasonality_num_points",
        "type": "choice",
        "values": [0, 1, 7],
        "value_type": "int",
        "is_ordered": True,
    },
]



GRID_SD_DICT: Dict[Type[DetectorModel], List[Dict[str, Any]]] = {
    SlowDriftDetectorModel: SLOWDRIFT_SPACE,
    StatSigDetectorModel: STATSIG_SPACE,
    ProphetTrendDetectorModel: PROPHET_TREND_SPACE,
    MKDetectorModel: MK_SPACE,
    ProphetDetectorModel: PROPHET_SPACE,
}

DETECTOR_ONLINE_ONLY: Dict[Union[Type[Detector], Type[DetectorModel]], bool] = {
    SlowDriftDetectorModel: True,
    StatSigDetectorModel: True,
    MKDetectorModel: True,
    ProphetTrendDetectorModel: False,
    ProphetDetectorModel: False,
}


class GenerateData:
    """
    Generate the data used to evaluate the parsing and the detector.
    """

    len_ts: int
    eg_df: pd.DataFrame
    num_secs_in_day: int = 3600 * 24

    def __init__(self, monthly: bool = True) -> None:
        self.eg_anno = {"1": [2, 6, 10], "2": [3, 6]}
        if monthly:
            date_range = pd.date_range(start="2010-02-01", end="2020-03-31", freq="M")
            date_range_start = [x + timedelta(days=1) for x in date_range]
            self.len_ts = len(date_range_start)
            y_m_d_str = [datetime.strftime(x, "%Y-%m-%d") for x in date_range_start]
            y_m_str = [datetime.strftime(x, "%Y-%m") for x in date_range_start]
            int_str = [str(x) for x in range(self.len_ts)]
            int_val = list(range(len(date_range_start)))

            val = np.random.randn(len(date_range_start))

            y_m_d_dict = {k: v for k, v in zip(y_m_d_str, val)}
            y_m_dict = {k: v for k, v in zip(y_m_str, val)}
            int_dict = {k: v for k, v in zip(int_str, val)}
            int_val_dict = {k: v for k, v in zip(int_val, val)}

            self.eg_df = pd.DataFrame(
                [
                    {
                        "dataset_name": "eg_1",
                        "time_series": str(y_m_d_dict),
                        "annotation": str(self.eg_anno),
                    },
                    {
                        "dataset_name": "eg_2",
                        "time_series": str(y_m_dict),
                        "annotation": str(self.eg_anno),
                    },
                    {
                        "dataset_name": "eg_3",
                        "time_series": str(int_dict),
                        "annotation": str(self.eg_anno),
                    },
                    {
                        "dataset_name": "eg_4",
                        "time_series": str(int_val_dict),
                        "annotation": str(self.eg_anno),
                    },
                ]
            )
        else:
            eg_start_unix_time = 1613764800

            date_range_daily = pd.date_range(
                start="2020-03-01", end="2020-03-31", freq="D"
            )
            date_range_start_daily = [x + timedelta(days=1) for x in date_range_daily]
            self.len_ts = len(date_range_start_daily)
            y_m_d_str_daily = [
                datetime.strftime(x, "%Y-%m-%d") for x in date_range_start_daily
            ]
            int_daily = [
                (eg_start_unix_time + x * self.num_secs_in_day)
                for x in range(len(date_range_start_daily))
            ]
            int_str_daily = [str(x) for x in int_daily]

            val_daily = np.random.randn(len(date_range_start_daily))

            y_m_d_dict_daily = {k: v for k, v in zip(y_m_d_str_daily, val_daily)}
            int_dict_daily = {k: v for k, v in zip(int_daily, val_daily)}
            int_str_dict_daily = {k: v for k, v in zip(int_str_daily, val_daily)}

            self.eg_df = pd.DataFrame(
                [
                    {
                        "dataset_name": "eg_1",
                        "time_series": str(y_m_d_dict_daily),
                        "annotation": str(self.eg_anno),
                    },
                    {
                        "dataset_name": "eg_3",
                        "time_series": str(int_dict_daily),
                        "annotation": str(self.eg_anno),
                    },
                    {
                        "dataset_name": "eg_4",
                        "time_series": str(int_str_dict_daily),
                        "annotation": str(self.eg_anno),
                    },
                ]
            )


class TestChangepointEvaluator(TestCase):
    def test_eval_agg(self) -> None:
        eval_1 = Evaluation(
            dataset_name="eg_1",
            precision=0.3,
            recall=0.5,
            f_score=0.6,
            delay=-2,
            TP=3,
            FN=3,
            FP=7,
        )

        eval_2 = Evaluation(
            dataset_name="eg_2",
            precision=0.3,
            recall=0.5,
            f_score=0.7,
            delay=1,
            TP=3,
            FN=3,
            FP=7,
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

    def test_parse_data(self) -> None:
        """
        tests the correctness of test_parse_data.
        """
        # Define an evaluator with any model just to access the _parse_data method.
        data_generator = GenerateData()
        eg_df = data_generator.eg_df
        turing_2 = TuringEvaluator(detector=Detector)
        for i, row in eg_df.iterrows():
            data_name, tsd, anno = turing_2._parse_data(row)
            self.assertEqual(data_name, eg_df.dataset_name.values[i])
            self.assertEqual(len(tsd), data_generator.len_ts)
            self.assertDictEqual(anno, data_generator.eg_anno)

    def test_evaluator(self) -> None:
        data_generator = GenerateData()
        eg_df = data_generator.eg_df
        model_params_mock = mock.MagicMock()

        # Test RobustStatDetector
        class RobustStatDetector_fake(Detector):
            def __init__(self, tsd=None):
                pass

            def detector(self, params=None):
                return []

        turing_2 = TuringEvaluator(detector=RobustStatDetector_fake)
        eval_agg_2_df = turing_2.evaluate(data=eg_df, model_params=model_params_mock)
        self.assertEqual(eval_agg_2_df.shape[0], eg_df.shape[0])

        # Test CUSUMDetector
        class CUSUMDetector_fake(Detector):
            def __init__(self, tsd=None):
                pass

            def detector(self, params=None):
                return []

        turing_3 = TuringEvaluator(detector=CUSUMDetector_fake)
        eval_agg_3_df = turing_3.evaluate(data=eg_df)
        self.assertEqual(eval_agg_3_df.shape[0], eg_df.shape[0])

        # Test BOCPDDetector
        class BOCPDDetector_fake(Detector):
            def __init__(self, tsd=None):
                pass

            def detector(self, params=None):
                return []

        turing_4 = TuringEvaluator(detector=BOCPDDetector_fake)
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
        eval_agg_5_df = turing_4.evaluate(data=None, model_params=model_params_mock)
        self.assertTrue(eval_agg_5_df.shape[0] > 0)

        # test ignore list
        eval_agg_6_df = turing_4.evaluate(
            data=eg_df,
            model_params=model_params_mock,
            ignore_list=[eg_df.dataset_name.values[1]],
        )
        self.assertEqual(eval_agg_6_df.shape[0], eg_df.shape[0] - 1)

        # test the detectormodels
        BocpdDetectorModelMock = mock.MagicMock()
        BocpdDetectorModelMock.return_value.fit_predict.return_value.scores.value.values = np.array(
            [0.0] * (data_generator.len_ts - 1) + [0.1]
        )
        turing_7 = TuringEvaluator(
            detector=BocpdDetectorModelMock, is_detector_model=True
        )
        eval_agg_7_df = turing_7.evaluate(
            data=eg_df,
            model_params=None,
            alert_style_cp=True,
            threshold_high=0.001,
        )
        self.assertEqual(eval_agg_7_df.shape[0], eg_df.shape[0])

        # test load data
        eval_agg_8_df = turing_7.evaluate(data=None, model_params=model_params_mock)
        self.assertTrue(eval_agg_8_df.shape[0] > 0)

        # test ignore list
        eval_agg_9_df = turing_7.evaluate(
            data=eg_df,
            model_params=model_params_mock,
            ignore_list=[eg_df.dataset_name.values[1]],
        )
        self.assertEqual(eval_agg_9_df.shape[0], eg_df.shape[0] - 1)

        # test Statsig
        num_secs_in_month = 86400 * 30
        statsig_model_params = {
            "n_control": 7 * num_secs_in_month,
            "n_test": 7 * num_secs_in_month,
            "time_unit": "sec",
        }
        StatSigDetectorModelMock = mock.MagicMock()
        StatSigDetectorModelMock.return_value.fit_predict.return_value.scores.value.values = np.array(
            [0.0] * data_generator.len_ts
        )
        turing_8 = TuringEvaluator(
            detector=StatSigDetectorModelMock,
            is_detector_model=True,
        )
        eval_agg_10_df = turing_8.evaluate(
            data=eg_df,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, float]]` for 2nd
            #  param but got `Dict[str, typing.Union[int, str]]`.
            model_params=statsig_model_params,
            alert_style_cp=False,
            threshold_low=-5.0,
            threshold_high=5.0,
        )
        self.assertEqual(eval_agg_10_df.shape[0], eg_df.shape[0])

        data_generator = GenerateData(monthly=False)
        eg_df = data_generator.eg_df

        cusum_model_params = {
            "scan_window": 8 * data_generator.num_secs_in_day,
            "historical_window": 8 * data_generator.num_secs_in_day,
            "threshold": 0.01,
            "delta_std_ratio": 1.0,
            "change_directions": ["increase", "decrease"],
            "score_func": CusumScoreFunction.percentage_change,
            "remove_seasonality": False,
        }

        CUSUMDetectorModelMock = mock.MagicMock()
        CUSUMDetectorModelMock.return_value.fit_predict.return_value.scores.value.values = np.array(
            [0.0] * data_generator.len_ts
        )

        turing_9 = TuringEvaluator(
            detector=CUSUMDetectorModelMock, is_detector_model=True
        )
        eval_agg_9_df = turing_9.evaluate(
            data=eg_df,
            # pyre-fixme[6]: For 2nd param expected `Optional[Dict[str, float]]` but
            #  got `Dict[str, Union[List[str], CusumScoreFunction, float]]`.
            model_params=cusum_model_params,
            alert_style_cp=True,
            threshold_low=-0.1,
            threshold_high=0.1,
        )

        self.assertEqual(eval_agg_9_df.shape[0], eg_df.shape[0])

    def test_eval_start_time(self) -> None:
        data_generator = GenerateData()
        eg_df = data_generator.eg_df.iloc[:2, :]

        for detect_model in list(GRID_SD_DICT.keys()):
            turing_model = TuringEvaluator(
                detector=detect_model,
                is_detector_model=True,
            )
            params = {list(x.values())[0]: list(x.values())[2][0] for x in GRID_SD_DICT[detect_model]}
            eval_agg = turing_model.evaluate(
                data=eg_df,
                model_params=params,
                alert_style_cp=True,
                eval_start_time_sec=500*24*3600,
                training_window_sec=500*24*3600,
                retrain_freq_sec=500*24*3600,
            )

            self.assertEqual(eval_agg.shape[0], eg_df.shape[0])

    def test_periodic_retraining_error(self) -> None:
        data_generator = GenerateData()
        eg_df = data_generator.eg_df

        # for time series data that is too short
        eg_df2 = pd.DataFrame(
                [
                    {
                        "dataset_name": "eg_2",
                        "time_series": str(TimeSeriesData()),
                        "annotation": str({"1": [2, 6, 10], "2": [3, 6]}),
                    },
                ]
            )

        # incorrect params
        for detect_model, x, y, z in [
            (ProphetTrendDetectorModel, 5, 500, 500),
            (ProphetTrendDetectorModel, 500, 5, 500),
            (ProphetTrendDetectorModel, 500, 500, 5),
        ]:
            turing_model = TuringEvaluator(
                detector=detect_model,
                is_detector_model=True,
            )
            params = {list(x.values())[0]: list(x.values())[2][0] for x in GRID_SD_DICT[detect_model]}
            with self.assertRaises(ValueError):
                _ = turing_model.evaluate(
                    data=eg_df,
                    model_params=params,
                    alert_style_cp=True,
                    eval_start_time_sec=x*24*3600,
                    training_window_sec=y*24*3600,
                    retrain_freq_sec=z*24*3600,
                )

            # for time series data that is too short
            turing_model = TuringEvaluator(
                detector=detect_model,
                is_detector_model=True,
            )
            with self.assertRaises(ValueError):
                _ = turing_model.evaluate(
                    data=eg_df2,
                    model_params=params,
                    alert_style_cp=True,
                    eval_start_time_sec=500*24*3600,
                    training_window_sec=500*24*3600,
                    retrain_freq_sec=500*24*3600,
                )

    def test_online_flag(self) -> None:
        for detect_model in GRID_SD_DICT:
            print(detect_model)
            turing_model = TuringEvaluator(
                detector=detect_model,
                is_detector_model=True,
            )
            self.assertEqual(DETECTOR_ONLINE_ONLY[detect_model], turing_model.onlineflag)
