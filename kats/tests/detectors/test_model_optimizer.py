# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import IntervalAnomaly, SearchMethodEnum, TimeSeriesData

# from kats.detectors.window_slo_detector import WindowSloDetectorModel
from kats.detectors.bocpd_model import BocpdDetectorModel

# from kats.detectors.stat_sig_detector import MultiStatSigDetectorModel
from kats.detectors.cusum_model import CUSUMDetectorModel
from kats.detectors.detector import Detector, DetectorModel
from kats.detectors.detector_benchmark import ModelBenchmark, ModelOptimizer
from kats.detectors.gm_detector import GMDetectorModel

# from kats.detectors.sprt_detector import SPRTDetectorModel
from kats.detectors.outlier_detector import OutlierDetectorModel
from kats.detectors.prophet_detector import (
    ProphetDetectorModel,
    ProphetTrendDetectorModel,
)
from kats.detectors.slow_drift_detector import SlowDriftDetectorModel
from kats.detectors.stat_sig_detector import StatSigDetectorModel
from kats.detectors.threshold_detector import StaticThresholdModel
from kats.detectors.trend_mk_model import MKDetectorModel
from kats.detectors.utils import DetectorModelSearchSpace
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.serialize import global_model_to_json
from kats.models.globalmodel.utils import GMParam
from kats.utils.simulator import Simulator

# search space

# Test 12 models: CUSUM, BOCPD, OUTLIER, MK, STATSIG, PROPHET_TREND, PROPHET, SLOWDRIFT, GM, StaticThresholdModel
# SQRT, SLOVIOLATION, MultiStatSigDetectorModel

COMMON_SPACE: List[Dict[str, Any]] = [
    {
        "name": "detection_window_sec",
        "type": "choice",
        "values": [3 * 24 * 3600, 3 * 24 * 3600],
        "value_type": "int",
        "is_ordered": False,
    },
]

CUSUM_SPACE: List[Dict[str, Any]] = [
    {
        "name": "scan_window",
        "type": "choice",
        "values": [i * 24 * 3600 for i in [2, 3, 5, 10]],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "historical_window",
        "type": "choice",
        "values": [i * 24 * 3600 for i in [5, 10, 15, 20]],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [0.0, 0.5],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

BOCPD_SPACE: List[Dict[str, Any]] = [
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [0.0, 0.5],  # 0.9],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.1, 0.5],  # 1.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

STATIC_SPACE: List[Dict[str, Any]] = [
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [0.0, 0.5],  # 0.9],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.1, 0.5],  # 1.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

OUTLIER_SPACE: List[Dict[str, Any]] = [
    {
        "name": "iqr_mult",
        "type": "choice",
        "values": [2.0, 3.0, 4.0],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [0.0, 0.5, 0.9],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [1, 1.5, 2.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

MK_SPACE: List[Dict[str, Any]] = [
    {
        "name": "window_size",
        "type": "choice",
        "values": [5, 10, 15, 20],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-1.0, -0.75, -0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.5, 0.75, 0.9, 1.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

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
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-3.0, -2],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [2, 3, 5.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

PROPHET_TREND_SPACE: List[Dict[str, Any]] = [
    {
        "name": "changepoint_prior_scale",
        "type": "choice",
        "values": [0.01, 0.05, 0.1],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-1.0, -0.75, -0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.5, 0.75, 0.9, 1.0],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

PROPHET_SPACE: List[Dict[str, Any]] = [
    {
        "name": "scoring_confidence_interval",
        "type": "choice",
        "values": [0.8, 0.9],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "outlier_threshold",
        "type": "choice",
        "values": [0.9, 0.99],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

SLOWDRIFT_SPACE: List[Dict[str, Any]] = [
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.1, 0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-0.5, -0.1],
        "value_type": "float",
        "is_ordered": True,
    },
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
] + COMMON_SPACE


# serialized model for GM
gmm = GMModel(
    GMParam(
        freq="D",
        input_window=10,
        fcst_window=10,
        seasonality=7,
        nn_structure=[[1]],
        state_size=10,
        h_size=5,
        quantile=[0.5, 0.01, 0.05, 0.95, 0.99],
    )
)
gmm._initiate_nn()
gm_str: str = global_model_to_json(gmm)

gmm2 = GMModel(
    GMParam(
        freq="D",
        input_window=10,
        fcst_window=10,
        seasonality=7,
        nn_structure=[[1]],
        state_size=10,
        h_size=3,
        quantile=[0.5, 0.01, 0.05, 0.95, 0.99],
    )
)
gmm2._initiate_nn()
gm_str2: str = global_model_to_json(gmm2)

GM_SPACE: List[Dict[str, Any]] = [
    {
        "name": "serialized_model",
        "type": "choice",
        "values": [gm_str, gm_str2],
        "value_type": "str",
        "is_ordered": True,
    },
    {
        "name": "scoring_confidence_interval",
        "type": "choice",
        "values": [0.9, 0.99],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "outlier_confidence_interval",
        "type": "choice",
        "values": [0.9, 0.99],
        "value_type": "float",
        "is_ordered": True,
    },
] + COMMON_SPACE

GRID_SD_DICT: Dict[Type[DetectorModel], List[Dict[str, Any]]] = {
    SlowDriftDetectorModel: SLOWDRIFT_SPACE,
    StatSigDetectorModel: STATSIG_SPACE,
    CUSUMDetectorModel: CUSUM_SPACE,
    BocpdDetectorModel: BOCPD_SPACE,
    StaticThresholdModel: STATIC_SPACE,
    OutlierDetectorModel: OUTLIER_SPACE,
    ProphetTrendDetectorModel: PROPHET_TREND_SPACE,
    MKDetectorModel: MK_SPACE,
    GMDetectorModel: GM_SPACE,
    ProphetDetectorModel: PROPHET_SPACE,
}


def _parse_ts(ts_df: TimeSeriesData) -> Dict[List[str], np.ndarray]:
    k = list(map(str, ts_df.time.values))
    v = ts_df.value.values
    return dict(zip(k, v))


def _gen_trend_shift(
    cp_arr: Optional[List[int]] = None,
    trend_arr: Optional[List[float]] = None,
    intercept: int = 30,
    noise: int = 50,
    seasonal_period: int = 7,
    seasonal_magnitude: int = 100,
    ts_len: int = 500,
    random_seed: int = 50,
) -> TimeSeriesData:
    if cp_arr is None:
        cp_arr = [125, 250, 375]
    if trend_arr is None:
        trend_arr = [2, 2.5, -2, 8]

    np.random.seed(random_seed)
    sim = Simulator(n=ts_len, start="1980-01-01")
    ts = sim.trend_shift_sim(
        cp_arr=cp_arr,
        trend_arr=trend_arr,
        intercept=intercept,
        noise=noise,
        seasonal_period=seasonal_period,
        seasonal_magnitude=seasonal_magnitude,
    )
    return ts


def _gen_synthetic_data(
    random_seed: int = 50,
) -> Tuple[TimeSeriesData, List[int], List[float]]:
    # hardcoded values
    ts_len = 200
    cp_interval = 100  # 125
    init_trend = 2.0
    slope_std = 10.0  # 8.0

    np.random.seed(random_seed)

    geom_prop = 1.0 / cp_interval
    cp_loc = 0
    cp_arr = []
    while cp_loc < ts_len:
        cp_loc += np.random.geometric(p=geom_prop, size=1)[0]
        if cp_loc < ts_len:
            cp_arr.append(cp_loc)

    trend_arr = []
    this_trend = init_trend

    for _ in range(len(cp_arr) + 1):
        this_trend = np.random.normal(loc=this_trend, scale=slope_std, size=1)[0]
        trend_arr.append(this_trend)

    ts = _gen_trend_shift(
        cp_arr=cp_arr,
        trend_arr=trend_arr,
        ts_len=ts_len,
        random_seed=random_seed,
    )

    return ts, cp_arr, trend_arr


def _generate_all_data() -> pd.DataFrame:
    N = 2
    cp_list = []
    trend_list = []
    ts_list = []

    for i in range(N):
        ts, true_cp, true_trend = _gen_synthetic_data(random_seed=100 * i)
        ts_list.append(ts)
        cp_list.append(true_cp)
        trend_list.append(true_trend)

    simulated_cp_df = pd.DataFrame(
        [
            {
                "dataset_name": f"trend_change_{i}",
                "time_series": str(_parse_ts(ts_list[i])),
                "annotation": str({"1": cp_list[i]}),
            }
            for i in range(N)
        ]
    )

    return simulated_cp_df


def generate_data_daily() -> pd.DataFrame:
    np.random.seed(0)
    val1 = np.random.normal(1.5, 3, 100)
    time1 = pd.date_range(start="2018-01-01 00:00:00", freq="d", periods=100)
    ts1 = TimeSeriesData(pd.DataFrame({"time": time1, "value": pd.Series(val1)}))
    anno1 = {
        "1": [
            IntervalAnomaly(
                *[
                    pd.to_datetime("2018-01-05 00:00:00"),
                    pd.to_datetime("2018-01-06 00:00:00"),
                ]
            ),
            IntervalAnomaly(
                *[
                    pd.to_datetime("2018-01-17 00:00:00"),
                    pd.to_datetime("2018-01-18 00:00:00"),
                ]
            ),
        ],
    }

    np.random.seed(0)
    val2 = np.random.normal(0.5, 3, 100)
    time2 = pd.date_range(start="2018-01-01 00:00:00", freq="d", periods=100)
    ts2 = TimeSeriesData(pd.DataFrame({"time": time2, "value": pd.Series(val2)}))
    anno2 = {
        "1": [
            IntervalAnomaly(
                *[
                    pd.to_datetime("2018-01-05 00:00:00"),
                    pd.to_datetime("2018-01-06 00:00:00"),
                ]
            ),
        ],
        "2": [],
    }

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


SIMULATED_TS_DF: pd.DataFrame = _generate_all_data()
SIMULATED_TS_DF2: pd.DataFrame = generate_data_daily()

# SIMULATED_TS_DF3 for unsupervised tests without annotation column.
SIMULATED_TS_DF3: pd.DataFrame = generate_data_daily()[["dataset_name", "time_series"]]


DETECTOR_ONLINE_ONLY: Dict[Union[Type[Detector], Type[DetectorModel]], bool] = {
    OutlierDetectorModel: True,
    SlowDriftDetectorModel: True,
    StatSigDetectorModel: True,
    CUSUMDetectorModel: True,
    BocpdDetectorModel: True,
    StaticThresholdModel: True,
    MKDetectorModel: True,
    GMDetectorModel: False,
    ProphetTrendDetectorModel: False,
    ProphetDetectorModel: False,
}

TEST_DETECTORS: Dict[Type[DetectorModel], str] = {
    OutlierDetectorModel: "outlier",
    SlowDriftDetectorModel: "slowdrift",
    StatSigDetectorModel: "statsig",
    CUSUMDetectorModel: "cusum",
    BocpdDetectorModel: "bocpd",
    StaticThresholdModel: "static",
    MKDetectorModel: "mk",
    GMDetectorModel: "gm",
    ProphetTrendDetectorModel: "prophet_trend",
    ProphetDetectorModel: "prophet",
}


class ModelOptimizerTest(TestCase):
    # Test 10 models: CUSUM, BOCPD, OUTLIER, MK, STATSIG, PROPHET_TREND, PROPHET, SLOWDRIFT, GM, Static
    def test_detectors(self) -> None:
        # test Turing evaluator
        for detect_model in TEST_DETECTORS:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric_list=["f_score"],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="min",  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()

            assert mopt.parameter_tuning_results_grid is not None
            self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)
            # name + 7 super + 1 obj + param
            self.assertEqual(mopt.prelim_res.shape, (4, 11))

            # supervised metrics
            mopt2 = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric_list=["f_score"],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF2,
                arm_count=2,
                margin=2 * 24 * 3600,
                eval_start_time_sec=25 * 24 * 3600,
                bulk_eval_method="min",  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=20 * 24 * 3600,
                perc_last_itv=0.1,
                abs_gap_sec=60,
                interval_based=True,
            )
            mopt2._evaluate()

            assert mopt2.parameter_tuning_results_grid is not None
            self.assertEqual(len(mopt2.parameter_tuning_results_grid), 2)
            # name*2 + 8 super + 5 unsup + 1 obj + param + 6 continuous metrics
            self.assertEqual(mopt2.prelim_res.shape, (4, 24))

        # unsupervised metrics
        mopt3 = ModelOptimizer(
            model_name="cusum",
            model=CUSUMDetectorModel,
            parameters_space=GRID_SD_DICT[CUSUMDetectorModel],
            optimization_metric_list=["health_loss"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=SIMULATED_TS_DF3,
            arm_count=2,
            margin=2 * 24 * 3600,
            eval_start_time_sec=20 * 24 * 3600,
            bulk_eval_method="min",  # mean, min, median, max
            training_window_sec=20 * 24 * 3600,
            retrain_freq_sec=100 * 24 * 3600,
            perc_last_itv=0.1,
            abs_gap_sec=60,
            interval_based=True,
        )
        mopt3._evaluate()

        assert mopt3.parameter_tuning_results_grid is not None
        self.assertEqual(len(mopt3.parameter_tuning_results_grid), 2)

        # test kwargs
        mopt4 = ModelOptimizer(
            model_name="cusum",
            model=CUSUMDetectorModel,
            parameters_space=GRID_SD_DICT[CUSUMDetectorModel],
            optimization_metric_list=["health_loss"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=SIMULATED_TS_DF3,
            arm_count=2,
            margin=2 * 24 * 3600,
            eval_start_time_sec=20 * 24 * 3600,
            bulk_eval_method="min",  # mean, min, median, max
            training_window_sec=20 * 24 * 3600,
            retrain_freq_sec=100 * 24 * 3600,
            perc_last_itv=0.1,
            abs_gap_sec=60,
            interval_based=True,
            shorty_len_sec=10 * 60,
            flappy_len_sec=10 * 60,
            threshold_spammy=10,
            threshold_shorty=4,
            threshold_flappy=4,
            threshold_alert_duration=0.2,
            time_normalization_interval_sec=86400,
            alpha=0.4,
        )
        mopt4._evaluate()

        assert mopt4.parameter_tuning_results_grid is not None
        self.assertEqual(len(mopt4.parameter_tuning_results_grid), 2)

    def test_detectors_otherparams(self) -> None:
        for detect_model, x, y in [
            (OutlierDetectorModel, "recall", "min"),
            (SlowDriftDetectorModel, "delay", "median"),
            (StatSigDetectorModel, "precision", "max"),
        ]:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric_list=[x],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method=y,  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()

            assert mopt.parameter_tuning_results_grid is not None
            self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)

    def test_preliminary_results(self) -> None:
        """
        Testing process consists of the following for both supervised and
        unsupervised metrics:
          1. Set varing optimization metrics and weights (test cases)
          2. Evaluate ModelOptimizer
          3. Compute the preliminary result of the ModelOptmizer
          4. Ensure preliminary result matches the expected value given the metrics
        """

        # Test the preliminary results of supervised metrics
        supervised_test_cases = [
            (
                ["recall"],
                [1.0],
            ),
            (
                ["running_time", "f_score"],
                [0.2, 0.8],
            ),
            (
                ["f_score", "precision", "running_time"],
                [0.7, 0.4, 0.1],
            ),
        ]

        for metrics, weigths in supervised_test_cases:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[OutlierDetectorModel],
                model=OutlierDetectorModel,
                parameters_space=GRID_SD_DICT[OutlierDetectorModel],
                optimization_metric_list=metrics,
                optimization_metric_weight=weigths,
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=4,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="mean",
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()

            assert mopt.prelim_res is not None

            x = np.array([mopt.prelim_res[m] for m in metrics]).T
            w = np.array([weigths])

            expected = np.sum(x * w, axis=1)
            result = np.array(mopt.prelim_res["cum_res"])

            np.testing.assert_array_equal(result, expected)

        # Test the preliminary results of unsupervised metrics
        unsupervised_test_cases = [
            (
                ["health_loss"],
                [1.0],
            ),
            (
                ["health_loss", "running_time"],
                [1.0, 0.001],
            ),
            (
                ["spammy_loss", "health_loss", "running_time"],
                [0.2, 0.8, 1.0],
            ),
        ]

        for metrics, weigths in unsupervised_test_cases:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[CUSUMDetectorModel],
                model=CUSUMDetectorModel,
                parameters_space=GRID_SD_DICT[CUSUMDetectorModel],
                optimization_metric_list=metrics,
                optimization_metric_weight=weigths,
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF3,
                arm_count=4,
                margin=2 * 24 * 3600,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="mean",
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
                perc_last_itv=0.1,
                abs_gap_sec=60,
                interval_based=True,
            )
            mopt._evaluate()

            assert mopt.prelim_res is not None

            x = np.array([mopt.prelim_res[m] for m in metrics]).T
            w = np.array([weigths])

            expected = np.sum(x * w, axis=1)
            result = np.array(mopt.prelim_res["cum_res"])

            np.testing.assert_array_equal(result, expected)

    def test_multi_obj(self) -> None:
        mopt = ModelOptimizer(
            model_name=TEST_DETECTORS[StatSigDetectorModel],
            model=StatSigDetectorModel,
            parameters_space=GRID_SD_DICT[StatSigDetectorModel],
            optimization_metric_list=["recall", "delay"],
            optimization_metric_weight=[1, 0.8],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=SIMULATED_TS_DF,
            arm_count=2,
            eval_start_time_sec=20 * 24 * 3600,
            bulk_eval_method="max",
            training_window_sec=20 * 24 * 3600,
            retrain_freq_sec=100 * 24 * 3600,
        )
        mopt._evaluate()

        assert mopt.parameter_tuning_results_grid is not None
        self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)

    def test_errors(self) -> None:
        for detect_model, x, y, z in [
            (ProphetTrendDetectorModel, -1, 20, 100),
            (GMDetectorModel, 20, 0, 100),
            (ProphetTrendDetectorModel, 20, 20, 0),
        ]:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric_list=["recall"],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=x * 24 * 3600,
                bulk_eval_method="min",  # mean, min, median, max
                training_window_sec=y * 24 * 3600,
                retrain_freq_sec=z * 24 * 3600,
            )
            with self.assertRaises(ValueError):
                mopt._evaluate()

        with self.assertRaises(ValueError):
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[StatSigDetectorModel],
                model=StatSigDetectorModel,
                parameters_space=GRID_SD_DICT[StatSigDetectorModel],
                optimization_metric_list=["other"],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="min",  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()

        with self.assertRaises(ValueError):
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[StatSigDetectorModel],
                model=StatSigDetectorModel,
                parameters_space=GRID_SD_DICT[StatSigDetectorModel],
                optimization_metric_list=["recall"],
                optimization_metric_weight=[1],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="other",  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()

        with self.assertRaises(ValueError):
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[StatSigDetectorModel],
                model=StatSigDetectorModel,
                parameters_space=GRID_SD_DICT[StatSigDetectorModel],
                optimization_metric_list=["recall"],
                optimization_metric_weight=[1, 2, 3],
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20 * 24 * 3600,
                bulk_eval_method="mean",  # mean, min, median, max
                training_window_sec=20 * 24 * 3600,
                retrain_freq_sec=100 * 24 * 3600,
            )
            mopt._evaluate()


class ModelOptimizerUtilsTest(TestCase):
    def setUp(self) -> None:
        val1 = np.random.normal(1.5, 3, 100)
        time1 = pd.date_range(start="2018-01-01 00:00:00", freq="d", periods=100)
        self.ts1 = TimeSeriesData(
            pd.DataFrame({"time": time1, "value": pd.Series(val1)})
        )
        # using two detectors as an example
        self.det_dict = {"bocpd": BocpdDetectorModel, "statsig": StatSigDetectorModel}

        self.params = {}
        self.params_thres = {}
        for det in self.det_dict:
            dmss = DetectorModelSearchSpace(self.det_dict[det], self.ts1)
            self.params[det] = dmss.get_params_search_space()
            self.params_thres[det] = dmss.get_params_thres_search_space()

    def test_recommend_params_search_space(self) -> None:
        my_params = {
            "bocpd": [
                {
                    "name": "slow_drift",
                    "type": "choice",
                    "values": [True, False],
                    "value_type": "bool",
                    "is_ordered": False,
                }
            ],
            "statsig": [
                {
                    "name": "n_control",
                    "type": "choice",
                    "values": [5, 10, 20, 30],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "n_test",
                    "type": "choice",
                    "values": [5, 10, 20, 30],
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
        }
        for det in self.det_dict:
            for i in range(len(my_params[det])):
                self.assertDictEqual(self.params[det][i], my_params[det][i])

    def test_recommend_params_thres_search_space(self) -> None:
        my_params = [
            {
                "name": "slow_drift",
                "type": "choice",
                "values": [True, False],
                "value_type": "bool",
                "is_ordered": False,
            },
            {
                "name": "threshold_low",
                "type": "choice",
                "values": [-2.0, -1.0, -0.5, -0.1],
                "value_type": "float",
                "is_ordered": True,
            },
            {
                "name": "threshold_high",
                "type": "choice",
                "values": [0.1, 0.5, 1.0, 2.0],
                "value_type": "float",
                "is_ordered": True,
            },
            {
                "name": "direction_1d",
                "type": "choice",
                "values": [1, 2, 3],
                "value_type": "int",
                "is_ordered": False,
            },
            {
                "name": "detection_window_sec",
                "type": "choice",
                "values": [172800, 259200, 432000, 864000],
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "fraction",
                "type": "choice",
                "values": [0.8, 0.9, 0.95, 1],
                "value_type": "float",
                "is_ordered": True,
            },
        ]
        for i in range(len(my_params)):
            self.assertDictEqual(self.params_thres["bocpd"][i], my_params[i])

    def test_recommend_params_for_models(self) -> None:
        for det in self.det_dict:
            model = self.det_dict[det]
            model_params_space = DetectorModelSearchSpace(
                detector=model,
                ts_data=self.ts1,
            ).get_params_search_space()

            model_params_first = {}
            for item in model_params_space:
                model_params_first[item["name"]] = item["values"][0]
            model(**model_params_first)

            model_params_last = {}
            for item in model_params_space:
                model_params_last[item["name"]] = item["values"][-1]
            ss_detector = model(**model_params_last)
            ss_detector.fit_predict(data=self.ts1)

    def test_recommend_params_for_modeloptimizer(self) -> None:
        model_params_space = DetectorModelSearchSpace(
            detector=StaticThresholdModel,
            ts_data=self.ts1,
        ).get_params_thres_search_space()

        df = pd.DataFrame(
            [
                {
                    "dataset_name": "x",
                    "time_series": self.ts1,
                    "annotation": [],
                }
            ]
        )
        mopt = ModelOptimizer(
            model_name="static",
            model=StaticThresholdModel,
            parameters_space=model_params_space,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=df,
            arm_count=2,
            eval_start_time_sec=20 * 24 * 3600,
            bulk_eval_method="min",
            training_window_sec=0,
            retrain_freq_sec=0,
            interval_based=True,
        )
        mopt._evaluate()

        assert mopt.parameter_tuning_results_grid is not None
        self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)

        model_params_space2 = DetectorModelSearchSpace(
            detector=CUSUMDetectorModel,
            ts_data=self.ts1,
        ).get_params_thres_search_space()

        mopt2 = ModelOptimizer(
            model_name="cusum",
            model=CUSUMDetectorModel,
            parameters_space=model_params_space2,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=df,
            arm_count=2,
            eval_start_time_sec=20 * 24 * 3600,
            bulk_eval_method="min",
            training_window_sec=0,
            retrain_freq_sec=0,
            interval_based=True,
        )
        mopt2._evaluate()

        assert mopt2.parameter_tuning_results_grid is not None
        self.assertEqual(len(mopt2.parameter_tuning_results_grid), 2)


class ModelBenchmarkTest(TestCase):
    def setUp(self) -> None:
        self.eg_df: pd.DataFrame = generate_data_daily()
        self.m_list: Dict[str, Tuple[Type[DetectorModel], List[Dict[str, Any]]]] = {
            "cusum": (CUSUMDetectorModel, CUSUM_SPACE),
            "slowdrift": (SlowDriftDetectorModel, SLOWDRIFT_SPACE),
        }
        self.mb: ModelBenchmark = ModelBenchmark(
            detect_models=self.m_list,
            data_df=self.eg_df,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            arm_count=2,
            margin=0,
            eval_start_time_sec=12,
            bulk_eval_method="mean",
            training_window_sec=0,
            retrain_freq_sec=0,
        )
        self.mb.evaluate()

    def test_modelbenchmark_recommend(self) -> None:
        self.assertEqual(self.mb.prelim_results.shape, (8, 25))
        self.assertEqual(self.mb.if_evaluated, True)
        self.assertEqual(self.mb.find_best_algos_params().shape, (1, 7))
        self.assertEqual(
            self.mb.compare_algos_on_metric(
                metric="health_loss",
                opt_for_min=True,
            ).shape,
            (1, 23),
        )

        self.assertEqual(self.mb.find_best_params_for_each_model().shape, (2, 23))
        self.assertEqual(
            self.mb.find_best_params_for_each_model_on_metric(
                "health_loss", True
            ).shape,
            (2, 23),
        )


class ModelOptimizeMultiprocessTest(TestCase):
    def setUp(self) -> None:
        self.eg_df: pd.DataFrame = generate_data_daily()
        self.mopt = ModelOptimizer(
            model_name="slowdrift",
            model=SlowDriftDetectorModel,
            parameters_space=SLOWDRIFT_SPACE,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=self.eg_df,
            arm_count=2,
            interval_based=True,
            multiprocessing=True,
        )
        self.mopt._evaluate()

        self.mopt2 = ModelOptimizer(
            model_name="cusum",
            model=CUSUMDetectorModel,
            parameters_space=CUSUM_SPACE,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=self.eg_df,
            arm_count=2,
            interval_based=True,
            multiprocessing=True,
        )
        self.mopt2._evaluate()

    def test_multiprocessing(self) -> None:
        # self.assertEqual(self.mopt.prelim_res.shape, (4, 23))
        assert self.mopt.parameter_tuning_results_grid is not None
        self.assertEqual(self.mopt.parameter_tuning_results_grid.shape, (2, 6))

        # self.assertEqual(self.mopt2.prelim_res.shape, (4, 23))
        assert self.mopt2.parameter_tuning_results_grid is not None
        self.assertEqual(self.mopt2.parameter_tuning_results_grid.shape, (2, 6))


class ModelBenchmarkNoSearchSpaceTest(TestCase):
    def setUp(self) -> None:
        self.eg_df: pd.DataFrame = generate_data_daily()
        self.m_list: Dict[str, Tuple[Type[DetectorModel], List[Dict[str, Any]]]] = {
            "cusum": (CUSUMDetectorModel, []),
            "slowdrift": (SlowDriftDetectorModel, []),
        }
        self.mb: ModelBenchmark = ModelBenchmark(
            detect_models=self.m_list,
            data_df=self.eg_df,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            arm_count=2,
        )
        self.mb.evaluate()

    def test_modelbenchmark_no_search_space_recommend(self) -> None:
        self.assertEqual(self.mb.prelim_results.shape, (8, 25))
        self.assertEqual(self.mb.find_best_algos_params().shape, (1, 7))
        self.assertEqual(
            self.mb.compare_algos_on_metric(
                metric="health_loss",
                opt_for_min=True,
            ).shape,
            (1, 23),
        )
        self.assertEqual(len(self.mb.detect_models["cusum"][-1]), 9)
        self.assertEqual(len(self.mb.detect_models["slowdrift"][-1]), 9)

        # infer margin based on data granularity -> 2 days
        self.assertEqual(self.mb.margin, 172800)


class MOForOutlierModel(TestCase):
    def setUp(self) -> None:
        self.eg_df: pd.DataFrame = generate_data_daily()

        val1 = np.random.normal(0, 3, 100)
        time1 = pd.date_range(start="2018-01-01 00:00:00", freq="d", periods=100)
        ts1 = TimeSeriesData(pd.DataFrame({"time": time1, "value": pd.Series(val1)}))

        val2 = np.random.normal(1500, 3, 100)
        time2 = pd.date_range(start="2018-01-01 00:00:00", freq="d", periods=100)
        ts2 = TimeSeriesData(pd.DataFrame({"time": time2, "value": pd.Series(val2)}))

        dmss = DetectorModelSearchSpace(OutlierDetectorModel, ts1)
        self.search_space1 = dmss.get_params_thres_search_space()

        dmss = DetectorModelSearchSpace(OutlierDetectorModel, ts2)
        self.search_space2 = dmss.get_params_thres_search_space()

        self.mopt = ModelOptimizer(
            model_name="outlier",
            model=OutlierDetectorModel,
            parameters_space=self.search_space1,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            data_df=self.eg_df,
            arm_count=2,
            margin=2 * 24 * 3600,
            eval_start_time_sec=25 * 24 * 3600,
            bulk_eval_method="min",  # mean, min, median, max
            training_window_sec=20 * 24 * 3600,
            retrain_freq_sec=20 * 24 * 3600,
            perc_last_itv=0.1,
            abs_gap_sec=60,
            interval_based=True,
        )

        self.m_list: Dict[str, Tuple[Type[DetectorModel], List[Dict[str, Any]]]] = {
            "outlier": (OutlierDetectorModel, self.search_space1),
        }
        self.mb: ModelBenchmark = ModelBenchmark(
            detect_models=self.m_list,
            data_df=self.eg_df,
            optimization_metric_list=["f_score"],
            optimization_metric_weight=[1],
            optimize_for_min=False,
            arm_count=2,
        )
        self.mb.evaluate()

    def test_modelbenchmark_for_outliermodel(self) -> None:
        self.assertEqual(self.mb.prelim_results.shape, (4, 25))
        self.assertEqual(self.mb.find_best_algos_params().shape, (1, 7))
        self.assertEqual(
            self.mb.compare_algos_on_metric(
                metric="health_loss",
                opt_for_min=True,
            ).shape,
            (1, 23),
        )
        # infer margin based on data granularity -> 2 days
        self.assertEqual(self.mb.margin, 172800)

    def test_modeloptimizer_for_outliermodel(self) -> None:
        self.mopt._evaluate()

        assert self.mopt.parameter_tuning_results_grid is not None
        self.assertEqual(len(self.mopt.parameter_tuning_results_grid), 2)
        self.assertEqual(self.mopt.prelim_res.shape, (4, 24))

    def test_recommend_search_space(self) -> None:
        self.assertEqual(len(self.search_space1), 6)
        self.assertEqual(len(self.search_space2), 7)
