# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Tuple
)
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import SearchMethodEnum, TimeSeriesData

# from kats.detectors.stat_sig_detector import MultiStatSigDetectorModel
from kats.detectors.cusum_model import CUSUMDetectorModel

# from kats.detectors.window_slo_detector import WindowSloDetectorModel
from kats.detectors.bocpd_model import BocpdDetectorModel
from kats.detectors.detector import Detector, DetectorModel
from kats.detectors.detector_benchmark import ModelOptimizer
from kats.detectors.gm_detector import GMDetectorModel

# from kats.detectors.sprt_detector import SPRTDetectorModel
from kats.detectors.outlier_detector import OutlierDetectorModel
from kats.detectors.prophet_detector import (
    ProphetDetectorModel,
    ProphetTrendDetectorModel,
)
from kats.detectors.slow_drift_detector import SlowDriftDetectorModel
from kats.detectors.stat_sig_detector import StatSigDetectorModel
from kats.detectors.trend_mk_model import MKDetectorModel
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.serialize import global_model_to_json
from kats.models.globalmodel.utils import GMParam
from kats.utils.simulator import Simulator


#search space

# Test 11 models: CUSUM, BOCPD, OUTLIER, MK, STATSIG, PROPHET_TREND, PROPHET, SLOWDRIFT, GM
# SQRT, SLOVIOLATION, MultiStatSigDetectorModel

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
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.6, 1.0],
        "value_type": "float",
        "is_ordered": True,
    },
]

BOCPD_SPACE: List[Dict[str, Any]] = [{
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
]

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
]

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
        "values": [-1., -0.75, -0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.5, 0.75, 0.9, 1.],
        "value_type": "float",
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
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-3., -2],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [2, 3 , 5.],
        "value_type": "float",
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
    {
        "name": "threshold_low",
        "type": "choice",
        "values": [-1., -0.75, -0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_high",
        "type": "choice",
        "values": [0.5, 0.75, 0.9, 1.],
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
    {
        "name": "outlier_threshold",
        "type": "choice",
        "values": [0.9, 0.99],
        "value_type": "float",
        "is_ordered": True,
    },
]

SLOWDRIFT_SPACE: List[Dict[str, Any]] = [
    {
        "name": "threshold_high",
        "type": "range",
        "bounds": [0.1, 0.5],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "threshold_low",
        "type": "range",
        "bounds": [-0.5, -0.1],
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
]


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
        "values": [0.9,0.99],
        "value_type": "float",
        "is_ordered": True,
    },
    {
        "name": "outlier_confidence_interval",
        "type": "choice",
        "values": [0.9,0.99],
        "value_type": "float",
        "is_ordered": True,
    },
]

GRID_SD_DICT: Dict[Type[DetectorModel], List[Dict[str, Any]]] = {
    SlowDriftDetectorModel: SLOWDRIFT_SPACE,
    StatSigDetectorModel: STATSIG_SPACE,
    CUSUMDetectorModel: CUSUM_SPACE,
    BocpdDetectorModel: BOCPD_SPACE,
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


def _gen_synthetic_data(random_seed: int = 50) -> Tuple[TimeSeriesData, List[int], List[float]]:
    # hardcoded values
    ts_len = 200
    cp_interval = 100 #125
    init_trend = 2.0
    slope_std = 10.0 #8.0

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

SIMULATED_TS_DF: pd.DataFrame = _generate_all_data()

DETECTOR_ONLINE_ONLY: Dict[Union[Type[Detector], Type[DetectorModel]], bool] = {
    OutlierDetectorModel: True,
    SlowDriftDetectorModel: True,
    StatSigDetectorModel: True,
    CUSUMDetectorModel: True,
    BocpdDetectorModel: True,
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
    MKDetectorModel: "mk",
    GMDetectorModel: "gm",
    ProphetTrendDetectorModel: "prophet_trend",
    ProphetDetectorModel: "prophet",
}

class ModelOptimizerTest(TestCase):
    # Test 9 models: CUSUM, BOCPD, OUTLIER, MK, STATSIG, PROPHET_TREND, PROPHET, SLOWDRIFT, GM
    def test_detectors(self) -> None:
        for detect_model in TEST_DETECTORS:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric="f_score",
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20*24*3600,
                bulk_eval_method="min", # mean, min, median, max
                training_window_sec=20*24*3600,
                retrain_freq_sec=100*24*3600,
            )
            mopt._evaluate()

            assert mopt.parameter_tuning_results_grid is not None
            self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)

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
                optimization_metric=x,
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20*24*3600,
                bulk_eval_method=y, # mean, min, median, max
                training_window_sec=20*24*3600,
                retrain_freq_sec=100*24*3600,
            )
            mopt._evaluate()

            assert mopt.parameter_tuning_results_grid is not None
            self.assertEqual(len(mopt.parameter_tuning_results_grid), 2)


    def test_errors(self) -> None:
        for detect_model, x, y, z in [
            (ProphetTrendDetectorModel, 0, 20, 100),
            (GMDetectorModel, 20, 0, 100),
            (ProphetTrendDetectorModel, 20, 20, 0),
        ]:
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[detect_model],
                model=detect_model,
                parameters_space=GRID_SD_DICT[detect_model],
                optimization_metric="recall",
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=x*24*3600,
                bulk_eval_method="min", # mean, min, median, max
                training_window_sec=y*24*3600,
                retrain_freq_sec=z*24*3600,
            )
            with self.assertRaises(ValueError):
                mopt._evaluate()

        with self.assertRaises(ValueError):
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[StatSigDetectorModel],
                model=StatSigDetectorModel,
                parameters_space=GRID_SD_DICT[StatSigDetectorModel],
                optimization_metric="other",
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20*24*3600,
                bulk_eval_method="min", # mean, min, median, max
                training_window_sec=20*24*3600,
                retrain_freq_sec=100*24*3600,
            )
            mopt._evaluate()

        with self.assertRaises(ValueError):
            mopt = ModelOptimizer(
                model_name=TEST_DETECTORS[StatSigDetectorModel],
                model=StatSigDetectorModel,
                parameters_space=GRID_SD_DICT[StatSigDetectorModel],
                optimization_metric="recall",
                optimize_for_min=False,
                search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                data_df=SIMULATED_TS_DF,
                arm_count=2,
                eval_start_time_sec=20*24*3600,
                bulk_eval_method="other", # mean, min, median, max
                training_window_sec=20*24*3600,
                retrain_freq_sec=100*24*3600,
            )
            mopt._evaluate()
