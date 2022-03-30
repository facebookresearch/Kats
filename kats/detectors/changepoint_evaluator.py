# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json
import re
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Type,
    Union,
    NamedTuple,
)

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector, DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from kats.utils.simulator import Simulator

from kats.detectors.prophet_detector import ProphetDetectorModel, ProphetTrendDetectorModel
from kats.detectors.stat_sig_detector import StatSigDetectorModel, MultiStatSigDetectorModel
from kats.detectors.cusum_model import CUSUMDetectorModel
from kats.detectors.window_slo_detector import WindowSloDetectorModel
from kats.detectors.bocpd_model import BocpdDetectorModel
from kats.detectors.sprt_detector import SPRTDetectorModel
from kats.detectors.trend_mk_model import MKDetectorModel
from kats.detectors.gm_detector import GMDetectorModel


# historical window related params' names for each dectectors
HISTORICAL_DATA_PARAM_NAME: Dict[Union[Type[Detector], Type[DetectorModel]], List[str]] = {
    StatSigDetectorModel: ['n_control', 'n_test'],
    MultiStatSigDetectorModel: ['n_control', 'n_test'],
    CUSUMDetectorModel: ['historical_window'],
    WindowSloDetectorModel: ['window_size'],
    # BocpdDetectorModel: [''],
    # SPRTDetectorModel: [''],
    MKDetectorModel: ['window_size'],
}

# historical window related params' default value for each dectectors
HISTORICAL_DATA_PARAM_DEFAULT: Dict[Union[Type[Detector], Type[DetectorModel]], Dict[str, int]] = {
    StatSigDetectorModel: {'n_control': 0, 'n_test': 1},
    MultiStatSigDetectorModel: {'n_control': 0, 'n_test': 1},
    CUSUMDetectorModel: {'historical_window': 0},
    WindowSloDetectorModel: {'window_size': 10},
    # BocpdDetectorModel: {'': 0},
    # SPRTDetectorModel: {'': 0},
    MKDetectorModel: {'window_size': 20},
}

# whether online only
DETECTOR_ONLINE_ONLY: Dict[Union[Type[Detector], Type[DetectorModel]], bool] = {
    StatSigDetectorModel: True,
    MultiStatSigDetectorModel: True,
    CUSUMDetectorModel: True,
    WindowSloDetectorModel: True,
    BocpdDetectorModel: True,
    SPRTDetectorModel: True,
    MKDetectorModel: True,
    GMDetectorModel: False,
    ProphetTrendDetectorModel: False,
    ProphetDetectorModel: False,
}

# get evaluation start time for detectors that have rolling window strategy
def get_appropriate_start_time_default(
    detector: Union[Type[Detector], Type[DetectorModel]],
    params: Optional[Dict[str, float]],
    tsd: TimeSeriesData,
) -> int:
    # return number of data points in TSD
    # [Multi]StatSigDetectorModel: n_control and n_test are in number of time_unit (points).
    # CUSUMDetectorModel: historical_window is in seconds.
    # WindowSloDetectorModel: window_size is in number of datapoints.
    # MKDetectorModel: window_size appears to be in number of datapoints.

    assert params is not None
    # for BocpdDetectorModel and SPRTDetectorModel
    if detector not in HISTORICAL_DATA_PARAM_NAME:
        return 0
    window_param_name_list = HISTORICAL_DATA_PARAM_NAME[detector]
    res = 0
    for wpn in window_param_name_list:
        if wpn in params:
            res += params[wpn]
        else:
            res += HISTORICAL_DATA_PARAM_DEFAULT[detector][wpn]

    if detector in [StatSigDetectorModel, MultiStatSigDetectorModel]:
        # for statsig and multistatsig, window = n_test + n_control - 1
        return int(res - 1)

    elif detector in [CUSUMDetectorModel]:
        # CUSUMDetectorModel: historical_window is in seconds.
        return get_points_from_sec(int(res), tsd)
    else:
        return int(res)

# get number of data points from number of seconds in TSD
def get_points_from_sec(
    eval_start_time_sec: int,
    tsd: TimeSeriesData,
) -> int:
    """
    eval_start_time_sec: evaluation start time in terms of second.
    tsd: TS data. Time is sorted.

    return: evaluation start point (index).
    """
    if eval_start_time_sec <= 0:
        raise ValueError(
                "Please use a larger eval_start_time_sec."
            )

    end = pd.to_datetime(tsd.time[0]) + pd.Timedelta(value=eval_start_time_sec, unit='s')
    for i in range(len(tsd)):
        if tsd.time[i] >= end:
            return i
    raise ValueError("Inappropriate eval_start_time_sec.")


# defining some helper functions
def get_cp_index(
    changepoints: Sequence[TimeSeriesChangePoint], tsd: TimeSeriesData
) -> List[int]:
    """
    Accepts the output of the Detector.detector() method which is a list of
    tuples of (TimeSeriesChangePoint, Metadata) and returns the index of the
    changepoints
    """
    cp_list = []
    tsd_df = tsd.to_dataframe()
    tsd_df["time_index"] = list(range(tsd_df.shape[0]))
    for cp in changepoints:
        tsd_row = tsd_df[tsd_df.time == cp.start_time]
        this_cp = tsd_row["time_index"].values[0]
        cp_list.append(this_cp)
    return cp_list


def crossed_threshold(val: float, threshold_low: float, threshold_high: float) -> bool:
    return (val < threshold_low) or (val > threshold_high)


def get_cp_index_from_alert_score(
    score_val: np.ndarray, threshold_low: float, threshold_high: float
) -> List[int]:
    cp_list = []
    alert_on = False
    for i in range(score_val.shape[0]):
        crossed_bool = crossed_threshold(score_val[i], threshold_low, threshold_high)

        # alarm went off
        if crossed_bool and (not alert_on):
            cp_list.append(i)
            alert_on = True

        # return back to normal
        if (not crossed_bool) and alert_on:
            cp_list.append(i)
            alert_on = False

    return cp_list


def get_cp_index_from_threshold_score(
    score_val: float, threshold_low: float, threshold_high: float
) -> List[int]:
    higher = np.where(score_val > threshold_high)[0]
    lower = np.where(score_val < threshold_low)[0]
    cp_list = list(set(higher).union(set(lower)))

    return cp_list


def get_cp_index_from_detector_model(
    anom_obj: AnomalyResponse,
    alert_style_cp: bool,
    threshold_low: float,
    threshold_high: float,
    onlineflag: bool,
    eval_start_time_point: int,
) -> List[int]:
    """
    Get change point index
    onlineflag: whether online only detectors
    eval_start_time_point: evaluation start point
    """
    if onlineflag:
        score_val = anom_obj.scores.value.values[eval_start_time_point:]
    else:
        # for GM and Prophet, anom_obj doesn't include first historical data
        score_val = anom_obj.scores.value.values
    if alert_style_cp:
        cp_list = get_cp_index_from_alert_score(
            score_val, threshold_low, threshold_high
        )
    else:
        cp_list = get_cp_index_from_threshold_score(
            score_val, threshold_low, threshold_high
        )

    return cp_list


# modified from https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/scripts/metrics.py
def true_positives(
    T: Set[int], X: Set[int], margin: int = 5, choose_earliest: bool = True
) -> Dict[int, int]:
    """Compute true positives without double counting.
    If there is multiple detected positives in the margin of a true positive and choose_earliest = True, we keep only the first one.
    If choose_earliest = False, we keep only the one closest to the true_positive.

    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1: 2, 10: -2, 20: 0}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1: 0, 10: -2, 20: 0}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1: 0, 10: -5, 20: 0}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20}, choose_earliest=False)
    {1: 0, 10: -2, 20: 0}
    >>> true_positives(set(), {1, 2, 3})
    dict()
    >>> true_positives({1, 2, 3}, set())
    dict()

    Args:
        T: true positives.
        X: detected positives.
        margin: threshold for absolute difference to be counted as different.

    Returns:
        TP: A dict with true positives as keys and
        the distance between the detected positives and the true positives as values.
    """
    # make a copy so we don't affect the caller
    X = copy.deepcopy(X)
    X = set(X)
    TP = {}
    for tau in T:
        close = [(x - tau, x) for x in X if abs(tau - x) <= margin]
        if choose_earliest:
            close.sort(key=lambda x: x[1])
        else:
            close.sort(key=lambda x: abs(x[0]))
        if not close:
            continue
        dist, xstar = close[0]
        TP[tau] = dist
        X.remove(xstar)
    return TP


# modified from https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/scripts/metrics.py
def measure(
    annotations: Dict[str, List[int]],
    predictions: List[int],
    margin: int = 5,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Compute the F-measure, precision, and recall based on human annotations.

    Remember that all CP locations are 0-based!

    >>> measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [4, 5]}, [10, 20], margin=5, alpha=0.5)
    {'f_score': 0.9565217391304348,
    'precision': 1.0,
    'recall': 0.9166666666666666,
    'delay': 0.5416666666666666,
    'TP': 3,
    'FP': 0,
    'FN': 0.2727272727272729}
    >>> measure({1: [], 2: [10], 3: [50]}, [10], margin=5, alpha=0.5)
    {'f_score': 0.9090909090909091,
    'precision': 1.0,
    'recall': 0.8333333333333333,
    'delay': 0.0,
    'TP': 2,
    'FP': 0,
    'FN': 0.4000000000000002}
    >>> measure({1: [], 2: [10], 3: [50]}, [], margin=5, alpha=0.5)
    {'f_score': 0.8,
    'precision': 1.0,
    'recall': 0.6666666666666666,
    'delay': 0.0,
    'TP': 1,
    'FP': 0,
    'FN': 0.5000000000000001}

    Args:
        annotations : dict from user_id to iterable of CP locations.
        predictions : iterable of predicted CP locations.
        margin : maximum distance between a true positive and an annotation.
        alpha : value for the F-measure, alpha=0.5 gives the F1-measure.
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    # Total number of detection num_detect.
    num_detect = len(X)
    # Compute num of true positives TP.
    TP = len(true_positives(Tstar, X, margin=margin))
    # Compute the precision as TP / (TP + FP) = TP / num_detect.
    P = TP / num_detect
    # Compute num of False positive FP.
    FP = num_detect - TP
    # Compute the number of true positives per annotator.
    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    # Recall (From Turing) computed as the average of each annotator recall over all annotators: R = <Rk>.
    R = 1 / K * sum(len(TPk[k].keys()) / len(Tks[k]) for k in Tks)
    # Compute the num of False Negative FN using the recall: R = TP / (TP + FN) -> FN = (1 - R) * TP / R.
    # Should never be 0 as the recall != 0 (as there is always a detection in 0)
    FN = (1 - R) * TP / R
    # Delay computed as the average of each annotator delay over all annotators: D = <Dk>
    D = 1 / K * sum(np.mean(list(TPk[k].values())) for k in Tks)
    # Compute F_score.
    F = P * R / (alpha * R + (1 - alpha) * P)

    score_dict = {
        "f_score": F,
        "precision": P,
        "recall": R,
        "delay": D,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }

    return score_dict


def generate_from_simulator(
    cp_arr: Optional[List[int]] = None,
    level_arr: Optional[List[float]] = None,
    ts_length: int = 450,
) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    if cp_arr is None:
        cp_arr = [100, 200, 350]
    if level_arr is None:
        level_arr = [1.35, 1.05, 1.35, 1.2]

    sim2 = Simulator(n=ts_length, start="2018-01-01")
    ts2 = sim2.level_shift_sim(
        cp_arr=cp_arr,
        level_arr=level_arr,
        noise=0.05,
        seasonal_period=7,
        seasonal_magnitude=0.075,
    )
    ts2_df = ts2.to_dataframe()
    ts2_dict = {str(row["time"]): row["value"] for _, row in ts2_df.iterrows()}
    # We are generating annotations that match the ground truth.
    ts2_anno = {"1": cp_arr}
    return ts2_dict, ts2_anno


class BenchmarkEvaluator(ABC):
    def __init__(self, detector: Union[Type[Detector], Type[DetectorModel]]) -> None:
        self.detector = detector

    @abstractmethod
    def evaluate(self) -> Union[pd.DataFrame, Dict[str, float]]:
        pass

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass


class Evaluation(NamedTuple):
    dataset_name: str
    precision: float
    recall: float
    f_score: float
    delay: float
    TP: float
    FP: float
    FN: float


class EvalAggregate:
    def __init__(self, eval_list: List[Evaluation]) -> None:
        self.eval_list = eval_list
        self.eval_df: Optional[pd.DataFrame] = None

    def get_eval_dataframe(self) -> pd.DataFrame:
        df_list = []

        for this_eval in self.eval_list:
            df_list.append(
                {
                    "dataset_name": this_eval.dataset_name,
                    "precision": this_eval.precision,
                    "recall": this_eval.recall,
                    "f_score": this_eval.f_score,
                    "delay": this_eval.delay,
                    "TP": this_eval.TP,
                    "FP": this_eval.FP,
                    "FN": this_eval.FN,
                }
            )

        self.eval_df = pd.DataFrame(df_list)
        return self.eval_df

    def _ensure_eval_df(self) -> None:
        if self.eval_df is None:
            _ = self.get_eval_dataframe()

    def get_avg_precision(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.precision)

    def get_avg_recall(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.recall)

    def get_avg_f_score(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.f_score)

    def get_avg_delay(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.delay)

    def get_avg_tp(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.TP)

    def get_avg_fp(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.FP)

    def get_avg_fn(self) -> float:
        self._ensure_eval_df()
        assert self.eval_df is not None, "Evaluation DataFrame is missing."
        return np.mean(self.eval_df.FN)


class TuringEvaluator(BenchmarkEvaluator):
    """
    Evaluates a changepoint detection algorithm. The evaluation
    follows the benchmarking method established in this paper:
    https://arxiv.org/pdf/2003.06222.pdf.
    By default, this evaluates the Turing changepoint benchmark,
    which is introduced in the above paper. This is the most comprehensive
    benchmark for changepoint detection algorithms.

    You can also evaluate your own dataset. The dataset should be a
    dataframe with 3 columns:

        'dataset_name': str,
        'time_series': str "{'0': 0.55, '1': 0.56}",
        'annotation': str "{'0':[1,2], '1':[2,3]}"

    Annotations allow different human beings to annotate a changepoints
    in a time series. Each key consists of one human labeler's label. This
    allows for uncertainty in labeling.
    Usage:

    >>> model_params = {'p_value_cutoff': 5e-3, 'comparison_window': 2}
    >>> turing_2 = TuringEvaluator(detector = RobustStatDetector)
    >>> eval_agg_df_2 = turing.evaluate(data=eg_df, model_params=model_params)
    The above produces a dataframe with scores for each dataset
    To get an average over all datasets you can do
    >>> eval_agg = turing.get_eval_aggregate()
    >>> avg_precision = eval_agg.get_avg_precision()
    """

    ts_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        detector: Union[Type[Detector], Type[DetectorModel]],
        is_detector_model: bool = False,
    ) -> None:
        super(TuringEvaluator, self).__init__(detector=detector)
        self.detector = detector
        self.is_detector_model = is_detector_model
        self.eval_agg: Optional[EvalAggregate] = None
        self.cp_dict: Dict[str, List[int]] = {}

        # Whether the algorithms can only operates in online mode
        self.onlineflag: bool = self._if_online_only()

    def _if_online_only(self) -> bool:
        # check whether the algorithms can only operates in online mode
        if self.detector not in DETECTOR_ONLINE_ONLY:
            print("The given detector model is not in detector model dictionary, using onlineflag=True")
        return DETECTOR_ONLINE_ONLY.get(self.detector, True)

    # get anomaly response for GM and Prophet (onlineonly = False)
    def _get_anomaly_response(
        self,
        tsdata: TimeSeriesData,
        eval_start_time_point: int,
        training_window_point: int,
        retrain_freq_point: int,
        model_params: Optional[Dict[str, float]] = None,
    ) -> AnomalyResponse:

        if eval_start_time_point < 0:
            raise ValueError(
                "Please use a larger evaluation start time (>=0)."
            )
        if training_window_point < 1:
            raise ValueError(
                "Please use a larger training window (>=1)."
            )
        if retrain_freq_point < 1:
            raise ValueError(
                "Please use a larger retraining frequency (>=1)."
            )
        if len(tsdata) < eval_start_time_point or len(tsdata) < training_window_point:
            raise ValueError(
                "Time series is too short for the given model or the given evalution start time or the training window."
            )

        timeindex = tsdata.time.values
        valuedata = tsdata.value.values

        remain_length = len(tsdata) - eval_start_time_point

        if retrain_freq_point >= remain_length:
            raise ValueError(
                "Time series is too short for the given retrain frequency."
            )

        response = None
        for i in range(0, remain_length, retrain_freq_point):
            # historical_data
            historical_data = TimeSeriesData(pd.DataFrame({
                'time': timeindex[
                    eval_start_time_point + i - training_window_point : eval_start_time_point + i
                ],
                'value': valuedata[
                    eval_start_time_point + i - training_window_point : eval_start_time_point + i
                ],
            }))

            test_data = TimeSeriesData(pd.DataFrame({
                'time': timeindex[eval_start_time_point + i : eval_start_time_point + i + retrain_freq_point],
                'value': valuedata[eval_start_time_point + i : eval_start_time_point + i + retrain_freq_point],
            }))

            # fit and predict
            # for GM, change serialized_model to bytes from str type
            if model_params and "serialized_model" in model_params and isinstance(model_params["serialized_model"], str):
                # pyre-fixme[16]: `float` has no attribute `encode`.
                model_params["serialized_model"] = model_params["serialized_model"].encode()

            # pyre-fixme[32]: Keyword argument `model_params` has type `Optional[Dict[str, float]]` but must be a mapping with string keys.
            detector = self.detector(**model_params)
            # pyre-fixme[16]: `Detector` has no attribute `fit_predict`.
            anom_obj = detector.fit_predict(
                data=test_data,
                historical_data=historical_data,
            )

            # concat anomaly response
            if not response:
                response = anom_obj
            else:
                response = AnomalyResponse(
                    scores=TimeSeriesData(
                        time=pd.concat([
                            pd.Series(response.scores.time.values),
                            pd.Series(anom_obj.scores.time.values)
                        ]),
                        value=pd.concat([
                            pd.Series(response.scores.value.values),
                            pd.Series(anom_obj.scores.value.values)
                        ]),
                    ),
                    confidence_band = None,
                    predicted_ts = None,
                    anomaly_magnitude_ts=TimeSeriesData(
                        time=pd.concat([
                            pd.Series(response.anomaly_magnitude_ts.time.values),
                            pd.Series(anom_obj.anomaly_magnitude_ts.time.values)
                        ]),
                        value=pd.concat([
                            pd.Series(response.anomaly_magnitude_ts.value.values),
                            pd.Series(anom_obj.anomaly_magnitude_ts.value.values)
                        ]),
                    ),
                    stat_sig_ts = None,
                )

        return response

    def evaluate(
        self,
        model_params: Optional[Dict[str, float]] = None,
        data: Optional[pd.DataFrame] = None,
        ignore_list: Optional[List[str]] = None,
        alert_style_cp: bool = False,
        threshold_low: float = 0.0,
        threshold_high: float = 1.0,
        margin: int = 5,
        eval_start_time_sec: Optional[int] = None,
        training_window_sec: Optional[int] = None,
        retrain_freq_sec: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        eval_start_time_sec: evaluation start time for anom_obj and anno
        training_window_sec: for GM and Prophet only
        retrain_freq_sec: for GM and Prophet only
        """

        if self.is_detector_model:
            return self._evaluate_detector_model(
                model_params=model_params,
                data=data,
                ignore_list=ignore_list,
                alert_style_cp=alert_style_cp,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                margin=margin,
                eval_start_time_sec=eval_start_time_sec,
                training_window_sec=training_window_sec,
                retrain_freq_sec=retrain_freq_sec,
            )
        else:
            return self._evaluate_detector(
                model_params=model_params,
                data=data,
                ignore_list=ignore_list,
                margin=margin,
            )

    def _evaluate_detector_model(
        self,
        model_params: Optional[Dict[str, float]] = None,
        data: Optional[pd.DataFrame] = None,
        ignore_list: Optional[List[str]] = None,
        alert_style_cp: bool = False,
        threshold_low: float = 0.0,
        threshold_high: float = 1.0,
        margin: int = 5,
        eval_start_time_sec: Optional[int] = None,
        training_window_sec: Optional[int] = None,
        retrain_freq_sec: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        eval_start_time_sec: evaluation start time for anom_obj and anno
        training_window_sec: for GM and Prophet only
        retrain_freq_sec: for GM and Prophet only
        """

        if not ignore_list:
            ignore_list = []

        if model_params is None:
            model_params = {}

        if data is None:
            data = self.load_data()
        self.ts_df = data

        eval_list = []

        for _, row in data.iterrows():
            this_dataset = row["dataset_name"]
            if this_dataset in ignore_list:
                continue
            data_name, tsd, anno = self._parse_data(row)

            if self.onlineflag:
                # if a user dosen't set eval_start_time
                if eval_start_time_sec is None:
                    eval_start_time_point = get_appropriate_start_time_default(
                        self.detector,
                        model_params,
                        tsd,
                    )
                    if eval_start_time_point >= len(tsd):
                        raise ValueError("Inappropriate parameters.")
                else:
                    # test if a user sets an appropriate eval_start_time
                    temp_start_time_point = get_appropriate_start_time_default(
                        self.detector,
                        model_params,
                        tsd,
                    )
                    if temp_start_time_point >= len(tsd):
                        raise ValueError("Inappropriate parameters.")

                    eval_start_time_point = get_points_from_sec(
                        eval_start_time_sec,
                        tsd,
                    )

                    if eval_start_time_point >= len(tsd):
                        raise ValueError("Inappropriate evaluation start time.")

                    if eval_start_time_point < temp_start_time_point:
                        print(f"Not approriate evaluation start time, using {tsd.time[temp_start_time_point]} instead.")
                        eval_start_time_point = temp_start_time_point

                # pyre-fixme[45]: Cannot instantiate abstract class `Detector`.
                detector = self.detector(**model_params)
                # pyre-fixme[16]: `Detector` has no attribute `fit_predict`.
                anom_obj = detector.fit_predict(tsd)
            else:
                assert eval_start_time_sec is not None
                assert training_window_sec is not None
                assert retrain_freq_sec is not None

                eval_start_time_point = get_points_from_sec(
                    eval_start_time_sec,
                    tsd,
                )
                training_window_point = get_points_from_sec(
                    training_window_sec,
                    tsd,
                )
                retrain_freq_point = get_points_from_sec(
                    retrain_freq_sec,
                    tsd,
                )
                # get anomaly response for GM and Prophet
                anom_obj = self._get_anomaly_response(
                    tsdata=tsd,
                    eval_start_time_point=eval_start_time_point,
                    training_window_point=training_window_point,
                    retrain_freq_point=retrain_freq_point,
                    model_params=model_params,
                )

            self.cp_dict[data_name] = get_cp_index_from_detector_model(
                anom_obj=anom_obj,
                alert_style_cp=alert_style_cp,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                onlineflag=self.onlineflag,
                eval_start_time_point=eval_start_time_point,
            )
            new_anno = {}
            for key in anno:
                new_anno[key] = [x for x in anno[key] if x >= eval_start_time_point]

            eval_dict = measure(
                annotations=new_anno, predictions=self.cp_dict[data_name], margin=margin
            )
            eval_list.append(
                Evaluation(
                    dataset_name=data_name,
                    precision=eval_dict["precision"],
                    recall=eval_dict["recall"],
                    f_score=eval_dict["f_score"],
                    delay=eval_dict["delay"],
                    TP=eval_dict["TP"],
                    FP=eval_dict["FP"],
                    FN=eval_dict["FN"],
                )
            )
        # break
        self.eval_agg = EvalAggregate(eval_list)
        eval_df = self.eval_agg.get_eval_dataframe()

        return eval_df

    def _evaluate_detector(
        self,
        model_params: Optional[Dict[str, float]] = None,
        data: Optional[pd.DataFrame] = None,
        ignore_list: Optional[List[str]] = None,
        margin: int = 5,
    ) -> pd.DataFrame:
        if model_params is None:
            model_params = {}
        if ignore_list is None:
            ignore_list = []

        if data is None:
            data = self.load_data()
        self.ts_df = data

        eval_list = []

        for _, row in data.iterrows():
            this_dataset = row["dataset_name"]
            if this_dataset in ignore_list:
                continue
            data_name, tsd, anno = self._parse_data(row)
            # pyre-fixme[45]: Cannot instantiate abstract class `Detector`.
            detector = self.detector(tsd)
            change_points = detector.detector(**model_params)
            self.cp_dict[data_name] = get_cp_index(change_points, tsd)
            eval_dict = measure(
                annotations=anno, predictions=self.cp_dict[data_name], margin=margin
            )
            eval_list.append(
                Evaluation(
                    dataset_name=data_name,
                    precision=eval_dict["precision"],
                    recall=eval_dict["recall"],
                    f_score=eval_dict["f_score"],
                    delay=eval_dict["delay"],
                    TP=eval_dict["TP"],
                    FP=eval_dict["FP"],
                    FN=eval_dict["FN"],
                )
            )
            # break
        self.eval_agg = EvalAggregate(eval_list)
        eval_df = self.eval_agg.get_eval_dataframe()

        return eval_df

    def get_eval_aggregate(self) -> EvalAggregate:
        """
        returns the EvalAggregate object, which can then be used for
        for further processing
        """
        assert self.eval_agg is not None, "EvalAggregate object is missing"
        return self.eval_agg

    def load_data(self) -> pd.DataFrame:
        """
        loads data, the source is either simulator or hive
        """
        return self._load_data_from_simulator()

    def _load_data_from_simulator(self) -> pd.DataFrame:
        ts_dict, ts_anno = generate_from_simulator()
        ts2_dict, ts2_anno = generate_from_simulator(
            cp_arr=[50, 100, 150], level_arr=[1.1, 1.05, 1.35, 1.2]
        )

        eg_df = pd.DataFrame(
            [
                {
                    "dataset_name": "eg_1",
                    "time_series": str(ts_dict),
                    "annotation": str(ts_anno),
                },
                {
                    "dataset_name": "eg_2",
                    "time_series": str(ts2_dict),
                    "annotation": str(ts2_anno),
                },
            ]
        )

        return eg_df

    def _parse_data(
        self, df_row: pd.Series
    ) -> Tuple[str, TimeSeriesData, Dict[str, List[int]]]:
        this_dataset = df_row["dataset_name"]
        this_ts = df_row["time_series"]
        this_anno = df_row["annotation"]

        this_anno_json_acc = this_anno.replace("'", '"')

        this_anno_dict = json.loads(this_anno_json_acc)

        this_ts_json_acc = this_ts.replace("'", '"')
        try:
            this_ts_dict = json.loads(this_ts_json_acc)
        except Exception:
            this_ts_json_acc = re.sub(r"(\w+):", r'"\1":', this_ts_json_acc)
            this_ts_dict = json.loads(this_ts_json_acc)

        ts = pd.DataFrame.from_dict(this_ts_dict, orient="index")
        ts.sort_index(inplace=True)
        ts.reset_index(inplace=True)
        ts.columns = ["time", "y"]
        first_time_val = ts.time.values[0]
        if re.match(r"\d{4}-\d{2}-\d{2}", first_time_val):
            fmt = "%Y-%m-%d"
            unit = None
        elif re.match(r"\d{4}-\d{2}", first_time_val):
            fmt = "%Y-%m"
            unit = None
        else:
            fmt = None
            unit = "s"
        ts["time"] = pd.to_datetime(ts["time"], format=fmt, unit=unit)

        tsd = TimeSeriesData(
            ts,
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
            time_col_name="time",
        )

        return this_dataset, tsd, this_anno_dict
