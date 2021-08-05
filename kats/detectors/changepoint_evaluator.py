# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import re
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector, DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from kats.utils.simulator import Simulator

# defining some helper functions
def get_cp_index(
    changepoints: List[Tuple[TimeSeriesChangePoint, Any]], tsd: TimeSeriesData
) -> List[int]:
    """
    Accepts the output of the Detector.detector() method which is a list of
    tuples of (TimeSeriesChangePoint, Metadata) and returns the index of the
    changepoints
    """
    cp_list = []
    tsd_df = tsd.to_dataframe()
    tsd_df["time_index"] = list(range(tsd_df.shape[0]))
    for cp, _ in changepoints:
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
) -> List[int]:

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
    If there is multiple detected positives in the margin of a trupe positive and choose_earliest = True, we keep only the first one.
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

    >>> measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> measure({1: [], 2: [10], 3: [50]}, [])
    0.8

    Args:
        annotations : dict from user_id to iterable of CP locations.
        predictions : iterable of predicted CP locations.
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

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k].keys()) / len(Tks[k]) for k in Tks)
    D = 1 / K * sum(np.mean(list(TPk[k].values())) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)

    score_dict = {"f_score": F, "precision": P, "recall": R, "delay": D}

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
    def __init__(self, detector: Union[Type[Detector], Type[DetectorModel]]):
        self.detector = detector

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def load_data(self):
        pass


Evaluation = namedtuple(
    "Evaluation", ["dataset_name", "precision", "recall", "f_score", "delay"]
)


class EvalAggregate:
    def __init__(self, eval_list: List[Evaluation]):
        self.eval_list = eval_list
        self.eval_df = None

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
                }
            )

        self.eval_df = pd.DataFrame(df_list)
        return self.eval_df

    def get_avg_precision(self) -> float:
        if self.eval_df is None:
            _ = self.get_eval_dataframe()
        avg_precision = np.mean(self.eval_df.precision)

        return avg_precision

    def get_avg_recall(self) -> float:
        if self.eval_df is None:
            _ = self.get_eval_dataframe()
        avg_recall = np.mean(self.eval_df.recall)

        return avg_recall

    def get_avg_f_score(self) -> float:
        if self.eval_df is None:
            _ = self.get_eval_dataframe()
        avg_f_score = np.mean(self.eval_df.f_score)
        return avg_f_score

    def get_avg_delay(self) -> float:
        if self.eval_df is None:
            _ = self.get_eval_dataframe()
        avg_delay = np.mean(self.eval_df.delay)
        return avg_delay


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
    ):
        super(TuringEvaluator, self).__init__(detector=detector)
        self.detector = detector
        self.eval_agg = None
        self.is_detector_model = is_detector_model
        self.cp_list = []

    def evaluate(
        self,
        model_params: Optional[Dict[str, float]] = None,
        data: Optional[pd.DataFrame] = None,
        ignore_list: Optional[List[str]] = None,
        alert_style_cp: bool = False,
        threshold_low: float = 0.0,
        threshold_high: float = 1.0,
    ) -> pd.DataFrame:

        if self.is_detector_model:
            return self._evaluate_detector_model(
                model_params=model_params,
                data=data,
                ignore_list=ignore_list,
                alert_style_cp=alert_style_cp,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
        else:
            return self._evaluate_detector(
                model_params=model_params, data=data, ignore_list=ignore_list
            )

    def _evaluate_detector_model(
        self,
        model_params: Optional[Dict[str, float]] = None,
        data: Optional[pd.DataFrame] = None,
        ignore_list: Optional[List[str]] = None,
        alert_style_cp: bool = False,
        threshold_low: float = 0.0,
        threshold_high: float = 1.0,
    ) -> pd.DataFrame:

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
            # pyre-fixme[45]: Cannot instantiate abstract class `Detector`.
            detector = self.detector(**model_params)
            # pyre-fixme[16]: `Detector` has no attribute `fit_predict`.
            anom_obj = detector.fit_predict(tsd)

            self.cp_list = get_cp_index_from_detector_model(
                anom_obj, alert_style_cp, threshold_low, threshold_high
            )

            eval_dict = measure(annotations=anno, predictions=self.cp_list)
            eval_list.append(
                Evaluation(
                    dataset_name=data_name,
                    precision=eval_dict["precision"],
                    recall=eval_dict["recall"],
                    f_score=eval_dict["f_score"],
                    delay=eval_dict["delay"],
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
            self.cp_list = get_cp_index(change_points, tsd)
            eval_dict = measure(annotations=anno, predictions=self.cp_list)
            eval_list.append(
                Evaluation(
                    dataset_name=data_name,
                    precision=eval_dict["precision"],
                    recall=eval_dict["recall"],
                    f_score=eval_dict["f_score"],
                    delay=eval_dict["delay"],
                )
            )
            # break
        self.eval_agg = EvalAggregate(eval_list)
        eval_df = self.eval_agg.get_eval_dataframe()

        return eval_df

    def get_eval_aggregate(self):
        """
        returns the EvalAggregate object, which can then be used for
        for further processing
        """
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

    def _parse_data(self, df_row: Any):
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
