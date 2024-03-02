# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from kats.consts import IntervalAnomaly, TimeSeriesData

from kats.detectors.utils import ChangeDirections
from scipy import stats

_log: logging.Logger = logging.getLogger("anomaly_post_processing")


class AnomalyPostProcessHandler:
    """
    AnomalyPostProcessHandler: an anomaly alerts analyzer module

    The alert analyzer includes severity score (alert volume, alert magnitude, and alert average height, etc)
    calculation, and alert type analysis to classify anomalies into 4 categories:
    Level shift, Volatility change, Individual Anomaly, and Trend change.

    Attributes:
        ts_data: TimeSeriesData. Raw time series data
        anomaly_scores: TimeSeriesData. Anomaly scores returned by Kats detection algorithm
        threshold_low: Optional[float] = None. Lower bound threshold for transferring anomaly scores into anomaly intervals.
            Threshold value is the value which anomaly scores are compared against.
        threshold_high: Optional[float] = None. Upper bound threshold for transferring anomaly scores into anomaly intervals
            Threshold value is the value which anomaly scores are compared against.
        level_shift_coefficient: Optional[float] = None. Coefficient for deciding attribute level_shift
        volatility_change_coefficient: Optional[float] = None. Coefficient for deciding attribute volatility_change
        direction: Optional[ChangeDirections] = None. Direction for transferring anomaly scores into anomaly intervals
            Direction means which direction we care about when anomaly scores are beyond thresholds
            (upward, downward, inside/outside range, etc.).
        detection_window_sec: Optional[int] = None. Detection_window size for transferring anomaly scores into anomaly intervals
            Detection window is the size of the window within which we calculate the fraction of anomaly
            scores that are beyond the threshold value in the direction specified.
        fraction: Optional[float] = None. Fraction for transferring anomaly scores into anomaly intervals

    >>> # Example usage:
    >>> from kats.detectors.anomaly_postprocessing import AnomalyPostProcessHandler
    >>> from kats.detectors.stat_sig_detector import StatSigDetectorModel
    >>> ts = TimeSeriesData(ts_df)
    >>> DetectorModel = StatSigDetectorModel(n_control=3*24, n_test=12)
    >>> detection_results = DetectorModel.fit_predict(data=ts)
    >>> app = AnomalyPostProcessHandler(
            ts_data=ts,
            anomaly_scores=detection_results.scores,
            threshold_low=-1,
            threshold_high=2,
        )
    >>> app.get_severity_score_df()
    >>> app.get_overall_severity_score()
    >>> app.get_each_anomaly_type()
    >>> app.plot()
    >>> app.plot(on_ts=False)
    """

    def __init__(
        self,
        ts_data: TimeSeriesData,
        anomaly_scores: TimeSeriesData,
        threshold_low: Optional[float] = None,
        threshold_high: Optional[float] = None,
        level_shift_coefficient: Optional[float] = None,
        volatility_change_coefficient: Optional[float] = None,
        direction: Optional[ChangeDirections] = None,
        detection_window_sec: Optional[int] = None,  # second
        fraction: Optional[float] = None,
    ) -> None:
        if not anomaly_scores.is_univariate() or not ts_data.is_univariate():
            raise ValueError("Only support post processing for univariate time series")
        self.anomaly_scores_df: pd.DataFrame = anomaly_scores.to_dataframe()
        self.anomaly_scores_df.columns = ["time", "val"]

        self.ts_data_df: pd.DataFrame = ts_data.to_dataframe()
        self.ts_data_df.columns = ["time", "val"]
        self.ts_data_df = self.ts_data_df.set_index("time")

        self.threshold_low: float = threshold_low if threshold_low is not None else -2.0
        self.threshold_high: float = (
            threshold_high if threshold_high is not None else 2.0
        )

        self.level_shift_coefficient: float = (
            level_shift_coefficient if level_shift_coefficient is not None else 1.5
        )

        self.volatility_change_coefficient: float = (
            volatility_change_coefficient
            if volatility_change_coefficient is not None
            else 2.0
        )

        self.direction: ChangeDirections = (
            direction if direction is not None else ChangeDirections.BOTH
        )
        self.detection_window_sec: int = (
            detection_window_sec if detection_window_sec is not None else 1
        )
        self.fraction: float = fraction if fraction is not None else 1.0

        self.anomaly_interval_list: Union[List[IntervalAnomaly], None] = None
        self.score_lists: Union[List[List[float]], None] = None
        self.magnitude_list: Union[List[float], None] = None
        self.prelim_res: Union[pd.DataFrame, None] = None

    def _crossed_threshold_df(
        self,
        score_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Test whether passing threshold
        """
        if self.direction.value == 1:
            col = np.asarray(score_df["val"]) > self.threshold_high
            return pd.DataFrame({"time": score_df["time"], "val": col}, copy=False)
        elif self.direction.value == 2:
            col = np.asarray(score_df["val"]) < self.threshold_low
            return pd.DataFrame({"time": score_df["time"], "val": col}, copy=False)
        else:
            col = (np.asarray(score_df["val"]) < self.threshold_low) | (
                np.asarray(score_df["val"]) > self.threshold_high
            )
        return pd.DataFrame({"time": score_df["time"], "val": col}, copy=False)

    def _start_alert_function(
        self,
        cross_thres_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For past detection_window_sec seconds, if the cross_thresholds_ratio is >= fraction,
        then start raising an alert.
        """
        cross_thres = cross_thres_df.set_index("time")

        # Nowadays, pandas.DataFrame.rolling can deal with irregular time series.
        # reference:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
        res = cross_thres.rolling(str(self.detection_window_sec) + "s").mean()
        res = res.reset_index(drop=False)
        res.columns = ["time", "val"]
        res["val"] = res["val"] >= self.fraction

        # for those who doesn't have enough rolling window
        res.loc[
            res.time < res.time[0] + pd.Timedelta(str(self.detection_window_sec) + "s"),
            "val",
        ] = False
        return res

    def _get_anomaly_intervals_prelim(
        self,
        score_df: pd.DataFrame,
    ) -> None:
        score_df.columns = ["time", "val"]
        cross_thres = self._crossed_threshold_df(score_df=score_df)
        # when to raise alert
        potential_alert = self._start_alert_function(cross_thres_df=cross_thres)

        score_df_series = score_df.set_index("time")

        threshold_high = self.threshold_high
        threshold_low = self.threshold_low
        if threshold_high == 0:
            threshold_high += 1e-8
        if threshold_low == 0:
            threshold_low -= 1e-8

        cp_list = []
        cp = []  # timestamp
        score_lists = []  # score lists for calculating anomaly volume
        magnitude_list = []
        for i in range(len(potential_alert)):
            if potential_alert.iloc[i, 1]:
                if not cp:
                    # start alert
                    cp.append(potential_alert.iloc[i, 0])
            elif len(cp) == 1:
                # clear alert
                cp.append(potential_alert.iloc[i, 0])
                ia = IntervalAnomaly(*cp)
                cp_list.append(ia)

                score_itv_array = np.asarray(score_df_series.loc[cp[0] : cp[1], "val"])
                if self.direction.value == 1:
                    score_above = score_itv_array - self.threshold_high
                    mag = abs(score_above.max() / threshold_high)
                    magnitude_list.append(mag)
                    score_lists.append(score_above)
                elif self.direction.value == 2:
                    score_below = -score_itv_array + self.threshold_low
                    mag = abs(score_below.max() / threshold_low)
                    magnitude_list.append(mag)
                    score_lists.append(score_below)
                else:
                    score_above = np.maximum(
                        np.zeros(len(score_itv_array)),
                        score_itv_array - self.threshold_high,
                    )
                    score_below = np.maximum(
                        np.zeros(len(score_itv_array)),
                        -score_itv_array + self.threshold_low,
                    )
                    mag_above = abs(score_above.max() / threshold_high)
                    mag_below = abs(score_below.max() / threshold_low)

                    score_overall = np.concatenate(
                        [score_above.reshape([1, -1]), score_below.reshape([1, -1])], 0
                    ).max(0)
                    mag = max(mag_above, mag_below)
                    magnitude_list.append(mag)
                    score_lists.append(score_overall)

                cp = []

        # for alerts without an end
        if len(cp) == 1 and cp[0] < potential_alert.iloc[-1, 0]:
            cp.append(potential_alert.iloc[-1, 0])
            ia = IntervalAnomaly(*cp)
            cp_list.append(ia)
            score_itv_array = np.asarray(score_df_series.loc[cp[0] : cp[1], "val"])
            if self.direction.value == 1:
                score_above = score_itv_array - self.threshold_high
                mag = abs(score_above.max() / threshold_high)
                magnitude_list.append(mag)
                score_lists.append(score_above)
            elif self.direction.value == 2:
                score_below = -score_itv_array + self.threshold_low
                mag = abs(score_below.max() / threshold_low)
                magnitude_list.append(mag)
                score_lists.append(score_below)
            else:
                score_above = np.maximum(
                    np.zeros(len(score_itv_array)),
                    score_itv_array - self.threshold_high,
                )
                score_below = np.maximum(
                    np.zeros(len(score_itv_array)),
                    -score_itv_array + self.threshold_low,
                )
                mag_above = abs(score_above.max() / threshold_high)
                mag_below = abs(score_below.max() / threshold_low)

                score_overall = np.concatenate(
                    [score_above.reshape([1, -1]), score_below.reshape([1, -1])], 0
                ).max(0)
                mag = max(mag_above, mag_below)
                magnitude_list.append(mag)
                score_lists.append(score_overall)

        self.anomaly_interval_list = cp_list
        self.score_lists = score_lists
        self.magnitude_list = magnitude_list

    def get_anomaly_intervals(self) -> List[IntervalAnomaly]:
        if self.anomaly_interval_list is None:
            self._get_anomaly_intervals_prelim(score_df=self.anomaly_scores_df)

        assert self.anomaly_interval_list is not None
        return self.anomaly_interval_list

    def get_severity_score_df(self) -> pd.DataFrame:
        if self.anomaly_interval_list is None:
            self._get_anomaly_intervals_prelim(score_df=self.anomaly_scores_df)

        volume_list = []
        total_secs_list = []
        height_list = []
        if self.score_lists is not None and len(self.score_lists) > 0:
            # pyre-ignore [6]
            for i in range(len(self.anomaly_interval_list)):
                # pyre-ignore [16]
                ia = self.anomaly_interval_list[i]
                # pyre-ignore [16]
                itv_scores = self.score_lists[i]
                total_sec = (ia.end - ia.start).total_seconds()
                total_secs_list.append(total_sec)
                itv_len = len(itv_scores) - 1
                sec_list = total_sec / itv_len * np.ones(itv_len)
                volume = np.sum(sec_list * np.asarray(itv_scores[:-1]))
                volume_list.append(volume)
                height = volume / total_sec if total_sec else 0
                height_list.append(height)

        anomaly_type = self._get_anomaly_type()

        # pyre-ignore [16]
        start_time_list = [ia.start for ia in self.anomaly_interval_list]
        end_time_list = [ia.end for ia in self.anomaly_interval_list]

        self.prelim_res = pd.DataFrame(
            {
                "alert_interval": self.anomaly_interval_list,
                "alert_start_time": start_time_list,
                "alert_end_time": end_time_list,
                "alert_average_height": height_list,
                "alert_volume": volume_list,
                "alert_magnitude": self.magnitude_list,
                "alert_total_sec": total_secs_list,
                "anomaly_type": anomaly_type,
            }
        )
        return self.prelim_res

    def get_overall_severity_score(self) -> pd.DataFrame:
        # calculate alert volume, alert average high, alert magnitude.
        if self.prelim_res is None:
            self.get_severity_score_df()

        # pyre-ignore [16]
        overall_alerts_volume = sum(self.prelim_res.alert_volume)
        # pyre-ignore [16]
        overall_alerts_secs = sum(self.prelim_res.alert_total_sec)
        overall_alerts_average_height = (
            overall_alerts_volume / overall_alerts_secs if overall_alerts_secs else 0
        )

        # pyre-ignore [16]
        overall_alerts_magnitude = max(self.prelim_res.alert_magnitude)
        # pyre-ignore [6]
        alerts_number = len(self.prelim_res)

        overall_res = pd.DataFrame(
            {
                "alerts_number": [alerts_number],
                "alerts_volume": [overall_alerts_volume],
                "alerts_duration": [overall_alerts_secs],
                "alerts_average_height": [overall_alerts_average_height],
                "alerts_magnitude": [overall_alerts_magnitude],
            }
        )

        return overall_res

    def _get_anomaly_type(self) -> List[Dict[str, bool]]:
        """
        Anomaly type is a combination of individual and level shift, trend and volatility
        """
        if self.anomaly_interval_list is None:
            self._get_anomaly_intervals_prelim(score_df=self.anomaly_scores_df)

        res = []
        # pyre-ignore [16]
        for ai in self.anomaly_interval_list:
            cur_res = {}
            cur_res["level_shift"] = self._if_level_shift(ai)
            cur_res["individual"] = self._if_individual_outlier(ai)
            cur_res["trend_change"] = self._if_trend_change(ai)
            cur_res["volatility_change"] = self._if_volatility_change(ai)
            res.append(cur_res)

        return res

    def _if_level_shift(self, anomaly: IntervalAnomaly) -> bool:
        group_a0 = np.asarray(self.ts_data_df.loc[: anomaly.end, "val"])[:-1]
        group_b0 = np.asarray(self.ts_data_df.loc[anomaly.end :, "val"])
        if len(group_a0) <= 1 or len(group_b0) <= 1:
            t0 = False
        else:
            t0 = stats.ttest_ind(group_a0, group_b0).pvalue <= 0.05
        if t0:
            x01 = np.arange(0, len(group_a0))
            z01 = np.polyfit(x01, group_a0, 1)[1]

            group_all = np.concatenate([group_a0, group_b0])
            x02 = np.arange(0, len(group_all))
            z02 = np.polyfit(x02, group_all, 1)[1]

            if (
                z01 < self.level_shift_coefficient * z02
                and z01 > z02 / self.level_shift_coefficient
            ):
                t0 = False
        if t0:
            return True

        group_a1 = np.asarray(self.ts_data_df.loc[: anomaly.start, "val"])[:-1]
        group_b1 = np.asarray(self.ts_data_df.loc[anomaly.start :, "val"])
        if len(group_a1) <= 1 or len(group_b1) <= 1:
            t1 = False
        else:
            t1 = stats.ttest_ind(group_a1, group_b1).pvalue <= 0.05
        if t1:
            x11 = np.arange(0, len(group_a1))
            z11 = np.polyfit(x11, group_a1, 1)[1]

            group_all = np.concatenate([group_a1, group_b1])
            x12 = np.arange(0, len(group_all))
            z12 = np.polyfit(x12, group_all, 1)[1]

            if (
                z11 < self.level_shift_coefficient * z12
                and z11 > z12 / self.level_shift_coefficient
            ):
                t1 = False

        return t1

    def _if_individual_outlier(self, anomaly: IntervalAnomaly) -> bool:
        ts_itv_array = np.asarray(
            self.ts_data_df.loc[anomaly.start : anomaly.end, "val"]
        )[:-1]
        return len(ts_itv_array) < 5

    def _if_trend_change(self, anomaly: IntervalAnomaly) -> bool:
        ts_itv_before = np.asarray(self.ts_data_df.loc[: anomaly.start, "val"])[:-1]
        if len(ts_itv_before) <= 1:
            return False
        x0 = np.arange(0, len(ts_itv_before))
        z0 = np.polyfit(x0, ts_itv_before, 1)[0]

        ts_itv_after = np.asarray(self.ts_data_df.loc[anomaly.end :, "val"])
        if len(ts_itv_after) <= 1:
            return False
        x1 = np.arange(0, len(ts_itv_after))
        z1 = np.polyfit(x1, ts_itv_after, 1)[0]

        return z0 * z1 < 0

    def _if_volatility_change(self, anomaly: IntervalAnomaly) -> bool:
        group_a0 = np.asarray(self.ts_data_df.loc[: anomaly.end, "val"])[:-1]
        group_b0 = np.asarray(self.ts_data_df.loc[anomaly.end :, "val"])

        if len(group_a0) <= 1 or len(group_b0) <= 1:
            t0 = False
        else:
            group_a0_std = np.std(group_a0)
            group_b0_std = np.std(group_b0)
            t0 = (
                group_a0_std >= self.volatility_change_coefficient * group_b0_std
                or group_a0_std <= group_b0_std / self.volatility_change_coefficient
            )

        if t0:
            return True

        group_a1 = np.asarray(self.ts_data_df.loc[: anomaly.start, "val"])[:-1]
        group_b1 = np.asarray(self.ts_data_df.loc[anomaly.start :, "val"])

        if len(group_a1) <= 1 or len(group_b1) <= 1:
            t1 = False
        else:
            group_a1_std = np.std(group_a1)
            group_b1_std = np.std(group_b1)
            t1 = (
                group_a1_std >= self.volatility_change_coefficient * group_b1_std
                or group_a1_std <= group_b1_std / self.volatility_change_coefficient
            )

        return t1

    def get_each_anomaly_type(self) -> pd.DataFrame:
        """
        Anomaly type is a combination of individual and level shift, trend and volatility
        """
        if self.anomaly_interval_list is None:
            self._get_anomaly_intervals_prelim(score_df=self.anomaly_scores_df)

        res = []
        # pyre-ignore [16]
        for ai in self.anomaly_interval_list:
            cur_res = {}
            cur_res["level_shift"] = self._if_level_shift(ai)
            cur_res["individual"] = self._if_individual_outlier(ai)
            cur_res["trend_change"] = self._if_trend_change(ai)
            cur_res["volatility_change"] = self._if_volatility_change(ai)
            res.append(cur_res)

        res = pd.DataFrame(res)
        res["alert_interval"] = self.anomaly_interval_list

        if len(res) > 0:
            return res[
                [
                    "alert_interval",
                    "level_shift",
                    "individual",
                    "trend_change",
                    "volatility_change",
                ]
            ]
        else:
            return pd.DataFrame()

    def plot(self, on_ts: bool = True) -> None:
        if on_ts:
            title = "Anomaly detection (plot on top of the time series)"
        else:
            title = "Anomaly detection (plot on top of the anomaly scores)"

        sns.set(rc={"figure.figsize": (10, 5)})
        sns.set_style("whitegrid")
        if on_ts:
            ax = sns.lineplot(self.ts_data_df, color="midnightblue", marker="o")
        else:
            ax = sns.lineplot(
                self.anomaly_scores_df.set_index("time"),
                color="midnightblue",
                marker="o",
            )

        first = True
        if self.anomaly_interval_list is not None:
            # pyre-ignore[16]
            for ia in self.anomaly_interval_list:
                if first:
                    ax.axvspan(
                        ia.start,
                        ia.end,
                        alpha=0.98,
                        color="pink",
                        label="Anomaly",
                    )
                    first = False
                else:
                    ax.axvspan(
                        ia.start,
                        ia.end,
                        alpha=0.98,
                        color="pink",
                    )

        ax.legend(loc=0)
        ax.set_title(title)
        plt.xticks(rotation=90)
        plt.show()

        return
