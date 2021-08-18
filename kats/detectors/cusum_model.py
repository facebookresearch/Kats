# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUSUMDetectorModel is a wraper of CUSUMDetector to detect multiple change points

Typical usage example:

>>> # Define CUSUMDetectorModel
>>> model = CUSUMDetectorModel(
        scan_window=43200,
        historical_window=604800,
        threshold=0.01,
        delta_std_ratio=1.0,
        serialized_model=None,
        change_directions=["increase"],
        score_func=CusumScoreFunction.percentage_change,
        remove_seasonality=True,
    )
>>> # Run detector
>>> respond = model.fit_predict(tsd)
>>> # Plot anomaly score
>>> respond.scores.plot(cols=['value'])
>>> # Get change points in unixtime
>>> change_points = model.cps
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from kats.consts import (
    DEFAULT_VALUE_NAME,
    TimeSeriesData,
)
from kats.detectors.cusum_detection import CUSUMDetector, CUSUM_DEFAULT_ARGS
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from kats.utils.decomposition import TimeSeriesDecomposition


NORMAL_TOLERENCE = 1  # number of window
CHANGEPOINT_RETENTION = 7 * 24 * 60 * 60  # in seconds
MAX_CHANGEPOINT = 10


def percentage_change(
    data: TimeSeriesData, pre_mean: float, **kwargs: Any
) -> TimeSeriesData:
    """
    Calculate percentage change absolute change / baseline change

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
    """

    return (data - pre_mean) / (pre_mean)


def change(data: TimeSeriesData, pre_mean: float, **kwargs: Any) -> TimeSeriesData:
    """
    Calculate absolute change

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
    """

    return data - pre_mean


def z_score(data: TimeSeriesData, pre_mean: float, pre_std: float) -> TimeSeriesData:
    """
    Calculate z score: absolute change / std

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
        pre_std: Baseline std
    """

    return (data - pre_mean) / (pre_std)


class CusumScoreFunction(Enum):
    change = "change"
    percentage_change = "percentage_change"
    z_score = "z_score"


# Score Function Constants
SCORE_FUNC_DICT = {
    CusumScoreFunction.change.value: change,
    CusumScoreFunction.percentage_change.value: percentage_change,
    CusumScoreFunction.z_score.value: z_score,
}
DEFAULT_SCORE_FUNCTION = CusumScoreFunction.change
STR_TO_SCORE_FUNC = {  # Used for param tuning
    "change": CusumScoreFunction.change,
    "percentage_change": CusumScoreFunction.percentage_change,
    "z_score": CusumScoreFunction.z_score,
}


class CUSUMDetectorModel(DetectorModel):
    """CUSUMDetectorModel for detecting multiple level shift change points

    CUSUMDetectorModel runs CUSUMDetector multiple times to detect multiple change
    points. In each run, CUSUMDetector will use historical_window + scan_window as
    input time series, and find change point in scan_window. The DetectorModel stores
    change points and returns anomaly score.

    Attributes:
        cps: Change points detected in unixtime.
        alert_fired: If a change point is detected and the anomaly still present.
        pre_mean: Previous baseline mean.
        pre_std: Previous baseline std.
        number_of_normal_scan: Number of scans with mean returned back to baseline.
        alert_change_direction: Increase or decrease.
        scan_window: Length in seconds of scan window.
        historical_window: Length in seconds of historical window.
        step_window: The time difference between CUSUM runs.
        threshold: CUSUMDetector threshold.
        delta_std_ratio: The mean delta have to larger than this parameter times std of
            the data to be consider as a change.
        magnitude_quantile: See in CUSUMDetector.
        magnitude_ratio: See in CUSUMDetector.
        score_func: The score function to calculate the anomaly score.
        remove_seasonality: If apply STL to remove seasonality.
    """

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        scan_window: Optional[int] = None,
        historical_window: Optional[int] = None,
        step_window: Optional[int] = None,
        threshold: float = CUSUM_DEFAULT_ARGS["threshold"],
        delta_std_ratio: float = CUSUM_DEFAULT_ARGS["delta_std_ratio"],
        magnitude_quantile: float = CUSUM_DEFAULT_ARGS["magnitude_quantile"],
        magnitude_ratio: float = CUSUM_DEFAULT_ARGS["magnitude_ratio"],
        change_directions: List[str] = CUSUM_DEFAULT_ARGS["change_directions"],
        score_func: Union[str, CusumScoreFunction] = DEFAULT_SCORE_FUNCTION,
        remove_seasonality: bool = CUSUM_DEFAULT_ARGS["remove_seasonality"],
    ):
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.cps = previous_model["cps"]
            self.alert_fired = previous_model["alert_fired"]
            self.pre_mean = previous_model["pre_mean"]
            self.pre_std = previous_model["pre_std"]
            self.number_of_normal_scan = previous_model["number_of_normal_scan"]
            self.alert_change_direction = previous_model["alert_change_direction"]
            self.scan_window = previous_model["scan_window"]
            self.historical_window = previous_model["historical_window"]
            self.step_window = previous_model["step_window"]
            self.threshold = previous_model["threshold"]
            self.delta_std_ratio = previous_model["delta_std_ratio"]
            self.magnitude_quantile = previous_model["magnitude_quantile"]
            self.magnitude_ratio = previous_model["magnitude_ratio"]
            self.change_directions = previous_model["change_directions"]
            self.score_func = previous_model["score_func"]
            if "remove_seasonality" in previous_model:
                self.remove_seasonality = previous_model["remove_seasonality"]
            else:
                self.remove_seasonality = remove_seasonality
        elif scan_window is not None and historical_window is not None:
            self.cps = []
            self.alert_fired = False
            self.pre_mean = 0
            self.pre_std = 1
            self.number_of_normal_scan = 0
            self.alert_change_direction = None
            self.scan_window = scan_window
            self.historical_window = historical_window
            self.step_window = step_window
            self.threshold = threshold
            self.delta_std_ratio = delta_std_ratio
            self.magnitude_quantile = magnitude_quantile
            self.magnitude_ratio = magnitude_ratio
            self.change_directions = change_directions
            self.remove_seasonality = remove_seasonality

            # We allow score_function to be a str for compatibility with param tuning
            if isinstance(score_func, str):
                if score_func in STR_TO_SCORE_FUNC:
                    score_func = STR_TO_SCORE_FUNC[score_func]
                else:
                    score_func = DEFAULT_SCORE_FUNCTION
            self.score_func = score_func.value

        else:
            raise ValueError(
                """
            You must either provide serialized model or values for scan_window and historical_window.
            """
            )
        # pyre-fixme[58]: `>=` is not supported for operand types `int` and
        #  `Optional[int]`.
        # pyre-fixme[58]: `>=` is not supported for operand types `int` and
        #  `Optional[int]`.
        if step_window is not None and step_window >= scan_window:
            raise ValueError(
                "Step window should smaller than scan window to ensure we have overlap for scan windows."
            )

    def __eq__(self, other):
        if isinstance(other, CUSUMDetectorModel):
            return (
                self.cps == other.cps
                and self.alert_fired == other.alert_fired
                and self.pre_mean == other.pre_mean
                and self.pre_std == other.pre_std
                and self.number_of_normal_scan == other.number_of_normal_scan
                and self.alert_change_direction == other.alert_change_direction
                and self.scan_window == other.scan_window
                and self.historical_window == other.historical_window
                and self.step_window == other.step_window
                and self.threshold == other.threshold
                and self.delta_std_ratio == other.delta_std_ratio
                and self.magnitude_quantile == other.magnitude_quantile
                and self.magnitude_ratio == other.magnitude_ratio
                and self.change_directions == other.change_directions
                and self.score_func == other.score_func
            )
        return False

    def serialize(self) -> bytes:
        """
        Retrun serilized model.
        """

        return str.encode(json.dumps(self.__dict__))

    def _set_alert_off(self) -> None:
        self.alert_fired = False
        self.number_of_normal_scan = 0

    def _set_alert_on(
        self, baseline_mean: float, baseline_std: float, alert_change_direction: str
    ) -> None:
        self.alert_fired = True
        self.alert_change_direction = alert_change_direction
        self.pre_mean = baseline_mean
        self.pre_std = baseline_std

    def _if_normal(self, cur_mean: float, change_directions: str) -> None:
        if change_directions is not None:
            increase, decrease = (
                "increase" in change_directions,
                "decrease" in change_directions,
            )
        else:
            increase, decrease = True, True

        if self.alert_change_direction == "increase":
            check_increase = 0 if increase else np.inf
            check_decrease = 1.0 if decrease else np.inf
        elif self.alert_change_direction == "decrease":
            check_increase = 1.0 if increase else np.inf
            check_decrease = 0 if decrease else np.inf

        return (
            self.pre_mean - check_decrease * self.pre_std
            <= cur_mean
            <= self.pre_mean + check_increase * self.pre_std
        )

    def _fit(
        self,
        data: TimeSeriesData,
        historical_data: TimeSeriesData,
        scan_window: int,
        threshold: float = CUSUM_DEFAULT_ARGS["threshold"],
        delta_std_ratio: float = CUSUM_DEFAULT_ARGS["delta_std_ratio"],
        magnitude_quantile: float = CUSUM_DEFAULT_ARGS["magnitude_quantile"],
        magnitude_ratio: float = CUSUM_DEFAULT_ARGS["magnitude_ratio"],
        change_directions: List[str] = CUSUM_DEFAULT_ARGS["change_directions"],
    ) -> None:
        """Fit CUSUM model.

        Args:
            data: the new data the model never seen
            historical_data: the historical data, `historical_data` have to end with the
                datapoint right before the first data point in `data`
            scan_window: scan window length in seconds, scan window is the window where
                cusum search for changepoint(s)
            threshold: changepoint significant level, higher the value more changepoints
                detected
            delta_std_ratio: the mean change have to larger than `delta_std_ratio` *
            `std(data[:changepoint])` to be consider as a change, higher the value
            less changepoints detected
            magnitude_quantile: float, the quantile for magnitude comparison, if
                none, will skip the magnitude comparison;
            magnitude_ratio: float, comparable ratio;
            change_directions: a list contain either or both 'increas' and 'decrease' to
                specify what type of change to detect;
        """
        historical_data.extend(data, validate=False)
        n = len(historical_data)
        scan_start_time = historical_data.time.iloc[-1] - pd.Timedelta(
            scan_window, unit="s"
        )
        scan_start_index = max(
            0, np.argwhere((historical_data.time >= scan_start_time).values).min()
        )
        if not self.alert_fired:
            # if scan window is less than 2 data poins and there is no alert fired
            # skip this scan
            if n - scan_start_index <= 1:
                return
            detector = CUSUMDetector(historical_data)
            changepoints = detector.detector(
                interest_window=[scan_start_index, n],
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            if len(changepoints) > 0:
                cp, meta = sorted(changepoints, key=lambda x: x[0].start_time)[0]
                self.cps.append(int(cp.start_time.value / 1e9))

                if len(self.cps) > MAX_CHANGEPOINT:
                    self.cps.pop(0)

                self._set_alert_on(
                    historical_data.value[: meta.cp_index + 1].mean(),
                    historical_data.value[: meta.cp_index + 1].std(),
                    meta.direction,
                )
        else:
            cur_mean = historical_data[scan_start_index:].value.mean()

            # pyre-fixme[6]: Expected `str` for 2nd param but got `List[str]`.
            if self._if_normal(cur_mean, change_directions):
                self.number_of_normal_scan += 1
                if self.number_of_normal_scan >= NORMAL_TOLERENCE:
                    self._set_alert_off()
            else:
                self.number_of_normal_scan = 0

            current_time = int(data.time.max().value / 1e9)
            if current_time - self.cps[-1] > CHANGEPOINT_RETENTION:
                self._set_alert_off()

    def _predict(
        self,
        data: TimeSeriesData,
        score_func: CusumScoreFunction = CusumScoreFunction.change,
    ) -> TimeSeriesData:
        """
        data: the new data for the anoamly score calculation.
        """
        if self.alert_fired:
            cp = self.cps[-1]
            tz = data.tz()
            if tz is None:
                change_time = pd.to_datetime(cp, unit="s")
            else:
                change_time = pd.to_datetime(cp, unit="s", utc=True).tz_convert(tz)

            if change_time >= data.time.iloc[0]:
                cp_index = data.time[data.time == change_time].index[0]
                data_pre = data[: cp_index + 1]
                score_pre = self._zeros_ts(data_pre)
                score_post = SCORE_FUNC_DICT[score_func](
                    data=data[cp_index + 1 :],
                    pre_mean=self.pre_mean,
                    pre_std=self.pre_std,
                )
                score_pre.extend(score_post, validate=False)
                return score_pre
            return SCORE_FUNC_DICT[score_func](
                data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
            )
        else:
            return self._zeros_ts(data)

    def _zeros_ts(self, data: TimeSeriesData) -> TimeSeriesData:
        return TimeSeriesData(
            time=data.time,
            value=pd.Series(
                np.zeros(len(data)),
                name=data.value.name if data.value.name else DEFAULT_VALUE_NAME,
            ),
        )

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """
        This function combines fit and predict and return anomaly socre for data. It
        requires scan_window > step_window.
        The relationship between two consective cusum runs in the loop is shown as below:

        >>> |---historical_window---|---scan_window---|
        >>>                                           |-step_window-|
        >>>               |---historical_window---|---scan_window---|

        * scan_window: the window size in seconds to detect change point
        * historical_window: the window size in seconds to provide historical data
        * step_window: the window size in seconds to specify the step size between two scans

        Args:
            data: :class:`kats.consts.TimeSeriesData` object representing the data
            historical_data: :class:`kats.consts.TimeSeriesData` object representing the history.

        Returns:
            The anomaly response contains the anomaly socres.
        """
        # get parameters
        scan_window = self.scan_window
        historical_window = self.historical_window
        step_window = self.step_window
        threshold = self.threshold
        delta_std_ratio = self.delta_std_ratio
        magnitude_quantile = self.magnitude_quantile
        magnitude_ratio = self.magnitude_ratio
        change_directions = self.change_directions
        score_func = self.score_func
        remove_seasonality = self.remove_seasonality

        scan_window = pd.Timedelta(scan_window, unit="s")
        historical_window = pd.Timedelta(historical_window, unit="s")

        # pull all the data in historical data
        if historical_data is not None:
            # make a copy of historical data
            historical_data = historical_data[:]
            historical_data.extend(data, validate=False)
        else:
            # When historical_data is not provided, will use part of data as
            # historical_data, and fill with zero anomaly score.
            historical_data = data[:]

        frequency = historical_data.freq_to_timedelta()
        if frequency is None or frequency is pd.NaT:
            # Use the top frequency if any, when not able to infer from data.
            freq_counts = (
                historical_data.time.diff().value_counts().sort_values(ascending=False)
            )
            if freq_counts.iloc[0] >= int(len(historical_data)) * 0.8 - 1:
                frequency = freq_counts.index[0]
            else:
                logging.debug(f"freq_counts: {freq_counts}")
                raise ValueError("Not able to infer freqency of the time series")

        if remove_seasonality:
            decomposer_input = historical_data.interpolate(frequency)

            # fixing the period to 24 hours as indicated in T81530775
            period = int(24 * 60 * 60 / frequency.total_seconds())

            decomposer = TimeSeriesDecomposition(
                decomposer_input,
                period=period,
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=int((period + 1) * 0.15),  # save run time
                seasonal_jump=1,
                trend_jump=int((period + 1) * 0.15),  # save run time
            )

            decomp = decomposer.decomposer()
            historical_data_time_idx = decomp["rem"].time.isin(historical_data.time)
            historical_data.value = pd.Series(
                decomp["rem"][historical_data_time_idx].value
                + decomp["trend"][historical_data_time_idx].value,
                name=historical_data.value.name,
            )

        smooth_window = int(scan_window.total_seconds() / frequency.total_seconds())
        if smooth_window > 1:
            smooth_historical_value = pd.Series(
                np.convolve(
                    historical_data.value.values, np.ones(smooth_window), mode="full"
                )[: 1 - smooth_window]
                / smooth_window,
                name=historical_data.value.name,
            )
            smooth_historical_data = TimeSeriesData(
                time=historical_data.time, value=smooth_historical_value
            )
        else:
            smooth_historical_data = historical_data

        anomaly_start_time = max(
            historical_data.time.iloc[0] + historical_window, data.time.iloc[0]
        )
        if anomaly_start_time > historical_data.time.iloc[-1]:
            # if len(all data) is smaller than historical window return zero score
            return AnomalyResponse(
                scores=self._predict(smooth_historical_data[-len(data) :], score_func),
                # pyre-fixme[6]: Expected `ConfidenceBand` for 2nd param but got `None`.
                confidence_band=None,
                # pyre-fixme[6]: Expected `TimeSeriesData` for 3rd param but got `None`.
                predicted_ts=None,
                anomaly_magnitude_ts=self._zeros_ts(data),
                # pyre-fixme[6]: Expected `TimeSeriesData` for 5th param but got `None`.
                stat_sig_ts=None,
            )
        anomaly_start_idx = self._time2idx(data, anomaly_start_time, "right")
        anomaly_start_time = data.time.iloc[anomaly_start_idx]
        score_tsd = self._zeros_ts(data[:anomaly_start_idx])

        if (
            historical_data.time.iloc[-1] - historical_data.time.iloc[0] + frequency
            <= scan_window
        ):
            # if len(all data) is smaller than scan data return zero score
            return AnomalyResponse(
                scores=self._predict(smooth_historical_data[-len(data) :], score_func),
                # pyre-fixme[6]: Expected `ConfidenceBand` for 2nd param but got `None`.
                confidence_band=None,
                # pyre-fixme[6]: Expected `TimeSeriesData` for 3rd param but got `None`.
                predicted_ts=None,
                anomaly_magnitude_ts=self._zeros_ts(data),
                # pyre-fixme[6]: Expected `TimeSeriesData` for 5th param but got `None`.
                stat_sig_ts=None,
            )

        if step_window is None:
            # if step window is not provide use the time range of data or
            # half of the scan_window.
            step_window = min(
                scan_window / 2,
                (data.time.iloc[-1] - data.time.iloc[0])
                + frequency,  # to include the last data point
            )
        else:
            step_window = pd.Timedelta(step_window, unit="s")

        for start_time in pd.date_range(
            anomaly_start_time,
            min(
                data.time.iloc[-1]
                + frequency
                - step_window,  # to include last data point
                data.time.iloc[-1],  # make sure start_time won't beyond last data time
            ),
            freq=step_window,
        ):
            logging.debug(f"start_time {start_time}")
            historical_start = self._time2idx(
                historical_data, start_time - historical_window, "right"
            )
            logging.debug(f"historical_start {historical_start}")
            historical_end = self._time2idx(historical_data, start_time, "right")
            logging.debug(f"historical_end {historical_end}")
            scan_end = self._time2idx(historical_data, start_time + step_window, "left")
            logging.debug(f"scan_end {scan_end}")
            in_data = historical_data[historical_end : scan_end + 1]
            if len(in_data) == 0:
                # skip if there is no data in the step_window
                continue
            in_hist = historical_data[historical_start:historical_end]
            self._fit(
                in_data,
                in_hist,
                # pyre-fixme[6]: Expected `int` for 3rd param but got `Timedelta`.
                scan_window=scan_window,
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            score_tsd.extend(
                self._predict(
                    smooth_historical_data[historical_end : scan_end + 1],
                    score_func=score_func,
                ),
                validate=False,
            )

        # Handle the remaining data
        remain_data_len = len(data) - len(score_tsd)
        if remain_data_len > 0:
            scan_end = len(historical_data)
            historical_end = len(historical_data) - remain_data_len
            historical_start = self._time2idx(
                historical_data,
                historical_data.time.iloc[historical_end] - historical_window,
                "left",
            )
            in_data = historical_data[historical_end:scan_end]
            in_hist = historical_data[historical_start:historical_end]
            self._fit(
                in_data,
                in_hist,
                # pyre-fixme[6]: Expected `ConfidenceBand` for 2nd param but got `None`.
                # pyre-fixme[6]: Expected `int` for 3rd param but got `Timedelta`.
                scan_window=scan_window,
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            score_tsd.extend(
                self._predict(
                    smooth_historical_data[historical_end:scan_end],
                    score_func=score_func,
                ),
                validate=False,
            )

        return AnomalyResponse(
            scores=score_tsd,
            # pyre-fixme[6]: Expected `ConfidenceBand` for 2nd param but got `None`.
            confidence_band=None,
            # pyre-fixme[6]: Expected `TimeSeriesData` for 3rd param but got `None`.
            predicted_ts=None,
            anomaly_magnitude_ts=self._zeros_ts(data),
            # pyre-fixme[6]: Expected `TimeSeriesData` for 5th param but got `None`.
            stat_sig_ts=None,
        )

    def _time2idx(self, tsd: TimeSeriesData, time: datetime, direction: str) -> int:
        """
        This function get the index of the TimeSeries data given a datatime.
        left takes the index on the left of the time stamp (inclusive)
        right takes the index on the right of the time stamp (exclusive)
        """
        if direction == "right":
            return np.argwhere((tsd.time >= time).values).min()
        elif direction == "left":
            return np.argwhere((tsd.time < time).values).max()
        else:
            raise ValueError("direction can only be right or left")

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel` inconsistently.
    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> None:
        self.fit_predict(data, historical_data)

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")
