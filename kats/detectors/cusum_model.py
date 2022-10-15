# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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
from typing import Any, cast, Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from kats.consts import DEFAULT_VALUE_NAME, IRREGULAR_GRANULARITY_ERROR, TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDefaultArgs, CUSUMDetector
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from kats.utils.decomposition import TimeSeriesDecomposition


NORMAL_TOLERENCE = 1  # number of window
CHANGEPOINT_RETENTION: int = 7 * 24 * 60 * 60  # in seconds
MAX_CHANGEPOINT = 10
SEASON_PERIOD_FREQ_MAP: Dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "yearly": 365,
}

_log: logging.Logger = logging.getLogger("cusum_model")


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
# pyre-fixme
SCORE_FUNC_DICT = {
    CusumScoreFunction.change.value: change,
    CusumScoreFunction.percentage_change.value: percentage_change,
    CusumScoreFunction.z_score.value: z_score,
}
DEFAULT_SCORE_FUNCTION: CusumScoreFunction = CusumScoreFunction.change
STR_TO_SCORE_FUNC: Dict[str, CusumScoreFunction] = {  # Used for param tuning
    "change": CusumScoreFunction.change,
    "percentage_change": CusumScoreFunction.percentage_change,
    "z_score": CusumScoreFunction.z_score,
}


class PredictFunctionValues(NamedTuple):
    score: TimeSeriesData
    absolute_change: TimeSeriesData


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
        season_period_freq: str, "daily"/"weekly"/"monthly"/"yearly"
    """

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        scan_window: Optional[int] = None,
        historical_window: Optional[int] = None,
        step_window: Optional[int] = None,
        threshold: float = CUSUMDefaultArgs.threshold,
        delta_std_ratio: float = CUSUMDefaultArgs.delta_std_ratio,
        magnitude_quantile: Optional[float] = CUSUMDefaultArgs.magnitude_quantile,
        magnitude_ratio: float = CUSUMDefaultArgs.magnitude_ratio,
        change_directions: Optional[
            Union[List[str], str]
        ] = CUSUMDefaultArgs.change_directions,
        score_func: Union[str, CusumScoreFunction] = DEFAULT_SCORE_FUNCTION,
        remove_seasonality: bool = CUSUMDefaultArgs.remove_seasonality,
        season_period_freq: str = "daily",
    ) -> None:
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.cps: List[int] = previous_model["cps"]
            self.alert_fired: bool = previous_model["alert_fired"]
            self.pre_mean: float = previous_model["pre_mean"]
            self.pre_std: float = previous_model["pre_std"]
            self.number_of_normal_scan: int = previous_model["number_of_normal_scan"]
            self.alert_change_direction: str = previous_model["alert_change_direction"]
            self.scan_window: int = previous_model["scan_window"]
            scan_window = previous_model["scan_window"]
            self.historical_window: int = previous_model["historical_window"]
            self.step_window: int = previous_model["step_window"]
            step_window = previous_model["step_window"]
            self.threshold: float = previous_model["threshold"]
            self.delta_std_ratio: float = previous_model["delta_std_ratio"]
            self.magnitude_quantile: Optional[float] = previous_model[
                "magnitude_quantile"
            ]
            self.magnitude_ratio: float = previous_model["magnitude_ratio"]
            self.change_directions: Optional[List[str]] = previous_model[
                "change_directions"
            ]
            self.score_func: CusumScoreFunction = previous_model["score_func"]
            if "remove_seasonality" in previous_model:
                self.remove_seasonality: bool = previous_model["remove_seasonality"]
            else:
                self.remove_seasonality: bool = remove_seasonality

            self.season_period_freq: str = previous_model.get(
                "season_period_freq", "daily"
            )

        elif scan_window is not None and historical_window is not None:
            self.cps = []
            self.alert_fired = False
            self.pre_mean = 0
            self.pre_std = 1
            self.number_of_normal_scan = 0
            self.alert_change_direction: Union[str, None] = None
            self.scan_window = scan_window
            self.historical_window = historical_window
            self.step_window = cast(int, step_window)
            self.threshold = threshold
            self.delta_std_ratio = delta_std_ratio
            self.magnitude_quantile = magnitude_quantile
            self.magnitude_ratio = magnitude_ratio

            if isinstance(change_directions, str):
                self.change_directions = [change_directions]
            else:
                # List[str]
                self.change_directions = change_directions

            self.remove_seasonality = remove_seasonality
            self.season_period_freq = season_period_freq

            # We allow score_function to be a str for compatibility with param tuning
            if isinstance(score_func, str):
                if score_func in STR_TO_SCORE_FUNC:
                    score_func = STR_TO_SCORE_FUNC[score_func]
                else:
                    score_func = DEFAULT_SCORE_FUNCTION
            self.score_func = score_func.value

        else:
            raise ValueError(
                "You must provide either serialized model or values for "
                "scan_window and historical_window."
            )

    def __eq__(self, other: object) -> bool:
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

    def _if_normal(
        self, cur_mean: float, change_directions: Optional[List[str]]
    ) -> bool:
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
        scan_window: Union[int, pd.Timedelta],
        threshold: float = CUSUMDefaultArgs.threshold,
        delta_std_ratio: float = CUSUMDefaultArgs.delta_std_ratio,
        magnitude_quantile: Optional[float] = CUSUMDefaultArgs.magnitude_quantile,
        magnitude_ratio: float = CUSUMDefaultArgs.magnitude_ratio,
        change_directions: Optional[List[str]] = CUSUMDefaultArgs.change_directions,
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
            change_directions: a list contain either or both 'increase' and 'decrease' to
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
                cp = sorted(changepoints, key=lambda x: x.start_time)[0]
                self.cps.append(int(cp.start_time.value / 1e9))

                if len(self.cps) > MAX_CHANGEPOINT:
                    self.cps.pop(0)

                self._set_alert_on(
                    historical_data.value[: cp.cp_index + 1].mean(),
                    historical_data.value[: cp.cp_index + 1].std(),
                    cp.direction,
                )
        else:
            cur_mean = historical_data[scan_start_index:].value.mean()

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
    ) -> PredictFunctionValues:
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
                change_pre = self._zeros_ts(data_pre)
                score_post = SCORE_FUNC_DICT[score_func](
                    data=data[cp_index + 1 :],
                    pre_mean=self.pre_mean,
                    pre_std=self.pre_std,
                )
                score_pre.extend(score_post, validate=False)

                change_post = SCORE_FUNC_DICT[CusumScoreFunction.change.value](
                    data=data[cp_index + 1 :],
                    pre_mean=self.pre_mean,
                    pre_std=self.pre_std,
                )
                change_pre.extend(change_post, validate=False)
                return PredictFunctionValues(score_pre, change_pre)
            return PredictFunctionValues(
                SCORE_FUNC_DICT[score_func](
                    data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
                ),
                SCORE_FUNC_DICT[CusumScoreFunction.change.value](
                    data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
                ),
            )
        else:
            return PredictFunctionValues(self._zeros_ts(data), self._zeros_ts(data))

    def _zeros_ts(self, data: TimeSeriesData) -> TimeSeriesData:
        return TimeSeriesData(
            time=data.time,
            value=pd.Series(
                np.zeros(len(data)),
                name=data.value.name if data.value.name else DEFAULT_VALUE_NAME,
                copy=False,
            ),
        )

    def _check_window_sizes(self, frequency_sec: int) -> None:
        """
        Function to check if historical_window, scan_window, and step_window
        are suitable for a given TS data and a given TS historical_data.
        We have already checked if self.step_window < self.scan_window in init func
        when self.step_window is not None.
        """
        if self.step_window is not None:
            if self.step_window >= self.scan_window:
                raise ValueError(
                    "Step window is supposed to be smaller than scan window to ensure we "
                    "have overlap for scan windows."
                )
            if self.step_window < frequency_sec:
                raise ValueError(
                    "Step window is supposed to be greater than TS granularity. "
                    f"TS granularity is: {frequency_sec} seconds. "
                    "Please provide a larger step window."
                )

        # if step_window is None, step_window = min(self.scan_window/2, *** + frequency_sec)
        # in order to make sure step_window >= frequency_sec, we need to make sure
        # self.scan_window >= 2 * frequency_sec

        if (
            self.scan_window < 2 * frequency_sec
            or self.historical_window < 2 * frequency_sec
        ):
            raise ValueError(
                "Scan window and historical window are supposed to be >= 2 * TS granularity. "
                f"TS granularity is: {frequency_sec} seconds. "
                "Please provide a larger scan window or historical_window."
            )

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        This function combines fit and predict and return anomaly socre for data. It
        requires scan_window > step_window.
        The relationship between two consective cusum runs in the loop is shown as below:

        >>> |---historical_window---|---scan_window---|
        >>>                                           |-step_window-|
        >>>               |---historical_window---|---scan_window---|

        # requirement: scan window > step window
        * scan_window: the window size in seconds to detect change point
        * historical_window: the window size in seconds to provide historical data
        * step_window: the window size in seconds to specify the step size between two scans

        Args:
            data: :class:`kats.consts.TimeSeriesData` object representing the data
            historical_data: :class:`kats.consts.TimeSeriesData` object representing the history.

        Returns:
            The anomaly response contains the anomaly scores.
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
        season_period_freq = self.season_period_freq

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
                _log.debug(f"freq_counts: {freq_counts}")
                raise ValueError(IRREGULAR_GRANULARITY_ERROR)

        # check if historical_window, scan_window, and step_window are suitable for given TSs
        self._check_window_sizes(frequency.total_seconds())

        if remove_seasonality:
            frequency_sec: int = int(frequency.total_seconds())
            frequency_sec_str = str(frequency_sec) + "s"

            # calculate resample base in second level
            # calculate remainder as resampling base
            resample_base_sec = (
                pd.to_datetime(historical_data.time[0]).day * 24 * 60 * 60
                + pd.to_datetime(historical_data.time[0]).hour * 60 * 60
                + pd.to_datetime(historical_data.time[0]).minute * 60
                + pd.to_datetime(historical_data.time[0]).second
            ) % frequency_sec

            decomposer_input = historical_data.interpolate(
                freq=frequency_sec_str,
                base=resample_base_sec,
            )

            data_time_idx = decomposer_input.time.isin(historical_data.time)
            if len(decomposer_input.time[data_time_idx]) != len(historical_data):
                raise ValueError(IRREGULAR_GRANULARITY_ERROR)

            # fixing the period to 24 hours as indicated in T81530775
            season_period_freq_multiplier = SEASON_PERIOD_FREQ_MAP[season_period_freq]
            period = int(
                24 * 60 * 60 * season_period_freq_multiplier / frequency.total_seconds()
            )

            if period < 2:
                period = 7

            decomposer = TimeSeriesDecomposition(
                decomposer_input,
                # statsmodels.STL requires that period must be a positive integer >= 2
                period=max(period, 2),
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                # statsmodels.STL requires that low_pass_jump must be a positive integer
                low_pass_jump=max(int((period + 1) * 0.15), 1),  # save run time
                seasonal_jump=1,
                # statsmodels.STL requires that trend_jump must be a positive integer
                trend_jump=max(int((period + 1) * 0.15), 1),  # save run time
            )

            decomp = decomposer.decomposer()
            historical_data_time_idx = decomp["rem"].time.isin(historical_data.time)

            historical_data.value = pd.Series(
                decomp["rem"][historical_data_time_idx].value
                + decomp["trend"][historical_data_time_idx].value,
                name=historical_data.value.name,
                copy=False,
            )

        smooth_window = int(scan_window.total_seconds() / frequency.total_seconds())
        if smooth_window > 1:
            smooth_historical_value = pd.Series(
                np.convolve(
                    historical_data.value.values, np.ones(smooth_window), mode="full"
                )[: 1 - smooth_window]
                / smooth_window,
                name=historical_data.value.name,
                copy=False,
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
            # Calling first _predict to poulate self.change_point_delta
            predict_results = self._predict(
                smooth_historical_data[-len(data) :], score_func
            )
            return AnomalyResponse(
                scores=predict_results.score,
                confidence_band=None,
                predicted_ts=None,
                anomaly_magnitude_ts=predict_results.absolute_change,
                stat_sig_ts=None,
            )
        anomaly_start_idx = self._time2idx(data, anomaly_start_time, "right")
        anomaly_start_time = data.time.iloc[anomaly_start_idx]
        score_tsd = self._zeros_ts(data[:anomaly_start_idx])
        change_tsd = self._zeros_ts(data[:anomaly_start_idx])

        if (
            historical_data.time.iloc[-1] - historical_data.time.iloc[0] + frequency
            <= scan_window
        ):
            # if len(all data) is smaller than scan data return zero score
            # Calling first _predict to poulate self.change_point_delta
            predict_results = self._predict(
                smooth_historical_data[-len(data) :], score_func
            )
            return AnomalyResponse(
                scores=predict_results.score,
                confidence_band=None,
                predicted_ts=None,
                anomaly_magnitude_ts=predict_results.absolute_change,
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
            _log.debug(f"start_time {start_time}")
            historical_start = self._time2idx(
                historical_data, start_time - historical_window, "right"
            )
            _log.debug(f"historical_start {historical_start}")
            historical_end = self._time2idx(historical_data, start_time, "right")
            _log.debug(f"historical_end {historical_end}")
            scan_end = self._time2idx(historical_data, start_time + step_window, "left")
            _log.debug(f"scan_end {scan_end}")
            in_data = historical_data[historical_end : scan_end + 1]
            if len(in_data) == 0:
                # skip if there is no data in the step_window
                continue
            in_hist = historical_data[historical_start:historical_end]
            self._fit(
                in_data,
                in_hist,
                scan_window=cast(Union[int, pd.Timedelta], scan_window),
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            predict_results = self._predict(
                smooth_historical_data[historical_end : scan_end + 1],
                score_func=score_func,
            )
            score_tsd.extend(
                predict_results.score,
                validate=False,
            )
            change_tsd.extend(predict_results.absolute_change, validate=False)

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
                scan_window=cast(Union[int, pd.Timedelta], scan_window),
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            predict_results = self._predict(
                smooth_historical_data[historical_end:scan_end],
                score_func=score_func,
            )
            score_tsd.extend(
                predict_results.score,
                validate=False,
            )
            change_tsd.extend(predict_results.absolute_change, validate=False)

        return AnomalyResponse(
            scores=score_tsd,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=change_tsd,
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

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        self.fit_predict(data, historical_data)

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")
