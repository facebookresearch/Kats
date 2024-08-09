# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
from typing import Any, cast, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from kats.consts import (
    DataIrregularGranularityError,
    DEFAULT_VALUE_NAME,
    InternalError,
    IRREGULAR_GRANULARITY_ERROR,
    ParameterError,
    TimeSeriesData,
)
from kats.detectors.cusum_detection import (
    CUSUMChangePoint,
    CUSUMDefaultArgs,
    CUSUMDetector,
    VectorizedCUSUMDetector,
)
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from kats.utils.decomposition import SeasonalityHandler


NORMAL_TOLERENCE = 1  # number of window
CHANGEPOINT_RETENTION: int = 7 * 24 * 60 * 60  # in seconds
MAX_CHANGEPOINT = 10

_log: logging.Logger = logging.getLogger("cusum_model")


def percentage_change(
    data: TimeSeriesData, pre_mean: Union[float, pd.Series], **kwargs: Any
) -> TimeSeriesData:
    """
    Calculate percentage change absolute change / baseline change

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
    """
    if type(data.value) == pd.DataFrame and data.value.shape[1] > 1:
        res = (data.value - pre_mean) / (pre_mean)
        return TimeSeriesData(value=res, time=data.time)
    else:
        return (data - pre_mean) / (pre_mean)


def change(
    data: TimeSeriesData, pre_mean: Union[float, pd.Series], **kwargs: Any
) -> TimeSeriesData:
    """
    Calculate absolute change

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
    """
    if type(data.value) == pd.DataFrame and data.value.shape[1] > 1:
        res = data.value - pre_mean
        return TimeSeriesData(value=res, time=data.time)
    else:
        return data - pre_mean


def z_score(
    data: TimeSeriesData,
    pre_mean: Union[float, pd.Series],
    pre_std: Union[float, pd.Series],
) -> TimeSeriesData:
    """
    Calculate z score
    The formula for calculating a z-score is is z = (x-mu)/sigma,
    where x is the raw score, mu is the population mean,
    and sigma is the population standard deviation.

    population standard deviation -> ddof = 0

    Args:
        data: The data need to calculate the score
        pre_mean: Baseline mean
        pre_std: Baseline std
    """
    if type(data.value) == pd.DataFrame and data.value.shape[1] > 1:
        res = (data.value - pre_mean) / (pre_std)
        return TimeSeriesData(value=res, time=data.time)
    else:
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
        vectorized: bool, transfer to multi-ts and call vectorized cusum model
        adapted_pre_mean: bool, whether using a rolling pre-mean and pre-std when calculating
            anomaly scores (when alert_fired = True)
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
        season_period_freq: Union[str, int] = "daily",
        vectorized: Optional[bool] = None,
        adapted_pre_mean: Optional[bool] = None,
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

            self.season_period_freq: Union[str, int] = previous_model.get(
                "season_period_freq", "daily"
            )

            # If vectorized is provided, it should supersede existing values
            if vectorized is not None:
                self.vectorized: bool = vectorized
            else:
                self.vectorized: bool = previous_model.get("vectorized", False)

            # If adapted_pre_mean is provided, it should supersede existing values
            if adapted_pre_mean is not None:
                self.adapted_pre_mean: bool = adapted_pre_mean
            else:
                self.adapted_pre_mean: bool = previous_model.get(
                    "adapted_pre_mean", False
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

            self.vectorized: bool = vectorized or False
            self.adapted_pre_mean: bool = adapted_pre_mean or False

        else:
            raise ParameterError(
                "You must provide either serialized model or values for "
                "scan_window and historical_window."
            )

        self.vectorized_trans_flag: bool = False

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
        self,
        cur_mean: float,
        change_directions: Optional[List[str]],
        delta_std_ratio: float = CUSUMDefaultArgs.delta_std_ratio,
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
            check_decrease = delta_std_ratio if decrease else np.inf
        elif self.alert_change_direction == "decrease":
            check_increase = delta_std_ratio if increase else np.inf
            check_decrease = 0 if decrease else np.inf

        return (
            # pyre-fixme[61]: `check_decrease` is undefined, or not always defined.
            self.pre_mean - check_decrease * self.pre_std
            <= cur_mean
            # pyre-fixme[61]: `check_increase` is undefined, or not always defined.
            <= self.pre_mean + check_increase * self.pre_std
        )

    def _fit_vec_row(
        self,
        vec_data_row: TimeSeriesData,
        # pyre-fixme[11]: Annotation `Timedelta` is not defined as a type.
        scan_window: Union[int, pd.Timedelta],
        changepoints: List[CUSUMChangePoint],  # len = 1 or 0
        time_adjust: pd.Timedelta,
        change_directions: Optional[List[str]] = CUSUMDefaultArgs.change_directions,
        delta_std_ratio: float = CUSUMDefaultArgs.delta_std_ratio,
    ) -> List[int]:
        scan_start_time = vec_data_row.time.iloc[-1] - pd.Timedelta(
            scan_window, unit="s"
        )
        scan_start_index = max(
            0, np.argwhere((vec_data_row.time >= scan_start_time).values).min()
        )

        # need to update vec_cusum.cps
        res_cp = []
        if not self.alert_fired:
            if len(changepoints) > 0:
                cp = changepoints[0]
                self.cps.append((cp.start_time + time_adjust).value // 10**9)
                res_cp.append(cp.start_time.value // 10**9)
                if len(self.cps) > MAX_CHANGEPOINT:
                    self.cps.pop(0)

                self._set_alert_on(
                    vec_data_row.value[: cp.cp_index + 1].mean(),
                    # Note: std() from Pandas has default ddof=1, while std() from numpy has default ddof=0
                    vec_data_row.value[: cp.cp_index + 1].std(ddof=0),
                    cp.direction,
                )
        else:
            cur_mean = vec_data_row[scan_start_index:].value.mean()

            if self.adapted_pre_mean:
                self.pre_mean = vec_data_row.value[:scan_start_index].mean()
                self.pre_std = vec_data_row.value[:scan_start_index].std(ddof=0)

            if self._if_normal(cur_mean, change_directions, delta_std_ratio):
                self.number_of_normal_scan += 1
                if self.number_of_normal_scan >= NORMAL_TOLERENCE:
                    self._set_alert_off()
            else:
                self.number_of_normal_scan = 0

            current_time = int((vec_data_row.time.max() + time_adjust).value / 1e9)
            if current_time - self.cps[-1] > CHANGEPOINT_RETENTION:
                self._set_alert_off()

        return res_cp

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
                    # Note: std() from Pandas has default ddof=1, while std() from numpy has default ddof=0
                    historical_data.value[: cp.cp_index + 1].std(ddof=0),
                    cp.direction,
                )

        else:
            cur_mean = historical_data[scan_start_index:].value.mean()

            if self.adapted_pre_mean:
                self.pre_mean = historical_data.value[:scan_start_index].mean()
                self.pre_std = historical_data.value[:scan_start_index].std(ddof=0)

            if self._if_normal(cur_mean, change_directions, delta_std_ratio):
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
                raise ParameterError(
                    "Step window is supposed to be smaller than scan window to ensure we "
                    "have overlap for scan windows."
                )
            if self.step_window < frequency_sec:
                raise ParameterError(
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
            raise ParameterError(
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
            org_hist_len = len(historical_data)
            historical_data = historical_data[:]
            historical_data.extend(data, validate=False)
        else:
            # When historical_data is not provided, will use part of data as
            # historical_data, and fill with zero anomaly score.
            historical_data = data[:]
            org_hist_len = 0

        frequency = historical_data.freq_to_timedelta()
        if frequency is None or frequency is pd.NaT:
            # Use the top frequency if any, when not able to infer from data.
            freq_counts = (
                historical_data.time.diff().value_counts().sort_values(ascending=False)
            )
            if freq_counts.iloc[0] >= int(len(historical_data)) * 0.5 - 1:
                frequency = freq_counts.index[0]
            else:
                _log.debug(f"freq_counts: {freq_counts}")
                raise DataIrregularGranularityError(IRREGULAR_GRANULARITY_ERROR)

        # check if historical_window, scan_window, and step_window are suitable for given TSs
        frequency_sec = frequency.total_seconds()
        self._check_window_sizes(frequency_sec)

        if remove_seasonality:
            sh_data = SeasonalityHandler(
                data=historical_data, seasonal_period=season_period_freq
            )
            historical_data = sh_data.remove_seasonality()

        smooth_window = int(scan_window.total_seconds() / frequency_sec)
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
            < scan_window
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

        # if need trans to multi-TS
        # TS needs to have regular granularity, otherwise cannot transfer uni-TS to multi-TS, because
        # columns might have different length

        n_hist_win_pts = int(
            np.ceil(historical_window.total_seconds() / frequency_sec)
        )  # match _time2idx --- right

        multi_ts_len = int(
            np.ceil(
                (historical_window.total_seconds() + step_window.total_seconds())
                / frequency_sec
            )
        )  # match _time2idx --- left + 1
        n_step_win_pts = multi_ts_len - n_hist_win_pts
        multi_dim = (
            len(historical_data[anomaly_start_idx + org_hist_len :]) // n_step_win_pts
        )

        if self.vectorized:
            if (
                step_window.total_seconds() % frequency_sec == 0
                and historical_window.total_seconds() % frequency_sec
                == 0  # otherwise in the loop around row 715, each iteration might have slightly different data length
                and pd.infer_freq(historical_data.time.values)
                is not None  # regular granularity
                and multi_dim >= 2
            ):
                self.vectorized_trans_flag = True
                _log.info("Using VectorizedCUSUMDetectorModel.")
            else:
                self.vectorized_trans_flag = False
                _log.info("Cannot transfer to multi-variate TS.")

        else:
            self.vectorized_trans_flag = False

        # if need trans to multi-TS
        if self.vectorized_trans_flag:
            end_idx = anomaly_start_idx + org_hist_len + multi_dim * n_step_win_pts
            new_historical_data = self._reorganize_big_data(
                historical_data[
                    max(anomaly_start_idx - n_hist_win_pts + org_hist_len, 0) : end_idx
                ],
                multi_ts_len,
                n_step_win_pts,
            )

            new_smooth_historical_data = self._reorganize_big_data(
                smooth_historical_data[
                    max(anomaly_start_idx - n_hist_win_pts + org_hist_len, 0) : end_idx
                ],
                multi_ts_len,
                n_step_win_pts,
            )

            # remaining_part = historical_data[end_idx:]
            ss_detect = VectorizedCUSUMDetectorModel(
                scan_window=self.scan_window,
                historical_window=self.historical_window,
                step_window=step_window.total_seconds(),
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
                score_func=score_func,
                remove_seasonality=False,  # already removed
                adapted_pre_mean=False,
            )

            column_index = new_historical_data.value.columns
            # pyre-ignore
            ss_detect.alert_fired: pd.Series = pd.Series(False, index=column_index)
            # pyre-ignore
            ss_detect.pre_mean: pd.Series = pd.Series(0, index=column_index)
            # pyre-ignore
            ss_detect.pre_std: pd.Series = pd.Series(1, index=column_index)
            # pyre-ignore
            ss_detect.alert_change_direction: pd.Series = pd.Series(
                "None", index=column_index
            )
            # pyre-ignore
            ss_detect.number_of_normal_scan: pd.Series = pd.Series(
                0, index=column_index
            )
            ss_detect.cps = [[] for _ in range(len(column_index))]
            ss_detect.cps_meta = [[] for _ in range(len(column_index))]

            ss_detect._fit(
                data=TimeSeriesData(),
                historical_data=new_historical_data,
                scan_window=cast(Union[int, pd.Timedelta], scan_window),
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )

            for c in range(new_historical_data.value.shape[1]):
                in_data = TimeSeriesData(
                    time=new_historical_data.time,
                    value=new_historical_data.value.iloc[:, c],
                )

                res_cp = self._fit_vec_row(
                    vec_data_row=in_data,
                    scan_window=cast(Union[int, pd.Timedelta], scan_window),
                    changepoints=ss_detect.cps_meta[c],
                    time_adjust=pd.Timedelta(c * step_window, "s"),
                    change_directions=change_directions,
                    delta_std_ratio=delta_std_ratio,
                )
                ss_detect.pre_mean[c] = self.pre_mean
                ss_detect.pre_std[c] = self.pre_std
                ss_detect.alert_fired[c] = self.alert_fired
                ss_detect.cps[c] = res_cp

            predict_results = ss_detect._predict(
                new_smooth_historical_data[-n_step_win_pts:],
                score_func=score_func,
            )
            score_tsd_vec, change_tsd_vec = self._reorganize_back(
                predict_results.score,
                predict_results.absolute_change,
                historical_data.value.name,
            )
            score_tsd.extend(
                score_tsd_vec,
                validate=False,
            )
            change_tsd.extend(change_tsd_vec, validate=False)

        else:

            for start_time in pd.date_range(
                anomaly_start_time,
                min(
                    data.time.iloc[-1]
                    + frequency
                    - step_window,  # to include last data point
                    data.time.iloc[
                        -1
                    ],  # make sure start_time won't beyond last data time
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
                scan_end = self._time2idx(
                    historical_data, start_time + step_window, "left"
                )
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

        score_tsd.time = data.time
        change_tsd.time = data.time

        return AnomalyResponse(
            scores=score_tsd,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=change_tsd,
            stat_sig_ts=None,
        )

    def _reorganize_big_data(
        self,
        org_data: TimeSeriesData,
        multi_ts_len: int,
        n_step_win_pts: int,
    ) -> TimeSeriesData:
        multi_ts_time_df = org_data[:multi_ts_len].time.copy()
        multi_ts_val = [list(org_data[:multi_ts_len].value)]
        for i in range(multi_ts_len, len(org_data), n_step_win_pts):
            multi_ts_val.append(
                list(
                    org_data[
                        i - multi_ts_len + n_step_win_pts : i + n_step_win_pts
                    ].value
                )
            )

        multi_ts_val_df = pd.DataFrame(multi_ts_val).T

        multi_ts_df = pd.concat([multi_ts_time_df, multi_ts_val_df], 1)
        df_names = ["val_" + str(i) for i in range(multi_ts_val_df.shape[1])]
        multi_ts_df.columns = ["time"] + df_names

        return TimeSeriesData(multi_ts_df)

    def _reorganize_back(
        self,
        scores: TimeSeriesData,
        magnitude_ts: TimeSeriesData,
        name: str,
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        anom_scores_val_array = np.asarray(scores.value)
        anom_mag_val_array = np.asarray(magnitude_ts.value)
        freq = scores.time[1] - scores.time[0]
        time_need = pd.date_range(
            start=scores.time.iloc[0],
            end=None,
            periods=anom_scores_val_array.shape[0] * anom_scores_val_array.shape[1],
            freq=freq,
        )

        anom_scores_val_1d = pd.Series(
            anom_scores_val_array.T.reshape([-1]),
            name=name,
        )

        anom_scores_ts = TimeSeriesData(time=time_need, value=anom_scores_val_1d)

        anom_mag_val_1d = pd.Series(
            anom_mag_val_array.T.reshape([-1]),
            name=name,
        )

        anom_mag_ts = TimeSeriesData(time=time_need, value=anom_mag_val_1d)

        return anom_scores_ts, anom_mag_ts

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
            raise InternalError("direction can only be right or left")

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
        raise InternalError("predict is not implemented, call fit_predict() instead")


class VectorizedCUSUMDetectorModel(CUSUMDetectorModel):
    """VectorizedCUSUMDetectorModel detects change points for multivariate input timeseries
    in vectorized form. The logic is based on CUSUMDetectorModel, and runs
    VectorizedCUSUMDetector.

    VectorizedCUSUMDetectorModel runs VectorizedCUSUMDetector multiple times to
    detect multiple change points.
    In each run, CUSUMDetector will use historical_window + scan_window as
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
        season_period_freq: str = "daily" for seasonaly decomposition.
        adapted_pre_mean: bool, whether using a rolling pre-mean and pre-std when calculating
            anomaly scores (when alert_fired = True).
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
        change_directions: Optional[List[str]] = CUSUMDefaultArgs.change_directions,
        score_func: Union[str, CusumScoreFunction] = DEFAULT_SCORE_FUNCTION,
        remove_seasonality: bool = CUSUMDefaultArgs.remove_seasonality,
        season_period_freq: str = "daily",
        adapted_pre_mean: Optional[bool] = None,
    ) -> None:
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.cps: List[List[int]] = previous_model["cps"]
            self.cps_meta: List[List[CUSUMChangePoint]] = previous_model["cps_meta"]
            self.alert_fired: pd.Series = previous_model["alert_fired"]
            self.pre_mean: pd.Series = previous_model["pre_mean"]
            self.pre_std: pd.Series = previous_model["pre_std"]
            self.number_of_normal_scan: pd.Series = previous_model[
                "number_of_normal_scan"
            ]
            self.alert_change_direction: pd.Series = previous_model[
                "alert_change_direction"
            ]
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

            # If adapted_pre_mean is provided, it should supersede existing values
            if adapted_pre_mean is not None:
                self.adapted_pre_mean: bool = adapted_pre_mean
            else:
                self.adapted_pre_mean: bool = previous_model.get(
                    "adapted_pre_mean", False
                )

        elif scan_window is not None and historical_window is not None:
            self.serialized_model: Optional[bytes] = serialized_model
            self.scan_window: int = scan_window
            self.historical_window: int = historical_window
            self.step_window: int = cast(int, step_window)
            self.threshold: float = threshold
            self.delta_std_ratio: float = delta_std_ratio
            self.magnitude_quantile: Optional[float] = magnitude_quantile
            self.magnitude_ratio: float = magnitude_ratio

            if isinstance(change_directions, str):
                self.change_directions = [change_directions]
            else:
                self.change_directions = change_directions

            self.remove_seasonality: bool = remove_seasonality
            self.season_period_freq = season_period_freq

            # We allow score_function to be a str for compatibility with param tuning
            if isinstance(score_func, str):
                if score_func in STR_TO_SCORE_FUNC:
                    score_func = STR_TO_SCORE_FUNC[score_func]
                else:
                    score_func = DEFAULT_SCORE_FUNCTION
            self.score_func: CusumScoreFunction = score_func.value
            self.adapted_pre_mean: bool = adapted_pre_mean or False

        else:
            raise ParameterError(
                "You must provide either serialized model or values for "
                "scan_window and historical_window."
            )

    def _set_alert_off_multi_ts(self, set_off_mask: pd.Series) -> None:
        self.alert_fired &= ~set_off_mask
        self.number_of_normal_scan[set_off_mask] = 0

    def _set_alert_on_multi_ts(
        self,
        set_on_mask: pd.Series,
        baseline_mean: pd.Series,
        baseline_std: pd.Series,
        alert_change_direction: pd.Series,
    ) -> None:
        self.alert_change_direction[set_on_mask] = alert_change_direction
        self.pre_mean[set_on_mask] = baseline_mean.combine_first(self.pre_mean)
        self.pre_std[set_on_mask] = baseline_std.combine_first(self.pre_std)

    def _if_back_to_normal(
        self,
        cur_mean: pd.Series,
        change_directions: Optional[List[str]],
        delta_std_ratio: float = CUSUMDefaultArgs.delta_std_ratio,
    ) -> pd.Series:
        if change_directions is not None:
            increase, decrease = (
                "increase" in change_directions,
                "decrease" in change_directions,
            )
        else:
            increase, decrease = True, True
        check_increase = np.array([])
        check_decrease = np.array([])
        for x in self.alert_change_direction:
            cur_increase = cur_decrease = 0
            if x == "increase":
                cur_increase = 0 if increase else np.inf
                cur_decrease = delta_std_ratio if decrease else np.inf
            elif x == "decrease":
                cur_increase = delta_std_ratio if increase else np.inf
                cur_decrease = 0 if decrease else np.inf
            check_increase = np.append(check_increase, cur_increase)
            check_decrease = np.append(check_decrease, cur_decrease)
        return (self.pre_mean - check_decrease * self.pre_std <= cur_mean) * (
            cur_mean <= self.pre_mean + check_increase * self.pre_std
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
        if len(data) > 0:
            historical_data.extend(data, validate=False)
        n = len(historical_data)
        scan_start_time = historical_data.time.iloc[-1] - pd.Timedelta(
            scan_window, unit="s"
        )
        scan_start_index = max(
            0, np.argwhere((historical_data.time >= scan_start_time).values).min()
        )

        n_pts = historical_data.value.shape[0]

        # separate into two cases: alert off and alert on

        # if scan window is less than 2 data poins and there is no alert fired
        # skip this scan
        alert_set_on_mask = (~self.alert_fired[:]) & (n - scan_start_index > 1)
        alert_set_off_mask = self.alert_fired.copy()

        # [Case 1] if alert is off (alert_fired[i] is False)
        if alert_set_on_mask.any():
            # Use VectorizedCUSUMDetector
            detector = VectorizedCUSUMDetector(historical_data)
            changepoints = detector.detector(
                interest_window=[scan_start_index, n],
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )

            cps = [
                (
                    sorted(x, key=lambda x: x.start_time)[0]
                    if x and alert_set_on_mask[i]
                    else None
                )
                for i, x in enumerate(changepoints)
            ]

            alert_set_on_mask &= np.array([x is not None for x in cps])

            # mask1 is used to calculate avg and std with different changepoint index
            mask1 = np.tile(alert_set_on_mask, (n_pts, 1))
            for i, cp in enumerate(cps):
                if cp is not None:
                    mask1[(cp.cp_index + 1) :, i] = False
                    self.cps[i].append(int(cp.start_time.value / 1e9))
                    self.cps_meta[i].append(cp)
                    if len(self.cps[i]) > MAX_CHANGEPOINT:
                        self.cps[i].pop(0)
                        self.cps_meta[i].pop(0)

            avg = np.divide(
                np.sum(np.multiply(historical_data.value, mask1), axis=0),
                np.sum(mask1, axis=0),
            )
            std = np.sqrt(
                np.divide(
                    np.sum(
                        np.multiply(np.square(historical_data.value - avg), mask1),
                        axis=0,
                    ),
                    np.sum(mask1, axis=0),
                )
            )
            self._set_alert_on_multi_ts(
                alert_set_on_mask,
                avg,
                std,
                # pyre-ignore
                [x.direction if x else None for x in cps],
            )
            self.alert_fired |= np.array([x is not None for x in cps])

        # [Case 2] if alert is on (alert_fired[i] is True)
        # set off alert when:
        # [2.1] the mean of current dataset is back to normal and num_normal_scan >= NORMAL_TOLERENCE
        # [2.2] alert retention > CHANGEPOINT_RETENTION

        if alert_set_off_mask.any():
            mask2 = np.tile(alert_set_off_mask, (n_pts, 1))
            mask2[:scan_start_index, :] = False
            cur_mean = np.divide(
                np.sum(np.multiply(historical_data.value, mask2), axis=0),
                np.sum(mask2, axis=0),
            )
            if self.adapted_pre_mean:
                mask3 = np.tile(alert_set_off_mask, (n_pts, 1))
                mask3[scan_start_index:, :] = False
                pre_mean = np.divide(
                    np.sum(np.multiply(historical_data.value, mask3), axis=0),
                    np.sum(mask3, axis=0),
                )

                pre_std = np.sqrt(
                    np.divide(
                        np.sum(
                            np.multiply(
                                np.square(historical_data.value - pre_mean), mask3
                            ),
                            axis=0,
                        ),
                        np.sum(mask3, axis=0),
                    )
                )

                self.pre_mean[alert_set_off_mask] = pre_mean.combine_first(
                    self.pre_mean
                )
                self.pre_std[alert_set_off_mask] = pre_std.combine_first(self.pre_std)

            is_normal = self._if_back_to_normal(
                cur_mean, change_directions, delta_std_ratio
            )
            # if current mean is normal, num_normal_scan increment 1, if not, num_normal_scan set to 0
            self.number_of_normal_scan += is_normal
            # set off alert
            # [case 2.1]
            tmp1 = self.number_of_normal_scan >= NORMAL_TOLERENCE
            self._set_alert_off_multi_ts(alert_set_off_mask & tmp1 & is_normal)
            self.number_of_normal_scan[~is_normal] = 0

            # [case 2.2]
            current_time = int(data.time.max().value / 1e9)
            tmp2 = np.asarray(
                [
                    current_time - x[-1] > CHANGEPOINT_RETENTION if len(x) else False
                    for x in self.cps
                ]
            )
            self._set_alert_off_multi_ts(alert_set_off_mask & tmp2)

    def _predict(
        self,
        data: TimeSeriesData,
        score_func: CusumScoreFunction = CusumScoreFunction.change,
    ) -> PredictFunctionValues:
        """
        data: the new data for the anoamly score calculation.
        """
        cp = [x[-1] if len(x) else None for x in self.cps]
        tz = data.tz()
        if tz is None:
            change_time = [pd.to_datetime(x, unit="s") if x else None for x in cp]
        else:
            change_time = [
                pd.to_datetime(x, unit="s", utc=True).tz_convert(tz) if x else None
                for x in cp
            ]
        n_pts = data.value.shape[0]
        first_ts = data.time.iloc[0]
        cp_index = [
            data.time.index[data.time == x][0] if x and x >= first_ts else None
            for x in change_time
        ]
        ret = PredictFunctionValues(
            SCORE_FUNC_DICT[score_func](
                data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
            ),
            SCORE_FUNC_DICT[CusumScoreFunction.change.value](
                data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
            ),
        )
        # in the following 2 cases, fill score with 0 by mask 0:
        # (i) if no alert fired for a timeseries (change_time is None)
        # (ii) if alert fired, fill 0 for time index before cp_index
        set_zero_mask = np.tile(~self.alert_fired, (n_pts, 1))
        for i, c in enumerate(cp_index):
            if c is not None:
                set_zero_mask[: (c + 1), i] = True
        ret.score.value[set_zero_mask] = 0
        ret.absolute_change.value[set_zero_mask] = 0
        return ret

    def _zeros_ts(self, data: TimeSeriesData) -> TimeSeriesData:
        if len(data) > 0:
            return TimeSeriesData(
                time=data.time,
                value=pd.DataFrame(
                    np.zeros(data.value.shape), columns=data.value.columns
                ),
            )
        else:
            return TimeSeriesData()

    def run_univariate_cusumdetectormodel(
        self, data: TimeSeriesData, historical_data: TimeSeriesData
    ) -> AnomalyResponse:
        d = CUSUMDetectorModel(
            serialized_model=self.serialized_model,
            scan_window=self.scan_window,
            historical_window=self.historical_window,
            step_window=self.step_window,
            threshold=self.threshold,
            delta_std_ratio=self.delta_std_ratio,
            magnitude_quantile=self.magnitude_quantile,
            magnitude_ratio=self.magnitude_ratio,
            change_directions=self.change_directions,
            score_func=self.score_func,
            remove_seasonality=self.remove_seasonality,
        )
        return d.fit_predict(data, historical_data)

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
        # init parameters after getting input data
        num_timeseries = data.value.shape[1] if type(data.value) == pd.DataFrame else 1
        if num_timeseries == 1:
            _log.info(
                "Input timeseries is univariate. CUSUMDetectorModel is preferred."
            )
            assert historical_data is not None
            return self.run_univariate_cusumdetectormodel(data, historical_data)

        self.alert_fired: pd.Series = pd.Series(False, index=data.value.columns)
        self.pre_mean: pd.Series = pd.Series(0, index=data.value.columns)
        self.pre_std: pd.Series = pd.Series(1, index=data.value.columns)
        self.alert_change_direction: pd.Series = pd.Series(
            "None", index=data.value.columns
        )
        self.number_of_normal_scan: pd.Series = pd.Series(0, index=data.value.columns)
        self.cps = [[] for _ in range(num_timeseries)]
        self.cps_meta = [[] for _ in range(num_timeseries)]
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
            if freq_counts.iloc[0] >= int(len(historical_data)) * 0.5 - 1:
                frequency = freq_counts.index[0]
            else:
                _log.debug(f"freq_counts: {freq_counts}")
                raise DataIrregularGranularityError(IRREGULAR_GRANULARITY_ERROR)

        # check if historical_window, scan_window, and step_window are suitable for given TSs
        self._check_window_sizes(frequency.total_seconds())

        if remove_seasonality:
            sh_data = SeasonalityHandler(
                data=historical_data, seasonal_period=season_period_freq
            )
            historical_data = sh_data.remove_seasonality()

        smooth_window = int(scan_window.total_seconds() / frequency.total_seconds())
        if smooth_window > 1:
            smooth_historical_value = (
                historical_data.value.apply(
                    lambda x: np.convolve(x, np.ones(smooth_window), mode="full"),
                    axis=0,
                )[: 1 - smooth_window]
                / smooth_window
            )
            smooth_historical_value = cast(pd.DataFrame, smooth_historical_value)
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
            < scan_window
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

        # rolling window
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
