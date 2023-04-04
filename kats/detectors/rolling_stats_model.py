# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Union

import numpy as np
import pandas as pd
from kats.consts import DataError, InternalError, ParameterError, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse

from kats.utils.decomposition import SeasonalityHandler
from scipy import stats


SEASON_PERIOD_SUPPORTED: Set[str] = {
    "hourly",
    "daily",
    "weekly",
    "biweekly",
    "monthly",
    "yearly",
}


class RollStatsFunction(Enum):
    iqr = "iqr"
    mad = "mad"
    z_score = "z_score"
    modified_z_score_mad = "modified_z_score_mad"
    modified_z_score_iqr = "modified_z_score_iqr"
    iqr_median_deviation = "iqr_median_deviation"


DEFAULT_STATS_SCORE_FUNCTION: RollStatsFunction = RollStatsFunction.z_score

STR_TO_STATS_SCORE_FUNC: Dict[str, RollStatsFunction] = {
    # Used for param tuning
    val.name: val
    for val in RollStatsFunction
}


def calculate_iqr(data_list: np.ndarray, **kwargs: Any) -> Union[float, np.ndarray]:
    """
    Calculate IQR = Q3 - Q1
    """
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))
    q3, q1 = np.nanpercentile(data_list_2dim, [75, 25], axis=1)
    result = q3 - q1
    return result[0] if len(result) == 1 else result


def calculate_z_scores(
    data_list: np.ndarray, **kwargs: Any
) -> Union[float, np.ndarray]:
    """
    Calculate the z-score of the last data point in data_list.
    Or calculate the z-score of the last data point in each row of the data_list.
    if data in the window (one row) is all nan, then return nan for that row.
    """
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))

    m = np.nanmean(data_list_2dim[:, :-1], axis=1)
    d = np.nanstd(data_list_2dim[:, :-1], axis=1)

    numerator = data_list_2dim[:, -1] - m
    result = np.divide(numerator, d, out=np.zeros_like(numerator), where=d != 0)

    return result[0] if len(result) == 1 else result


def calculate_mad(data_list: np.ndarray, **kwargs: Any) -> Union[float, np.ndarray]:
    """
    Calculate MAD: the mean (average) distance between each data
    value and the mean of the data set.
    """
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))
    result = stats.median_abs_deviation(data_list_2dim, axis=1, nan_policy="omit")

    # pyre-ignore[6, 16]
    return result[0] if len(result) == 1 else result


def calculate_modified_z_scores_mad(
    data_list: np.ndarray, **kwargs: Any
) -> Union[float, np.ndarray]:
    """
    Calculate Modified z-score: (x-median)/MAD.
    x: the last point of data_list.
    median: median of data_list except the last point
    MAD: MAD of data_list except the last point
    """
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))
    m = np.nanmedian(data_list_2dim[:, :-1], axis=1)
    d = calculate_mad(data_list_2dim[:, :-1])

    numerator = data_list_2dim[:, -1] - m
    result = np.divide(numerator, d, out=np.zeros_like(numerator), where=d != 0)

    return result[0] if len(result) == 1 else result


def calculate_modified_z_scores_iqr(
    data_list: np.ndarray, **kwargs: Any
) -> Union[float, np.ndarray]:
    """
    Calculate Modified z-score (iqr version): (x-median)/IQR
    x: the last point of data_list.
    median: median of data_list except the last point
    IQR: IQR of data_list except the last point
    """
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))
    m = np.nanmedian(data_list_2dim[:, :-1], axis=1)
    d = calculate_iqr(data_list_2dim[:, :-1])

    numerator = data_list_2dim[:, -1] - m
    result = np.divide(numerator, d, out=np.zeros_like(numerator), where=d != 0)

    return result[0] if len(result) == 1 else result


def calculate_iqr_median_deviation(
    data_list: np.ndarray, **kwargs: Any
) -> Union[float, np.ndarray]:
    """
    Calculate IQR based median-deviation scores.

    median_deviation is defined as ABS(Current_value - Historic_median) / Historic_median.
    IQR is defined as Q3 - Q1.
    If the current point is >= Q3 + iqr_base * IQR, then it gets a postive multipler (+1),
    if the current point is < Q1 - iqr_base * IQR, then it gets a negative multipler (-1),
    otherwise, it gets a neutral multipler (0).

    Final results: multipler * median_deviation
    """

    iqr_base = kwargs.get("iqr_base", 1.5)
    data_list_2dim = data_list.reshape((-1, data_list.shape[-1]))
    m = np.nanmedian(data_list_2dim[:, :-1], axis=1)
    q3, q1 = np.nanpercentile(data_list_2dim[:, :-1], [75, 25], axis=1)

    iqr_range = iqr_base * (q3 - q1)
    ge_upper_bound = data_list_2dim[:, -1] >= (q3 + iqr_range)
    l_lower_bound = data_list_2dim[:, -1] < (q1 - iqr_range)
    neutral = 1 - (ge_upper_bound | l_lower_bound)

    iqr_sign = (-1) ** l_lower_bound * 0**neutral * (1) ** ge_upper_bound

    numerator = abs(data_list_2dim[:, -1] - m)
    result = iqr_sign * np.divide(
        numerator, m, out=np.zeros_like(numerator), where=m != 0
    )

    return result[0] if len(result) == 1 else result


# pyre-ignore[24]
SCORE_FUNC_DICT: Dict[str, Callable] = {
    RollStatsFunction.iqr.value: calculate_iqr,
    RollStatsFunction.z_score.value: calculate_z_scores,
    RollStatsFunction.mad.value: calculate_mad,
    RollStatsFunction.modified_z_score_mad.value: calculate_modified_z_scores_mad,
    RollStatsFunction.modified_z_score_iqr.value: calculate_modified_z_scores_iqr,
    RollStatsFunction.iqr_median_deviation.value: calculate_iqr_median_deviation,
}

_log: logging.Logger = logging.getLogger("rolling_Stats_model")


class RollingStatsModel(DetectorModel):
    """RollingStatsModel
    It includes calculating z-scores, IQR-scores, MAD-scores, IQR/MAD based modified z-scores,
    and IQR based median-deviation scores.

    Attributes:
        rolling_window: int. Either in terms of seconds or number of points.
            rolling window size for calculating MAD, IQR, std, average.
        serialized_model: Optional[bytes] = None.
        statistics: Union[str, RollStatsFunction]. Default is z_score.
        remove_seasonality: bool = False. Whether removing seasonality when running the detection model.
        point_based: bool = True. Matched with parameter rolling_window's unit.
        seasonality_period: str = "daily". Seasonality period for seasonality decomposition.
        score_base: float = 1.0. For modified z scores. Multiplier for the denominator.
        iqr_base: float. Default is 1.5. 1.5XIQR rule for outlier detection.

    Example:
    >>> model = RollingStatsModel(
            rolling_window=10,
            statistics="z_score",
            remove_seasonality=True,
            point_based=True,
        )
    >>> anom = model.fit_predict(historical_data=hist_ts, data=test_ts)
    >>> anom.scores.plot()
    """

    def __init__(
        self,
        rolling_window: int,
        serialized_model: Optional[bytes] = None,
        statistics: Union[str, RollStatsFunction] = DEFAULT_STATS_SCORE_FUNCTION,
        remove_seasonality: bool = False,
        point_based: bool = True,
        seasonality_period: str = "daily",
        score_base: float = 1.0,
        iqr_base: float = 1.5,
    ) -> None:
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.rolling_window: int = previous_model["rolling_window"]
            self.statistics: RollStatsFunction = previous_model["statistics"]
            self.remove_seasonality: bool = previous_model["remove_seasonality"]
            self.point_based: bool = previous_model["point_based"]
            self.seasonality_period: str = previous_model["seasonality_period"]
            self.score_base: float = previous_model["score_base"]
            self.iqr_base: float = previous_model["iqr_base"]

        else:
            self.rolling_window: int = rolling_window

            # We allow stats_score_function to be a str for compatibility with param tuning
            if isinstance(statistics, str):
                if statistics.lower() in STR_TO_STATS_SCORE_FUNC:
                    self.statistics: RollStatsFunction = STR_TO_STATS_SCORE_FUNC[
                        statistics.lower()
                    ]
                else:
                    self.statistics: RollStatsFunction = DEFAULT_STATS_SCORE_FUNCTION
                    _log.info(
                        "Invalid Statstics name. Using default score function: z_score."
                    )
            else:
                self.statistics: RollStatsFunction = statistics

            self.remove_seasonality: bool = remove_seasonality
            self.point_based: bool = point_based

            if seasonality_period not in SEASON_PERIOD_SUPPORTED:
                raise ParameterError(
                    f"Invalid seasonality period: {seasonality_period}. We are supporting {SEASON_PERIOD_SUPPORTED}."
                )
            self.seasonality_period: str = seasonality_period

            # For denominator == MAD, it should be 0.6745
            self.score_base: float = score_base

            # default is 1.5
            self.iqr_base: float = iqr_base

        # We're creating 2 types of scores:
        # IQR/MAD: rolling stats that are computed on a window of length w inclusive of the current point
        # Z-scores: scores on the current point compared against a mean/median and stddev
        #     or equivalent computed on a window of length w exclusive of the current point.
        # Thus, for Z_scores, rolling window will be extended by 1.
        if self.statistics.value in {"mad", "iqr"}:
            self.extend_rolling_window: bool = False
        else:
            self.extend_rolling_window: bool = True

    def serialize(self) -> bytes:
        """
        Retrun serilized model.
        """
        return str.encode(json.dumps(self.__dict__))

    def _point_based_vectorized_data(self, data: np.ndarray) -> np.ndarray:
        """
        For Z-scores and its variants,
        reshape the input data to shape (-1, self.rolling_window + 1)
        Each row consists of [history window | one datapoint to evaluate]

        For IQR and MAD,
        reshape the input data to shape (-1, self.rolling_window)
        """
        if self.extend_rolling_window:
            shape = data.shape[:-1] + (
                data.shape[-1] - self.rolling_window,
                self.rolling_window + 1,
            )
        else:
            shape = data.shape[:-1] + (
                data.shape[-1] - self.rolling_window + 1,
                self.rolling_window,
            )
        strides = data.strides + (data.strides[-1],)
        return np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides, writeable=False
        )

    def _fit_predict_point_based(
        self,
        data_interp: TimeSeriesData,
        n_points_hist_data: int,
    ) -> AnomalyResponse:
        if self.extend_rolling_window:
            rolling_window = self.rolling_window
        else:
            rolling_window = self.rolling_window - 1

        if n_points_hist_data <= rolling_window:
            data_value = data_interp.value.values
            reorg_data = self._point_based_vectorized_data(
                np.concatenate(
                    [
                        np.nan * np.zeros(rolling_window - n_points_hist_data),
                        data_value,
                    ],
                    0,
                )
            )
        else:
            data_value = data_interp.value.values[
                (n_points_hist_data - rolling_window) :
            ]
            reorg_data = self._point_based_vectorized_data(data_value)

        scores_array = (
            SCORE_FUNC_DICT[self.statistics.value](reorg_data, iqr_base=self.iqr_base)
            * self.score_base
        )
        scores_tsd = TimeSeriesData(
            time=data_interp.time[n_points_hist_data:], value=pd.Series(scores_array)
        )

        return AnomalyResponse(
            scores=scores_tsd,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=TimeSeriesData(
                time=scores_tsd.time, value=pd.Series(np.zeros(len(scores_tsd.time)))
            ),
            stat_sig_ts=None,
        )

    def _check_window_sizes(self, frequency_sec: int) -> None:
        if self.rolling_window / frequency_sec < 3:
            error_info = (
                "Please use a larger rolling window size, which at least "
                f"includes 3 data points in the time series data, i.e., rolling_window > {3*frequency_sec}."
            )
            _log.error(error_info)
            raise ParameterError(error_info)

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        if not data.is_univariate():
            raise DataError(
                "Multiple time series not supported for Rolling-Stats algorithm."
            )

        if historical_data and not historical_data.is_univariate():
            raise DataError(
                "Multiple time series not supported for Rolling-Stats algorithm."
            )

        # pull all the data in historical data
        if historical_data is not None:
            n_points_hist_data = len(historical_data)
            historical_data.extend(data, validate=False)
        else:
            # When historical_data is not provided, will use part of data as
            # historical_data, and fill with zero anomaly score.
            historical_data = data[:]
            n_points_hist_data = 0

        if self.remove_seasonality:
            sh_data = SeasonalityHandler(
                data=historical_data, seasonal_period=self.seasonality_period
            )
            historical_data = sh_data.remove_seasonality()

        if self.point_based:
            return self._fit_predict_point_based(
                data_interp=historical_data,
                n_points_hist_data=n_points_hist_data,
            )

        # pd.timedelta
        frequency = historical_data.infer_freq_robust()
        frequency_sec = int(frequency.total_seconds())
        self._check_window_sizes(frequency_sec)

        if (
            self.rolling_window % frequency_sec == 0
            and not historical_data.is_data_missing()
        ):
            self.rolling_window = self.rolling_window // frequency_sec
            return self._fit_predict_point_based(
                data_interp=historical_data,
                n_points_hist_data=n_points_hist_data,
            )

        else:
            total_data_df = historical_data.to_dataframe()
            total_data_df.columns = ["time", "value"]
            total_data_df = total_data_df.set_index("time")

            scores = []
            for i in range(len(data.time)):
                # TODO: this loop could be further optimized by vectorization
                test_end_dt = pd.to_datetime(data.time[i])
                if self.extend_rolling_window:
                    test_start_dt = test_end_dt - pd.Timedelta(
                        str(self.rolling_window) + "s"
                    )
                else:
                    test_start_dt = test_end_dt - pd.Timedelta(
                        str(self.rolling_window - frequency_sec) + "s"
                    )
                window_data_array = total_data_df.loc[
                    test_start_dt:test_end_dt
                ].value.values
                if len(window_data_array) >= 2:
                    score = (
                        SCORE_FUNC_DICT[self.statistics.value](window_data_array)
                        * self.score_base
                    )
                else:
                    score = 0
                scores.append(score)

            scores_tsd = TimeSeriesData(
                pd.DataFrame({"time": data.time, "value": scores})
            )

            return AnomalyResponse(
                scores=scores_tsd,
                confidence_band=None,
                predicted_ts=None,
                anomaly_magnitude_ts=TimeSeriesData(
                    time=data.time, value=pd.Series([0] * len(data.time))
                ),
                stat_sig_ts=None,
            )

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """
        fit is not implemented
        """
        raise InternalError("fit is not implemented, call fit_predict() instead")

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
