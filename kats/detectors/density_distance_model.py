# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
from typing import Any, Optional, Set, Tuple

import numpy as np
import pandas as pd
from kats.consts import (
    DataError,
    DataIrregularGranularityError,
    InternalError,
    ParameterError,
    TimeSeriesData,
)

from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse
from scipy.spatial import distance

from scipy.spatial.distance import _METRIC_INFOS


# Supported metrics for calculating distance
# details: https://github.com/scipy/scipy/blob/v1.10.1/scipy/spatial/distance.py#L2672
SUPPORTED_DISTANCE_METRICS: Set[str] = set(
    {info.canonical_name for info in _METRIC_INFOS}
)


_log: logging.Logger = logging.getLogger("density_distance_model")


def _merge_percentile(l1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    handle equal percentile:
    [-2.5, -1.3, -1.3, 1.2, 1.2] -> [-2.5, -1.3, 1.2] with prob [0.2, 0.4, 0.4]
    """
    l1_perc = []
    l1_merge = []
    n_perc = 1.0 / len(l1)
    for i in range(len(l1)):
        if len(l1_perc) == 0 or l1[i] > l1[i - 1]:
            l1_perc.append(n_perc)
            l1_merge.append(l1[i])
        else:
            l1_perc[-1] += n_perc
    return np.array(l1_merge), np.array(l1_perc)


def _percentile_to_prob(
    l1: np.ndarray, l2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    convert decile to probability density: ([3,4,5,6,7], [1,2,3,4,5])
    to ([0.2, 0.2, 0.2, 0.2, 0.2], [0.6, 0.2, 0.2, 0, 0])
    """
    assert len(l1) == len(l2) and len(l1) > 1
    n_perc = 1.0 / len(l1)
    l1_merge, l1_perc = _merge_percentile(l1)
    l2_merge, l2_perc = _merge_percentile(l2)

    # make the final compare list as long as possible
    b_reverse_flag = False
    if len(l1_merge) < len(l2_merge):
        b_reverse_flag = True
        l1_merge, l2_merge = l2_merge, l1_merge
        l1_perc, l2_perc = l2_perc, l1_perc

    # make perc of l2 to cumulative
    l2_perc = np.cumsum(l2_perc)

    # cumulative result (cmf) for l2
    l_res_2 = np.array([1.0] * len(l1_merge))

    for i in range(len(l1_merge)):
        n_idx_at2 = np.where(l1_merge[i] < l2_merge)[0]

        # if no value in l2 is larger than l1
        if len(n_idx_at2) == 0:
            break

        n_idx_at2 = n_idx_at2[0]

        # if all values in l2 are lager than l1
        if n_idx_at2 == 0:

            # use distance from l2[0] to l2[j] as base
            j = 1
            while j < len(l2_merge) and l2_merge[j] == l2_merge[0]:
                j += 1

            if j < len(l2_merge):
                n_base_distance = l2_merge[j] - l2_merge[0]  # positive number
                n_distance_1_2 = l1_merge[i] - l2_merge[0]  # negative number
                n_start = 0
            else:
                l_res_2[i] = 0
                continue

        else:
            n_base_distance = (
                l2_merge[n_idx_at2] - l2_merge[n_idx_at2 - 1]
            )  # positive number
            n_distance_1_2 = l1_merge[i] - l2_merge[n_idx_at2 - 1]  # positive number
            n_start = l2_perc[n_idx_at2 - 1]

        l_res_2[i] = n_start + n_perc * n_distance_1_2 / n_base_distance
        if l_res_2[i] < 0:
            l_res_2[i] = 0

    if len(l_res_2) > 1:
        l_res_2 = [l_res_2[0]] + list(l_res_2[1:] - l_res_2[0:-1])

    if b_reverse_flag:
        return np.asarray(l_res_2), l1_perc
    else:
        return l1_perc, np.asarray(l_res_2)


class DensityDistanceModel(DetectorModel):
    """DensityDistanceModel

    The input of this algorithm is multivariate time series data.
    For non-density-based distance calculation, it doesn't have requirements
        for each data point (a vector).
    For density-based distance calculation, each data point of the input should
        be a non-decreasing vector, usually percentiles. For example, if input is
        deciles, then each data point would be a vector with length 10.

    The algorithm performs distance calculation as a multivariate analysis over
        the input data between the current data point and a point in the past
        -- Distance(current, current - window_size).

    Attributes:
        window_size: int, in terms of seconds.
        serialized_model: bytes, optional.
        distance_metric: str. Default is "jensenshannon".
        density_based_distance: bool, optional. True when distance_metric is "jensenshannon".
        validate_monotonic: bool, optional. True when distance_metric is "jensenshannon".
        jsd_base : float, optional. The base of the logarithm used to compute the output
            if not given, then the routine uses the default base of scipy.stats.entropy.

    Example:
    >>> from kats.detectors.density_distance_model import DensityDistanceModel
    >>> model = DensityDistanceModel(window_size=24*3600)
    >>> anom = model.fit_predict(historical_data=hist_ts, data=test_ts)
    >>> anom.scores.plot()
    """

    def __init__(
        self,
        window_size: int,
        serialized_model: Optional[bytes] = None,
        distance_metric: str = "jensenshannon",
        jsd_base: float = 2,
        density_based_distance: Optional[bool] = None,
        validate_monotonic: Optional[bool] = None,
    ) -> None:
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.window_size: int = previous_model["window_size"]
            self.distance_metric: str = previous_model["distance_metric"]
            self.jsd_base: int = previous_model["jsd_base"]
            self.density_based_distance: bool = previous_model["density_based_distance"]
            self.validate_monotonic: bool = previous_model["validate_monotonic"]

        else:
            self.window_size: int = window_size

            if distance_metric not in SUPPORTED_DISTANCE_METRICS:
                raise ParameterError(
                    f"Supported metrics for evaluating detector are: {SUPPORTED_DISTANCE_METRICS}"
                )

            self.distance_metric: str = distance_metric
            self.jsd_base: float = max(2, jsd_base)

            if distance_metric == "jensenshannon":
                self.density_based_distance: bool = True
                self.validate_monotonic: bool = True
            else:
                self.density_based_distance: bool = (
                    density_based_distance is not None
                ) & False
                self.validate_monotonic: bool = (validate_monotonic is not None) & False

    def serialize(self) -> bytes:
        """
        Retrun serilized model.
        """
        return str.encode(json.dumps(self.__dict__))

    def _validate_ts_input_data(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> None:
        if data.is_univariate():
            raise DataError(
                "This algorithm is supporting multivariate time series data only."
            )

        if historical_data is not None and historical_data.is_univariate():
            raise DataError(
                "This algorithm is supporting multivariate time series data only."
            )

        if (
            historical_data is not None
            and historical_data.value.shape[1] != data.value.shape[1]
        ):
            raise DataError("Unmatched dimension of historical data and data!")

    def _validate_monotonic(self, ts_df: pd.DataFrame) -> None:
        if not ts_df.diff(axis=1).iloc[:, 1:].ge(0).all(axis=1).all():
            raise DataError("Each row of input data must be non-decreasing.")

    def _validate_data_granularity(self, ts_df: pd.DataFrame) -> None:
        if ts_df.isna().sum().sum() > 0:
            raise DataIrregularGranularityError(
                "Can't find a time index which is close enough to compare against."
            )

    def _js_div_func(self, x: pd.Series) -> float:
        n = len(x) // 2
        prob_a, prob_b = _percentile_to_prob(x[:n], x[n:])
        return distance.jensenshannon(prob_a, prob_b, base=self.jsd_base)

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        self._validate_ts_input_data(data, historical_data)

        # pull all the data in historical data
        if historical_data is not None:
            historical_data = historical_data[:]
            historical_data.extend(data, validate=False)
        else:
            # When historical_data is not provided, will use part of data as
            # historical_data, and fill with zero anomaly score.
            historical_data = data[:]

        if (
            historical_data.time.iloc[-1] - historical_data.time.iloc[0]
        ).total_seconds() < self.window_size:
            raise DataError("Window size is greater than the data range.")

        total_data_df = historical_data.to_dataframe()
        total_data_df = total_data_df.set_index(total_data_df.columns[0])
        if self.validate_monotonic:
            self._validate_monotonic(total_data_df)

        total_data_df_group0 = total_data_df.rolling(
            window=str(self.window_size) + "s",
            closed="both",
        ).agg(
            lambda rows: rows[0]
            if (rows.index[-1] - rows.index[0]).total_seconds()
            > 0.9 * self.window_size  # tolerance
            else np.nan
        )

        # exclude the beginning part of NANs
        start_time_index = total_data_df_group0.first_valid_index()
        if not start_time_index:
            raise DataError("Window size is greater than the data range.")
        start_time_index = max(start_time_index, data.time[0])

        # validate if we can find a time index which is close enough to compare against
        self._validate_data_granularity(total_data_df_group0.loc[start_time_index:])
        self._validate_data_granularity(total_data_df.loc[start_time_index:])

        if self.distance_metric == "jensenshannon":
            total_df = pd.concat(
                [
                    total_data_df_group0.loc[start_time_index:],
                    total_data_df.loc[start_time_index:],
                ],
                1,
                copy=False,
            )
            scores = total_df.apply(self._js_div_func, axis=1)
        else:
            group_a = total_data_df_group0.loc[start_time_index:].values
            group_b = total_data_df.loc[start_time_index:].values
            # using np.diag(distance.cdist()) is faster than a loop func
            scores = np.diag(
                distance.cdist(
                    XA=group_a,
                    XB=group_b,
                    metric=self.distance_metric,
                )
            )

        scores_tsd = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": list(data.time),
                    "value": [0] * (len(data) - len(scores)) + list(scores),
                }
            )
        )

        return AnomalyResponse(
            scores=scores_tsd,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=TimeSeriesData(
                time=scores_tsd.time, value=pd.Series([0] * len(scores_tsd))
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
