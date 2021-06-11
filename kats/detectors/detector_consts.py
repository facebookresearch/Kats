#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from scipy.stats import norm, t, ttest_ind  # @manual
from statsmodels.stats import multitest

# from np.typing import ArrayLike
ArrayLike = np.ndarray


# Single Spike object
@attr.s(auto_attribs=True)
class SingleSpike:
    time: datetime
    value: float
    n_sigma: float

    @property
    def time_str(self) -> str:
        return datetime.strftime(self.time, "%Y-%m-%d")


# Changepoint Interval object
@attr.s(auto_attribs=True)
class ChangePointInterval:
    start_time: datetime
    end_time: datetime
    previous_interval: Optional[ChangePointInterval] = attr.ib(default=None, init=False)
    _all_spikes: Union[
        Optional[List[SingleSpike]], Optional[List[List[SingleSpike]]]
    ] = attr.ib(default=None, init=False)
    spike_std_threshold: float = attr.ib(default=2.0, init=False)
    data_df: Optional[pd.DataFrame] = attr.ib(None, init=False)
    _ts_cols: List[str] = attr.ib(factory=lambda: ["value"], init=False)
    num_series: int = 1

    @property
    def data(self) -> Optional[ArrayLike]:
        df = self.data_df
        if df is None:
            return None
        elif self.num_series == 1:
            return df.value.values
        else:
            return df[self._ts_cols].values

    @data.setter
    def data(self, data: TimeSeriesData) -> None:
        if not data.is_univariate():
            self._ts_cols = list(data.value.columns)
            self.num_series = len(self._ts_cols)
        all_data_df = data.to_dataframe()
        all_data_df.columns = ["time"] + self._ts_cols
        all_data_df["time"] = pd.to_datetime(all_data_df["time"])
        all_data_df = all_data_df.loc[
            (all_data_df.time >= self.start_time) & (all_data_df.time < self.end_time)
        ]
        self.data_df = all_data_df

    def _detect_spikes(self) -> Union[List[SingleSpike], List[List[SingleSpike]]]:
        df = self.data_df
        if df is None:
            raise ValueError("data must be set before spike detection")

        if self.num_series == 1:
            df["z_score"] = (df.value - self.mean_val) / np.sqrt(self.variance_val)

            spike_df = df.query(f"z_score >={self.spike_std_threshold}")
            return [
                SingleSpike(
                    time=row["time"], value=row["value"], n_sigma=row["z_score"]
                )
                for counter, row in spike_df.iterrows()
            ]
        else:
            spikes = []
            for i, c in enumerate(self._ts_cols):
                # pyre-fixme[16]: `float` has no attribute `__getitem__`.
                df[f"z_score_{c}"] = (df[c] - self.mean_val[i]) / np.sqrt(
                    self.variance_val[i]
                )

                spike_df = df.query(f"z_score_{c} >={self.spike_std_threshold}")

                if spike_df.shape[0] == 0:
                    continue
                else:
                    spikes.append(
                        [
                            SingleSpike(
                                time=row["time"],
                                value=row[c],
                                n_sigma=row[f"z_score_{c}"],
                            )
                            for counter, row in spike_df.iterrows()
                        ]
                    )
            return spikes

    def extend_data(self, data: TimeSeriesData) -> None:
        """
        extends the data.
        """
        new_data_df = data.to_dataframe()
        new_data_df.columns = ["time"] + self._ts_cols
        df = self.data_df
        if df is not None:
            new_data_df = pd.concat([df, new_data_df])
        self.data_df = new_data_df.loc[
            (new_data_df.time >= self.start_time) & (new_data_df.time < self.end_time)
        ]

    @property
    def start_time_str(self) -> str:
        return datetime.strftime(self.start_time, "%Y-%m-%d")

    @property
    def end_time_str(self) -> str:
        return datetime.strftime(self.end_time, "%Y-%m-%d")

    @property
    def mean_val(self) -> Union[float, ArrayLike]:
        if self.num_series == 1:
            vals = self.data
            return 0.0 if vals is None else np.mean(vals)
        else:
            data_df = self.data_df
            if data_df is None:
                return np.zeros(self.num_series)
            return np.array([np.mean(data_df[c].values) for c in self._ts_cols])

    @property
    def variance_val(self) -> Union[float, ArrayLike]:
        if self.num_series == 1:
            vals = self.data
            return 0.0 if vals is None else np.var(vals)
        else:
            data_df = self.data_df
            if data_df is None:
                return np.zeros(self.num_series)
            return np.array([np.var(data_df[c].values) for c in self._ts_cols])

    def __len__(self) -> int:
        df = self.data_df
        return 0 if df is None else len(df)

    @property
    def spikes(self) -> Union[List[SingleSpike], List[List[SingleSpike]]]:
        spikes = self._all_spikes
        if spikes is None:
            spikes = self._detect_spikes()
            self._all_spikes = spikes
        return spikes


# Percentage Change Object
class PercentageChange:
    def __init__(
        self,
        current: ChangePointInterval,
        previous: ChangePointInterval,
        method="fdr_bh",
    ):
        self.current = current
        self.previous = previous

        self.upper = None
        self.lower = None
        self._t_score = None
        self._p_value = None
        self.alpha = 0.05
        self.method = method
        self.num_series = self.current.num_series

    @property
    def ratio_estimate(self) -> Union[float, np.ndarray]:
        # pyre-ignore[6]: Expected float for 1st positional only parameter to call float.__truediv__ but got Union[float, np.ndarray].
        return self.current.mean_val / self.previous.mean_val

    @property
    def perc_change(self) -> float:
        return (self.ratio_estimate - 1.0) * 100.0

    @property
    def perc_change_upper(self) -> float:
        if self.upper is None:
            self._delta_method()
        return (self.upper - 1) * 100.0

    @property
    def perc_change_lower(self) -> float:
        if self.lower is None:
            self._delta_method()
        return (self.lower - 1) * 100.0

    @property
    def direction(self) -> Union[str, ArrayLike]:
        if self.num_series > 1:
            return np.vectorize(lambda x: "up" if x > 0 else "down")(self.perc_change)
        elif self.perc_change > 0.0:
            return "up"
        else:
            return "down"

    @property
    def stat_sig(self) -> Union[bool, ArrayLike]:
        if self.upper is None:
            self._delta_method()
        if self.num_series > 1:
            return np.array(
                [
                    False if self.upper[i] > 1.0 and self.lower[i] < 1 else True
                    for i in range(self.current.num_series)
                ]
            )
        # not stat sig e.g. [0.88, 1.55]
        return not (self.upper > 1.0 and self.lower < 1.0)

    @property
    def score(self) -> float:
        if self._t_score is None:
            self._ttest()

        return self._t_score

    @property
    def p_value(self) -> float:
        if self._p_value is None:
            self._ttest()

        return self._p_value

    @property
    def mean_previous(self) -> Union[float, np.ndarray]:
        return self.previous.mean_val

    @property
    def mean_difference(self) -> Union[float, np.ndarray]:
        # pyre-fixme[6]: Expected `float` for 1st param but got `Union[float,
        #  np.ndarray]`.
        _mean_diff = self.current.mean_val - self.previous.mean_val
        return _mean_diff

    @property
    def ci_upper(self) -> float:
        sp_mean = self._pooled_stddev()
        df = self._get_df()

        # the minus sign here is non intuitive.
        # this is because, for example, t.ppf(0.025, 30) ~ -1.96
        _ci_upper = self.previous.mean_val - t.ppf(self.alpha / 2, df) * sp_mean

        return _ci_upper

    @property
    def ci_lower(self) -> float:
        sp_mean = self._pooled_stddev()
        df = self._get_df()
        # the plus sign here is non-intuitive. See comment
        # above
        _ci_lower = self.previous.mean_val + t.ppf(self.alpha / 2, df) * sp_mean

        return _ci_lower

    def _get_df(self) -> float:
        """
        degree of freedom of t-test
        """
        n_1 = len(self.previous)
        n_2 = len(self.current)
        df = n_1 + n_2 - 2

        return df

    def _pooled_stddev(self) -> float:
        """
        This calculates the pooled standard deviation for t-test
        as defined in https://online.stat.psu.edu/stat500/lesson/7/7.3/7.3.1/7.3.1.1
        """

        s_1_sq = self.previous.variance_val
        s_2_sq = self.current.variance_val
        n_1 = len(self.previous)
        n_2 = len(self.current)

        if n_1 == 0 or n_2 == 0:
            return 0.0

        # pyre-ignore[58]: * is not supported for operand types int and Union[float, np.ndarray].
        s_p = np.sqrt(((n_1 - 1) * s_1_sq + (n_2 - 1) * s_2_sq) / (n_1 + n_2 - 2))

        # s_p_mean = s_p * np.sqrt((1. / n_1) + (1./ n_2))

        return s_p

    def _ttest_manual(self) -> Tuple[float, float]:
        """
        scipy's t-test gives nan when one of the arrays has a
        size of 1.
        To repro, run:
        >>> ttest_ind(np.array([1,2,3,4]), np.array([11]), equal_var=True, nan_policy='omit')
        This is implemented to fix this issue
        """
        sp_mean = self._pooled_stddev()
        df = self._get_df()

        # pyre-ignore[6]: Expected float for 1st positional only parameter to call float.__sub__ but got Union[float, np.ndarray].
        t_score = (self.current.mean_val - self.previous.mean_val) / sp_mean
        p_value = t.sf(np.abs(t_score), df) * 2  # sf = 1 - cdf

        return t_score, p_value

    def _ttest(self) -> None:
        if self.num_series > 1:
            self._ttest_multivariate()
            return

        n_1 = len(self.previous)
        n_2 = len(self.current)

        # if both control and test have one value
        # then using a t test does not make any sense
        if n_1 == 1 and n_2 == 1:
            self._t_score = np.inf
            self._p_value = 0.0

        # when sample size is 1, scipy's t test gives nan,
        # hence we separately handle this case
        # if n_1 == 1 or n_2 == 1:
        #     self._t_score, self._p_value = self._ttest_manual()
        # else:
        #     self._t_score, self._p_value = ttest_ind(
        #         current_data, prev_data, equal_var=True, nan_policy='omit'
        #     )

        # Always use ttest_manual because we changed the std to not include
        # np.sqrt((1. / n_1) + (1./ n_2))
        self._t_score, self._p_value = self._ttest_manual()

    def _ttest_multivariate(self) -> None:
        num_series = self.num_series
        p_value_start = np.zeros(num_series)
        t_value_start = np.zeros(num_series)

        n_1 = len(self.previous)
        n_2 = len(self.current)

        if n_1 == 1 and n_2 == 1:
            self._t_score = np.inf * np.ones(num_series)
            self._p_value = np.zeros(num_series)
            return
        elif n_1 == 1 or n_2 == 1:
            t_value_start, p_value_start = self._ttest_manual()
        else:
            current_data = self.current.data
            prev_data = self.previous.data
            if current_data is None or prev_data is None:
                raise ValueError("Interval data not set")
            for i in range(num_series):
                current_slice = current_data[:, i]
                prev_slice = prev_data[:, i]
                t_value_start[i], p_value_start[i] = ttest_ind(
                    current_slice, prev_slice, equal_var=True, nan_policy="omit"
                )

        # The new p-values are the old p-values rescaled so that self.alpha is still the threshold for rejection
        _, self._p_value, _, _ = multitest.multipletests(
            p_value_start, alpha=self.alpha, method=self.method
        )
        self._t_score = np.zeros(num_series)
        # We are using a two-sided test here, so we take inverse_tcdf(self._p_value / 2) with df = len(self.current) + len(self.previous) - 2
        for i in range(self.current.num_series):
            if t_value_start[i] < 0:
                self._t_score[i] = t.ppf(self._p_value[i] / 2, self._get_df())
            else:
                self._t_score[i] = t.ppf(1 - self._p_value[i] / 2, self._get_df())

    def _calc_cov(self) -> float:
        """
        Calculates the covariance of x and y
        """
        current = self.current.data
        previous = self.previous.data
        if current is None or previous is None:
            return np.nan
        n_min = min(len(current), len(previous))
        if n_min == 0:
            return np.nan
        current = current[-n_min:-1]
        previous = previous[-n_min:-1]

        return np.cov(current, previous)[0, 1] / n_min

    def _delta_method(self) -> None:
        test_mean = self.current.mean_val
        control_mean = self.previous.mean_val
        test_var = self.current.variance_val
        control_var = self.previous.variance_val

        n_test = len(self.current)
        n_control = len(self.previous)

        cov_xy = self._calc_cov()

        sigma_sq_ratio = (
            test_var / (n_test * (control_mean ** 2))
            - 2 * (test_mean * cov_xy) / (control_mean ** 3)
            + (control_var * (test_mean ** 2)) / (n_control * (control_mean ** 4))
        )
        # the signs appear flipped because norm.ppf(0.025) ~ -1.96
        self.lower = self.ratio_estimate + norm.ppf(self.alpha / 2) * np.sqrt(
            abs(sigma_sq_ratio)
        )
        self.upper = self.ratio_estimate - norm.ppf(self.alpha / 2) * np.sqrt(
            abs(sigma_sq_ratio)
        )


@dataclass
class ConfidenceBand:
    lower: TimeSeriesData
    upper: TimeSeriesData


@dataclass
class AnomalyResponse:
    scores: TimeSeriesData
    confidence_band: ConfidenceBand
    predicted_ts: TimeSeriesData
    anomaly_magnitude_ts: TimeSeriesData
    stat_sig_ts: TimeSeriesData

    def update(
        self,
        time: datetime,
        score: float,
        ci_upper: float,
        ci_lower: float,
        pred: float,
        anom_mag: float,
        stat_sig: float,
    ) -> None:
        """
        Add one more point and remove the last point
        """
        self.scores = self._update_ts_slice(self.scores, time, score)
        self.confidence_band = ConfidenceBand(
            lower=self._update_ts_slice(self.confidence_band.lower, time, ci_lower),
            upper=self._update_ts_slice(self.confidence_band.upper, time, ci_upper),
        )

        self.predicted_ts = self._update_ts_slice(self.predicted_ts, time, pred)
        self.anomaly_magnitude_ts = self._update_ts_slice(
            self.anomaly_magnitude_ts, time, anom_mag
        )
        self.stat_sig_ts = self._update_ts_slice(self.stat_sig_ts, time, stat_sig)

    def _update_ts_slice(
        self, ts: TimeSeriesData, time: datetime, value: float
    ) -> TimeSeriesData:
        time = ts.time.iloc[1:].append(pd.Series(time))
        time.reset_index(drop=True, inplace=True)
        value = ts.value.iloc[1:].append(pd.Series(value))
        value.reset_index(drop=True, inplace=True)
        return TimeSeriesData(time=time, value=value)

    def inplace_update(
        self,
        time: datetime,
        score: float,
        ci_upper: float,
        ci_lower: float,
        pred: float,
        anom_mag: float,
        stat_sig: float,
    ) -> None:
        """
        Add one more point and remove the last point
        """
        self._inplace_update_ts(self.scores, time, score)
        self._inplace_update_ts(self.confidence_band.lower, time, ci_lower),
        self._inplace_update_ts(self.confidence_band.upper, time, ci_upper)

        self._inplace_update_ts(self.predicted_ts, time, pred)
        self._inplace_update_ts(self.anomaly_magnitude_ts, time, anom_mag)
        self._inplace_update_ts(self.stat_sig_ts, time, stat_sig)

    def _inplace_update_ts(
        self, ts: TimeSeriesData, time: datetime, value: float
    ) -> None:
        ts.value.loc[ts.time == time] = value

    def get_last_n(self, N: int) -> AnomalyResponse:
        """
        returns the response for the last N days
        """

        return AnomalyResponse(
            scores=self.scores[-N:],
            confidence_band=ConfidenceBand(
                upper=self.confidence_band.upper[-N:],
                lower=self.confidence_band.lower[-N:],
            ),
            predicted_ts=self.predicted_ts[-N:],
            anomaly_magnitude_ts=self.anomaly_magnitude_ts[-N:],
            stat_sig_ts=self.stat_sig_ts[-N:],
        )

    def __str__(self) -> str:
        str_ret = f"""
        Time: {self.scores.time.values},
        Scores: {self.scores.value.values},
        Upper Confidence Bound: {self.confidence_band.upper.value.values},
        Lower Confidence Bound: {self.confidence_band.lower.value.values},
        Predicted Time Series: {self.predicted_ts.value.values},
        stat_sig:{self.stat_sig_ts.value.values}
        """

        return str_ret


class MultiAnomalyResponse:
    def __init__(
        self,
        scores: TimeSeriesData,
        confidence_band: ConfidenceBand,
        predicted_ts: TimeSeriesData,
        anomaly_magnitude_ts: TimeSeriesData,
        stat_sig_ts: TimeSeriesData,
    ):
        self.scores = scores
        self.key_mapping = {
            index: key for index, key in enumerate(scores.value.columns)
        }
        self.response_objects = {}
        for key in scores.value.columns:
            self.response_objects[key] = AnomalyResponse(
                scores=TimeSeriesData(
                    pd.DataFrame({"time": scores.time, "value": scores.value[key]})
                ),
                confidence_band=ConfidenceBand(
                    upper=TimeSeriesData(
                        pd.DataFrame(
                            {
                                "time": confidence_band.upper.time,
                                "value": confidence_band.upper.value[key],
                            }
                        )
                    ),
                    lower=TimeSeriesData(
                        pd.DataFrame(
                            {
                                "time": confidence_band.lower.time,
                                "value": confidence_band.lower.value[key],
                            }
                        )
                    ),
                ),
                predicted_ts=TimeSeriesData(
                    pd.DataFrame(
                        {"time": predicted_ts.time, "value": predicted_ts.value[key]}
                    )
                ),
                anomaly_magnitude_ts=TimeSeriesData(
                    pd.DataFrame(
                        {
                            "time": anomaly_magnitude_ts.time,
                            "value": anomaly_magnitude_ts.value[key],
                        }
                    )
                ),
                stat_sig_ts=TimeSeriesData(
                    pd.DataFrame(
                        {"time": stat_sig_ts.time, "value": stat_sig_ts.value[key]}
                    )
                ),
            )

    def update(
        self,
        time: datetime,
        score: ArrayLike,
        ci_upper: ArrayLike,
        ci_lower: ArrayLike,
        pred: ArrayLike,
        anom_mag: ArrayLike,
        stat_sig: ArrayLike,
    ) -> None:
        """
        Add one more point and remove the last point
        """
        for i in range(len(score)):
            self.response_objects[self.key_mapping[i]].update(
                time,
                score[i],
                ci_upper[i],
                ci_lower[i],
                pred[i],
                anom_mag[i],
                stat_sig[i],
            )

    def inplace_update(
        self,
        time: datetime,
        score: ArrayLike,
        ci_upper: ArrayLike,
        ci_lower: ArrayLike,
        pred: ArrayLike,
        anom_mag: ArrayLike,
        stat_sig: ArrayLike,
    ) -> None:
        """
        Add one more point and remove the last point
        """
        for i in range(len(score)):
            self.response_objects[self.key_mapping[i]].inplace_update(
                time,
                score[i],
                ci_upper[i],
                ci_lower[i],
                pred[i],
                anom_mag[i],
                1.0 if stat_sig[i] else 0.0,
            )

    def _get_time_series_data_components(
        self,
    ) -> Tuple[
        datetime,
        TimeSeriesData,
        TimeSeriesData,
        TimeSeriesData,
        TimeSeriesData,
        TimeSeriesData,
        TimeSeriesData,
    ]:

        time = next(iter(self.response_objects.values())).scores.time

        score_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].scores.value
                        for key in self.response_objects
                    },
                }
            )
        )

        upper_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].confidence_band.upper.value
                        for key in self.response_objects
                    },
                }
            )
        )

        lower_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].confidence_band.lower.value
                        for key in self.response_objects
                    },
                }
            )
        )

        predicted_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].predicted_ts.value
                        for key in self.response_objects
                    },
                }
            )
        )

        anomaly_magnitude_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].anomaly_magnitude_ts.value
                        for key in self.response_objects
                    },
                }
            )
        )

        stat_sig_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time},
                    **{
                        key: self.response_objects[key].stat_sig_ts.value
                        for key in self.response_objects
                    },
                }
            )
        )

        return (
            time,
            score_ts,
            upper_ts,
            lower_ts,
            predicted_ts,
            anomaly_magnitude_ts,
            stat_sig_ts,
        )

    def get_last_n(self, N: int) -> MultiAnomalyResponse:
        """
        returns the response for the last N days
        """

        (
            _,
            score_ts,
            upper_ts,
            lower_ts,
            predicted_ts,
            anomaly_magnitude_ts,
            stat_sig_ts,
        ) = self._get_time_series_data_components()

        return MultiAnomalyResponse(
            scores=score_ts[-N:],
            confidence_band=ConfidenceBand(upper=upper_ts[-N:], lower=lower_ts[-N:]),
            predicted_ts=predicted_ts[-N:],
            anomaly_magnitude_ts=anomaly_magnitude_ts[-N:],
            stat_sig_ts=stat_sig_ts[-N:],
        )

    def get_anomaly_response(self) -> AnomalyResponse:
        (
            time,
            score_ts,
            upper_ts,
            lower_ts,
            predicted_ts,
            anomaly_magnitude_ts,
            stat_sig_ts,
        ) = self._get_time_series_data_components()

        return AnomalyResponse(
            scores=score_ts,
            confidence_band=ConfidenceBand(upper=upper_ts, lower=lower_ts),
            predicted_ts=predicted_ts,
            anomaly_magnitude_ts=anomaly_magnitude_ts,
            stat_sig_ts=stat_sig_ts,
        )

    def __str__(self) -> str:
        str_ret = f"""
        Time: {self.scores.time.values},
        Scores: {self.scores.value.values}
        """

        return str_ret
