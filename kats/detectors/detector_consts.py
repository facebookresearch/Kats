#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from kats.consts import TimeSeriesData
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm, t, ttest_ind # @manual
from statsmodels.stats import multitest

from dataclasses import dataclass


#Single Spike object
class SingleSpike(object):
    def __init__(self, time: datetime, value: float, n_sigma: float):
        self.time = time
        self.value = value
        self.n_sigma = n_sigma

    @property
    def time_str(self):
        return datetime.strftime(self.time, '%Y-%m-%d')


# Changepoint Interval object
class ChangePointInterval(object):
    def __init__(self, cp_start: datetime, cp_end: datetime):
        self.start_time = cp_start
        self.end_time = cp_end

        self._previous_interval = None
        self.all_spikes = None

        self.spike_std_threshold = 2.
        self.data_df = None

    @property
    def data(self):
        if self.data_df is not None:
            return self.data_df.value.values
        else:
            return None

    @data.setter
    def data(self, data: TimeSeriesData):
        all_data_df = data.to_dataframe()
        all_data_df['time'] = pd.to_datetime(all_data_df['time'])
        self.data_df = all_data_df[
            (all_data_df.time >= self.cp_start)
            & (all_data_df.time < self.cp_end)
        ]
        self.data_df.columns = ['time', 'value']

    @property
    def previous_interval(self):
        return self._previous_interval

    @previous_interval.setter
    def previous_interval(self, prev_interval):
        self._previous_interval = prev_interval

    def _detect_spikes(self):
        self.data_df['z_score'] = (
            (self.data_df.value - self.mean_val)/np.sqrt(self.variance_val)
        )

        spike_df = self.data_df.query(f'z_score >={self.spike_std_threshold}')
        if spike_df.shape[0] == 0:
            self.all_spikes = []
        else:
            self.all_spikes = [
                SingleSpike(time=row['time'], value=row['value'], n_sigma=row['z_score'])
                for counter, row in spike_df.iterrows()
            ]

    def extend_data(self, data: TimeSeriesData):
        """
        extends the data.
        """
        new_data_df = data.to_dataframe()
        new_data_df.columns = ['time', 'value']

        self.data_df = pd.concat([self.data_df, new_data_df])
        self.data_df = self.data_df[
            (self.data_df.time >= self.cp_start)
            & (self.data_df.time < self.cp_end)
        ]


    @property
    def start_time(self):
        return self.cp_start

    @start_time.setter
    def start_time(self, cp_start: datetime):
        self.cp_start = cp_start

    @property
    def end_time(self):
        return self.cp_end

    @end_time.setter
    def end_time(self, cp_end: datetime):
        self.cp_end = cp_end

    @property
    def start_time_str(self):
        return datetime.strftime(self.cp_start, '%Y-%m-%d')

    @property
    def end_time_str(self):
        return datetime.strftime(self.cp_end, '%Y-%m-%d')

    @property
    def mean_val(self):
        if self.data_df is not None:
            return np.mean(self.data_df.value.values)
        else:
            return 0.
    @property
    def variance_val(self):
        if self.data_df is not None:
            return np.var(self.data_df.value.values)
        else:
            return 0.

    def __len__(self) -> int:
        if self.data_df is not None:
            return self.data_df.shape[0]
        else:
            return 0

    @property
    def spikes(self):
        if self.all_spikes is None:
            self._detect_spikes()
        return self.all_spikes



# Percentage Change Object
class PercentageChange(object):
    def __init__(self, current:ChangePointInterval,
                 previous: ChangePointInterval):
        self.current = current
        self.previous = previous

        self.upper = None
        self.lower = None
        self._t_score = None
        self._p_value = None
        self.alpha = 0.05

    @property
    def ratio_estimate(self):
        return (self.current.mean_val / self.previous.mean_val)

    @property
    def perc_change(self):
        return  (self.ratio_estimate - 1.) * 100.

    @property
    def perc_change_upper(self):
        if self.upper is None:
            self._delta_method()
        return (self.upper - 1) * 100.

    @property
    def perc_change_lower(self):
        if self.lower is None:
            self._delta_method()
        return (self.lower - 1) * 100.

    @property
    def direction(self):
        if self.perc_change > 0.:
            return 'up'
        else:
            return 'down'

    @property
    def stat_sig(self):
        if self.upper is None:
            self._delta_method()

        # not stat sig e.g. [0.88, 1.55]
        if self.upper  > 1. and self.lower < 1.:
            return False
        else:
            return True

    @property
    def score(self):
        if self._t_score is None:
            self._ttest()

        return self._t_score

    @property
    def p_value(self):
        if self._p_value is None:
            self._ttest()

        return self._p_value

    @property
    def mean_previous(self):
        return self.previous.mean_val

    @property
    def mean_difference(self):
        _mean_diff = self.current.mean_val - self.previous.mean_val
        return _mean_diff

    @property
    def ci_upper(self):
        sp_mean = self._pooled_stddev()
        df = self._get_df()

        # the minus sign here is non intuitive.
        # this is because, for example, t.ppf(0.025, 30) ~ -1.96
        _ci_upper = (
            self.previous.mean_val
            - t.ppf(self.alpha/2, df) * sp_mean
        )

        return _ci_upper

    @property
    def ci_lower(self):
        sp_mean = self._pooled_stddev()
        df = self._get_df()
        # the plus sign here is non-intuitive. See comment
        # above
        _ci_lower = (
            self.previous.mean_val
            + t.ppf(self.alpha/2, df) * sp_mean
        )

        return _ci_lower

    def _get_df(self):
        """
        degree of freedom of t-test
        """
        n_1 = len(self.previous)
        n_2 = len(self.current)
        df = n_1 + n_2 - 2

        return df

    def _pooled_stddev(self):
        """
        This calculates the pooled standard deviation for t-test
        as defined in https://online.stat.psu.edu/stat500/lesson/7/7.3/7.3.1/7.3.1.1
        """

        s_1_sq = self.previous.variance_val
        s_2_sq = self.current.variance_val
        n_1 = len(self.previous)
        n_2 = len(self.current)

        if n_1 == 0 or n_2 == 0:
            return 0.

        s_p = np.sqrt(
            ((n_1 - 1) * s_1_sq + (n_2 - 1) * s_2_sq)
            / (n_1 + n_2 - 2)
        )

        # s_p_mean = s_p * np.sqrt((1. / n_1) + (1./ n_2))

        return s_p

    def _ttest_manual(self):
        """
        scipy's t-test gives nan when one of the arrays has a
        size of 1.
        To repro, run:
        >>> ttest_ind(np.array([1,2,3,4]), np.array([11]), equal_var=True, nan_policy='omit')
        This is implemented to fix this issue
        """
        sp_mean = self._pooled_stddev()
        df = self._get_df()

        t_score = (self.current.mean_val - self.previous.mean_val)/sp_mean
        p_value = t.sf(np.abs(t_score), df) * 2   # sf = 1 - cdf

        return t_score, p_value

    def _ttest(self):
        current_data = self.current.data
        prev_data = self.previous.data

        n_1 = len(self.previous)
        n_2 = len(self.current)

        # if both control and test have one value
        # then using a t test does not make any sense
        if n_1 == 1 and n_2 == 1:
            self._t_score = np.inf
            self._p_value = 0.

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

    def _calc_cov(self) -> float:
        """
        Calculates the covariance of x and y
        """
        n_min = min(len(self.current), len(self.previous))
        current_data = self.current.data[len(self.current) - n_min:-1]
        prev_data = self.previous.data[len(self.previous) - n_min:-1]

        return np.cov(current_data,prev_data)[0,1]/n_min

    def _delta_method(self):
        test_mean = self.current.mean_val
        control_mean = self.previous.mean_val
        test_var = self.current.variance_val
        control_var = self.previous.variance_val

        n_test = len(self.current)
        n_control = len(self.previous)

        cov_xy = self._calc_cov()

        sigma_sq_ratio =  (
            test_var / (n_test * (control_mean**2))
            - 2 * (test_mean * cov_xy) / (control_mean**3)
            + (control_var * (test_mean**2))/(n_control * (control_mean**4))
        )
        # the signs appear flipped because norm.ppf(0.025) ~ -1.96
        self.lower = self.ratio_estimate + norm.ppf(self.alpha/2) * np.sqrt(abs(sigma_sq_ratio))
        self.upper = self.ratio_estimate - norm.ppf(self.alpha/2) * np.sqrt(abs(sigma_sq_ratio))

# Multivariable Changepoint Interval object
class MultiChangePointInterval(ChangePointInterval):
    def __init__(self,cp_start: datetime, cp_end: datetime):
        ChangePointInterval.__init__(self, cp_start, cp_end)
        self.num_series = -1
        self.ts_cols = None

    @property
    def data(self):
        if self.data_df is not None:
            return self.data_df[[c for c in self.data_df.columns if c != 'time']].values
        else:
            return None

    @data.setter
    def data(self, data: TimeSeriesData):
        all_data_df = data.to_dataframe()
        self.ts_cols = list(data.value.columns)
        self.num_series = len(self.ts_cols)
        all_data_df['time'] = pd.to_datetime(all_data_df['time'])
        self.data_df = all_data_df[
            (all_data_df.time >= self.cp_start)
            & (all_data_df.time <= self.cp_end)
        ]
        self.data_df.columns = ['time']  + self.ts_cols

    def _detect_spikes(self):
        self.all_spikes = [[] for _ in range(self.num_series)]

        for i, c in enumerate(self.ts_cols):
            self.data_df[f'z_score_{c}'] = (
                (self.data_df[c] - self.mean_val[i])/np.sqrt(self.variance_val[i])
            )

            spike_df = self.data_df.query(f'z_score_{c} >={self.spike_std_threshold}')

            if spike_df.shape[0] == 0:
                continue
            else:
                self.all_spikes[i] = [
                    SingleSpike(time=row['time'], value=row[c], n_sigma=row[f'z_score_{c}'])
                    for counter, row in spike_df.iterrows()
                ]

    def extend_data(self, data: TimeSeriesData):
        """
        extends the data.
        """
        new_data_df = data.to_dataframe()
        self.data_df.columns =  ['time']  + self.ts_cols


        self.data_df = pd.concat([self.data_df, new_data_df])
        self.data_df = self.data_df[
            (self.data_df.time >= self.cp_start)
            & (self.data_df.time <= self.cp_end)
        ]

    @property
    def mean_val(self):
        if self.data_df is not None:
            return np.array([np.mean(self.data_df[c].values) for c in self.ts_cols])
        else:
            return np.zeros(self.num_series)

    @property
    def variance_val(self):
        if self.data_df is not None:
            return np.array([np.var(self.data_df[c].values) for c in self.ts_cols])
        else:
            return np.zeros(self.num_series)


# Multivariable Percentage Change Object
class MultiPercentageChange(PercentageChange):
    def __init__(self,
                 current:MultiChangePointInterval,
                 previous: MultiChangePointInterval,
                 method='fdr_bh'):
        PercentageChange.__init__(self, current, previous)
        self.method = method

    @property
    def direction(self):
        return np.vectorize(lambda x: 'up' if x > 0 else 'down')(self.perc_change)

    @property
    def stat_sig(self):
        if self.upper is None:
            self._delta_method()

        return np.array([False if self.upper[i]  > 1. and self.lower[i] < 1 else True for i in range(self.current.num_series)])

    def _ttest(self):
        p_value_start = np.zeros(self.current.num_series)
        t_value_start = np.zeros(self.current.num_series)

        n_1 = len(self.previous)
        n_2 = len(self.current)

        if n_1 == 1 and n_2 == 1:
            self._t_score = np.inf * np.ones(self.current.num_series)
            self._p_value = np.zeros(self.current.num_series)
            return
        elif n_1 == 1 or n_2 == 1:
            t_value_start, p_value_start = self._ttest_manual()
        else:
            for i in range(self.current.num_series):
                current_data = self.current.data[:,i]
                prev_data = self.previous.data[:,i]
                t_value_start[i], p_value_start[i] = ttest_ind(
                    current_data, prev_data, equal_var=True, nan_policy='omit'
                )

        # The new p-values are the old p-values rescaled so that self.alpha is still the threshold for rejection
        _, self._p_value, _, _ = multitest.multipletests(p_value_start, alpha = self.alpha, method = self.method)
        self._t_score = np.zeros(self.current.num_series)
        # We are using a two-sided test here, so we take inverse_tcdf(self._p_value / 2) with df = len(self.current) + len(self.previous) - 2
        for i in range(self.current.num_series):
            if t_value_start[i] < 0:
                self._t_score[i] = t.ppf(self._p_value[i]/2, self._get_df())
            else:
                self._t_score[i] = t.ppf(1-self._p_value[i]/2, self._get_df())


@dataclass
class ConfidenceBand():
    lower: TimeSeriesData
    upper: TimeSeriesData


@dataclass
class AnomalyResponse():
    scores: TimeSeriesData
    confidence_band: ConfidenceBand
    predicted_ts: TimeSeriesData
    anomaly_magnitude_ts: TimeSeriesData
    stat_sig_ts: TimeSeriesData

    def update(self, time: datetime, score: float, ci_upper: float,
               ci_lower: float, pred: float, anom_mag: float, stat_sig: float):
        """
        Add one more point and remove the last point
        """
        self.scores = self._update_ts_slice(self.scores, time, score)
        self.confidence_band = ConfidenceBand(
            lower=self._update_ts_slice(self.confidence_band.lower, time, ci_lower),
            upper=self._update_ts_slice(self.confidence_band.upper, time, ci_upper)
        )

        self.predicted_ts = self._update_ts_slice(self.predicted_ts, time, pred)
        self.anomaly_magnitude_ts = self._update_ts_slice(self.anomaly_magnitude_ts, time, anom_mag)
        self.stat_sig_ts =  self._update_ts_slice(self.stat_sig_ts, time, stat_sig)

    def _update_ts_slice(self, ts: TimeSeriesData, time: datetime, value: float):
        time = ts.time.iloc[1:].append(pd.Series(time))
        time.reset_index(drop=True, inplace=True)
        value=ts.value.iloc[1:].append(pd.Series(value))
        value.reset_index(drop=True, inplace=True)
        return TimeSeriesData(
            time=time,
            value=value
        )

    def inplace_update(self, time: datetime, score: float, ci_upper: float,
               ci_lower: float, pred: float, anom_mag: float, stat_sig: float):
        """
        Add one more point and remove the last point
        """
        self._inplace_update_ts(self.scores, time, score)
        self._inplace_update_ts(self.confidence_band.lower, time, ci_lower),
        self._inplace_update_ts(self.confidence_band.upper, time, ci_upper)

        self._inplace_update_ts(self.predicted_ts, time, pred)
        self._inplace_update_ts(self.anomaly_magnitude_ts, time, anom_mag)
        self._inplace_update_ts(self.stat_sig_ts, time, stat_sig)

    def _inplace_update_ts(self, ts: TimeSeriesData, time: datetime, value: float):
        ts.value.loc[ts.time == time] = value

    def get_last_n(self, N: int):
        """
        returns the response for the last N days
        """

        return AnomalyResponse(
            scores=self.scores[-N:],
            confidence_band=ConfidenceBand(
                upper=self.confidence_band.upper[-N:],
                lower=self.confidence_band.lower[-N:]
            ),
            predicted_ts=self.predicted_ts[-N:],
            anomaly_magnitude_ts=self.anomaly_magnitude_ts[-N:],
            stat_sig_ts=self.stat_sig_ts[-N:]
        )

    def __str__(self):
        str_ret = f"""
        Time: {self.scores.time.values},
        Scores: {self.scores.value.values},
        Upper Confidence Bound: {self.confidence_band.upper.value.values},
        Lower Confidence Bound: {self.confidence_band.lower.value.values},
        Predicted Time Series: {self.predicted_ts.value.values},
        stat_sig:{self.stat_sig_ts.value.values}
        """

        return str_ret


class MultiAnomalyResponse(object):

    def __init__(self, scores: TimeSeriesData, confidence_band: ConfidenceBand, predicted_ts: TimeSeriesData, anomaly_magnitude_ts: TimeSeriesData, stat_sig_ts: TimeSeriesData):
        self.key_mapping = {index:key for index, key in enumerate(scores.value.columns)}
        self.response_objects = {}
        for key in scores.value.columns:
            self.response_objects[key] = AnomalyResponse(
                scores = TimeSeriesData(pd.DataFrame({"time": scores.time, "value": scores.value[key]})),
                confidence_band = ConfidenceBand(
                upper=TimeSeriesData(pd.DataFrame({"time": confidence_band.upper.time, "value": confidence_band.upper.value[key]})),
                lower=TimeSeriesData(pd.DataFrame({"time": confidence_band.lower.time, "value": confidence_band.lower.value[key]})) ),
                predicted_ts = TimeSeriesData(pd.DataFrame({"time": predicted_ts.time, "value": predicted_ts.value[key]})),
                anomaly_magnitude_ts = TimeSeriesData(pd.DataFrame({"time": anomaly_magnitude_ts.time, "value": anomaly_magnitude_ts.value[key]})),
                stat_sig_ts=TimeSeriesData(pd.DataFrame({"time": stat_sig_ts.time, "value": stat_sig_ts.value[key]})),
            )

    def update(self, time: datetime, score: np.array, ci_upper: np.array,
               ci_lower: np.array, pred: np.array, anom_mag: np.array, stat_sig: np.array):
        """
        Add one more point and remove the last point
        """
        for i in range(len(score)):
            self.response_objects[self.key_mapping[i]].update(time, score[i], ci_upper[i], ci_lower[i], pred[i], anom_mag[i], stat_sig[i])


    def inplace_update(self, time: datetime, score: np.array, ci_upper: np.array,
               ci_lower: np.array, pred: np.array, anom_mag: np.array, stat_sig: np.array):
        """
        Add one more point and remove the last point
        """
        for i in range(len(score)):
            self.response_objects[self.key_mapping[i]].inplace_update(time, score[i],
                ci_upper[i], ci_lower[i], pred[i], anom_mag[i], 1.0 if stat_sig[i] else 0.0)

    def _get_time_series_data_components(self):

        time =  next(iter(self.response_objects.values())).scores.time

        score_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].scores.value for key in self.response_objects}
        }))

        upper_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].confidence_band.upper.value for key in self.response_objects}
        }))

        lower_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].confidence_band.lower.value for key in self.response_objects}
        }))

        predicted_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].predicted_ts.value for key in self.response_objects}
        }))

        anomaly_magnitude_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].anomaly_magnitude_ts.value for key in self.response_objects}
        }))

        stat_sig_ts = TimeSeriesData(pd.DataFrame({
            **{'time': time},
            **{key: self.response_objects[key].stat_sig_ts.value for key in self.response_objects}
        }))

        return time, score_ts, upper_ts, lower_ts, predicted_ts, anomaly_magnitude_ts, stat_sig_ts

    def get_last_n(self, N: int):
        """
        returns the response for the last N days
        """

        time, score_ts, upper_ts, lower_ts, predicted_ts, anomaly_magnitude_ts, stat_sig_ts = self._get_time_series_data_components()

        return MultiAnomalyResponse(
            scores=score_ts[-N:],
            confidence_band=ConfidenceBand(
                upper=upper_ts[-N:],
                lower=lower_ts[-N:]
            ),
            predicted_ts=predicted_ts[-N:],
            anomaly_magnitude_ts=anomaly_magnitude_ts[-N:],
            stat_sig_ts=stat_sig_ts[-N:]
        )

    def get_anomaly_response(self):
        time, score_ts, upper_ts, lower_ts, predicted_ts, anomaly_magnitude_ts, stat_sig_ts = self._get_time_series_data_components()

        return AnomalyResponse(
            scores=score_ts,
            confidence_band=ConfidenceBand(upper=upper_ts, lower=lower_ts),
            predicted_ts=predicted_ts,
            anomaly_magnitude_ts=anomaly_magnitude_ts,
            stat_sig_ts=stat_sig_ts,
        )

    def __str__(self):
        str_ret = f"""
        Time: {self.scores.time.values},
        Scores: {self.scores.value.values}
        """

        return str_ret
