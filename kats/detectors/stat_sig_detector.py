# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from datetime import datetime
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from kats.consts import IRREGULAR_GRANULARITY_ERROR, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import (
    AnomalyResponse,
    ChangePointInterval,
    ConfidenceBand,
    PercentageChange,
)
from kats.utils.decomposition import TimeSeriesDecomposition


"""Statistical Significance Detector Module
This module contains simple detectors that apply a t-test over a rolling window to compare
check if there is a statistically significant increase or decrease between the control and test
time periods.  In addition to the univariate version of this test, this module includes a
multivariate version that uses a false discovery rate (FDR) controlling procedure to reduce noise.
"""


class StatSigDetectorModel(DetectorModel):
    """
    StatSigDetectorModel is a simple detector, which compares a control and test period.
    The detector assumes that the time series data comes from a iid normal distribution,
    and applies a t-test to check if the means between the control and test period are
    significantly different.

    We start with the history data, and then as for the current data, we apply a rolling
    window, adding one data point at a time from the current data, and detecting significant
    change. We return the t-statistic as a score, which reflects the severity of the
    change.
    We suggest using n_control >= 30 to get good estimates

    Attributes:
        n_control: number of data points(or time units) of history to compare with
        n_test: number of points(or time_units) to compare the history with
        serialized_model: serialized json containing the parameters
        time_units: units of time used to measure the intervals. If not provided
                    we infer it from the provided data.
        rem_season: default value is False, if remove seasonality for historical data and data
        seasonal_period: str, default value is 'weekly'. Other possible values: 'daily', 'biweekly', 'monthly', 'yearly'
        use_corrected_scores: bool, default value is False, using original t-scores or correct t-scores.
        max_split_ts_length: int, default value is 500. If the given TS (except historical part) is longer than max_split_ts_length,
                    we will transform a long univariate TS into a multi-variate TS and then use multistatsig detector, which is faster,
        anomaly_scores_only: bool = False. Only calculate anomaly scores without using advanced classes, which is much faster.
        min_perc_change: float, minimum percentage change, for a non zero score. Score will be clipped to zero if the absolute value of the percentage chenge is less than this value
    >>> # Example usage:
    >>> # history and ts_pt are TimeSeriesData objects and history is larger
    >>> # than (n_control + n_test) so that we have sufficient history to
    >>> # run the detector
    >>> n_control = 28
    >>> n_test = 7
    >>> import random
    >>> control_time = pd.date_range(start='2018-01-01', freq='D', periods=(n_control + n_test))
    >>> test_time = pd.date_range(start='2018-02-05', freq='D', periods=n_test)
    >>> control_val = [random.normalvariate(100,10) for _ in range(n_control + n_test)]
    >>> test_val = [random.normalvariate(120,10) for _ in range(n_test)]
    >>> hist_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))
    >>> data_ts = TimeSeriesData(time=test_time, value=pd.Series(test_val))
    >>> ss_detect = StatSigDetectorModel(n_control=n_control, n_test=n_test)
    >>> anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)
    >>> # if only caculate anomaly scores
    >>> ss_detect = StatSigDetectorModel(n_control=n_control, n_test=n_test, anomaly_scores_only=True)
    >>> anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)
    """

    data: Optional[TimeSeriesData] = None

    def __init__(
        self,
        n_control: Optional[int] = None,
        n_test: Optional[int] = None,
        serialized_model: Optional[bytes] = None,
        time_unit: Optional[str] = None,
        rem_season: bool = False,
        seasonal_period: str = "weekly",
        use_corrected_scores: bool = True,
        max_split_ts_length: int = 500,
        anomaly_scores_only: bool = False,
        min_perc_change: float = 0.0,
    ) -> None:

        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.n_test: int = model_dict["n_test"]
            self.n_control: int = model_dict["n_control"]
            self.time_unit: str = model_dict["time_unit"]
            # for seasonality
            self.rem_season: bool = model_dict.get("rem_season", rem_season)
            self.seasonal_period: str = model_dict.get(
                "seasonal_period", seasonal_period
            )

            # for big data and correct t-scores
            self.use_corrected_scores: bool = model_dict.get(
                "use_corrected_scores", use_corrected_scores
            )
            # threshold for splitting long TS
            self.max_split_ts_length: int = model_dict.get(
                "max_split_ts_length", max_split_ts_length
            )

            self.min_perc_change: float = model_dict.get(
                "min_perc_change", min_perc_change
            )

        else:
            self.n_test: Optional[int] = n_test
            self.n_control: Optional[int] = n_control
            self.time_unit: Optional[str] = time_unit
            # for seasonality
            self.rem_season: bool = rem_season
            self.seasonal_period: str = seasonal_period
            # big data and t-scores
            self.use_corrected_scores: bool = use_corrected_scores
            # threshold for splitting long TS
            self.max_split_ts_length: int = max_split_ts_length
            self.min_perc_change: float = min_perc_change

        if (self.n_control is None) or (self.n_test is None):
            raise ValueError(
                "You must either provide serialized model or values for control "
                "and test intervals."
            )

        self.control_interval: Optional[ChangePointInterval] = None
        self.test_interval: Optional[ChangePointInterval] = None
        self.response: Optional[AnomalyResponse] = None

        self.is_initialized = False  # flag on whether initialized or not
        self.last_N = 0  # this is the size of the last chunk of data we saw
        self.data_history: Optional[TimeSeriesData] = None

        # for seasonality
        self.data_season: Optional[TimeSeriesData] = None

        # big data strategy
        self.bigdata_trans_flag: Optional[bool] = None
        self.remaining: Optional[int] = None

        # validate time_unit
        if self.time_unit is not None:
            try:
                # for digit+str ('3D', '10h') cases
                _ = pd.Timedelta(self.time_unit)
            except ValueError:
                try:
                    _ = pd.Timedelta(1, unit=self.time_unit)
                    assert self.time_unit is not None
                    self.time_unit = "1" + self.time_unit
                    _ = pd.Timedelta(self.time_unit)
                except ValueError:
                    raise ValueError("Invalide time_unit value.")

        self.anomaly_scores_only: bool = anomaly_scores_only

    def serialize(self) -> bytes:
        """
        Serializes by putting model parameters in a json
        """
        model_dict = {
            "n_control": self.n_control,
            "n_test": self.n_test,
            "time_unit": self.time_unit,
            "rem_season": self.rem_season,
            "seasonal_period": self.seasonal_period,
            "use_corrected_scores": self.use_corrected_scores,
            "max_split_ts_length": self.max_split_ts_length,
            "min_perc_change": self.min_perc_change,
        }

        return json.dumps(model_dict).encode("utf-8")

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        This is the main working function.
        The function returns an AnomalyResponse object of length
        equal to the length of the data.
        We require len(historical_data) > (n_control + n_test).

        Args:
            data: TimeSeriesData, A univariate TimeSeriesData for which we are running the StatSigDetectorModel
            historical_data: Optional[TimeSeriesData] Historical data used to do detection for initial points in data

        """

        if not data.is_univariate():
            msg = "Input is multivariate but StatSigDetector expected univariate input."
            logging.error(msg)
            raise ValueError(msg)

        self._set_time_unit(data=data, historical_data=historical_data)
        self.last_N = len(data)

        # this ensures we start with a default response of
        # the size of the data
        self._init_response(data)
        response = self.response
        assert response is not None

        # when there is no need to update
        # just return the initial response of zeros
        if not self._should_update(data=data, historical_data=historical_data):
            return response

        # if we only need anomaly scores
        if self.anomaly_scores_only:
            return self._get_anomaly_scores_only(
                historical_data=historical_data,
                data=data,
            )

        # handle cases where there is either no historical data, or
        # not enough historical data
        data, historical_data = self._handle_not_enough_history(
            data=data,
            historical_data=historical_data,
        )

        # remove seasonality
        if self.rem_season:
            sh_data = SeasonalityHandler(
                data=data, seasonal_period=self.seasonal_period
            )

            self.data_season = sh_data.get_seasonality()
            data = sh_data.remove_seasonality()

            if historical_data:
                sh_hist_data = SeasonalityHandler(
                    data=historical_data,
                    seasonal_period=self.seasonal_period,
                )
                historical_data = sh_hist_data.remove_seasonality()

        self.data = data
        # first initialize this with the historical data
        self._init_data(historical_data)

        # if using new t-scores
        if self.use_corrected_scores:
            if (
                len(data) > self.max_split_ts_length
                # pyre-ignore[16]: `Optional` has no attribute `infer_freq_robust`.
                and historical_data.infer_freq_robust() == data.infer_freq_robust()
            ):
                self.bigdata_trans_flag = True
            else:
                self.bigdata_trans_flag = False
        else:
            self.bigdata_trans_flag = False

        # if need trans to multi-TS
        if self.bigdata_trans_flag:
            new_data_ts = self._reorganize_big_data(self.max_split_ts_length)
            ss_detect = MultiStatSigDetectorModel(
                n_control=self.n_control,
                n_test=self.n_test,
                time_unit=self.time_unit,
                rem_season=False,
                seasonal_period=self.seasonal_period,
                skip_rescaling=True,
                use_corrected_scores=self.use_corrected_scores,
            )
            anom = ss_detect.fit_predict(data=new_data_ts)
            self._reorganize_back(anom)

        else:
            self._init_control_test(
                data if historical_data is None else historical_data
            )
            # set the flag to true
            self.is_initialized = True

            # now run through the data to get the prediction
            for i in range(len(data)):
                current_time = data.time.iloc[i]

                ts_pt = TimeSeriesData(
                    time=pd.Series(current_time, copy=False),
                    value=pd.Series(data.value.iloc[i], copy=False),
                )

                self._update_data(ts_pt)
                self._update_control_test(ts_pt)
                self._update_response(ts_pt.time.iloc[0])

        # add seasonality back
        if self.rem_season:
            data_season = self.data_season
            confidence_band = response.confidence_band
            predicted_ts = response.predicted_ts
            assert data_season is not None
            assert confidence_band is not None
            assert predicted_ts is not None

            start_idx = len(response.scores) - len(data_season)
            datatime = response.scores.time

            self.response = AnomalyResponse(
                scores=response.scores,
                confidence_band=ConfidenceBand(
                    upper=TimeSeriesData(
                        time=datatime,
                        value=pd.concat(
                            [
                                pd.Series(
                                    confidence_band.upper.value.values[:start_idx],
                                    copy=False,
                                ),
                                pd.Series(
                                    np.asarray(
                                        confidence_band.upper.value.values[start_idx:]
                                    )
                                    + np.asarray(data_season.value.values),
                                    copy=False,
                                ),
                            ],
                            copy=False,
                        ),
                    ),
                    lower=TimeSeriesData(
                        time=datatime,
                        value=pd.concat(
                            [
                                pd.Series(
                                    confidence_band.lower.value.values[:start_idx],
                                    copy=False,
                                ),
                                pd.Series(
                                    np.asarray(
                                        confidence_band.lower.value.values[start_idx:]
                                    )
                                    + np.asarray(data_season.value.values),
                                    copy=False,
                                ),
                            ],
                            copy=False,
                        ),
                    ),
                ),
                predicted_ts=TimeSeriesData(
                    time=datatime,
                    value=pd.concat(
                        [
                            pd.Series(
                                predicted_ts.value.values[:start_idx], copy=False
                            ),
                            pd.Series(
                                np.asarray(predicted_ts.value.values[start_idx:])
                                + np.asarray(data_season.value.values),
                                copy=False,
                            ),
                        ],
                        copy=False,
                    ),
                ),
                anomaly_magnitude_ts=response.anomaly_magnitude_ts,
                stat_sig_ts=response.stat_sig_ts,
            )
        assert self.response is not None
        return self.response.get_last_n(self.last_N)

    def _get_anomaly_scores_only(
        self,
        historical_data: Optional[TimeSeriesData],
        data: TimeSeriesData,
    ) -> AnomalyResponse:
        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data, validate=False)
            total_data = historical_data

        total_data_df = total_data.to_dataframe()
        total_data_df.columns = ["time", "value"]
        total_data_df = total_data_df.set_index("time")

        res = []
        i = 0
        while i < len(data.time):
            test_end_dt = data.time[i]

            assert self.n_test is not None
            test_start_dt = test_end_dt - (self.n_test - 1) * pd.Timedelta(
                self.time_unit
            )

            control_end_dt = test_start_dt
            assert self.n_control is not None
            control_start_dt = control_end_dt - self.n_control * pd.Timedelta(
                self.time_unit
            )

            if control_end_dt in total_data_df.index:
                # exclude index control_end_dt
                group1 = total_data_df[control_start_dt:control_end_dt].value.to_list()[
                    :-1
                ]
            else:
                group1 = total_data_df[control_start_dt:control_end_dt].value.to_list()
            group2 = total_data_df[test_start_dt:test_end_dt].value.to_list()

            # 2 sample t-test
            if control_start_dt >= total_data.time[0]:
                res.append(stats.ttest_ind(a=group2, b=group1, equal_var=True)[0])
            else:
                res.append(0)

            i += 1

        scores = TimeSeriesData(pd.DataFrame({"time": list(data.time), "value": res}))

        return AnomalyResponse(
            scores=scores,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=TimeSeriesData(),
            stat_sig_ts=None,
        )

    def _reorganize_big_data(self, max_split_ts_length: int) -> TimeSeriesData:
        data_history = self.data_history
        data = self.data
        assert data_history is not None
        assert data is not None

        first_half_len = len(data_history)
        n_seq = len(data) // max_split_ts_length + int(
            len(data) % max_split_ts_length > 0
        )
        remaining = (max_split_ts_length * n_seq - len(data)) % max_split_ts_length

        time_need = pd.concat(
            [data_history.time[:], data.time[:max_split_ts_length]],
            copy=False,
        )

        new_ts = [
            list(
                pd.concat(
                    [data_history.value[:], data.value[:max_split_ts_length]],
                    copy=False,
                )
            )
        ]
        for i in range(max_split_ts_length, len(data), max_split_ts_length):
            new_ts.append(
                new_ts[-1][-first_half_len:]
                + list(data.value[i : i + max_split_ts_length])
            )
        new_ts[-1] += [1] * remaining

        new_data_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": time_need},
                    **{f"ts_{i}": new_ts[i] for i in range(len(new_ts))},
                },
                copy=False,
            )
        )
        self.remaining = remaining
        return new_data_ts

    def _reorganize_back(self, anom: AnomalyResponse) -> None:
        data_history = self.data_history
        remaining = self.remaining
        anom_predicted_ts = anom.predicted_ts
        anom_confidence_band = anom.confidence_band
        anom_stat_sig_ts = anom.stat_sig_ts
        response = self.response
        assert data_history is not None
        assert remaining is not None
        assert anom_predicted_ts is not None
        assert anom_confidence_band is not None
        assert anom_stat_sig_ts is not None
        assert response is not None
        response_predicted_ts = response.predicted_ts
        assert response_predicted_ts is not None

        start_point = len(data_history)
        res_score_val = pd.Series(
            pd.DataFrame(anom.scores.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        res_predicted_ts_val = pd.Series(
            pd.DataFrame(anom_predicted_ts.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        res_anomaly_magnitude_ts_val = pd.Series(
            pd.DataFrame(anom.anomaly_magnitude_ts.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        res_stat_sig_ts_val = pd.Series(
            pd.DataFrame(anom_stat_sig_ts.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        res_confidence_band_lower_val = pd.Series(
            pd.DataFrame(anom_confidence_band.lower.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        res_confidence_band_upper_val = pd.Series(
            pd.DataFrame(anom_confidence_band.upper.value, copy=False)
            .iloc[start_point:, :]
            .values.T.flatten()[:-remaining],
            copy=False,
        )

        datatime = response.scores.time
        zeros = pd.Series(np.zeros(len(datatime) - len(res_score_val)), copy=False)
        datavalues = pd.Series(
            response_predicted_ts.value.values[: len(datatime) - len(res_score_val)],
            copy=False,
        )

        self.response = AnomalyResponse(
            scores=TimeSeriesData(
                time=datatime, value=pd.concat([zeros, res_score_val], copy=False)
            ),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=datatime,
                    value=pd.concat(
                        [datavalues, res_confidence_band_upper_val], copy=False
                    ),
                ),
                lower=TimeSeriesData(
                    time=datatime,
                    value=pd.concat(
                        [datavalues, res_confidence_band_lower_val], copy=False
                    ),
                ),
            ),
            predicted_ts=TimeSeriesData(
                time=datatime,
                value=pd.concat([datavalues, res_predicted_ts_val], copy=False),
            ),
            anomaly_magnitude_ts=TimeSeriesData(
                time=datatime,
                value=pd.concat([zeros, res_anomaly_magnitude_ts_val], copy=False),
            ),
            stat_sig_ts=TimeSeriesData(
                time=datatime, value=pd.concat([zeros, res_stat_sig_ts_val], copy=False)
            ),
        )

    def visualize(self) -> None:
        """Function to visualize the result of the StatSigDetectorModel."""
        self.plot()

    def plot(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        line_color: str = "k",
        line_style: str = "-",
        prediction_color: str = "r",
        prediction_style: str = "--",
        fill_color: Optional[str] = "blue",
        response_color: str = "r",
        response_style: str = "-",
        **kwargs: Any,
    ) -> Tuple[plt.Axes, plt.Axes]:
        """Plot the results.

        Args:
            figsize: figsize to use. If None, use (10, 8).
            line_color: color to use for the original data line.
            line_style: linestyle to use for the original data.
            prediction_color: color to use for the predicted data.
            prediction_style: linestyle to use for the predicted data.
            fill_color: color to use for the confidence interval. If None, don't
                plot the confidence interval.
            response_color: color to use for the response plot.
            response_style: linestyle to use for the response plot.
        """
        data = self.data
        if data is None:
            raise ValueError("Call fit_predict() before visualize()")
        response = self.response
        assert response is not None
        predicted_ts = response.predicted_ts
        confidence_band = response.confidence_band
        assert predicted_ts is not None
        assert confidence_band is not None
        if figsize is None:
            figsize = (10, 8)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        N_data = len(data)

        ax1.plot(data.time, data.value.values, color=line_color, ls=line_style)
        ax1.plot(
            predicted_ts.time.iloc[-N_data:],
            predicted_ts.value.values[-N_data:],
            color=prediction_color,
            ls=prediction_style,
        )
        upper_ci = confidence_band.upper
        lower_ci = confidence_band.lower

        if fill_color is not None:
            ax1.fill_between(
                upper_ci.time.iloc[-N_data:],
                lower_ci.value.values[-N_data:],
                upper_ci.value.values[-N_data:],
                facecolor=fill_color,
                alpha=0.25,
            )

        ax2.plot(
            response.scores.time.iloc[-N_data:],
            response.scores.value.values[-N_data:],
            color=response_color,
            ls=response_style,
        )
        return ax1, ax2

    def _set_time_unit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData]
    ) -> None:
        """
        if the time unit is not set, this function tries to set it.
        """
        if not self.time_unit:
            if len(data) >= 3:
                # frequency is pd.TimeDelta
                frequency = data.infer_freq_robust()
            elif historical_data and len(historical_data) >= 3:
                frequency = historical_data.infer_freq_robust()
            else:
                raise ValueError(
                    "Not able to infer freqency of the time series. "
                    "Please use longer time series data or pass the time_unit parameter to the initializer."
                )

            # Timedelta string
            self.time_unit = f"{frequency.total_seconds()}S"

        # validate the time_unit
        try:
            # for digit+str ('3D', '10h') cases
            _ = pd.Timedelta(self.time_unit)
        except ValueError:
            try:
                _ = pd.Timedelta(1, unit=self.time_unit)
                assert self.time_unit is not None
                self.time_unit = "1" + self.time_unit
                _ = pd.Timedelta(self.time_unit)
            except ValueError:
                raise ValueError("Invalide time_unit value.")

    def _should_update(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> bool:
        """
        Flag on whether we should update responses, or send a default response of zeros.
        """

        if self.time_unit is None:
            raise ValueError("time_unit variable cannot be None")

        if not historical_data:
            start_time = data.time.iloc[0]
        else:
            start_time = historical_data.time.iloc[0]

        end_time = data.time.iloc[-1]
        assert self.n_control is not None
        assert self.n_test is not None

        return end_time >= (
            start_time
            + (self.n_control + self.n_test - 1) * pd.Timedelta(self.time_unit)
        )

    def _handle_not_enough_history(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData]
    ) -> Tuple[TimeSeriesData, Optional[TimeSeriesData]]:
        """
        Handles the case when we don't have enough historical data.
        If we don't need to update, this does not do anything.
        If we need to update, this divides up the data accordingly.
        """
        assert self.time_unit is not None
        assert self.n_control is not None
        assert self.n_test is not None
        num_hist_points = self.n_control + self.n_test - 1

        # if we have enough history, we should not do anything
        if historical_data:
            history_first = historical_data.time.iloc[0]
            history_last = historical_data.time.iloc[-1]
            min_history_last = history_first + num_hist_points * pd.Timedelta(
                self.time_unit
            )

            if history_last >= min_history_last:
                return data, historical_data

        # when no historical data, divide the data into historical and not
        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data, validate=False)
            total_data = historical_data

        first_dt = total_data.time.iloc[0]  # first date of the data

        last_dt = first_dt + num_hist_points * pd.Timedelta(self.time_unit)

        historical_data = TimeSeriesData(
            time=total_data.time[total_data.time < last_dt],
            value=total_data.value[total_data.time < last_dt],
        )

        data = TimeSeriesData(
            time=total_data.time[total_data.time >= last_dt],
            value=total_data.value[total_data.time >= last_dt],
        )

        return data, historical_data

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit can be called during priming. It's a noop for us.
        """

        return

    def _init_response(self, data: TimeSeriesData) -> None:
        """
        Initializes a default response.
        """

        zeros = np.zeros(len(data))
        self.response = AnomalyResponse(
            scores=TimeSeriesData(time=data.time, value=pd.Series(zeros, copy=False)),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=data.time, value=pd.Series(data.value.values, copy=False)
                ),
                lower=TimeSeriesData(
                    time=data.time, value=pd.Series(data.value.values, copy=False)
                ),
            ),
            predicted_ts=TimeSeriesData(
                time=data.time, value=pd.Series(data.value.values, copy=False)
            ),
            anomaly_magnitude_ts=TimeSeriesData(
                time=data.time, value=pd.Series(zeros, copy=False)
            ),
            stat_sig_ts=TimeSeriesData(
                time=data.time, value=pd.Series(zeros, copy=False)
            ),
        )

    def _update_response(self, date: pd.Timestamp) -> None:
        """
        Updates the current response with data from date.
        """
        assert self.test_interval is not None
        assert self.control_interval is not None
        perc_change = PercentageChange(
            current=self.test_interval,
            previous=self.control_interval,
            use_corrected_scores=self.use_corrected_scores,
            min_perc_change=self.min_perc_change,
        )
        assert self.response is not None
        self.response.inplace_update(
            time=date,
            score=perc_change.score,
            ci_upper=perc_change.ci_upper,
            ci_lower=perc_change.ci_lower,
            pred=perc_change.mean_previous,
            anom_mag=perc_change.mean_difference,
            stat_sig=1.0 if perc_change.stat_sig else 0.0,
        )

    def _get_start_end_dates(
        self, data: TimeSeriesData
    ) -> Tuple[datetime, datetime, datetime, datetime]:
        """
        Gets the start and end dates of the initial interval.
        """

        last_dt = data.time.iloc[-1]

        test_end_dt = last_dt + pd.Timedelta(self.time_unit)

        assert self.n_test is not None
        test_start_dt = test_end_dt - self.n_test * pd.Timedelta(self.time_unit)

        assert self.n_test is not None
        assert self.n_control is not None
        control_start_dt = test_end_dt - (self.n_test + self.n_control) * pd.Timedelta(
            self.time_unit
        )

        control_end_dt = test_start_dt

        return control_start_dt, control_end_dt, test_start_dt, test_end_dt

    def _init_control_test(self, data: TimeSeriesData) -> None:
        """
        initializes the control and test intervals.
        """

        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)
        data_history = self.data_history
        assert data_history is not None

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = data_history

    def _update_control_test(self, data: TimeSeriesData) -> None:
        """
        Updates control and test with new data.
        """

        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)
        data_history = self.data_history
        assert data_history is not None
        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = data_history

    def _init_data(self, data: Optional[TimeSeriesData]) -> None:
        self.data_history = data

    def _update_data(self, data: TimeSeriesData) -> None:
        """
        Updates the data with new data.
        """
        data_history = self.data_history
        assert data_history is not None
        self.data_history = TimeSeriesData(
            time=pd.concat([data_history.time, data.time], copy=False),
            value=pd.concat([data_history.value, data.value], copy=False),
        )

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        Predict is not implemented.
        """

        raise ValueError("predict is not implemented, call fit_predict() instead")


class MultiStatSigDetectorModel(StatSigDetectorModel):

    """
    MultiStatSigDetectorModel is a multivariate version of the StatSigDetector.  It applies a univariate
    t-test to each of the components of the multivariate time series to see if the means between the control
    and test periods are significantly different.  Then it uses a false discovery rate controlling procedure
    rate (FDR) controlling procedure (https://en.wikipedia.org/wiki/False_discovery_rate#Controlling_procedure)
    to adjust the p-values, reducing the noise the the alerts that are triggered by the detector.  The default
    FDR controlling procedure is the Benjamini-Hochberg procedure, but this can be adjusted when initializing
    the model.

    Like with the StatSigDetector, we start with the history data, and then as for the current data,
    we apply a rolling window, adding one data point at a time from the current data,
    and detecting significant change. The T-statistics we return here are based on the adjusted p-values
    from the FDR controlling procedure.

    We suggest using n_control >= 30 to get good estimates

    Attributes:
        n_control: int, number of data points(or time units) of history to compare with
        n_test: int, number of points(or time_units) to compare the history with
        serialized_model: Optional, serialized json containing the parameters
        time_units: str, units of time used to measure the intervals. If not provided
                    we infer it from the provided data
        method: str, indicates the FDR controlling method used for adjusting the p-values.
            Defaults to 'fdr_bh' for Benjamini-Hochberg.  Inputs for other FDR controlling methods
            can be found at https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        rem_season: default value is False, if remove seasonality for historical data and data
        seasonal_period: str, default value is 'weekly'. Other possible values: 'daily', 'biweekly', 'monthly', 'yearly'
        skip_rescaling: bool. If we'd like skip rescaling p-values for multivariate timeseires when calling Percentagechange class
        use_corrected_scores: bool, default value is False, using original t-scores or correct t-scores.
        min_perc_change: float, minimum percentage change, for a non zero score. Score will be clipped to zero if the absolute value of the percentage chenge is less than this value
    >>> # Example usage:
    >>> # history and ts_pt are TimeSeriesData objects and history is larger
    >>> # than (n_control + n_test) so that we have sufficient history to
    >>> # run the detector
    >>> n_control = 28
    >>> n_test = 7
    >>> import random
    >>> control_time = pd.date_range(start='2018-01-01', freq='D', periods=(n_control + n_test))
    >>> test_time = pd.date_range(start='2018-02-05', freq='D', periods=n_test)
    >>> num_seq = 5
    >>> control_val = [np.random.randn(len(control_time)) for _ in range(num_seq)]
    >>> test_val = [np.random.randn(len(test_time)) for _ in range(num_seq)]
    >>> hist_ts =
        TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": control_time},
                    **{f"ts_{i}": control_val[i] for i in range(num_seq)},
                }
            )
        )
    >>> data_ts =
        TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": test_time},
                    **{f"ts_{i}": test_val[i] for i in range(num_seq)},
                }
            )
        )
    >>> ss_detect = MultiStatSigDetectorModel(n_control=n_control, n_test=n_test)
    >>> anom = ss_detect.fit_predict(data=data_ts, historical_data=hist_ts)
    """

    def __init__(
        self,
        n_control: Optional[int] = None,
        n_test: Optional[int] = None,
        serialized_model: Optional[bytes] = None,
        time_unit: Optional[str] = None,
        rem_season: bool = False,
        seasonal_period: str = "weekly",
        method: str = "fdr_bh",
        skip_rescaling: bool = False,
        use_corrected_scores: bool = False,
        min_perc_change: float = 0.0,
    ) -> None:

        StatSigDetectorModel.__init__(
            self,
            n_control=n_control,
            n_test=n_test,
            serialized_model=serialized_model,
            time_unit=time_unit,
            rem_season=rem_season,
            seasonal_period=seasonal_period,
            min_perc_change=min_perc_change,
        )
        self.method = method
        self.skip_rescaling = skip_rescaling
        self.use_corrected_scores = use_corrected_scores

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        This is the main working function.
        The function returns an AnomalyResponse object of length
        equal to the length of the data
        We require len(historical_data) > (n_control + n_test)

        Args:
            data: TimeSeriesData, A multivariate TimeSeriesData for which we are running the MultiStatSigDetectorModel
            historical_data: Optional[TimeSeriesData] Historical data used to do detection for initial points in data
        """

        if data.is_univariate():
            msg = "Input is univariate but MultiStatSigDetector expected multivariate input."
            logging.error(msg)
            raise ValueError(msg)

        self._set_time_unit(data=data, historical_data=historical_data)
        self.last_N = len(data)

        # this ensures we start with a default response of
        # the size of the data
        self._init_response(data)
        response = self.response
        assert response is not None
        # when there is no need to update
        # just return the initial response of zeros
        if not self._should_update(data=data, historical_data=historical_data):
            return response.get_last_n(self.last_N)

        # handle cases where there is either no historical  data, or
        # not enough historical data
        data, historical_data = self._handle_not_enough_history(
            data=data,
            historical_data=historical_data,
        )
        self.data = data

        # remove seasonality
        if self.rem_season:
            sh_data = SeasonalityHandler(
                data=data, seasonal_period=self.seasonal_period
            )

            self.data_season = sh_data.get_seasonality()
            data = sh_data.remove_seasonality()

            if historical_data:
                sh_hist_data = SeasonalityHandler(
                    data=historical_data,
                    seasonal_period=self.seasonal_period,
                )
                historical_data = sh_hist_data.remove_seasonality()

        # first initialize this with the historical data
        self._init_data(historical_data)
        self._init_control_test(data if historical_data is None else historical_data)
        # set the flag to true
        self.is_initialized = True

        # now run through the data to get the prediction

        for i in range(len(data)):
            ts_pt = TimeSeriesData(
                pd.DataFrame(
                    {
                        **{"time": data.time.iloc[i]},
                        **{c: data.value[c].iloc[i] for c in data.value.columns},
                    },
                    index=[0],
                    copy=False,
                )
            )
            self._update_data(ts_pt)
            self._update_control_test(ts_pt)
            self._update_response(ts_pt.time.iloc[0])

        # add seasonality back
        if self.rem_season:
            data_season = self.data_season
            confidence_band = response.confidence_band
            predicted_ts = response.predicted_ts
            assert data_season is not None
            assert confidence_band is not None
            assert predicted_ts is not None

            start_idx = len(response.scores) - len(data_season)
            datatime = response.scores.time

            self.response = AnomalyResponse(
                scores=response.scores,
                confidence_band=ConfidenceBand(
                    upper=TimeSeriesData(
                        time=datatime,
                        value=pd.concat(
                            [
                                pd.DataFrame(
                                    confidence_band.upper.value.values[:start_idx, :],
                                    copy=False,
                                ),
                                pd.DataFrame(
                                    np.asarray(
                                        confidence_band.upper.value.values[
                                            start_idx:, :
                                        ]
                                    )
                                    + np.asarray(data_season.value.values),
                                    copy=False,
                                ),
                            ],
                            copy=False,
                        ),
                    ),
                    lower=TimeSeriesData(
                        time=datatime,
                        value=pd.concat(
                            [
                                pd.DataFrame(
                                    confidence_band.lower.value.values[:start_idx, :],
                                    copy=False,
                                ),
                                pd.DataFrame(
                                    np.asarray(
                                        confidence_band.lower.value.values[
                                            start_idx:, :
                                        ]
                                    )
                                    + np.asarray(data_season.value.values),
                                    copy=False,
                                ),
                            ],
                            copy=False,
                        ),
                    ),
                ),
                predicted_ts=TimeSeriesData(
                    time=datatime,
                    value=pd.concat(
                        [
                            pd.DataFrame(
                                predicted_ts.value.values[:start_idx, :], copy=False
                            ),
                            pd.DataFrame(
                                np.asarray(predicted_ts.value.values[start_idx:, :])
                                + np.asarray(data_season.value.values),
                                copy=False,
                            ),
                        ],
                        copy=False,
                    ),
                ),
                anomaly_magnitude_ts=response.anomaly_magnitude_ts,
                stat_sig_ts=response.stat_sig_ts,
            )
        assert self.response is not None
        return self.response.get_last_n(self.last_N)

    def _init_response(self, data: TimeSeriesData) -> None:

        zeros_df = pd.DataFrame(
            {
                **{"time": data.time},
                **{c: pd.Series(np.zeros(len(data))) for c in data.value.columns},
            },
            copy=False,
        )

        init_df = pd.DataFrame(
            {
                **{"time": data.time},
                **{c: data.value[c].values for c in data.value.columns},
            },
            copy=False,
        )

        self.response = AnomalyResponse(
            scores=TimeSeriesData(zeros_df.copy()),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(init_df.copy()),
                lower=TimeSeriesData(init_df.copy()),
            ),
            predicted_ts=TimeSeriesData(init_df.copy()),
            anomaly_magnitude_ts=TimeSeriesData(zeros_df.copy()),
            stat_sig_ts=TimeSeriesData(zeros_df.copy()),
        )

    def _update_response(self, date: pd.Timestamp) -> None:
        """
        updates the current response with data from date
        """
        assert self.test_interval is not None
        assert self.control_interval is not None
        perc_change = PercentageChange(
            current=self.test_interval,
            previous=self.control_interval,
            method=self.method,
            skip_rescaling=self.skip_rescaling,
            use_corrected_scores=self.use_corrected_scores,
            min_perc_change=self.min_perc_change,
        )
        assert self.response is not None
        self.response.inplace_update(
            time=date,
            score=perc_change.score,
            ci_upper=perc_change.ci_upper,
            ci_lower=perc_change.ci_lower,
            pred=perc_change.mean_previous,
            anom_mag=perc_change.mean_difference,
            stat_sig=perc_change.stat_sig,
        )

    def _init_control_test(self, data: TimeSeriesData) -> None:
        """
        initializes the control and test intervals
        """
        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)
        data_history = self.data_history
        assert data_history is not None

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = data_history

    def _update_control_test(self, data: TimeSeriesData) -> None:
        """
        updates control and test with new data
        """
        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)
        data_history = self.data_history
        assert data_history is not None

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = data_history


class SeasonalityHandler:
    """
    SeasonalityHandler is a class that do timeseries STL decomposition for detecors
    Attributes:
        data: TimeSeriesData that need to be decomposed
        seasonal_period: str, default value is 'weekly'. Other possible values: 'daily', 'biweekly', 'monthly', 'yearly'

    >>> # Example usage:
    >>> from kats.utils.simulator import Simulator
    >>> sim = Simulator(n=120, start='2018-01-01')
    >>> ts = sim.level_shift_sim(cp_arr = [60], level_arr=[1.35, 1.05], noise=0.05, seasonal_period=7, seasonal_magnitude=0.575)
    >>> sh = SeasonalityHandler(data=ts, seasonal_period='weekly')
    >>> sh.get_seasonality()
    >>> sh.remove_seasonality()
    """

    def __init__(
        self,
        data: TimeSeriesData,
        seasonal_period: str = "weekly",
        **kwargs: Any,
    ) -> None:
        self.data = data

        _map = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30, "yearly": 365}
        if seasonal_period not in _map:
            msg = "Invalid seasonal_period, possible values are 'daily', 'weekly', 'biweekly', 'monthly', and 'yearly'"
            logging.error(msg)
            raise ValueError(msg)
        self.seasonal_period: int = _map[seasonal_period] * 24  # change to hours

        self.low_pass_jump_factor: float = kwargs.get("lpj_factor", 0.15)
        self.trend_jump_factor: float = kwargs.get("tj_factor", 0.15)

        self.frequency: pd.Timedelta = self.data.infer_freq_robust()
        self.frequency_sec: int = int(self.frequency.total_seconds())
        self.frequency_sec_str: str = str(self.frequency_sec) + "s"

        # calculate resample base in second level
        time0 = pd.to_datetime(self.data.time[0])
        # calculate remainder as resampling base
        resample_base_sec = (
            time0.day * 24 * 60 * 60
            + time0.hour * 60 * 60
            + time0.minute * 60
            + time0.second
        ) % self.frequency_sec

        self.decomposer_input: TimeSeriesData = self.data.interpolate(
            freq=self.frequency_sec_str,
            base=resample_base_sec,
        )

        data_time_idx = self.decomposer_input.time.isin(self.data.time)
        if len(self.decomposer_input.time[data_time_idx]) != len(self.data):
            raise ValueError(IRREGULAR_GRANULARITY_ERROR)

        self.period = int(
            self.seasonal_period * 60 * 60 / self.frequency.total_seconds()
        )

        if (
            int((self.period + 1) * self.low_pass_jump_factor) < 1
            or int((self.period + 1) * self.trend_jump_factor) < 1
        ):
            msg = "Invalid low_pass_jump or trend_jump_factor value, try a larger factor or try a larger seasonal_period"
            logging.error(msg)
            raise ValueError(msg)

        self.decomp: Optional[dict[str, Any]] = None

        self.ifmulti: bool = False
        # for multi-variate TS
        if len(self.data.value.values.shape) != 1:
            self.ifmulti = True
            self.num_seq: int = self.data.value.values.shape[1]

        self.data_season = TimeSeriesData(time=self.data.time, value=self.data.value)
        self.data_nonseason = TimeSeriesData(time=self.data.time, value=self.data.value)

    def _decompose(self) -> None:
        if not self.ifmulti:
            decomposer = TimeSeriesDecomposition(
                self.decomposer_input,
                period=self.period,
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=int((self.period + 1) * self.low_pass_jump_factor),
                seasonal_jump=1,
                trend_jump=int((self.period + 1) * self.trend_jump_factor),
            )

            self.decomp = decomposer.decomposer()
            return

        self._decompose_multi()

    def _decompose_multi(self) -> None:
        self.decomp = {}
        for i in range(self.num_seq):
            temp_ts = TimeSeriesData(
                time=self.decomposer_input.time,
                value=pd.Series(self.decomposer_input.value.values[:, i], copy=False),
            )
            decomposer = TimeSeriesDecomposition(
                temp_ts,
                period=self.period,
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=int((self.period + 1) * self.low_pass_jump_factor),
                seasonal_jump=1,
                trend_jump=int((self.period + 1) * self.trend_jump_factor),
            )
            assert self.decomp is not None
            self.decomp[str(i)] = decomposer.decomposer()

    def remove_seasonality(self) -> TimeSeriesData:
        if self.decomp is None:
            self._decompose()
        if not self.ifmulti:
            decomp = self.decomp
            assert decomp is not None
            data_time_idx = decomp["rem"].time.isin(self.data_nonseason.time)

            self.data_nonseason.value = pd.Series(
                decomp["rem"][data_time_idx].value
                + decomp["trend"][data_time_idx].value,
                name=self.data_nonseason.value.name,
                copy=False,
            )
            return self.data_nonseason
        decomp = self.decomp
        assert decomp is not None
        data_time_idx = decomp[str(0)]["rem"].time.isin(self.data_nonseason.time)
        for i in range(self.num_seq):
            self.data_nonseason.value.iloc[:, i] = pd.Series(
                decomp[str(i)]["rem"][data_time_idx].value
                + decomp[str(i)]["trend"][data_time_idx].value,
                name=self.data_nonseason.value.iloc[:, i].name,
                copy=False,
            )

        return self.data_nonseason

    def get_seasonality(self) -> TimeSeriesData:

        if self.decomp is None:
            self._decompose()
        decomp = self.decomp
        assert decomp is not None
        if not self.ifmulti:
            data_time_idx = decomp["seasonal"].time.isin(self.data_season.time)
            self.data_season.value = pd.Series(
                decomp["seasonal"][data_time_idx].value,
                name=self.data_season.value.name,
                copy=False,
            )
            return self.data_season

        data_time_idx = decomp[str(0)]["seasonal"].time.isin(self.data_season.time)
        for i in range(self.num_seq):
            self.data_season.value.iloc[:, i] = pd.Series(
                decomp[str(i)]["seasonal"][data_time_idx].value,
                name=self.data_season.value.iloc[:, i].name,
                copy=False,
            )

        return self.data_season
