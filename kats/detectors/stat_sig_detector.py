# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kats.consts import TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import (
    AnomalyResponse,
    ChangePointInterval,
    ConfidenceBand,
    PercentageChange,
)

"""Statistical Significance Detector Module
This module contains simple detectors that apply a t-test over a rolling window to compare
check if there is a statistically significant increase or decrease between the control and test
time periods.  In addition to the univariate version of this test, this module includes a
multivariate version that uses a false discovery rate (FDR) controlling procedure to reduce noise.
"""


def to_datetime(dt: np.datetime64) -> datetime:
    """
    Helper function to convert from np.datetime64 which
    is used by pandas pd.to_datetime to datetime in datetime
    library
    """

    return datetime.utcfromtimestamp(dt.tolist() / 1e9)


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
    """

    def __init__(
        self,
        n_control: Optional[int] = None,
        n_test: Optional[int] = None,
        serialized_model: Optional[bytes] = None,
        # pyre-fixme[9]: time_unit has type `str`; used as `None`.
        time_unit: str = None,
    ) -> None:

        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.n_test = model_dict["n_test"]
            self.n_control = model_dict["n_control"]
            self.time_unit = model_dict["time_unit"]

        else:
            self.n_test = n_test
            self.n_control = n_control
            self.time_unit = time_unit

        if (self.n_control is None) or (self.n_test is None):
            raise ValueError(
                """
            You must either provide serialized model or values for control and test
            intervals.
            """
            )

        self.control_interval = None
        self.test_interval = None
        self.response = None

        self.is_initialized = False  # flag on whether initialized or not
        self.last_N = 0  # this is the size of the last chunk of data we saw
        self.data_history = None

    def serialize(self) -> bytes:
        """
        Serializes by putting model parameters in a json
        """
        model_dict = {
            "n_control": self.n_control,
            "n_test": self.n_test,
            "time_unit": self.time_unit,
        }

        return json.dumps(model_dict).encode("utf-8")

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
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

        # pyre-fixme[6]: Expected `TimeSeriesData` for 2nd param but got
        #  `Optional[TimeSeriesData]`.
        self._set_time_unit(data=data, historical_data=historical_data)

        self.last_N = len(data)

        # this ensures we start with a default response of
        # the size of the data
        self._init_response(data)

        # when there is no need to update
        # just return the initial response of zeros
        if not self._should_update(data=data, historical_data=historical_data):
            return self.response

        # handle cases where there is either no historical  data, or
        # not enough historical data
        data, historical_data = self._handle_not_enough_history(
            data=data,
            # pyre-fixme[6]: Expected `TimeSeriesData` for 2nd param but got
            #  `Optional[TimeSeriesData]`.
            historical_data=historical_data,
        )
        # pyre-fixme[16]: `StatSigDetectorModel` has no attribute `data`.
        self.data = data

        # first initialize this with the historical data
        self._init_data(historical_data)
        self._init_control_test(historical_data)
        # set the flag to true
        self.is_initialized = True

        # now run through the data to get the prediction

        for i in range(len(data)):
            current_time = data.time.iloc[i]

            ts_pt = TimeSeriesData(
                time=pd.Series(current_time), value=pd.Series(data.value.iloc[i])
            )

            self._update_data(ts_pt)
            self._update_control_test(ts_pt)
            self._update_response(ts_pt.time.iloc[0])

        return self.response.get_last_n(self.last_N)

    def visualize(self):
        """Function to visualize the result of the StatSigDetectorModel."""
        sns.set()
        plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(211)
        N_data = len(self.data)

        ax1.plot(self.data.time, self.data.value.values, "k-")
        ax1.plot(
            self.response.predicted_ts.time.iloc[-N_data:],
            self.response.predicted_ts.value.values[-N_data:],
            "r--",
        )
        upper_ci = self.response.confidence_band.upper
        lower_ci = self.response.confidence_band.lower

        ax1.fill_between(
            upper_ci.time.iloc[-N_data:],
            lower_ci.value.values[-N_data:],
            upper_ci.value.values[-N_data:],
            facecolor="blue",
            alpha=0.25,
        )

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(
            self.response.scores.time.iloc[-N_data:],
            self.response.scores.value.values[-N_data:],
            "r-",
        )

    def _set_time_unit(self, data: TimeSeriesData, historical_data: TimeSeriesData):
        """
        if the time unit is not set, this function tries to set it.
        """

        # first try historical data
        if not self.time_unit:
            self.time_unit = pd.infer_freq(data.time)
        if not self.time_unit and historical_data:
            pd.infer_freq(historical_data.time)

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

        if end_time < (
            start_time
            + pd.Timedelta(
                value=(self.n_control + self.n_test - 1), unit=self.time_unit
            )
        ):
            return False
        else:
            return True

    def _handle_not_enough_history(
        self, data: TimeSeriesData, historical_data: TimeSeriesData
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        """
        Handles the case when we don't have enough historical data.
        If we don't need to update, this does not do anything.
        If we need to update, this divides up the data accordingly.
        """

        if self.time_unit is None:
            raise ValueError("time_unit variable cannot be None")

        # if we are not upating, we should not do anything
        if not self._should_update(data=data, historical_data=historical_data):
            return data, historical_data

        num_hist_points = self.n_control + self.n_test - 1

        # if we have enough history, we should not do anything
        if historical_data:
            history_first = historical_data.time.iloc[0]
            history_last = historical_data.time.iloc[-1]
            min_history_last = history_first + pd.Timedelta(
                value=num_hist_points, unit=self.time_unit
            )

            if history_last >= min_history_last:
                return data, historical_data

        # when no historical data, divide the data into historical and not
        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data)
            total_data = historical_data

        first_dt = total_data.time.iloc[0]  # first date of the data

        last_dt = first_dt + pd.Timedelta(value=num_hist_points, unit=self.time_unit)

        historical_data = TimeSeriesData(
            time=total_data.time[total_data.time < last_dt],
            value=total_data.value[total_data.time < last_dt],
        )

        data = TimeSeriesData(
            time=total_data.time[total_data.time >= last_dt],
            value=total_data.value[total_data.time >= last_dt],
        )

        return data, historical_data

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel` inconsistently.
    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """
        Fit can be called during priming. It's a noop for us.
        """

        return

    def _init_response(self, data: TimeSeriesData):
        """
        Initializes a default response.
        """

        zeros = np.zeros(len(data))
        self.response = AnomalyResponse(
            scores=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=data.time, value=pd.Series(data.value.values)
                ),
                lower=TimeSeriesData(
                    time=data.time, value=pd.Series(data.value.values)
                ),
            ),
            predicted_ts=TimeSeriesData(
                time=data.time, value=pd.Series(data.value.values)
            ),
            anomaly_magnitude_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
            stat_sig_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
        )

    def _update_response(self, date: pd.Timestamp):
        """
        Updates the current response with data from date.
        """

        perc_change = PercentageChange(
            current=self.test_interval, previous=self.control_interval
        )

        self.response.inplace_update(
            time=date,
            score=perc_change.score,
            ci_upper=perc_change.ci_upper,
            ci_lower=perc_change.ci_lower,
            pred=perc_change.mean_previous,
            anom_mag=perc_change.mean_difference,
            stat_sig=1.0 if perc_change.stat_sig else 0.0,
        )

    def _get_start_end_dates(self, data: TimeSeriesData):
        """
        Gets the start and end dates of the initial interval.
        """

        last_dt = data.time.iloc[-1]

        test_end_dt = last_dt + pd.Timedelta(value=1, unit=self.time_unit)
        test_start_dt = test_end_dt - pd.Timedelta(
            value=self.n_test, unit=self.time_unit
        )
        control_start_dt = test_end_dt - pd.Timedelta(
            value=(self.n_test + self.n_control), unit=self.time_unit
        )
        control_end_dt = test_start_dt

        return control_start_dt, control_end_dt, test_start_dt, test_end_dt

    def _init_control_test(self, data: TimeSeriesData):
        """
        initializes the control and test intervals.
        """

        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = self.data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = self.data_history

    def _update_control_test(self, data: TimeSeriesData):
        """
        Updates control and test with new data.
        """

        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)

        self.test_interval.data = self.data_history

        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)

        self.control_interval.data = self.data_history

    def _init_data(self, data: TimeSeriesData):
        self.data_history = data

    def _update_data(self, data: TimeSeriesData):
        """
        Updates the data with new data.
        """

        self.data_history = TimeSeriesData(
            # pyre-fixme[6]: Expected `Union[None,
            #  pd.core.indexes.datetimes.DatetimeIndex, pd.core.series.Series]` for 1st
            #  param but got `DataFrame`.
            time=pd.concat([self.data_history.time, data.time]),
            value=pd.concat([self.data_history.value, data.value]),
        )

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
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
        # pyre-fixme[9]: time_unit has type `str`; used as `None`.
        time_unit: str = None,
        method: str = "fdr_bh",
    ) -> None:

        StatSigDetectorModel.__init__(
            self, n_control, n_test, serialized_model, time_unit
        )
        self.method = method

    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
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

        # pyre-fixme[6]: Expected `TimeSeriesData` for 2nd param but got
        #  `Optional[TimeSeriesData]`.
        self._set_time_unit(data=data, historical_data=historical_data)

        self.last_N = len(data)

        # this ensures we start with a default response of
        # the size of the data
        self._init_response(data)

        # when there is no need to update
        # just return the initial response of zeros
        if not self._should_update(data=data, historical_data=historical_data):
            return self.response.get_last_n(self.last_N)

        # handle cases where there is either no historical  data, or
        # not enough historical data
        data, historical_data = self._handle_not_enough_history(
            data=data,
            # pyre-fixme[6]: Expected `TimeSeriesData` for 2nd param but got
            #  `Optional[TimeSeriesData]`.
            historical_data=historical_data,
        )
        # pyre-fixme[16]: `MultiStatSigDetectorModel` has no attribute `data`.
        self.data = data

        # first initialize this with the historical data
        self._init_data(historical_data)
        self._init_control_test(historical_data)
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
                )
            )
            self._update_data(ts_pt)
            self._update_control_test(ts_pt)
            self._update_response(ts_pt.time.iloc[0])

        return self.response.get_last_n(self.last_N)

    def _init_response(self, data: TimeSeriesData):

        zeros_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": data.time},
                    **{c: pd.Series(np.zeros(len(data))) for c in data.value.columns},
                }
            )
        )

        init_ts = TimeSeriesData(
            pd.DataFrame(
                {
                    **{"time": data.time},
                    **{c: data.value[c].values for c in data.value.columns},
                }
            )
        )

        self.response = AnomalyResponse(
            scores=zeros_ts,
            confidence_band=ConfidenceBand(upper=init_ts, lower=init_ts),
            predicted_ts=init_ts,
            anomaly_magnitude_ts=zeros_ts,
            stat_sig_ts=zeros_ts,
        )

    # pyre-fixme[14]: `_update_response` overrides method defined in
    #  `StatSigDetectorModel` inconsistently.
    def _update_response(self, date: datetime):
        """
        updates the current response with data from date
        """
        perc_change = PercentageChange(
            current=self.test_interval,
            previous=self.control_interval,
            method=self.method,
        )

        self.response.inplace_update(
            time=date,
            score=perc_change.score,
            ci_upper=perc_change.ci_upper,
            ci_lower=perc_change.ci_lower,
            pred=perc_change.mean_previous,
            anom_mag=perc_change.mean_difference,
            stat_sig=perc_change.stat_sig,
        )

    def _init_control_test(self, data: TimeSeriesData):
        """
        initializes the control and test intervals
        """
        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = self.data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = self.data_history

    def _update_control_test(self, data: TimeSeriesData):
        """
        updates control and test with new data
        """
        (
            control_start_dt,
            control_end_dt,
            test_start_dt,
            test_end_dt,
        ) = self._get_start_end_dates(data)

        self.test_interval = ChangePointInterval(test_start_dt, test_end_dt)
        self.test_interval.data = self.data_history
        self.control_interval = ChangePointInterval(control_start_dt, control_end_dt)
        self.control_interval.data = self.data_history
