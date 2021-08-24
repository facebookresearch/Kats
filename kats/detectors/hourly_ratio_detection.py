# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector

# pyre-ignore[21]: Could not find a name `chi2` defined in module `scipy.stats`.
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

"""A module for detecting abnormal in hourly ratio.

This module contains the class :class:`HourlyRatioDetector`, which detects the abnormal intra-day hourly ratio patterns.
"""


class HourlyRatioDetector(Detector):
    """Hourly Ratio Anormaly detector.

    This detector detects the abnormal intra-day hourly ratio patterns. This detector takes TimeSeriesDataas input and returns a list of TimeSeriesChangePoint representing the abnormal dates.
    The detection algorithm assumes that the hourly ratio of each day should follow a multivariate normal distribution, and we utilize Mahalanobis distance tests to detect abnormal days.
    This class provides detector and plot.

    Attributes:
        data: A :class:`kats.consts.TimeSeriesData` object representing the data to be examed, which should be of hour-level granularity.
        freq: Optional; A string or a `pandas.Timedelta` object representing the data frequency (following the naming conventions of pandas). Can be 'H' (hourly frequency), 'T' minutely frequency, 'S' secondly frequency or any other frequency finer than hourly frequency.
            Default is None, in which case the frequency will be infered by infer_freq_robust.
        aggregate: Optional; A string representing the aggregation method for aggregating data to hourly level data. Can be 'min', 'max', 'sum', 'mean' or None. Default is None, which means no aggregation.

    Sample Usage:
        >>> hr = HourlyRatioDetector(data)
        >>> anomlies=hr.detector()
        >>> hr = hr.plot(weekday = 3) # Plot the anomalies of weekday 3
    """

    def __init__(
        self,
        data: TimeSeriesData,
        freq: Union[str, pd.Timedelta, None] = None,
        aggregate: Optional[str] = None,
    ) -> None:
        super(HourlyRatioDetector, self).__init__(data=data)
        if len(data) == 0:
            msg = "Input data is empty."
            logging.error(msg)
            raise ValueError(msg)

        if not self.data.is_univariate():
            msg = "Only support univariate time series, but get {}.".format(
                type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self._ratiodf = None
        self.incomplete_dates = None
        self.anomaly_dates = None
        self.freq = freq
        self.aggregate = aggregate
        self._valid_frequency()

    def _valid_frequency(self) -> None:
        """Valid frequency of the input timeseries.

        If freq is given by user, then use the freq defined by user, otherwise we use ts.infer_freq_robust() to infer data frequencey.
        Data freq should be at least hourly level. For data with the granularity finner than hourly level, aggregate function should be given using aggregate attribuate.
        Now only support aggregation functions: min, max, sum, mean.
        """

        lower_granularity = ["T", "S", "L", "U", "N"]
        if self.freq is None:
            self.freq = self.data.infer_freq_robust()
        if self.freq == "H" or (
            isinstance(self.freq, pd.Timedelta) and self.freq.value == 3600000000000
        ):
            msg = "Input data is hourly data."
            logging.info(msg)
            return
        if isinstance(self.freq, str):
            for level in lower_granularity:
                # pyre-fixme[58]: `in` is not supported for right operand type
                #  `Optional[str]`.
                if level in self.freq:
                    msg = "Input data granularity is {} and we can continue processing using aggregation function.".format(
                        self.freq
                    )
                    logging.info(msg)
        elif isinstance(self.freq, pd.Timedelta) and self.freq.value < 3600000000000:
            pass
        else:
            msg = "Time series should be of hourly or finer granularity."
            logging.error(msg)
            raise ValueError(msg)

        if self.aggregate is None:
            msg = "Aggregation method is missing."
            logging.error(msg)
            raise ValueError(msg)
        elif self.aggregate in ["min", "max", "sum", "mean"]:
            msg = "Aggregation method is {}.".format(self.aggregate)
            logging.info(msg)
            return
        else:
            msg = "Aggregation methd {} is not implemented.".format(self.aggregate)
            logging.error(msg)
            raise ValueError(msg)

    def _preprocess(self):
        """preprocess input data.

        There are two steps for preprocess: 1) filter out dates with incomplete data, aggregate data to hourly level if necessary; and 2) calculate hourly ratio.
        """

        if self._ratiodf is None:
            df = self.data.to_dataframe()
            df.dropna(inplace=True)
            df.sort_values("time", inplace=True)
            df["date"] = df["time"].dt.date
            df["hour"] = df["time"].dt.hour
            df["weekday"] = df["time"].dt.weekday
            # aggregate the data to hourly level.
            if self.freq != "H" and self.aggregate is not None:
                df = (
                    df.groupby(["date", "hour", "weekday"])["value"]
                    .agg(self.aggregate)
                    .reset_index()
                )
                msg = "Successfully aggregate data to hourly level using {}".format(
                    self.aggregate
                )
                logging.info(msg)
            df["counts"] = df.groupby("date")["hour"].transform("count")
            # filter out dates with less than 24 observations
            incomplete_dates = df["date"][df["counts"] < 24].unique()
            self.incomplete_dates = [
                TimeSeriesChangePoint(t, t, 1.0) for t in incomplete_dates
            ]
            df = df[df["counts"] == 24]
            if len(df) == 0:
                msg = "Data should have hour-level granularity."
                logging.error(msg)
                raise ValueError(msg)
            df["hourly_mean"] = df.groupby("date")["value"].transform("sum")
            df["hourly_ratio"] = df["value"] / df["hourly_mean"]
            self._ratiodf = df
        return

    def _mahalanobis_test(
        self,
        obs: np.ndarray,
        median: np.ndarray,
        cov: np.ndarray,
        alpha: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """mahalanobis test function.

        Args:
            obs: A :class:`numpy.ndarray` object storing the data to be tested.
            median: A :class:`numpy.ndarray` object storing the medians used to build centeralize test data.
            cov: A :class:`numpy.ndarray` object representing the covariance matrix.
            alpha: A float representing the significance level for the Mahalanobis test. We take the instance with pvalue<=alpha as an abnormal point.

        Returns:
            lab: A :class:`numpy.ndarray` object of booleans representing whether the corresponding instance is abnormal or not.
            pvalue: A :class:`numpy.ndarray` object storing the pvalues of tests of each instance.
        """

        diff = obs - median
        scores = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
        # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
        pvalue = 1.0 - chi2(df=diff.shape[1]).cdf(scores)
        lab = pvalue <= alpha
        # pyre-fixme[7]: Expected `Tuple[np.ndarray, np.ndarray]` but got `Tuple[bool, float]`.
        return (lab, pvalue)

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self, support_fraction=0.9) -> List[TimeSeriesChangePoint]:
        """Run detection algorithm.

        Args:
            support_fraction: Optional; A float representing the support_fraction for MinCovDet class from scikit-learn. Default is 0.9.
                              See https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html for more details.

        Returns:
            A list of TimeSeriesChangePoint representing the anormal dates.
        """

        if self._ratiodf is None:
            self._preprocess()
        anomaly = []
        pvalues = []
        for w in range(7):
            obs = self._ratiodf[self._ratiodf["weekday"] == w][
                "hourly_ratio"
            ].values.reshape(-1, 24)
            dates = np.unique(
                self._ratiodf[self._ratiodf["weekday"] == w]["date"].values
            )
            # we omit the last dimension due to linearity constrant
            median = np.median(obs, axis=0)
            median = (median / np.sum(median) * 24)[:-1]
            cov = MinCovDet(
                assume_centered=True, support_fraction=support_fraction
            ).fit(obs[:, :-1] - median)
            lab, p = self._mahalanobis_test(obs[:, :-1], median, cov.covariance_)
            anomaly.extend(list(dates[lab]))
            pvalues.extend(p[lab])
        anomaly = [
            TimeSeriesChangePoint(anomaly[i], anomaly[i], 1.0 - pvalues[i])
            for i in range(len(anomaly))
        ]
        self.anomaly_dates = anomaly
        return anomaly

    def plot(self, weekday: int = 0) -> None:
        """plot the detection results.

        Args:
            weekday: Optional; An integer representing the weekday label, which should be in [0,6]. Default is 0.

        Returns:
            None.
        """
        if self.anomaly_dates is None:
            msg = "Please run detector method first."
            logging.error(msg)
            raise ValueError(msg)
        anomaly_dates = [t.start_time for t in self.anomaly_dates]
        anomaly_dates = set(anomaly_dates)
        obs = self._ratiodf[self._ratiodf["weekday"] == weekday][
            "hourly_ratio"
        ].values.reshape(-1, 24)
        dates = np.unique(
            self._ratiodf[self._ratiodf["weekday"] == weekday]["date"].values
        )
        labs = [(t in anomaly_dates) for t in dates]
        logging.info("# of anomaly dates: {}".format(np.sum(labs)))
        for i in range(len(obs)):
            if not labs[i]:
                plt.plot(obs[i], "--", color="silver", alpha=0.5)
            else:
                plt.plot(obs[i], "--", color="navy", label=str(dates[i]))
        plt.title("Hourly Ratio Plot for Weeday {}".format(weekday))
        plt.legend()
        plt.show()
        return
