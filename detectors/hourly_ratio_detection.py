#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from infrastrategy.kats.consts import TimeSeriesChangePoint, TimeSeriesData
from infrastrategy.kats.detector import Detector
from sklearn.covariance import MinCovDet


class HourlyRatioDetector(Detector):
    """
    Hourly Ratio Anormaly detector

    This detector is used to detect abnormal intra-day hourly ratio patterns. This detector takes TimeSeriesData
    as input and returns a list of TimeSeriesChangePoint representing the abnormal dates.

    :Parameters:
    data: TimeSeriesData
        The data to be examed. It should have hour-level granularity.
    freq: Optional[str]
        The data frequency (following the naming conventions of pandas).
        This will be inferred if it is not specified by user. Should be at least of hourly granularity.
        Common frequency expressions:
            'H': hourly frequency
            'T': minutely frequency
            'S': secondly frequency

    aggregate: Optional[str]
        The aggregation method for aggregating data to hourly level data. We currently support: min, max, sum, mean.

    :Example:
    >>> from infrastrategy.kats.detectors.hourly_ratio_detection import HourlyRatioDetector
    >>> # create the object
    >>> hr = HourlyRatioDetector(data)
    >>> # or one can create the object with more info: hr = HourlyRatioDetector(data, freq = "T", aggregate = 'mean')
    >>> # run detect method
    >>> anomlies=hr.detector()
    >>> # plot anomalies of weekday 3
    >>> hr = hr.plot(weekday = 3)

    :Methods:
    """

    def __init__(
        self,
        data: TimeSeriesData,
        freq: Optional[str] = None,
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
        self.ratiodf = None
        self.incomplete_dates = None
        self.anomaly_dates = None
        self.freq = freq
        self.aggregate = aggregate
        self._valid_frequency()

    def _valid_frequency(self) -> None:
        """
        Valid frequency of the input timeseries. If freq is given by user, then use the freq defined by user, otherwise we use pd.infer_freq to infer data frequencey.
        Data freq should be at least hourly level. For data with the granularity finner than hourly level, aggregate function should be given using aggregate attribuate.
        Now only support aggregation functions: min, max, sum, mean.
        """
        lower_granularity = ["T", "S", "L", "U", "N"]
        if self.freq is None:
            freq = pd.infer_freq(self.data.time)
            if freq is None:
                msg = "Cannot infer data frequency, please specify it."
                logging.error(msg)
                raise ValueError(msg)
            else:
                self.freq = freq
        if self.freq == "H":
            msg = "Input data is hourly data."
            logging.info(msg)
            return
        else:
            for level in lower_granularity:
                if level in self.freq:
                    msg = "Input data granularity is {} and we can continue processing using aggregation function.".format(
                        self.freq
                    )
                    logging.info(msg)
                    if self.aggregate is None:
                        msg = "Aggregation method is missing."
                        logging.error(msg)
                        raise ValueError(msg)
                    elif self.aggregate in ["min", "max", "sum", "mean"]:
                        msg = "Aggregation method is {}.".format(self.aggregate)
                        logging.info(msg)
                        return
                    else:
                        msg = "Aggregation methd {} is not implemented.".format(
                            self.aggregate
                        )
                        logging.error(msg)
                        raise ValueError(msg)
            msg = "Input data granularity is {}, which should be at least hourly data.".format(
                self.freq
            )
            logging.error(msg)
            raise ValueError(msg)

    def _preprocess(self):
        """
        preprocess input data: filter out dates with incomplete data, aggregate data to hourly level if necessary
        and calculate hourly ratio.
        """
        if self.ratiodf is None:
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
            self.ratiodf = df
        return

    def _mahalanobis_test(
        self, obs: np.array, median: np.array, cov: np.array, alpha: float = 0.01
    ) -> Tuple[np.array, np.array]:
        """
        mahalanobis test function

        :Parameters:
        obs: np.array
            data to be tested
        median: np.array
            medians used to build centeralize test data
        cov: np.array
            covariance matrix
        alpha: float
            significance level for the mahalanobis test. We take the instance with pvalue<=alpha as
            an abnormal point.
        :Returns:
        lab: np.array
            whether the corresponding instance is abnormal
        pvalue: np.array
            pvalues for each instance
        """
        diff = obs - median
        scores = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
        pvalue = 1.0 - scipy.stats.chi2(df=diff.shape[1]).cdf(scores)
        lab = pvalue <= alpha
        return (lab, pvalue)

    def detector(self, *args, **kwargs) -> List[TimeSeriesChangePoint]:
        if self.ratiodf is None:
            self._preprocess()
        anomaly = []
        pvalues = []
        for w in range(7):
            obs = self.ratiodf[self.ratiodf["weekday"] == w][
                "hourly_ratio"
            ].values.reshape(-1, 24)
            dates = np.unique(self.ratiodf[self.ratiodf["weekday"] == w]["date"].values)
            # we omit the last dimension due to linearity constrant
            median = np.median(obs, axis=0)
            median = (median / np.sum(median) * 24)[:-1]
            kwargs["assume_centered"] = True
            kwargs["support_fraction"] = kwargs.get("support_fraction", 0.9)
            cov = MinCovDet(**kwargs).fit(obs[:, :-1] - median)
            lab, p = self._mahalanobis_test(obs[:, :-1], median, cov.covariance_)
            anomaly.extend(list(dates[lab]))
            pvalues.extend(p[lab])
        anomaly = [
            TimeSeriesChangePoint(anomaly[i], anomaly[i], 1.0 - pvalues[i])
            for i in range(len(anomaly))
        ]
        self.anomaly_dates = anomaly
        return anomaly

    def plot(self, weekday: int = 0):
        """
        plot function for the results

        :Parameters:
        weekday: int
            weekday label which should be an int and should be in [0,6]
        """
        if self.anomaly_dates is None:
            msg = "Please run detector method first."
            logging.error(msg)
            raise ValueError(msg)
        anomaly_dates = [t.start_time for t in self.anomaly_dates]
        anomaly_dates = set(anomaly_dates)
        obs = self.ratiodf[self.ratiodf["weekday"] == weekday][
            "hourly_ratio"
        ].values.reshape(-1, 24)
        dates = np.unique(
            self.ratiodf[self.ratiodf["weekday"] == weekday]["date"].values
        )
        labs = [(t in anomaly_dates) for t in dates]
        print("# of anomaly dates: {}".format(np.sum(labs)))
        for i in range(len(obs)):
            if not labs[i]:
                plt.plot(obs[i], "--", color="silver", alpha=0.5)
            else:
                plt.plot(obs[i], "--", color="navy", label=str(dates[i]))
        plt.title("Hourly Ratio Plot for Weeday {}".format(weekday))
        plt.legend()
        plt.show()
        return
