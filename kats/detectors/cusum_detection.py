#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUSUM stands for cumulative sum, it is a changepoint detection algorithm.

In the Kats implementation, it has two main components:
  1. Locate the change point: The algorithm iteratively estimates the means before and
  after the change point and finds the change point maximizing/minimizing the cusum
  value until the change point has converged. The starting point for the change point is
  at the middle.

  2. Hypothesis testing: Conducting log likelihood ratio test where the null hypothesis has
  no change point with one mean and the alternative hypothesis has a change point with
  two means.
And here are a few things worth mentioning:
  * We assume there is only one increase/decrease change point;
  * We use Gaussian distribution as the underlying model to calculate the cusum value and
  conduct the hypothesis test;

Typical usage example:

>>> # Univariate CUSUM
>>> timeseries = TimeSeriesData(...)
>>> detector = CusumDetector(timeseries)
>>> #Run detector
>>> changepoints = detector.detector()
>>> # Plot the results
>>> detector.plot(changepoints)

The usage is the same for multivariate CUSUM except that the time series needs to be multivariate
and that the plotting functions are not yet supported for this use case.

"""

import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import (
    TimeSeriesChangePoint,
    TimeSeriesData,
)
from kats.detectors.detector import Detector
# pyre-fixme[21]: Could not find name `chi2` in `scipy.stats`.
from scipy.stats import chi2  # @manual


pd.options.plotting.matplotlib.register_converters = True

# Constants
CUSUM_DEFAULT_ARGS = {
    "threshold": 0.01,
    "max_iter": 10,
    "delta_std_ratio": 1.0,
    "min_abs_change": 0,
    "start_point": None,
    "change_directions": None,
    "interest_window": None,
    "magnitude_quantile": None,
    "magnitude_ratio": 1.3,
    "magnitude_comparable_day": 0.5,
    "return_all_changepoints": False,
    "remove_seasonality": False,
}


class CUSUMMetadata:
    """CUSUM metadata

    This is the metadata of the changepoint returned by CusumDetectors

    Attributes:
        direction: a str stand for the changepoint change direction 'increase' or
            'decrease'.
        cp_index: an int for changepoint index.
        _mu0: a float indicates the mean before changepoint.
        _mu1: a float indicates the mean after changepoint.
        delta: _mu1 - _mu0.
        llr: log likelihood ratio.
        llr_int: log likelihood ratio in the interest window.
        regression_detected: a bool indicates if regression detected.
        stable_changepoint: a bool indicates if we have a stable changepoint when locating
            the changepoint.
        p_value: p_value of the changepoint.
        p_value_int: p_value of the changepoint in the interest window.
    """

    def __init__(
        self,
        direction: str,
        cp_index: int,
        mu0: float,
        mu1: float,
        delta: float,
        llr_int: float,
        llr: float,
        regression_detected: bool,
        stable_changepoint: bool,
        p_value: float,
        p_value_int: float,
    ):
        self._direction = direction
        self._cp_index = cp_index
        self._mu0 = mu0
        self._mu1 = mu1
        self._delta = delta
        self._llr_int = llr_int
        self._llr = llr
        self._regression_detected = regression_detected
        self._stable_changepoint = stable_changepoint
        self._p_value = p_value
        self._p_value_int = p_value_int

    @property
    def direction(self):
        return self._direction

    @property
    def cp_index(self):
        return self._cp_index

    @property
    def mu0(self):
        return self._mu0

    @property
    def mu1(self):
        return self._mu1

    @property
    def delta(self):
        return self._delta

    @property
    def llr(self):
        return self._llr

    @property
    def llr_int(self):
        return self._llr_int

    @property
    def regression_detected(self):
        return self._regression_detected

    @property
    def stable_changepoint(self):
        return self._stable_changepoint

    @property
    def p_value(self):
        return self._p_value

    @property
    def p_value_int(self):
        return self._p_value_int

    def __str__(self):
        return f"CUSUMMetadata(direction: {self.direction}, index: {self.cp_index}, delta: {self.delta}, regression_detected: {self.regression_detected}, stable_changepoint: {self.stable_changepoint})"


class CUSUMDetector(Detector):
    """Univariate CUSUM detector for level shifts

    Use cusum to detect changes, the algorithm is based on likelihood ratio cusum.
    See https://www.fs.isy.liu.se/Edu/Courses/TSFS06/PDFs/Basseville.pdf for details.
    These detector is used to detect mean changes in Normal Distribution.

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`; The input time series data
        is_multivariate: Optional; bool; should be False unless running MultiCUSUMDetector
    """

    def __init__(self, data: TimeSeriesData, is_multivariate: bool = False) -> None:
        super(CUSUMDetector, self).__init__(data=data)
        if not self.data.is_univariate() and not is_multivariate:
            msg = "CUSUMDetector only supports univariate time series, but got {type}.  For multivariate time series, use MultiCUSUMDetector".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def _get_change_point(
        self, ts: np.ndarray, max_iter: int, start_point: int, change_direction: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find change point in the timeseries
        """
        # pyre-fixme[16]: `CUSUMDetector` has no attribute `interest_window`.
        interest_window = self.interest_window
        # locate the change point using cusum method
        if change_direction == "increase":
            changepoint_func = np.argmin
            logging.debug("Detecting increase changepoint.")
        if change_direction == "decrease":
            changepoint_func = np.argmax
            logging.debug("Detecting decrease changepoint.")
        n = 0
        # use the middle point as the initial change point to esitimat mu0 and mu1
        if interest_window:
            ts_int = ts[self.interest_window[0] : interest_window[1]]
        else:
            ts_int = ts

        if start_point is None:
            cusum_ts = np.cumsum(ts_int - np.mean(ts_int))
            changepoint = min(changepoint_func(cusum_ts), len(ts_int) - 2)
        else:
            changepoint = start_point

        # iterate until the changepoint converage
        while n < max_iter:
            n += 1
            mu0 = np.mean(ts_int[: (changepoint + 1)])
            mu1 = np.mean(ts_int[(changepoint + 1) :])
            mean = (mu0 + mu1) / 2
            # here is where cusum is happening
            cusum_ts = np.cumsum(ts_int - mean)
            next_changepoint = max(1, min(changepoint_func(cusum_ts), len(ts_int) - 2))
            if next_changepoint == changepoint:
                break
            else:
                changepoint = next_changepoint

        if n == max_iter:
            logging.info("Max iteration reached and no stable changepoint found.")
            stable_changepoint = False
        else:
            stable_changepoint = True

        # llr in interest window
        if self.interest_window:
            llr_int = self._get_llr(
                ts_int, {"mu0": mu0, "mu1": mu1, "changepoint": changepoint}
            )
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            pval_int = 1 - chi2.cdf(llr_int, 2)
            delta_int = mu1 - mu0
        else:
            llr_int = np.inf
            pval_int = np.NaN
            delta_int = None

        # full time chnagepoint and mean
        changepoint += interest_window[0] if interest_window else 0
        mu0 = np.mean(ts[: (changepoint + 1)])
        mu1 = np.mean(ts[(changepoint + 1) :])
        return {
            "changepoint": changepoint,
            "mu0": mu0,
            "mu1": mu1,
            "changetime": self.data.time[changepoint],
            "stable_changepoint": stable_changepoint,
            "delta": mu1 - mu0,
            "llr_int": llr_int,
            "p_value_int": pval_int,
            "delta_int": delta_int,
        }

    def _get_llr(self, ts: np.ndarray, change_meta: Dict[str, Dict[str, Any]]):
        """
        Calculate the log likelihood ratio
        """
        mu0 = change_meta["mu0"]
        mu1 = change_meta["mu1"]
        changepoint = change_meta["changepoint"]
        scale = np.sqrt(
            (
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `int`.
                np.sum((ts[: (changepoint + 1)] - mu0) ** 2)
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `int`.
                + np.sum((ts[(changepoint + 1) :] - mu1) ** 2)
            )
            / (len(ts) - 2)
        )
        mu_tilde, sigma_tilde = np.mean(ts), np.std(ts)

        if scale == 0:
            scale = sigma_tilde

        llr = -2 * (
            # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
            #  typing.Any]` and `int`.
            # pyre-fixme[6]: Expected `float` for 4th param but got `Dict[str,
            #  typing.Any]`.
            self._log_llr(ts[: (changepoint + 1)], mu_tilde, sigma_tilde, mu0, scale)
            # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
            #  typing.Any]` and `int`.
            # pyre-fixme[6]: Expected `float` for 4th param but got `Dict[str,
            #  typing.Any]`.
            + self._log_llr(ts[(changepoint + 1) :], mu_tilde, sigma_tilde, mu1, scale)
        )
        return llr

    def _log_llr(
        self, x: np.ndarray, mu0: float, sigma0: float, mu1: float, sigma1: float
    ):
        """Helper function to calculate log likelihood ratio

        This function calculate the log likelihood ratio of two Gaussian distribution
        log(l(0)/l(1))

        Args:
            x: the data value
            mu0: mean of model 0
            sigma0: std of model 0
            mu1: mean of model 1
            sigma1: std of model 1

        Returns:
            the value of log likelihood ratio
        """

        return np.sum(
            np.log(sigma1 / sigma0)
            + 0.5 * (((x - mu1) / sigma1) ** 2 - ((x - mu0) / sigma0) ** 2)
        )

    def _magnitude_compare(self, ts: np.ndarray) -> float:
        """
        Compare daily magnitude to avoid daily seasonality false positives
        """
        time = self.data.time
        # pyre-fixme[16]: `CUSUMDetector` has no attribute `interest_window`.
        interest_window = self.interest_window

        # get number of days in historical window
        days = (time.max() - time.min()).days

        # get interest window magnitude
        mag_int = self._get_time_series_magnitude(
            ts[interest_window[0] : interest_window[1]]
        )

        comparable_mag = 0

        for i in range(days):
            start_time = time[interest_window[0]] - pd.Timedelta("{}D".format(i))
            end_time = time[interest_window[1]] - pd.Timedelta("{}D".format(i))
            start_idx = time[time == start_time].index[0]
            end_idx = time[time == end_time].index[0]

            hist_int = self._get_time_series_magnitude(ts[start_idx:end_idx])
            # pyre-fixme[16]: `CUSUMDetector` has no attribute `magnitude_ratio`.
            if mag_int / hist_int >= self.magnitude_ratio:
                comparable_mag += 1

        return comparable_mag / days

    def _get_time_series_magnitude(self, ts: np.ndarray) -> float:
        """
        Calcualte the magnitude of a time series
        """
        # pyre-fixme[16]: `CUSUMDetector` has no attribute `magnitude_quantile`.
        magnitude = np.quantile(ts, self.magnitude_quantile, interpolation="nearest")
        return magnitude

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    # pyre-fixme[15]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self, **kwargs) -> List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]:
        """
        Find the change point and calculate related statistics

        Args:
            threshold: Optional; float; significance level, default: 0.01;
            max_iter: Optional; int, maximun iteration in finding the changepoint;
            delta_std_ratio: Optional; float; the mean delta have to larger than this parameter
                times std of the data to be consider as a change;
            min_abs_change: Optional; int; minimal absolute delta between mu0 and mu1
            start_point: Optional; int; the start idx of the changepoint, if None means
                the middle of the time series;
            change_directions: Optional; list<str>; a list contain either or both 'increas' and
                'decrease' to specify what type of change want to detect;
            interest_window: Optional; list<int, int>, a list contian the start and end of
                interest window where we will look for change poin. Note that the
                llr will still be calculated using all data points;
            magnitude_quantile: Optional; float; the quantile for magnitude comparison, if
                none, will skip the magnitude comparison;
            magnitude_ratio: Optional; float; comparable ratio;
            magnitude_comparable_day: Optional; float; maximal percentage of days can have
                comparable magnitude to be considered as regression.
            return_all_changepoints: Optional; bool; return all the changepoints found, even the
                insignificant ones.

        Returns:
            A list of tuple of TimeSeriesChangePoint and CUSUMMetadata
        """
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", CUSUM_DEFAULT_ARGS["threshold"])
        max_iter = kwargs.get("max_iter", CUSUM_DEFAULT_ARGS["max_iter"])
        delta_std_ratio = kwargs.get(
            "delta_std_ratio", CUSUM_DEFAULT_ARGS["delta_std_ratio"]
        )
        min_abs_change = kwargs.get(
            "min_abs_change", CUSUM_DEFAULT_ARGS["min_abs_change"]
        )
        start_point = kwargs.get("start_point", CUSUM_DEFAULT_ARGS["start_point"])
        change_directions = kwargs.get(
            "change_directions", CUSUM_DEFAULT_ARGS["change_directions"]
        )
        interest_window = kwargs.get(
            "interest_window", CUSUM_DEFAULT_ARGS["interest_window"]
        )
        magnitude_quantile = kwargs.get(
            "magnitude_quantile", CUSUM_DEFAULT_ARGS["magnitude_quantile"]
        )
        magnitude_ratio = kwargs.get(
            "magnitude_ratio", CUSUM_DEFAULT_ARGS["magnitude_ratio"]
        )
        magnitude_comparable_day = kwargs.get(
            "magnitude_comparable_day", CUSUM_DEFAULT_ARGS["magnitude_comparable_day"]
        )
        return_all_changepoints = kwargs.get(
            "return_all_changepoints", CUSUM_DEFAULT_ARGS["return_all_changepoints"]
        )

        # pyre-fixme[16]: `CUSUMDetector` has no attribute `interest_window`.
        self.interest_window = interest_window
        # pyre-fixme[16]: `CUSUMDetector` has no attribute `magnitude_quantile`.
        self.magnitude_quantile = magnitude_quantile
        # pyre-fixme[16]: `CUSUMDetector` has no attribute `magnitude_ratio`.
        self.magnitude_ratio = magnitude_ratio

        # Use array to store the data
        ts = self.data.value.to_numpy()
        ts = ts.astype("float64")
        changes_meta = {}

        if change_directions is None:
            change_directions = ["increase", "decrease"]

        for change_direction in change_directions:

            assert change_direction in ["increase", "decrease"]

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
                change_direction=change_direction,
            )
            change_meta["llr"] = self._get_llr(ts, change_meta)
            # pyre-fixme[6]: Expected `Dict[str, typing.Any]` for 2nd param but got
            #  `int`.
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            change_meta["p_value"] = 1 - chi2.cdf(change_meta["llr"], 2)

            # compare magnitude on interest_window and historical_window
            if np.min(ts) >= 0:
                if magnitude_quantile and interest_window:
                    if change_direction == "increase":
                        mag_change = (
                            self._magnitude_compare(ts) >= magnitude_comparable_day
                        )
                    else:
                        mag_change = (
                            self._magnitude_compare(-ts) >= magnitude_comparable_day
                        )
                else:
                    mag_change = True
            elif magnitude_quantile:
                logging.warning(
                    "the minimal value is less than 0 cannot perform magnitude comparison"  # NOQA: B950
                )
                mag_change = True
            else:
                mag_change = True

            # pyre-fixme[58]: `>` is not supported for operand types `Dict[str,
            #  typing.Any]` and `Any`.
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            if_significant = change_meta["llr"] > chi2.ppf(1 - threshold, 2)
            # pyre-fixme[58]: `>` is not supported for operand types `Dict[str,
            #  typing.Any]` and `Any`.
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            if_significant_int = change_meta["llr_int"] > chi2.ppf(1 - threshold, 2)
            larger_than_min_abs_change = (
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `Any`.
                change_meta["mu0"] + min_abs_change < change_meta["mu1"]
                if change_direction == "increase"
                # pyre-fixme[58]: `>` is not supported for operand types `Dict[str,
                #  typing.Any]` and `Any`.
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `Any`.
                else change_meta["mu0"] > change_meta["mu1"] + min_abs_change
            )
            larger_than_std = (
                np.abs(change_meta["delta"])
                > np.std(ts[: change_meta["changepoint"]]) * delta_std_ratio
            )

            change_meta["regression_detected"] = (
                if_significant
                and if_significant_int
                and larger_than_min_abs_change
                and larger_than_std
                and mag_change
            )
            changes_meta[change_direction] = change_meta

        # pyre-fixme[16]: `CUSUMDetector` has no attribute `changes_meta`.
        self.changes_meta = changes_meta

        return self._convert_cusum_changepoints(changes_meta, return_all_changepoints)

    def _convert_cusum_changepoints(
        self,
        cusum_changepoints: Dict[str, Dict[str, Any]],
        return_all_changepoints: bool,
    ) -> List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]:
        """
        Convert the output from the other kats cusum algorithm into TimeSeriesChangePoint type
        """
        converted = []
        detected_cps = cusum_changepoints

        for direction in detected_cps:
            dir_cps = detected_cps[direction]
            if (
                dir_cps["regression_detected"] or return_all_changepoints
            ):  # we have a change point
                change_point = TimeSeriesChangePoint(
                    start_time=dir_cps["changetime"],
                    end_time=dir_cps["changetime"],
                    confidence=1 - dir_cps["p_value"],
                )
                metadata = CUSUMMetadata(
                    direction=direction,
                    cp_index=dir_cps["changepoint"],
                    mu0=dir_cps["mu0"],
                    mu1=dir_cps["mu1"],
                    delta=dir_cps["delta"],
                    llr_int=dir_cps["llr_int"],
                    llr=dir_cps["llr"],
                    regression_detected=dir_cps["regression_detected"],
                    stable_changepoint=dir_cps["stable_changepoint"],
                    p_value=dir_cps["p_value"],
                    p_value_int=dir_cps["p_value_int"],
                )
                converted.append((change_point, metadata))

        return converted

    def plot(
        self, change_points: List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]
    ) -> None:
        """Plot detection results from CUSUM

        Args:
            change_points: A list of tuple of TimeSeriesChangePoint and CUSUMMetadata

        Returns:
            None
        """
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        plt.plot(data_df[time_col_name], data_df[val_col_name])

        if len(change_points) == 0:
            logging.warning("No change points detected!")

        for change in change_points:
            if change[1].regression_detected:
                plt.axvline(x=change[0].start_time, color="red")

        # pyre-fixme[16]: `CUSUMDetector` has no attribute `interest_window`.
        if self.interest_window:
            plt.axvspan(
                pd.to_datetime(self.data.time)[self.interest_window[0]],
                pd.to_datetime(self.data.time)[self.interest_window[1] - 1],
                alpha=0.3,
                label="interets_window",
            )

        plt.show()


class MultiCUSUMDetector(CUSUMDetector):
    """
    MultiCUSUM is similar to univariate CUSUM, but we use MultiCUSUM to find a changepoint
    in multivariate time series.  The detector is used to detect changepoints in the multivariate
    mean of the time series.  The cusum values and likelihood ratio test calculations assume
    the underlying distribution has a Multivariate Guassian distriubtion.

    Attributes:
        data: The input time series data from TimeSeriesData
    """

    def __init__(self, data: TimeSeriesData) -> None:
        super(MultiCUSUMDetector, self).__init__(data=data, is_multivariate=True)

    def detector(self, **kwargs) -> List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]:
        """
        Overwrite the detector method for MultiCUSUMDetector
        Args:
            threshold: Optional; float; significance level, default: 0.01;
            max_iter: Optional; int, maximun iteration in finding the changepoint;
            start_point: Optional; int; the start idx of the changepoint, if None means
                the middle of the time series;
        """

        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", CUSUM_DEFAULT_ARGS["threshold"])
        max_iter = kwargs.get("max_iter", CUSUM_DEFAULT_ARGS["max_iter"])
        start_point = kwargs.get("start_point", CUSUM_DEFAULT_ARGS["start_point"])

        # TODO: Add support for interest windows

        return_all_changepoints = kwargs.get(
            "return_all_changepoints", CUSUM_DEFAULT_ARGS["return_all_changepoints"]
        )

        # Use array to store the data
        ts = self.data.value.to_numpy()
        ts = ts.astype("float64")
        changes_meta = {}

        # We will always be looking for increases in the CUSUM values for multivariate detection
        # We keep using change_direction = "increase" here to have
        # consistent CUSUMMetadata with the univariate detector
        for change_direction in ["increase"]:

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
            )
            change_meta["llr"] = self._get_llr(ts, change_meta)
            # pyre-fixme[6]: Expected `Dict[str, typing.Any]` for 2nd param but got
            #  `int`.
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            change_meta["p_value"] = 1 - chi2.cdf(change_meta["llr"], ts.shape[1] + 1)

            # pyre-fixme[58]: `>` is not supported for operand types `Dict[str,
            #  typing.Any]` and `Any`.
            # pyre-fixme[16]: Module `stats` has no attribute `chi2`.
            if_significant = change_meta["llr"] > chi2.ppf(
                1 - threshold, ts.shape[1] + 1
            )

            change_meta["regression_detected"] = if_significant
            changes_meta[change_direction] = change_meta

        # pyre-fixme[16]: `MultiCUSUMDetector` has no attribute `changes_meta`.
        self.changes_meta = changes_meta

        return self._convert_cusum_changepoints(changes_meta, return_all_changepoints)

    def _get_llr(self, ts: np.ndarray, change_meta: Dict[str, Dict[str, Any]]):
        mu0 = change_meta["mu0"]
        mu1 = change_meta["mu1"]
        sigma0 = change_meta["sigma0"]
        sigma1 = change_meta["sigma1"]
        changepoint = change_meta["changepoint"]

        mu_tilde = np.mean(ts, axis=0)
        sigma_pooled = np.cov(ts, rowvar=False)
        llr = -2 * (
            self._log_llr_multi(
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `int`.
                # pyre-fixme[6]: Expected `ndarray` for 4th param but got `Dict[str,
                #  typing.Any]`.
                ts[: (changepoint + 1)], mu_tilde, sigma_pooled, mu0, sigma0
            )
            - self._log_llr_multi(
                # pyre-fixme[58]: `+` is not supported for operand types `Dict[str,
                #  typing.Any]` and `int`.
                # pyre-fixme[6]: Expected `ndarray` for 4th param but got `Dict[str,
                #  typing.Any]`.
                ts[(changepoint + 1) :], mu_tilde, sigma_pooled, mu1, sigma1
            )
        )
        return llr

    def _log_llr_multi(
        self,
        x: np.ndarray,
        mu0: np.ndarray,
        # pyre-fixme[11]: Annotation `matrix` is not defined as a type.
        sigma0: np.matrix,
        mu1: np.ndarray,
        sigma1: np.matrix,
    ):
        try:
            sigma0_inverse = np.linalg.inv(sigma0)
            sigma1_inverse = np.linalg.inv(sigma1)
            log_det_sigma0 = np.log(np.linalg.det(sigma0))
            log_det_sigma1 = np.log(np.linalg.det(sigma1))
        except np.linalg.linalg.LinAlgError:
            msg = "One or more covariance matrix is singular."
            logging.error(msg)
            raise ValueError(msg)

        return len(x) / 2 * (log_det_sigma0 - log_det_sigma1) + np.sum(
            -np.matmul(np.matmul(x[i] - mu1, sigma1_inverse), (x[i] - mu1).T)
            + np.matmul(np.matmul(x[i] - mu0, sigma0_inverse), (x[i] - mu0).T)
            for i in range(len(x))
        )

    # pyre-fixme[14]: `_get_change_point` overrides method defined in
    #  `CUSUMDetector` inconsistently.
    def _get_change_point(
        self, ts: np.ndarray, max_iter: int, start_point: int
    ) -> Dict[str, Dict[str, Any]]:

        # locate the change point using cusum method
        changepoint_func = np.argmin
        n = 0
        ts_int = ts

        if start_point is None:
            start_point = len(ts_int) // 2
            changepoint = start_point

        # iterate until the changepoint converage
        while n < max_iter:
            n += 1
            data_before_changepoint = ts_int[: (changepoint + 1)]
            data_after_changepoint = ts_int[(changepoint + 1) :]

            mu0 = np.mean(data_before_changepoint, axis=0)
            mu1 = np.mean(data_after_changepoint, axis=0)

            # TODO: replace pooled variance with sample variances before and after changepoint
            # sigma0 = np.cov(data_before_changepoint, rowvar=False)
            # sigma1 = np.cov(data_after_changepoint, rowvar=False)
            sigma0 = sigma1 = np.cov(ts_int, rowvar=False)

            try:
                log_det_sigma0 = np.log(np.linalg.det(sigma0))
                log_det_sigma1 = np.log(np.linalg.det(sigma1))
                sigma0_inverse = np.linalg.inv(sigma0)
                sigma1_inverse = np.linalg.inv(sigma1)
            except np.linalg.linalg.LinAlgError:
                msg = "One or more covariance matrix is singular."
                logging.error(msg)
                raise ValueError(msg)

            si_values = np.diag(
                -(1 / 2) * log_det_sigma1
                - np.matmul(np.matmul(ts_int - mu1, sigma1_inverse), (ts_int - mu1).T)
                + (1 / 2) * log_det_sigma0
                + np.matmul(np.matmul(ts_int - mu0, sigma0_inverse), (ts_int - mu0).T)
            )

            cusum_ts = np.cumsum(si_values)
            next_changepoint = max(
                1, min(changepoint_func(cusum_ts), len(cusum_ts) - 2)
            )

            if next_changepoint == changepoint:
                break
            else:
                changepoint = next_changepoint

        if n == max_iter:
            logging.info("Max iteration reached and no stable changepoint found.")
            stable_changepoint = False
        else:
            stable_changepoint = True

        llr_int = np.inf
        pval_int = np.NaN
        delta_int = None

        # full time changepoint and mean
        mu0 = np.mean(ts[: (changepoint + 1)], axis=0)
        mu1 = np.mean(ts[(changepoint + 1) :], axis=0)
        sigma0 = sigma1 = np.cov(ts, rowvar=False)

        return {
            "changepoint": changepoint,
            "mu0": mu0,
            "mu1": mu1,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "changetime": self.data.time[changepoint],
            "stable_changepoint": stable_changepoint,
            "delta": mu1 - mu0,
            "llr_int": llr_int,
            "p_value_int": pval_int,
            "delta_int": delta_int,
        }
