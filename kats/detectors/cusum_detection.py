# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
CUSUM stands for cumulative sum, it is a changepoint detection algorithm.

In the Kats implementation, it has two main components:

  1. Locate the change point: The algorithm iteratively estimates the means
      before and after the change point and finds the change point
      maximizing/minimizing the cusum value until the change point has
      converged. The starting point for the change point is at the middle.

  2. Hypothesis testing: Conducting log likelihood ratio test where the null
      hypothesis has no change point with one mean and the alternative
      hypothesis has a change point with two means.

And here are a few things worth mentioning:

  * We assume there is only one increase/decrease change point;
  * We use Gaussian distribution as the underlying model to calculate the cusum
      value and conduct the hypothesis test;

Typical usage example:

>>> # Univariate CUSUM
>>> timeseries = TimeSeriesData(...)
>>> detector = CusumDetector(timeseries)
>>> #Run detector
>>> changepoints = detector.detector()
>>> # Plot the results
>>> detector.plot(changepoints)

The usage is the same for multivariate CUSUM except that the time series needs
to be multivariate and that the plotting functions are not yet supported for
this use case.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector
from scipy.stats import chi2  # @manual

pd.options.plotting.matplotlib.register_converters = True

_log: logging.Logger = logging.getLogger("cusum_detection")


@dataclass
class CUSUMDefaultArgs:
    threshold: float = 0.01
    max_iter: int = 10
    delta_std_ratio: float = 1.0
    min_abs_change: int = 0
    start_point: Optional[int] = None
    change_directions: Optional[List[str]] = None
    interest_window: Optional[int] = None
    magnitude_quantile: Optional[float] = None
    magnitude_ratio: float = 1.3
    magnitude_comparable_day: float = 0.5
    return_all_changepoints: bool = False
    remove_seasonality: bool = False


@dataclass
class CUSUMChangePointVal:
    changepoint: int
    mu0: float
    mu1: float
    changetime: List[float]
    stable_changepoint: bool
    delta: float
    llr_int: float
    p_value_int: float
    delta_int: Optional[float]
    sigma0: Optional[float] = None
    sigma1: Optional[float] = None
    llr: Optional[float] = None
    p_value: Optional[float] = None
    regression_detected: Optional[bool] = None


@dataclass
class VectorizedCUSUMChangePointVal:
    changepoint: List[int]
    mu0: List[float]
    mu1: List[float]
    changetime: List[float]
    stable_changepoint: List[bool]
    delta: List[float]
    llr_int: List[float]
    p_value_int: List[float]
    delta_int: Optional[List[float]]
    sigma0: Optional[List[float]] = None
    sigma1: Optional[List[float]] = None
    llr: Optional[List[float]] = None
    p_value: Optional[List[float]] = None
    regression_detected: Optional[List[bool]] = None


class CUSUMChangePoint(TimeSeriesChangePoint):
    """CUSUM change point.

    This is a changepoint detected by CUSUMDetector.

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
        direction: a str stand for the changepoint change direction 'increase'
            or 'decrease'.
        cp_index: an int for changepoint index.
        mu0: a float indicates the mean before changepoint.
        mu1: a float indicates the mean after changepoint.
        delta: mu1 - mu0.
        llr: log likelihood ratio.
        llr_int: log likelihood ratio in the interest window.
        regression_detected: a bool indicates if regression detected.
        stable_changepoint: a bool indicates if we have a stable changepoint
            when locating the changepoint.
        p_value: p_value of the changepoint.
        p_value_int: p_value of the changepoint in the interest window.
    """

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
        direction: str,
        cp_index: int,
        mu0: Union[float, np.ndarray],
        mu1: Union[float, np.ndarray],
        delta: Union[float, np.ndarray],
        llr_int: float,
        llr: float,
        regression_detected: bool,
        stable_changepoint: bool,
        p_value: float,
        p_value_int: float,
    ) -> None:
        super().__init__(start_time, end_time, confidence)
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
    def direction(self) -> str:
        return self._direction

    @property
    def cp_index(self) -> int:
        return self._cp_index

    @property
    def mu0(self) -> Union[float, np.ndarray]:
        return self._mu0

    @property
    def mu1(self) -> Union[float, np.ndarray]:
        return self._mu1

    @property
    def delta(self) -> Union[float, np.ndarray]:
        return self._delta

    @property
    def llr(self) -> float:
        return self._llr

    @property
    def llr_int(self) -> float:
        return self._llr_int

    @property
    def regression_detected(self) -> bool:
        return self._regression_detected

    @property
    def stable_changepoint(self) -> bool:
        return self._stable_changepoint

    @property
    def p_value(self) -> float:
        return self._p_value

    @property
    def p_value_int(self) -> float:
        return self._p_value_int

    def __repr__(self) -> str:
        return (
            f"CUSUMChangePoint(start_time: {self._start_time}, end_time: "
            f"{self._end_time}, confidence: {self._confidence}, direction: "
            f"{self._direction}, index: {self._cp_index}, delta: {self._delta}, "
            f"regression_detected: {self._regression_detected}, "
            f"stable_changepoint: {self._stable_changepoint}, mu0: {self._mu0}, "
            f"mu1: {self._mu1}, llr: {self._llr}, llr_int: {self._llr_int}, "
            f"p_value: {self._p_value}, p_value_int: {self._p_value_int})"
        )


class CUSUMDetector(Detector):
    interest_window: Optional[Tuple[int, int]] = None
    magnitude_quantile: Optional[float] = None
    magnitude_ratio: Optional[float] = None
    changes_meta: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(
        self,
        data: TimeSeriesData,
        is_multivariate: bool = False,
        is_vectorized: bool = False,
    ) -> None:
        """Univariate CUSUM detector for level shifts

        Use cusum to detect changes, the algorithm is based on likelihood ratio
        cusum. See https://www.fs.isy.liu.se/Edu/Courses/TSFS06/PDFs/Basseville.pdf
        for details. This detector is used to detect mean changes in Normal
        Distribution.

        Args:

            data: :class:`kats.consts.TimeSeriesData`; The input time series data.
            is_multivariate: Optional; bool; should be False unless running
                MultiCUSUMDetector,
        """
        super(CUSUMDetector, self).__init__(data=data)
        if not self.data.is_univariate() and not is_multivariate and not is_vectorized:
            msg = (
                "CUSUMDetector only supports univariate time series, but got "
                f"{type(self.data.value)}.  For multivariate time series, use "
                "MultiCUSUMDetector or VectorizedCUSUMDetector"
            )
            _log.error(msg)
            raise ValueError(msg)

    def _get_change_point(
        self, ts: np.ndarray, max_iter: int, start_point: int, change_direction: str
    ) -> CUSUMChangePointVal:
        """
        Find change point in the timeseries.
        """
        interest_window = self.interest_window

        # locate the change point using cusum method
        if change_direction == "increase":
            changepoint_func = np.argmin
            _log.debug("Detecting increase changepoint.")
        else:
            assert change_direction == "decrease"
            changepoint_func = np.argmax
            _log.debug("Detecting decrease changepoint.")
        n = 0
        # use the middle point as initial change point to estimate mu0 and mu1
        if interest_window is not None:
            ts_int = ts[interest_window[0] : interest_window[1]]
        else:
            ts_int = ts

        if start_point is None:
            cusum_ts = np.cumsum(ts_int - np.mean(ts_int))
            changepoint = min(changepoint_func(cusum_ts), len(ts_int) - 2)
        else:
            changepoint = start_point

        mu0 = mu1 = None
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
            changepoint = next_changepoint

        if n == max_iter:
            _log.info("Max iteration reached and no stable changepoint found.")
            stable_changepoint = False
        else:
            stable_changepoint = True

        # llr in interest window
        if interest_window is None:
            llr_int = np.inf
            pval_int = np.NaN
            delta_int = None
        else:
            llr_int = self._get_llr(ts_int, mu0, mu1, changepoint)
            pval_int = 1 - chi2.cdf(llr_int, 2)
            delta_int = mu1 - mu0
            changepoint += interest_window[0]

        # full time changepoint and mean
        mu0 = np.mean(ts[: (changepoint + 1)])
        mu1 = np.mean(ts[(changepoint + 1) :])

        return CUSUMChangePointVal(
            changepoint=changepoint,
            mu0=mu0,
            mu1=mu1,
            changetime=self.data.time[changepoint],
            stable_changepoint=stable_changepoint,
            delta=mu1 - mu0,
            llr_int=llr_int,
            p_value_int=pval_int,
            delta_int=delta_int,
        )

    def _get_llr(
        self,
        ts: np.ndarray,
        mu0: float,
        mu1: float,
        changepoint: int,
        sigma0: Optional[float] = None,
        sigma1: Optional[float] = None,
    ) -> float:
        """
        Calculate the log likelihood ratio
        """
        scale = np.sqrt(
            (
                np.sum((ts[: (changepoint + 1)] - mu0) ** 2)
                + np.sum((ts[(changepoint + 1) :] - mu1) ** 2)
            )
            / (len(ts) - 2)
        )
        mu_tilde, sigma_tilde = np.mean(ts), np.std(ts)

        if scale == 0:
            scale = sigma_tilde * 0.01

        llr = -2 * (
            self._log_llr(ts[: (changepoint + 1)], mu_tilde, sigma_tilde, mu0, scale)
            + self._log_llr(ts[(changepoint + 1) :], mu_tilde, sigma_tilde, mu1, scale)
        )
        return llr

    def _log_llr(
        self, x: np.ndarray, mu0: float, sigma0: float, mu1: float, sigma1: float
    ) -> float:
        """Helper function to calculate log likelihood ratio.

        This function calculate the log likelihood ratio of two Gaussian
        distribution log(l(0)/l(1)).

        Args:
            x: the data value.
            mu0: mean of model 0.
            sigma0: std of model 0.
            mu1: mean of model 1.
            sigma1: std of model 1.

        Returns:
            the value of log likelihood ratio.
        """

        return np.sum(
            np.log(sigma1 / sigma0)
            + 0.5 * (((x - mu1) / sigma1) ** 2 - ((x - mu0) / sigma0) ** 2)
        )

    def _magnitude_compare(self, ts: np.ndarray) -> float:
        """
        Compare daily magnitude to avoid daily seasonality false positives.
        """
        time = self.data.time
        interest_window = self.interest_window
        magnitude_ratio = self.magnitude_ratio
        if interest_window is None:
            raise ValueError("detect must be called first")
        assert magnitude_ratio is not None

        # get number of days in historical window
        days = (time.max() - time.min()).days

        # get interest window magnitude
        mag_int = self._get_time_series_magnitude(
            ts[interest_window[0] : interest_window[1]]
        )

        comparable_mag = 0

        for i in range(days):
            start_time = time[interest_window[0]] - pd.Timedelta(f"{i}D")
            end_time = time[interest_window[1]] - pd.Timedelta(f"{i}D")
            start_idx = time[time == start_time].index[0]
            end_idx = time[time == end_time].index[0]

            hist_int = self._get_time_series_magnitude(ts[start_idx:end_idx])
            if mag_int / hist_int >= magnitude_ratio:
                comparable_mag += 1

        return comparable_mag / days

    def _get_time_series_magnitude(self, ts: np.ndarray) -> float:
        """
        Calculate the magnitude of a time series.
        """
        magnitude = np.quantile(ts, self.magnitude_quantile, interpolation="nearest")
        return magnitude

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self, **kwargs: Any) -> Sequence[CUSUMChangePoint]:
        """
        Find the change point and calculate related statistics.

        Args:

            threshold: Optional; float; significance level, default: 0.01.
            max_iter: Optional; int, maximum iteration in finding the
                changepoint.
            delta_std_ratio: Optional; float; the mean delta have to larger than
                this parameter times std of the data to be consider as a change.
            min_abs_change: Optional; int; minimal absolute delta between mu0
                and mu1.
            start_point: Optional; int; the start idx of the changepoint, if
                None means the middle of the time series.
            change_directions: Optional; list<str>; a list contain either or
                both 'increase' and 'decrease' to specify what type of change
                want to detect.
            interest_window: Optional; list<int, int>, a list containing the
                start and end of interest windows where we will look for change
                points. Note that llr will still be calculated using all data
                points.
            magnitude_quantile: Optional; float; the quantile for magnitude
                comparison, if none, will skip the magnitude comparison.
            magnitude_ratio: Optional; float; comparable ratio.
            magnitude_comparable_day: Optional; float; maximal percentage of
                days can have comparable magnitude to be considered as
                regression.
            return_all_changepoints: Optional; bool; return all the changepoints
                found, even the insignificant ones.

        Returns:
            A list of CUSUMChangePoint.
        """
        defaultArgs = CUSUMDefaultArgs()
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", defaultArgs.threshold)
        max_iter = kwargs.get("max_iter", defaultArgs.max_iter)
        delta_std_ratio = kwargs.get("delta_std_ratio", defaultArgs.delta_std_ratio)
        min_abs_change = kwargs.get("min_abs_change", defaultArgs.min_abs_change)
        start_point = kwargs.get("start_point", defaultArgs.start_point)
        change_directions = kwargs.get(
            "change_directions", defaultArgs.change_directions
        )
        interest_window = kwargs.get("interest_window", defaultArgs.interest_window)
        magnitude_quantile = kwargs.get(
            "magnitude_quantile", defaultArgs.magnitude_quantile
        )
        magnitude_ratio = kwargs.get("magnitude_ratio", defaultArgs.magnitude_ratio)
        magnitude_comparable_day = kwargs.get(
            "magnitude_comparable_day", defaultArgs.magnitude_comparable_day
        )
        return_all_changepoints = kwargs.get(
            "return_all_changepoints", defaultArgs.return_all_changepoints
        )

        self.interest_window = interest_window
        self.magnitude_quantile = magnitude_quantile
        self.magnitude_ratio = magnitude_ratio

        # Use array to store the data
        ts = self.data.value.to_numpy()
        ts = ts.astype("float64")
        changes_meta = {}

        if change_directions is None:
            change_directions = ["increase", "decrease"]

        for change_direction in change_directions:
            if change_direction not in {"increase", "decrease"}:
                raise ValueError(
                    "Change direction must be 'increase' or 'decrease.' "
                    f"Got {change_direction}"
                )

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
                change_direction=change_direction,
            )
            change_meta.llr = llr = self._get_llr(
                ts,
                change_meta.mu0,
                change_meta.mu1,
                change_meta.changepoint,
                change_meta.sigma0,
                change_meta.sigma1,
            )
            change_meta.p_value = 1 - chi2.cdf(llr, 2)

            # compare magnitude on interest_window and historical_window
            if np.min(ts) >= 0:
                if magnitude_quantile and interest_window:
                    change_ts = ts if change_direction == "increase" else -ts
                    mag_change = (
                        self._magnitude_compare(change_ts) >= magnitude_comparable_day
                    )
                else:
                    mag_change = True
            else:
                mag_change = True
                if magnitude_quantile:
                    _log.warning(
                        (
                            "The minimal value is less than 0. Cannot perform "
                            "magnitude comparison."
                        )
                    )

            if_significant = llr > chi2.ppf(1 - threshold, 2)
            if_significant_int = change_meta.llr_int > chi2.ppf(1 - threshold, 2)
            if change_direction == "increase":
                larger_than_min_abs_change = (
                    change_meta.mu0 + min_abs_change < change_meta.mu1
                )
            else:
                larger_than_min_abs_change = (
                    change_meta.mu0 > change_meta.mu1 + min_abs_change
                )
            larger_than_std = (
                np.abs(change_meta.delta)
                > np.std(ts[: change_meta.changepoint]) * delta_std_ratio
            )

            change_meta.regression_detected = (
                if_significant
                and if_significant_int
                and larger_than_min_abs_change
                and larger_than_std
                and mag_change
            )
            changes_meta[change_direction] = asdict(change_meta)

        self.changes_meta = changes_meta

        return self._convert_cusum_changepoints(changes_meta, return_all_changepoints)

    def _convert_cusum_changepoints(
        self,
        cusum_changepoints: Dict[str, Dict[str, Any]],
        return_all_changepoints: bool,
    ) -> List[CUSUMChangePoint]:
        """
        Convert the output from the other kats cusum algorithm into
        CUSUMChangePoint type.
        """
        converted = []
        detected_cps = cusum_changepoints

        for direction in detected_cps:
            dir_cps = detected_cps[direction]
            if dir_cps["regression_detected"] or return_all_changepoints:
                # we have a change point
                change_point = CUSUMChangePoint(
                    start_time=dir_cps["changetime"],
                    end_time=dir_cps["changetime"],
                    confidence=1 - dir_cps["p_value"],
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
                converted.append(change_point)

        return converted

    def plot(
        self, change_points: Sequence[CUSUMChangePoint], **kwargs: Any
    ) -> plt.Axes:
        """Plot detection results from CUSUM.

        Args:
            change_points: A list of CUSUMChangePoint.
            kwargs: other arguments to pass to subplots.

        Returns:
            The matplotlib Axes.
        """
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        _, ax = plt.subplots(**kwargs)
        ax.plot(data_df[time_col_name], data_df[val_col_name])

        changepoint_annotated = False
        for change in change_points:
            if change.regression_detected:
                ax.axvline(x=change.start_time, color="red")
                changepoint_annotated = True
        if not changepoint_annotated:
            _log.warning("No change points detected!")

        interest_window = self.interest_window
        if interest_window is not None:
            ax.axvspan(
                pd.to_datetime(self.data.time)[interest_window[0]],
                pd.to_datetime(self.data.time)[interest_window[1] - 1],
                alpha=0.3,
                label="interets_window",
            )

        return ax


class MultiCUSUMDetector(CUSUMDetector):
    """
    MultiCUSUM is similar to univariate CUSUM, but we use MultiCUSUM to find a
    changepoint in multivariate time series.  The detector is used to detect
    changepoints in the multivariate mean of the time series.  The cusum values
    and likelihood ratio test calculations assume the underlying distribution
    has a Multivariate Guassian distriubtion.

    Attributes:
        data: The input time series data from TimeSeriesData
    """

    def __init__(self, data: TimeSeriesData) -> None:
        super(MultiCUSUMDetector, self).__init__(data=data, is_multivariate=True)

    def detector(self, **kwargs: Any) -> List[CUSUMChangePoint]:
        """
        Overwrite the detector method for MultiCUSUMDetector.

        Args:
            threshold: Optional; float; significance level, default: 0.01.
            max_iter: Optional; int, maximum iteration in finding the
                changepoint.
            start_point: Optional; int; the start idx of the changepoint, if
                None means the middle of the time series.
        """
        defaultArgs = CUSUMDefaultArgs()
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", defaultArgs.threshold)
        max_iter = kwargs.get("max_iter", defaultArgs.max_iter)
        start_point = kwargs.get("start_point", defaultArgs.start_point)

        # TODO: Add support for interest windows

        return_all_changepoints = kwargs.get(
            "return_all_changepoints", defaultArgs.return_all_changepoints
        )

        # Use array to store the data
        ts = self.data.value.to_numpy()
        ts = ts.astype("float64")
        changes_meta = {}

        # We will always be looking for increases in the CUSUM values for
        # multivariate detection. We keep using change_direction = "increase"
        # here to be consistent with the univariate detector.
        for change_direction in ["increase"]:

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
            )
            change_meta.llr = llr = self._get_llr(
                ts,
                change_meta.mu0,
                change_meta.mu1,
                change_meta.changepoint,
                change_meta.sigma0,
                change_meta.sigma1,
            )
            change_meta.p_value = 1 - chi2.cdf(llr, ts.shape[1] + 1)

            if_significant = llr > chi2.ppf(1 - threshold, ts.shape[1] + 1)

            change_meta.regression_detected = if_significant
            changes_meta[change_direction] = asdict(change_meta)

        self.changes_meta = changes_meta

        return self._convert_cusum_changepoints(changes_meta, return_all_changepoints)

    def _get_llr(
        self,
        ts: np.ndarray,
        mu0: float,
        mu1: float,
        changepoint: int,
        sigma0: Optional[float],
        sigma1: Optional[float],
    ) -> float:

        mu_tilde = np.mean(ts, axis=0)
        sigma_pooled = np.cov(ts, rowvar=False)
        llr = -2 * (
            self._log_llr_multi(
                ts[: (changepoint + 1)],
                mu_tilde,
                sigma_pooled,
                mu0,
                sigma0,  # pyre-fixme
            )
            - self._log_llr_multi(
                ts[(changepoint + 1) :],
                mu_tilde,
                sigma_pooled,
                mu1,
                sigma1,  # pyre-fixme
            )
        )
        return llr

    def _log_llr_multi(
        self,
        x: np.ndarray,
        mu0: Union[float, np.ndarray],
        sigma0: Union[float, np.ndarray],
        mu1: Union[float, np.ndarray],
        sigma1: Union[float, np.ndarray],
    ) -> float:
        try:
            sigma0_inverse = np.linalg.inv(sigma0)
            sigma1_inverse = np.linalg.inv(sigma1)
            log_det_sigma0 = np.log(np.linalg.det(sigma0))
            log_det_sigma1 = np.log(np.linalg.det(sigma1))
        except np.linalg.linalg.LinAlgError:
            msg = "One or more covariance matrix is singular."
            _log.error(msg)
            raise ValueError(msg)

        return len(x) / 2 * (log_det_sigma0 - log_det_sigma1) + np.sum(
            -np.matmul(np.matmul(x[i] - mu1, sigma1_inverse), (x[i] - mu1).T)
            + np.matmul(np.matmul(x[i] - mu0, sigma0_inverse), (x[i] - mu0).T)
            for i in range(len(x))
        )

    def _get_change_point(
        self,
        ts: np.ndarray,
        max_iter: int,
        start_point: int,
        change_direction: str = "increase",
    ) -> CUSUMChangePointVal:

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

            # TODO: replace pooled variance with sample variances before and
            # after changepoint.
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
                _log.error(msg)
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
            _log.info("Max iteration reached and no stable changepoint found.")
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

        return CUSUMChangePointVal(
            changepoint=changepoint,
            mu0=mu0,
            mu1=mu1,
            changetime=self.data.time[changepoint],
            stable_changepoint=stable_changepoint,
            delta=mu1 - mu0,
            llr_int=llr_int,
            p_value_int=pval_int,
            delta_int=delta_int,
            sigma0=sigma0,
            sigma1=sigma1,
        )


class VectorizedCUSUMDetector(CUSUMDetector):
    """
    VectorizedCUSUM is the vecteorized version of CUSUM. It can take
    multiple time series as an input and run CUSUM algorithm on each time series
    in a vectorized manner.

    Attributes:
        data: The input time series data from TimeSeriesData
    """

    changes_meta_list: Optional[List[Dict[str, Dict[str, Any]]]] = None

    def __init__(self, data: TimeSeriesData) -> None:
        super(VectorizedCUSUMDetector, self).__init__(
            data=data, is_multivariate=False, is_vectorized=True
        )

    def detector(self, **kwargs: Any) -> Sequence[CUSUMChangePoint]:
        msg = "VectorizedCUSUMDetector is in beta and please use detector_()"
        _log.error(msg)
        raise ValueError(msg)

    def detector_(self, **kwargs: Any) -> List[List[CUSUMChangePoint]]:
        """
        Detector method for vectorized version of CUSUM

        Args:
            threshold: Optional; float; significance level, default: 0.01.
            max_iter: Optional; int, maximum iteration in finding the
                changepoint.
            delta_std_ratio: Optional; float; the mean delta have to larger than
                this parameter times std of the data to be consider as a change.
            min_abs_change: Optional; int; minimal absolute delta between mu0
                and mu1.
            change_directions: Optional; list<str>; a list contain either or
                both 'increase' and 'decrease' to specify what type of change
                want to detect.
            interest_window: Optional; list<int, int>, a list containing the
                start and end of interest windows where we will look for change
                points. Note that llr will still be calculated using all data
                points.
            magnitude_quantile: Optional; float; the quantile for magnitude
                comparison, if none, will skip the magnitude comparison.
            magnitude_ratio: Optional; float; comparable ratio.
            magnitude_comparable_day: Optional; float; maximal percentage of
                days can have comparable magnitude to be considered as
                regression.
            return_all_changepoints: Optional; bool; return all the changepoints
                found, even the insignificant ones.

        Returns:
            A list of tuple of TimeSeriesChangePoint and CUSUMMetadata.
        """
        defaultArgs = CUSUMDefaultArgs()
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", defaultArgs.threshold)
        max_iter = kwargs.get("max_iter", defaultArgs.max_iter)
        delta_std_ratio = kwargs.get("delta_std_ratio", defaultArgs.delta_std_ratio)
        min_abs_change = kwargs.get("min_abs_change", defaultArgs.min_abs_change)
        change_directions = kwargs.get(
            "change_directions", defaultArgs.change_directions
        )
        interest_window = kwargs.get("interest_window", defaultArgs.interest_window)
        magnitude_quantile = kwargs.get(
            "magnitude_quantile", defaultArgs.magnitude_quantile
        )
        magnitude_ratio = kwargs.get("magnitude_ratio", defaultArgs.magnitude_ratio)
        magnitude_comparable_day = kwargs.get(
            "magnitude_comparable_day", defaultArgs.magnitude_comparable_day
        )
        return_all_changepoints = kwargs.get(
            "return_all_changepoints", defaultArgs.return_all_changepoints
        )
        self.interest_window = interest_window
        self.magnitude_quantile = magnitude_quantile
        self.magnitude_ratio = magnitude_ratio

        # Use array to store the data
        ts_all = self.data.value.to_numpy()
        ts_all = ts_all.astype("float64")
        if ts_all.ndim == 1:
            ts_all = ts_all[:, np.newaxis]
        changes_meta_list = []

        if change_directions is None:
            change_directions = ["increase", "decrease"]

        change_meta_all = {}
        for change_direction in change_directions:
            if change_direction not in {"increase", "decrease"}:
                raise ValueError(
                    "Change direction must be 'increase' or 'decrease.' "
                    f"Got {change_direction}"
                )

            change_meta_all[change_direction] = asdict(
                self._get_change_point_multiple_ts(
                    ts_all,
                    max_iter=max_iter,
                    change_direction=change_direction,
                )
            )

        ret = []
        for col_idx in np.arange(ts_all.shape[1]):
            ts = ts_all[:, col_idx]
            changes_meta = {}
            for change_direction in change_directions:
                change_meta_ = change_meta_all[change_direction]
                # if no change points are detected, skip
                if not list(change_meta_["changepoint"]):
                    continue
                change_meta = {
                    k: change_meta_[k][col_idx]
                    if isinstance(change_meta_[k], np.ndarray)
                    or isinstance(change_meta_[k], list)
                    else change_meta_[k]
                    for k in change_meta_
                }
                change_meta["llr"] = llr = self._get_llr(
                    ts,
                    change_meta["mu0"],
                    change_meta["mu1"],
                    change_meta["changepoint"],
                    change_meta["sigma0"],
                    change_meta["sigma1"],
                )
                change_meta["p_value"] = 1 - chi2.cdf(change_meta["llr"], 2)

                # compare magnitude on interest_window and historical_window
                if np.min(ts) >= 0:
                    if magnitude_quantile and interest_window:
                        change_ts = ts if change_direction == "increase" else -ts
                        mag_change = (
                            self._magnitude_compare(change_ts)
                            >= magnitude_comparable_day
                        )
                    else:
                        mag_change = True
                else:
                    mag_change = True
                    if magnitude_quantile:
                        _log.warning(
                            (
                                "The minimal value is less than 0. Cannot perform "
                                "magnitude comparison."
                            )
                        )

                if_significant = llr > chi2.ppf(1 - threshold, 2)
                if_significant_int = change_meta["llr_int"] > chi2.ppf(1 - threshold, 2)
                if change_direction == "increase":
                    larger_than_min_abs_change = (
                        change_meta["mu0"] + min_abs_change < change_meta["mu1"]
                    )
                else:
                    larger_than_min_abs_change = (
                        change_meta["mu0"] > change_meta["mu1"] + min_abs_change
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
            changes_meta_list.append(changes_meta)
            ret.append(
                self._convert_cusum_changepoints(changes_meta, return_all_changepoints)
            )
        self.changes_meta_list = changes_meta_list
        return ret

    def _get_change_point_multiple_ts(
        self, ts: np.ndarray, max_iter: int, change_direction: str
    ) -> VectorizedCUSUMChangePointVal:
        """
        Find change points in a list of time series
        """
        interest_window = self.interest_window

        # locate the change point using cusum method
        if change_direction == "increase":
            changepoint_func = np.argmin
            _log.debug("Detecting increase changepoint.")
        else:
            assert change_direction == "decrease"
            changepoint_func = np.argmax
            _log.debug("Detecting decrease changepoint.")

        n = 0
        if interest_window is not None:
            ts_int = ts[interest_window[0] : interest_window[1], :]
        else:
            ts_int = ts
        n_ts = ts_int.shape[1]
        n_pts = ts_int.shape[0]
        if n_pts == 0:
            return VectorizedCUSUMChangePointVal(
                changepoint=[],
                mu0=[],
                mu1=[],
                changetime=[],
                stable_changepoint=[],
                delta=[],
                llr_int=[],
                p_value_int=[],
                delta_int=[],
            )
        # use the middle point as initial change point to estimate mu0 and mu1
        tmp = ts_int - np.tile(np.mean(ts_int, axis=0), (n_pts, 1))
        cusum_ts = np.cumsum(tmp, axis=0)
        changepoint = np.minimum(changepoint_func(cusum_ts, axis=0), n_pts - 2)
        mu0 = mu1 = None
        stable_changepoint = [False] * len(changepoint)
        # iterate until the changepoint converage
        while n < max_iter:
            mask = np.zeros((n_pts, n_ts), dtype=bool)
            for i, c in enumerate(changepoint):
                mask[: (c + 1), i] = True
            n += 1
            mu0 = np.divide(
                np.sum(np.multiply(ts_int, mask), axis=0), np.sum(mask, axis=0)
            )
            mu1 = np.divide(
                np.sum(np.multiply(ts_int, ~mask), axis=0), np.sum(~mask, axis=0)
            )
            mean = (mu0 + mu1) / 2
            # here is where cusum is happening
            tmp = ts_int - np.tile(mean, (n_pts, 1))
            cusum_ts = np.cumsum(tmp, axis=0)
            next_changepoint = np.maximum(
                np.minimum(changepoint_func(cusum_ts, axis=0), n_pts - 2), 1
            )
            stable_changepoint = np.equal(changepoint, next_changepoint)
            if all(stable_changepoint):
                break
            changepoint = next_changepoint

        # full time changepoint and mean
        mask = np.zeros((n_pts, n_ts), dtype=bool)
        for i, c in enumerate(changepoint):
            mask[: (c + 1), i] = True
        mu0 = np.divide(np.sum(np.multiply(ts_int, mask), axis=0), np.sum(mask, axis=0))
        mu1 = np.divide(
            np.sum(np.multiply(ts_int, ~mask), axis=0), np.sum(~mask, axis=0)
        )

        # llr in interest window
        if interest_window is None:
            llr_int = [np.inf for _ in np.arange(n_ts)]
            pval_int = [np.NaN for _ in np.arange(n_ts)]
            delta_int = None
        else:
            llr_int = []
            pval_int = []
            delta_int = []
            for col_idx in np.arange(n_ts):
                _llr_int = self._get_llr(
                    ts_int[:, col_idx], mu0[col_idx], mu1[col_idx], changepoint[col_idx]
                )
                _pval_int = 1 - chi2.cdf(_llr_int, 2)
                _delta_int = mu1[col_idx] - mu0[col_idx]
                llr_int.append(_llr_int)
                pval_int.append(_pval_int)
                delta_int.append(_delta_int)
                changepoint[col_idx] += interest_window[0]

        return VectorizedCUSUMChangePointVal(
            changepoint=changepoint,
            mu0=mu0,
            mu1=mu1,
            changetime=[self.data.time[c] for c in changepoint],
            stable_changepoint=stable_changepoint,
            delta=mu1 - mu0,
            llr_int=llr_int,
            p_value_int=pval_int,
            delta_int=delta_int,
        )
