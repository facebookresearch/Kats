import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import (
    DEFAULT_VALUE_NAME,
    TimeSeriesChangePoint,
    TimeSeriesData,
)
from infrastrategy.kats.detector import Detector, DetectorModel
from infrastrategy.kats.detectors.detector_consts import AnomalyResponse
from infrastrategy.kats.utils.decomposition import TimeSeriesDecomposition
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


def log_llr(x, mu0, sigma0, mu1, sigma1):
    return np.sum(
        np.log(sigma1 / sigma0)
        + 0.5 * (((x - mu1) / sigma1) ** 2 - ((x - mu0) / sigma0) ** 2)
    )


def log_llr_multi(
    x: np.ndarray,
    mu0: np.ndarray,
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


class CUSUMMetadata:
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
    """
    Use cusum to detect changes, the algorithm is based on likelihood ratio cusum.
    See https://www.fs.isy.liu.se/Edu/Courses/TSFS06/PDFs/Basseville.pdf for details.
    These detector is used to detect mean changes in Normal Distribution.
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
        # Calculate the log likelihood ratio
        mu0 = change_meta["mu0"]
        mu1 = change_meta["mu1"]
        changepoint = change_meta["changepoint"]
        scale = np.sqrt(
            (
                np.sum((ts[: (changepoint + 1)] - mu0) ** 2)
                + np.sum((ts[(changepoint + 1) :] - mu1) ** 2)
            )
            / (len(ts) - 2)
        )
        mu_tilde, sigma_tilde = np.mean(ts), np.std(ts)

        if scale == 0:
            scale = sigma_tilde

        llr = -2 * (
            log_llr(ts[: (changepoint + 1)], mu_tilde, sigma_tilde, mu0, scale)
            + log_llr(ts[(changepoint + 1) :], mu_tilde, sigma_tilde, mu1, scale)
        )
        return llr

    def _magnitude_compare(self, ts: np.ndarray) -> float:
        time = self.data.time
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
            if mag_int / hist_int >= self.magnitude_ratio:
                comparable_mag += 1

        return comparable_mag / days

    def _get_time_series_magnitude(self, ts: np.ndarray) -> float:
        magnitude = np.quantile(ts, self.magnitude_quantile, interpolation="nearest")
        return magnitude

    def detector(self, **kwargs) -> List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]:
        """
        Find the change point and calculate related statistics
        Args:
            threshold: float, significance level;
            max_iter: int, maximun iteration in finding the changepoint;
            delta_std_ratio: float, the mean delta have to larger than this parameter
                times std of the data to be consider as a change;
            min_abs_change: int, minimal absolute delta between mu0 and mu1
            start_point: int, the start idx of the changepoint, if None means
                the middle of the time series;
            change_directions: list<str>, a list contain either or both 'increas' and
                'decrease' to specify what type of change want to detect;
            interest_window: list<int, int>, a list contian the start and end of
                interest window where we will look for change poin. Note that the
                llr will still be calculated using all data points;
            magnitude_quantile: float, the quantile for magnitude comparison, if
                none, will skip the magnitude comparison;
            magnitude_ratio: float, comparable ratio;
            magnitude_comparable_day: float, maximal percentage of days can have
                comparable magnitude to be considered as regression.
            return_all_changepoints: bool, return all the changepoints found, even the
                insignificant ones.
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

            assert change_direction in ["increase", "decrease"]

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
                change_direction=change_direction,
            )
            change_meta["llr"] = self._get_llr(ts, change_meta)
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

            if_significant = change_meta["llr"] > chi2.ppf(1 - threshold, 2)
            if_significant_int = change_meta["llr_int"] > chi2.ppf(1 - threshold, 2)
            larger_than_min_abs_change = (
                change_meta["mu0"] + min_abs_change < change_meta["mu1"]
                if change_direction == "increase"
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

    # TODO: hmm... the plot method on old CUSUM is pretty good -- is there a way to transfer that over?
    def plot(
        self, change_points: List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]
    ) -> None:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        plt.plot(data_df[time_col_name], data_df[val_col_name])

        if len(change_points) == 0:
            logging.warning("No change points detected!")

        for change in change_points:
            if change[1].regression_detected:
                plt.axvline(x=change[0].start_time, color="red")

        if self.interest_window:
            plt.axvspan(
                pd.to_datetime(self.data.time)[self.interest_window[0]],
                pd.to_datetime(self.data.time)[self.interest_window[1] - 1],
                alpha=0.3,
                label="interets_window",
            )

        plt.show()


NORMAL_TOLERENCE = 1  # number of window
CHANGEPOINT_RETENTION = 7 * 24 * 60 * 60  # in seconds
MAX_CHANGEPOINT = 10


def percentage_change(
    data: TimeSeriesData, pre_mean: float, **kwargs: Any
) -> TimeSeriesData:
    return (data - pre_mean) / (pre_mean)


def change(data: TimeSeriesData, pre_mean: float, **kwargs: Any) -> TimeSeriesData:
    return data - pre_mean


def z_score(data: TimeSeriesData, pre_mean: float, pre_std: float) -> TimeSeriesData:
    return (data - pre_mean) / (pre_std)


class CusumScoreFunction(Enum):
    change = "change"
    percentage_change = "percentage_change"
    z_score = "z_score"


SCORE_FUNC_DICT = {
    CusumScoreFunction.change.value: change,
    CusumScoreFunction.percentage_change.value: percentage_change,
    CusumScoreFunction.z_score.value: z_score,
}


class MultiCUSUMDetector(CUSUMDetector):
    def __init__(self, data: TimeSeriesData, is_multivariate: bool = True) -> None:
        super(MultiCUSUMDetector, self).__init__(data=data, is_multivariate=True)

    def detector(self, **kwargs) -> List[Tuple[TimeSeriesChangePoint, CUSUMMetadata]]:
        """
        Overwrite the detector method for MultiCUSUMDetector
        """
        # Extract all arg values or assign defaults from default vals constant
        threshold = kwargs.get("threshold", CUSUM_DEFAULT_ARGS["threshold"])
        max_iter = kwargs.get("max_iter", CUSUM_DEFAULT_ARGS["max_iter"])
        start_point = kwargs.get("start_point", CUSUM_DEFAULT_ARGS["start_point"])

        #TODO: Add support for interest windows

        return_all_changepoints = kwargs.get(
            "return_all_changepoints", CUSUM_DEFAULT_ARGS["return_all_changepoints"]
        )

        # Use array to store the data
        ts = self.data.value.to_numpy()
        ts = ts.astype("float64")
        changes_meta = {}

        for change_direction in ["increase"]:

            change_meta = self._get_change_point(
                ts,
                max_iter=max_iter,
                start_point=start_point,
            )
            change_meta["llr"] = self._get_llr(ts, change_meta)
            change_meta["p_value"] = 1 - chi2.cdf(change_meta["llr"], ts.shape[1] + 1)

            if_significant = change_meta["llr"] > chi2.ppf(
                1 - threshold, ts.shape[1] + 1
            )

            change_meta["regression_detected"] = if_significant
            changes_meta[change_direction] = change_meta

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
            log_llr_multi(ts[: (changepoint + 1)], mu_tilde, sigma_pooled, mu0, sigma0)
            - log_llr_multi(
                ts[(changepoint + 1) :], mu_tilde, sigma_pooled, mu1, sigma1
            )
        )
        return llr

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

        # full time chnagepoint and mean
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


class CUSUMDetectorModel(DetectorModel):
    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        scan_window: Optional[int] = None,
        historical_window: Optional[int] = None,
        step_window: Optional[int] = None,
        threshold: float = CUSUM_DEFAULT_ARGS["threshold"],
        delta_std_ratio: float = CUSUM_DEFAULT_ARGS["delta_std_ratio"],
        magnitude_quantile: float = CUSUM_DEFAULT_ARGS["magnitude_quantile"],
        magnitude_ratio: float = CUSUM_DEFAULT_ARGS["magnitude_ratio"],
        change_directions: List[str] = CUSUM_DEFAULT_ARGS["change_directions"],
        score_func: CusumScoreFunction = CusumScoreFunction.change,
        remove_seasonality: bool = CUSUM_DEFAULT_ARGS["remove_seasonality"],
    ):
        if serialized_model:
            previous_model = json.loads(serialized_model)
            self.cps = previous_model["cps"]
            self.alert_fired = previous_model["alert_fired"]
            self.pre_mean = previous_model["pre_mean"]
            self.pre_std = previous_model["pre_std"]
            self.number_of_normal_scan = previous_model["number_of_normal_scan"]
            self.alert_change_direction = previous_model["alert_change_direction"]
            self.scan_window = previous_model["scan_window"]
            self.historical_window = previous_model["historical_window"]
            self.step_window = previous_model["step_window"]
            self.threshold = previous_model["threshold"]
            self.delta_std_ratio = previous_model["delta_std_ratio"]
            self.magnitude_quantile = previous_model["magnitude_quantile"]
            self.magnitude_ratio = previous_model["magnitude_ratio"]
            self.change_directions = previous_model["change_directions"]
            self.score_func = previous_model["score_func"]
            if "remove_seasonality" in previous_model:
                self.remove_seasonality = previous_model["remove_seasonality"]
            else:
                self.remove_seasonality = remove_seasonality
        elif scan_window is not None and historical_window is not None:
            self.cps = []
            self.alert_fired = False
            self.pre_mean = 0
            self.pre_std = 1
            self.number_of_normal_scan = 0
            self.alert_change_direction = None
            self.scan_window = scan_window
            self.historical_window = historical_window
            self.step_window = step_window
            self.threshold = threshold
            self.delta_std_ratio = delta_std_ratio
            self.magnitude_quantile = magnitude_quantile
            self.magnitude_ratio = magnitude_ratio
            self.change_directions = change_directions
            self.score_func = score_func.value
            self.remove_seasonality = remove_seasonality
        else:
            raise ValueError(
                """
            You must either provide serialized model or values for scan_window and historical_window.
            """
            )
        if step_window is not None and step_window >= scan_window:
            raise ValueError(
                "Step window should smaller than scan window to ensure we have overlap for scan windows."
            )

    def __eq__(self, other):
        if isinstance(other, CUSUMDetectorModel):
            return (
                self.cps == other.cps
                and self.alert_fired == other.alert_fired
                and self.pre_mean == other.pre_mean
                and self.pre_std == other.pre_std
                and self.number_of_normal_scan == other.number_of_normal_scan
                and self.alert_change_direction == other.alert_change_direction
                and self.scan_window == other.scan_window
                and self.historical_window == other.historical_window
                and self.step_window == other.step_window
                and self.threshold == other.threshold
                and self.delta_std_ratio == other.delta_std_ratio
                and self.magnitude_quantile == other.magnitude_quantile
                and self.magnitude_ratio == other.magnitude_ratio
                and self.change_directions == other.change_directions
                and self.score_func == other.score_func
            )
        return False

    def serialize(self) -> bytes:
        return str.encode(json.dumps(self.__dict__))

    def set_alert_off(self):
        self.alert_fired = False
        self.number_of_normal_scan = 0

    def set_alert_on(self, baseline_mean, baseline_std, alert_change_direction):
        self.alert_fired = True
        self.alert_change_direction = alert_change_direction
        self.pre_mean = baseline_mean
        self.pre_std = baseline_std

    def if_normal(self, cur_mean, change_directions):
        if change_directions is not None:
            increase, decrease = (
                "increase" in change_directions,
                "decrease" in change_directions,
            )
        else:
            increase, decrease = True, True

        if self.alert_change_direction == "increase":
            check_increase = 0 if increase else np.inf
            check_decrease = 1.0 if decrease else np.inf
        elif self.alert_change_direction == "decrease":
            check_increase = 1.0 if increase else np.inf
            check_decrease = 0 if decrease else np.inf

        return (
            self.pre_mean - check_decrease * self.pre_std
            <= cur_mean
            <= self.pre_mean + check_increase * self.pre_std
        )

    def _fit(
        self,
        data: TimeSeriesData,
        historical_data: TimeSeriesData,
        scan_window: int,
        threshold: float = CUSUM_DEFAULT_ARGS["threshold"],
        delta_std_ratio: float = CUSUM_DEFAULT_ARGS["delta_std_ratio"],
        magnitude_quantile: float = CUSUM_DEFAULT_ARGS["magnitude_quantile"],
        magnitude_ratio: float = CUSUM_DEFAULT_ARGS["magnitude_ratio"],
        change_directions: List[str] = CUSUM_DEFAULT_ARGS["change_directions"],
    ):
        """
        data: the new data the model never seen
        historical_data: the historical data, `historical_data` have to end with the
            datapoint right before the first data point in `data`
        scan_window: scan window length in seconds, scan window is the window where
            cusum search for changepoint(s)
        threshold: changepoint significant level, higher the value more changepoints
            detected
        delta_std_ratio: the mean change have to larger than `delta_std_ratio` *
           `std(data[:changepoint])` to be consider as a change, higher the value
           less changepoints detected
        magnitude_quantile: float, the quantile for magnitude comparison, if
            none, will skip the magnitude comparison;
        magnitude_ratio: float, comparable ratio;
        change_directions: a list contain either or both 'increas' and 'decrease' to
            specify what type of change to detect;
        """
        historical_data.extend(data, validate=False)
        n = len(historical_data)
        scan_start_time = historical_data.time.iloc[-1] - pd.Timedelta(
            scan_window, unit="s"
        )
        scan_start_index = max(
            0, np.argwhere((historical_data.time >= scan_start_time).values).min()
        )
        if not self.alert_fired:
            # if scan window is less than 2 data poins and there is no alert fired
            # skip this scan
            if n - scan_start_index <= 1:
                return
            detector = CUSUMDetector(historical_data)
            changepoints = detector.detector(
                interest_window=[scan_start_index, n],
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            if len(changepoints) > 0:
                cp, meta = sorted(changepoints, key=lambda x: x[0].start_time)[0]
                self.cps.append(int(cp.start_time.value / 1e9))

                if len(self.cps) > MAX_CHANGEPOINT:
                    self.cps.pop(0)

                self.set_alert_on(
                    historical_data.value[: meta.cp_index + 1].mean(),
                    historical_data.value[: meta.cp_index + 1].std(),
                    meta.direction,
                )
        else:
            cur_mean = historical_data[scan_start_index:].value.mean()

            if self.if_normal(cur_mean, change_directions):
                self.number_of_normal_scan += 1
                if self.number_of_normal_scan >= NORMAL_TOLERENCE:
                    self.set_alert_off()
            else:
                self.number_of_normal_scan = 0

            current_time = int(data.time.max().value / 1e9)
            if current_time - self.cps[-1] > CHANGEPOINT_RETENTION:
                self.set_alert_off()

    def _predict(
        self,
        data: TimeSeriesData,
        score_func: CusumScoreFunction = CusumScoreFunction.change,
    ) -> TimeSeriesData:
        """
        data: the new data for the anoamly score calculation
        """
        if self.alert_fired:
            cp = self.cps[-1]
            tz = data.tz()
            if tz is None:
                change_time = pd.to_datetime(cp, unit="s")
            else:
                change_time = pd.to_datetime(cp, unit="s", utc=True).tz_convert(tz)

            if change_time >= data.time.iloc[0]:
                cp_index = data.time[data.time == change_time].index[0]
                data_pre = data[: cp_index + 1]
                score_pre = self._zeros_ts(data_pre)
                score_post = SCORE_FUNC_DICT[score_func](
                    data=data[cp_index + 1 :],
                    pre_mean=self.pre_mean,
                    pre_std=self.pre_std,
                )
                score_pre.extend(score_post, validate=False)
                return score_pre
            return SCORE_FUNC_DICT[score_func](
                data=data, pre_mean=self.pre_mean, pre_std=self.pre_std
            )
        else:
            return self._zeros_ts(data)

    def _zeros_ts(self, data):
        return TimeSeriesData(
            time=data.time,
            value=pd.Series(
                np.zeros(len(data)),
                name=data.value.name if data.value.name else DEFAULT_VALUE_NAME,
            ),
        )

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """
        This function combines fit and predict and return anomaly socre for data. It
        requires scan_window > step_window.
        The definition of windows in each cusum run in the loop is shown as below:
        |-----historical_window----|-step_window-|
                             |----scan_window----|
        scan_window: the window size in seconds to detect change point
        historical_window: the window size in seconds to provide historical data
        step_window: the window size in seconds to specify the step size between two
            scans
        """
        # get parameters
        scan_window = self.scan_window
        historical_window = self.historical_window
        step_window = self.step_window
        threshold = self.threshold
        delta_std_ratio = self.delta_std_ratio
        magnitude_quantile = self.magnitude_quantile
        magnitude_ratio = self.magnitude_ratio
        change_directions = self.change_directions
        score_func = self.score_func
        remove_seasonality = self.remove_seasonality

        scan_window = pd.Timedelta(scan_window, unit="s")
        historical_window = pd.Timedelta(historical_window, unit="s")

        # pull all the data in historical data
        if historical_data is not None:
            # make a copy of historical data
            historical_data = historical_data[:]
            historical_data.extend(data, validate=False)
        else:
            # When historical_data is not provided, will use part of data as
            # historical_data, and fill with zero anomaly score.
            historical_data = data[:]

        frequency = historical_data.freq_to_timedelta()
        if frequency is None or frequency is pd.NaT:
            # Use the top frequency if any, when not able to infer from data.
            freq_counts = (
                historical_data.time.diff().value_counts().sort_values(ascending=False)
            )
            if freq_counts.iloc[0] >= int(len(historical_data)) * 0.8 - 1:
                frequency = freq_counts.index[0]
            else:
                logging.debug(f"freq_counts: {freq_counts}")
                raise ValueError("Not able to infer freqency of the time series")

        if remove_seasonality:
            decomposer_input = historical_data.interpolate(frequency)

            # fixing the period to 24 hours as indicated in T81530775
            period = int(24 * 60 * 60 / frequency.total_seconds())

            decomposer = TimeSeriesDecomposition(
                decomposer_input,
                period=period,
                robust=True,
                seasonal_deg=0,
                trend_deg=1,
                low_pass_deg=1,
                low_pass_jump=int((period + 1) * 0.15),  # save run time
                seasonal_jump=1,
                trend_jump=int((period + 1) * 0.15),  # save run time
            )

            decomp = decomposer.decomposer()
            historical_data_time_idx = decomp["rem"].time.isin(historical_data.time)
            historical_data.value = pd.Series(
                decomp["rem"][historical_data_time_idx].value
                + decomp["trend"][historical_data_time_idx].value,
                name=historical_data.value.name,
            )

        smooth_window = int(scan_window.total_seconds() / frequency.total_seconds())
        if smooth_window > 1:
            smooth_historical_value = pd.Series(
                np.convolve(
                    historical_data.value.values, np.ones(smooth_window), mode="full"
                )[: 1 - smooth_window]
                / smooth_window,
                name=historical_data.value.name,
            )
            smooth_historical_data = TimeSeriesData(
                time=historical_data.time, value=smooth_historical_value
            )
        else:
            smooth_historical_data = historical_data

        anomaly_start_time = max(
            historical_data.time.iloc[0] + historical_window, data.time.iloc[0]
        )
        if anomaly_start_time > historical_data.time.iloc[-1]:
            # if len(all data) is smaller than historical window return zero score
            return AnomalyResponse(
                scores=self._predict(smooth_historical_data[-len(data) :], score_func),
                confidence_band=None,
                predicted_ts=None,
                anomaly_magnitude_ts=self._zeros_ts(data),
                stat_sig_ts=None,
            )
        anomaly_start_idx = self._time2idx(data, anomaly_start_time, "right")
        anomaly_start_time = data.time.iloc[anomaly_start_idx]
        score_tsd = self._zeros_ts(data[:anomaly_start_idx])

        if (
            historical_data.time.iloc[-1] - historical_data.time.iloc[0] + frequency
            <= scan_window
        ):
            # if len(all data) is smaller than scan data return zero score
            return AnomalyResponse(
                scores=self._predict(smooth_historical_data[-len(data) :], score_func),
                confidence_band=None,
                predicted_ts=None,
                anomaly_magnitude_ts=self._zeros_ts(data),
                stat_sig_ts=None,
            )

        if step_window is None:
            # if step window is not provide use the time range of data or
            # half of the scan_window.
            step_window = min(
                scan_window / 2,
                (data.time.iloc[-1] - data.time.iloc[0])
                + frequency,  # to include the last data point
            )
        else:
            step_window = pd.Timedelta(step_window, unit="s")

        for start_time in pd.date_range(
            anomaly_start_time,
            min(
                data.time.iloc[-1]
                + frequency
                - step_window,  # to include last data point
                data.time.iloc[-1],  # make sure start_time won't beyond last data time
            ),
            freq=step_window,
        ):
            logging.debug(f"start_time {start_time}")
            historical_start = self._time2idx(
                historical_data, start_time - historical_window, "right"
            )
            logging.debug(f"historical_start {historical_start}")
            historical_end = self._time2idx(historical_data, start_time, "right")
            logging.debug(f"historical_end {historical_end}")
            scan_end = self._time2idx(historical_data, start_time + step_window, "left")
            logging.debug(f"scan_end {scan_end}")
            in_data = historical_data[historical_end : scan_end + 1]
            if len(in_data) == 0:
                # skip if there is no data in the step_window
                continue
            in_hist = historical_data[historical_start:historical_end]
            self._fit(
                in_data,
                in_hist,
                scan_window=scan_window,
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            score_tsd.extend(
                self._predict(
                    smooth_historical_data[historical_end : scan_end + 1],
                    score_func=score_func,
                ),
                validate=False,
            )

        # Handle the remaining data
        remain_data_len = len(data) - len(score_tsd)
        if remain_data_len > 0:
            scan_end = len(historical_data)
            historical_end = len(historical_data) - remain_data_len
            historical_start = self._time2idx(
                historical_data,
                historical_data.time.iloc[historical_end] - historical_window,
                "left",
            )
            in_data = historical_data[historical_end:scan_end]
            in_hist = historical_data[historical_start:historical_end]
            self._fit(
                in_data,
                in_hist,
                scan_window=scan_window,
                threshold=threshold,
                delta_std_ratio=delta_std_ratio,
                magnitude_quantile=magnitude_quantile,
                magnitude_ratio=magnitude_ratio,
                change_directions=change_directions,
            )
            score_tsd.extend(
                self._predict(
                    smooth_historical_data[historical_end:scan_end],
                    score_func=score_func,
                ),
                validate=False,
            )

        return AnomalyResponse(
            scores=score_tsd,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=self._zeros_ts(data),
            stat_sig_ts=None,
        )

    def _time2idx(self, tsd: TimeSeriesData, time: datetime, direction: str):
        """
        This function get the index of the TimeSeries data given a datatime.
        left takes the index on the left of the time stamp (inclusive)
        right takes the index on the right of the time stamp (exclusive)
        """
        if direction == "right":
            return np.argwhere((tsd.time >= time).values).min()
        elif direction == "left":
            return np.argwhere((tsd.time < time).values).max()
        else:
            raise ValueError("direction can only be right or left")

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> None:
        self.fit_predict(data, historical_data)

    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")
