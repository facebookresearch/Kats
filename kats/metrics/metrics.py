# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import cast, Dict, Generator, Optional, Sequence, Union

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # pragma: no cover

import numpy as np

with warnings.catch_warnings():
    # suppress patsy warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from statsmodels.distributions.empirical_distribution import ECDF

# from numpy.typing import ArrayLike
ArrayLike = Union[np.ndarray, Sequence[float]]

# Type aliases
#
# Most metrics have the shape:
#
# def metric(y_true: ArrayLike,
#            y_pred: ArrayLike) -> float:
#
# defined as Metric or
#
# def metric(y_true: ArrayLike,
#            y_pred: ArrayLike,
#            sample_weight: Optional[ArrayLike] = None) -> float:
#
# defined as WeightedMetric.
#
# Other shapes require their own custom protocols.
#
class ArrayMetric(Protocol):
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = ...,
    ) -> np.ndarray:
        ...  # pragma: no cover


class Metric(Protocol):
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        ...  # pragma: no cover


class WeightedMetric(Protocol):
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = ...,
    ) -> float:
        ...  # pragma: no cover


class MultiOutputMetric(Protocol):
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike] = ...,
        multioutput: Union[str, ArrayLike] = ...,
    ) -> float:
        ...  # pragma: no cover


class ThresholdMetric(Protocol):
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        threshold: float,
    ) -> float:
        ...  # pragma: no cover


KatsMetric = Union[
    Metric, ArrayMetric, WeightedMetric, MultiOutputMetric, ThresholdMetric
]

# Imported metrics

from kats.compat.sklearn import (  # noqa
    # mean_absolute_error,  # doesn't handle empty arrays, nan values
    # mean_pinball_loss
    mean_squared_error as sklearn_mse,
    mean_squared_log_error as sklearn_msle,
)


def _arrays(*args: Optional[ArrayLike]) -> Generator[np.ndarray, None, None]:
    """Ensure all arguments are arrays of matching size.

    Args:
        args: zero or more array-like values.

    Yields:
        The values converted to numpy arrays.

    Raises:
        ValueError if any arrays are different sizes.
    """
    n_samples = None
    for a in args:
        if a is None:
            continue
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if n_samples is None:
            n_samples = a.shape[0]
        elif a.shape[0] != n_samples:
            raise ValueError(
                f"Arrays have different number of samples ({a.shape}, expected {n_samples})"
            )
        yield a


def _safe_divide(
    num: np.ndarray,
    denom: Union[np.ndarray, float],
    negative_infinity: float = -1.0,
    positive_infinity: float = 1.0,
    indeterminate: float = 0.0,
    nan: float = np.nan,
) -> np.ndarray:
    """Safely divide one array by another or a float.

    Args:
        num: the numerator
        denom: the denominator
        negative_infinity: the value to replace negative infinity with
        positive_infinity: the value to replace positive infinity with
        indeterminate: the value to replace indeterminates with
        nan: the value to replace nan values with

    Returns:
        The numerator divided by the denominator.
    """
    nans = np.isnan(num)
    if ~np.isscalar(denom):
        nans &= np.isnan(denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(num, denom)

    # compute locations up-front
    neg_inf = np.isneginf(result)
    pos_inf = np.isposinf(result)
    ind = np.isnan(result)

    # Replace values, nans last
    result[neg_inf] = negative_infinity
    result[pos_inf] = positive_infinity
    result[ind] = indeterminate
    result[nans] = nan
    return result


#
# Array Metrics
#

# Each is defined as a helper on the already-cast values, in case other metrics
# methods need to use those.


def _err(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    if sample_weight is None:
        return y_true - y_pred
    return _safe_divide(
        np.multiply(y_true - y_pred, sample_weight), np.nansum(sample_weight)
    )


def error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None
) -> np.ndarray:
    """Compute the error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The error array.
    """
    return _err(*_arrays(y_true, y_pred, sample_weight))


def _abs_err(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    err = np.abs(y_true - y_pred)
    if sample_weight is None:
        return err

    err = np.multiply(err, sample_weight)
    return _safe_divide(err, np.nansum(sample_weight))


def absolute_error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None
) -> np.ndarray:
    """Compute the absolute error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The absolute error array.
    """
    return _abs_err(*_arrays(y_true, y_pred, sample_weight))


def _pct_err(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    err = y_true - y_pred
    if sample_weight is None:
        return _safe_divide(err, y_true)

    err = np.multiply(err, sample_weight)
    return _safe_divide(err, y_true * np.nansum(sample_weight))


def percentage_error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None
) -> np.ndarray:
    """Compute the percentage error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The percentage error array.
    """
    return _pct_err(*_arrays(y_true, y_pred, sample_weight))


def _abs_pct_err(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    err = np.abs(y_true - y_pred)
    if sample_weight is None:
        return _safe_divide(err, y_true)

    err = np.multiply(err, sample_weight)
    return _safe_divide(err, y_true * np.nansum(sample_weight))


def absolute_percentage_error(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None
) -> np.ndarray:
    """Compute the absolute percentage error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The absolute percentage error array.
    """
    return _abs_pct_err(*_arrays(y_true, y_pred, sample_weight))


#
# Metrics
#


def continuous_rank_probability_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the continuous rank in probability score (CRPS).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The linear error in probability score (LEPS) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    # If the arrays are empty or all values are missing, return nan.
    if not y_true.size:
        return np.nan
    idx = np.where(~np.isnan(y_pred))[0]
    if not idx.size:
        return np.nan
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    # Otherwise, drop missing entries and compute the ECDF...
    ecdf = ECDF(y_true)
    # ...and finally compute CRPS.
    return np.nanmean((ecdf(y_pred) - ecdf(y_true)) ** 2)


# todo
# def cross_entropy(y_true: ArrayLike,
#                   y_pred: ArrayLike) -> float:


def frequency_exceeds_relative_threshold(
    y_true: ArrayLike, y_pred: ArrayLike, threshold: float
) -> float:
    """Compute the fraction of true that exceeds threshold times prediction.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        threshold: the threshold value.

    Returns:
        The fraction of true values that exceeds threshold times prediction.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    return np.nanmean(y_true >= threshold * y_pred)


def linear_error_in_probability_space(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the linear error in probability space (LEPS).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The linear error in probability space (LEPS) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    # If the arrays are empty or all values are missing, return nan.
    if not y_true.size:
        return np.nan
    idx = np.where(~np.isnan(y_pred))[0]
    if not idx.size:
        return np.nan
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    # Otherwise, drop missing entries and compute the ECDF...
    ecdf = ECDF(y_true)
    # ...and finally compute LEPS.
    return np.nanmean(np.abs(ecdf(y_pred) - ecdf(y_true)))


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the median absolute error (MdAE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The median absolute error (MdAE) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    if not y_true.size:
        return np.nan
    return np.nanmedian(_abs_err(y_true, y_pred))


def median_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the median absolute percentage error (MdAPE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The median absolute percentage error (MdAPE) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    if not y_true.size:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(_abs_pct_err(y_true, y_pred))


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Union[str, ArrayLike] = "uniform_average",
) -> float:
    """Compute the mean absolute error (MAE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.
        multioutput: raw_values, uniform_avareage, or array shape.

    Returns:
        The mean absolute error (MAE) value.
    """
    if isinstance(multioutput, str):
        if multioutput not in {"raw_values", "uniform_average"}:
            raise ValueError(
                "multioutput must be 'raw_values', 'uniform_average', or an array of floats"
            )

    err = absolute_error(y_true, y_pred)
    if not err.shape[0]:
        return np.nan
    ma = np.ma.MaskedArray(err, np.isnan(err))
    err = np.ma.average(ma, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return err
        multioutput = None
    return np.average(err, weights=multioutput)


def mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean absolute percentage error (MAPE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The mean absolute percentage error (MAPE) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    if not y_true.size:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(_abs_pct_err(y_true, y_pred))


def mean_absolute_scaled_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean absolute scaled error (MASE).

    MASE is computed by comparing mean absolute error of the predictions to
    the mean absolute error obtained from a lag-1 forecast.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The mean absolute scaled error (MASE) value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    if not y_true.size:
        return np.nan
    denom = np.nanmean(np.abs(np.diff(y_true)))
    if np.isclose(denom, 0.0):
        return np.nan
    return mean_absolute_error(y_true, y_pred) / denom


def mean_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The mean error value.
    """
    return np.nanmean(error(y_true, y_pred))


# todo
# def mean_interval_score(y_true: ArrayLike,
#                         y_pred: ArrayLike) -> float:
#


def mean_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean percentage error (MPE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The mean percentage error value.
    """
    return np.nanmean(percentage_error(y_true, y_pred))


def mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the mean squared error (RSE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The mean squared error (MSE) value.
    """
    if not len(y_true):
        return np.nan
    # pyre-ignore[6]: For 1st param expected `bool` but got `ndarray`.
    # pyre-ignore[6]: For 1st param expected `str` but got `ndarray`.
    return sklearn_mse(*_arrays(y_true, y_pred, sample_weight))


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the root mean squared error (RMSE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The root mean squared error (RMSE) value.
    """
    if not len(y_true):
        return np.nan
    # pyre-ignore[6]: For 1st param expected `str` but got `ndarray`.
    return sklearn_mse(*_arrays(y_true, y_pred, sample_weight), squared=False)


def root_mean_squared_log_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the root mean squared log error (RMSLE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The root mean squared log error (RMSLE) value.
    """
    if not len(y_true):
        return np.nan
    # pyre-ignore[6]: For 1st param expected `str` but got `ndarray`.
    return sklearn_msle(*_arrays(y_true, y_pred, sample_weight), squared=False)


def root_mean_squared_percentage_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the root mean squared percentage error (RMSPE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.
        sample_weight: optional weights.

    Returns:
        The root mean squared percentage error (RMSPE) value.
    """
    if not len(y_true):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.sqrt(
            np.nanmean(_pct_err(*_arrays(y_true, y_pred, sample_weight)) ** 2)
        )


def scaled_symmetric_mean_absolute_percentage_error(
    y_true: ArrayLike, y_pred: ArrayLike
) -> float:
    """Compute the scaled symmetric mean absolute percentage error (SMAPE).

    Traditionally, SMAPE goes from 0 to 200%. This function goes from
    0 to 100%, and is thus equal to SMAPE divided by 2.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The symmetric mean absolute percentage error (SMAPE) value.
    """
    num = absolute_error(y_true, y_pred)
    return np.nanmean(_safe_divide(num, np.abs(y_true) + np.abs(y_pred)))


def symmetric_bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the symmetric bias (sbias).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The the symmetric bias (sbias) value.
    """
    return -2 * np.nanmean(
        _safe_divide(error(y_true, y_pred), np.abs(y_true) + np.abs(y_pred))
    )


def symmetric_mean_absolute_percentage_error(
    y_true: ArrayLike, y_pred: ArrayLike
) -> float:
    """Compute the symmetric mean absolute percentage error (SMAPE).

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The symmetric mean absolute percentage error (SMAPE) value.
    """
    return 2.0*scaled_symmetric_mean_absolute_percentage_error(y_true, y_pred)


def tracking_signal(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the tracking signal.

    Tracking signal is the ratio of the sum of the residuals to the
    mean absolute error.

    Args:
        y_true: the actual values.
        y_pred: the predicted values.

    Returns:
        The tracking signal value.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    err = mean_absolute_error(y_true, y_pred)
    return np.nan if err == 0 else np.sum(y_true - y_pred) / err


# Name aliases (sorted alphabetically by alias)

ae: ArrayMetric = absolute_error
ape: ArrayMetric = absolute_percentage_error
bias: Metric = mean_error
crps: Metric = continuous_rank_probability_score
leps: Metric = linear_error_in_probability_space
# mad - don't abbreviate this, it's ambiguous whether mean or median
mae: MultiOutputMetric = mean_absolute_error
mape: Metric = mean_absolute_percentage_error
mase: Metric = mean_absolute_scaled_error
mdae: Metric = median_absolute_error
mdape: Metric = median_absolute_percentage_error
me: Metric = mean_error
mean_absolute_deviation: MultiOutputMetric = mean_absolute_error
median_absolute_deviation: Metric = median_absolute_error
# mis: Metric = mean_interval_score  # todo
mpe: Metric = mean_percentage_error
mse: WeightedMetric = mean_squared_error
residual: ArrayMetric = error
rmse: WeightedMetric = root_mean_squared_error
rmsle: WeightedMetric = root_mean_squared_log_error
rmspe: WeightedMetric = root_mean_squared_percentage_error
sbias: Metric = symmetric_bias
scaled_smape: Metric = scaled_symmetric_mean_absolute_percentage_error
smape: Metric = symmetric_mean_absolute_percentage_error

ALL_METRICS: Dict[str, KatsMetric] = {
    # Array Metrics
    "error": error,
    "absolute_error": absolute_error,
    "percentage_error": percentage_error,
    "absolute_percentage_error": absolute_percentage_error,
    # Other Metrics
    "continuous_rank_probability_score": continuous_rank_probability_score,
    "frequency_exceeds_relative_threshold": frequency_exceeds_relative_threshold,
    "linear_error_in_probability_space": linear_error_in_probability_space,
    "median_absolute_error": median_absolute_error,
    "median_absolute_percentage_error": median_absolute_percentage_error,
    "mean_absolute_error": mean_absolute_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "mean_absolute_scaled_error": mean_absolute_scaled_error,
    "mean_error": mean_error,
    "mean_percentage_error": mean_percentage_error,
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "root_mean_squared_log_error": root_mean_squared_log_error,
    "root_mean_squared_percentage_error": root_mean_squared_percentage_error,
    "symmetric_bias": symmetric_bias,
    "symmetric_mean_absolute_percentage_error": symmetric_mean_absolute_percentage_error,
    "tracking_signal": tracking_signal,
    # Aliases
    "ae": ae,
    "ape": ape,
    "bias": bias,
    "crps": crps,
    "leps": leps,
    "mae": mae,
    "mape": mape,
    "mase": mase,
    "mdae": mdae,
    "mdape": mdape,
    "me": me,
    "mean_absolute_deviation": mean_absolute_deviation,
    "median_absolute_deviation": median_absolute_deviation,
    "mpe": mpe,
    "mse": mse,
    "residual": residual,
    "rmse": rmse,
    "rmsle": rmsle,
    "rmspe": rmspe,
    "sbias": sbias,
    "smape": smape,
}


def metric(name: str) -> KatsMetric:
    """Convenience method to look up a kats.metric by name.

    Args:
        name: the metric name (or abbreviation)

    Returns:
        The metric function.
    """
    try:
        return ALL_METRICS[name]
    except KeyError:
        raise ValueError(f"Could not find metric named {name}")


# These are metrics that return a float and can be invoked with just (y_true, y_pred).
CoreMetric = Union[Metric, WeightedMetric, MultiOutputMetric]

CORE_METRICS: Dict[str, CoreMetric] = {
    name: cast(CoreMetric, method) for name, method in ALL_METRICS.items()
    if name not in {
        "error",
        "absolute_error",
        "percentage_error",
        "absolute_percentage_error",
        "frequency_exceeds_relative_threshold",
        "ae",
        "ape",
    }
}


def core_metric(name: str) -> CoreMetric:
    """Convenience method to look up a kats.metric by name.

    Args:
        name: the metric name (or abbreviation)

    Returns:
        The metric function.
    """
    try:
        return CORE_METRICS[name]
    except KeyError:
        raise ValueError(f"Could not find core metric named {name}")
