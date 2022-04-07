# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Dict

import numpy as np


class MetricType(Enum):
    """Metrics can be scores, errors, or neither."""

    NONE = 0
    """Neither score nor error"""

    SCORE = 1
    """Larger is better (1 is better than 0)"""

    ERROR = 2
    """Closer to zero is better (0 is better than 1 or -1)"""


class Directionality(IntEnum):
    """Metrics can improve in a direction (up or down) or lack clear direction.

    Non-negative error metrics are negative and vice-versa because lower values
    are also closer to zero. However, metrics that can result in negative values
    cannot be simultaneously NEGATIVE and ERROR.
    """

    NONE = 0
    """Neither positive nor negative."""

    POSITIVE = 1
    """Larger is better  (1 is better than 0)"""

    NEGATIVE = -1
    """Smaller is better (-1 better than 0 better than 1)"""


@dataclass
class MetricMetadata:
    """Metadata about a metric method."""

    type: MetricType
    """The type of metric (score, error, neither)."""

    direction: Directionality
    """The directionality of a metric (positive, negative, or neither is better)."""

    lower_bound: float
    """All values returned by this metric are at least the lower bound."""

    upper_bound: float
    """All values returned by this metric are at most the upper bound."""


_NONNEGATIVE_ERROR_METRIC = MetricMetadata(
    MetricType.ERROR, Directionality.NEGATIVE, 0.0, np.inf
)
_ERROR_RATE_METRIC = MetricMetadata(MetricType.ERROR, Directionality.NEGATIVE, 0.0, 1.0)
_UNBOUNDED_ERROR_METRIC = MetricMetadata(
    MetricType.ERROR, Directionality.NONE, np.NINF, np.inf
)
_UNBOUNDED_METRIC = MetricMetadata(
    MetricType.NONE, Directionality.NONE, np.NINF, np.inf
)
_SCORE_MAX_1 = MetricMetadata(MetricType.SCORE, Directionality.POSITIVE, np.NINF, 1.0)
_SCORE_MIN_0 = MetricMetadata(MetricType.SCORE, Directionality.POSITIVE, 0.0, np.inf)


# These metadata values apply only when unweighted. When unconstrained weights
# are used, most metrics become unbounded.
@lru_cache(None)
def _metadata() -> Dict[str, MetricMetadata]:
    metadata = {
        "continuous_rank_probability_score": _ERROR_RATE_METRIC,
        "cross_entropy": _NONNEGATIVE_ERROR_METRIC,
        "frequency_exceed_relative_threshold": _ERROR_RATE_METRIC,
        "linear_error_in_probability_space": _ERROR_RATE_METRIC,
        "max_error": _UNBOUNDED_ERROR_METRIC,
        "mean_absolute_error": _NONNEGATIVE_ERROR_METRIC,
        "mean_absolute_percentage_error": _NONNEGATIVE_ERROR_METRIC,
        "mean_absolute_scaled_error": _NONNEGATIVE_ERROR_METRIC,
        "mean_error": _UNBOUNDED_ERROR_METRIC,
        "mean_interval_score": _SCORE_MIN_0,
        "mean_percentage_error": _UNBOUNDED_ERROR_METRIC,
        "mean_squared_error": _NONNEGATIVE_ERROR_METRIC,
        "mean_squared_log_error": _NONNEGATIVE_ERROR_METRIC,
        "median_absolute_error": _UNBOUNDED_ERROR_METRIC,
        "median_absolute_percentage_error": _NONNEGATIVE_ERROR_METRIC,
        "r2_score": _SCORE_MAX_1,
        "root_mean_squared_error": _NONNEGATIVE_ERROR_METRIC,
        "root_mean_squared_log_error": _NONNEGATIVE_ERROR_METRIC,
        "root_mean_squared_percentage_error": _NONNEGATIVE_ERROR_METRIC,
        "symmetric_bias": _UNBOUNDED_METRIC,
        "symmetric_mean_absolute_percentage_error": _ERROR_RATE_METRIC,
        "tracking_signal": _UNBOUNDED_ERROR_METRIC,
    }
    # aliases
    metadata["bias"] = metadata["mean_error"]
    metadata["crps"] = metadata["continuous_rank_probability_score"]
    metadata["leps"] = metadata["linear_error_in_probability_space"]
    metadata["mae"] = metadata["mean_absolute_error"]
    metadata["mape"] = metadata["mean_absolute_percentage_error"]
    metadata["mase"] = metadata["mean_absolute_scaled_error"]
    metadata["mdae"] = metadata["median_absolute_error"]
    metadata["mdape"] = metadata["median_absolute_percentage_error"]
    metadata["me"] = metadata["mean_error"]
    metadata["median_absolute_deviation"] = metadata["median_absolute_error"]
    metadata["mean_absolute_deviation"] = metadata["mean_absolute_error"]
    metadata["mis"] = metadata["mean_interval_score"]
    metadata["mpe"] = metadata["mean_percentage_error"]
    metadata["mse"] = metadata["mean_squared_error"]
    metadata["r2"] = metadata["r2_score"]
    metadata["rmse"] = metadata["root_mean_squared_error"]
    metadata["rmsle"] = metadata["root_mean_squared_log_error"]
    metadata["rmspe"] = metadata["root_mean_squared_percentage_error"]
    metadata["sbias"] = metadata["symmetric_bias"]
    metadata["smape"] = metadata["symmetric_mean_absolute_percentage_error"]
    return metadata


def inspect(name: str) -> MetricMetadata:
    """Get metadata about a metric, by name.

    Args:
        name: the metric name

    Returns:
        Metadata about that metric.

    Raises:
        ValueError if the metric name is not a Kats metric.
    """
    metadata = _metadata()
    try:
        return metadata[name]
    except KeyError:
        raise ValueError(f"Unknown metric {name}")
