# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from typing import List, Optional, Union
from unittest import TestCase

from kats.metrics import metadata
from kats.metrics.metadata import inspect, MetricMetadata
from parameterized.parameterized import parameterized


class MetricsMetadataTest(TestCase):

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("continuous_rank_probability_score", metadata._ERROR_RATE_METRIC),
            ("cross_entropy", metadata._NONNEGATIVE_ERROR_METRIC),
            ("frequency_exceed_relative_threshold", metadata._ERROR_RATE_METRIC),
            ("linear_error_in_probability_space", metadata._ERROR_RATE_METRIC),
            ("max_error", metadata._UNBOUNDED_ERROR_METRIC),
            ("mean_absolute_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("mean_absolute_scaled_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("mean_error", metadata._UNBOUNDED_ERROR_METRIC),
            ("mean_interval_score", metadata._SCORE_MIN_0),
            ("mean_percentage_error", metadata._UNBOUNDED_ERROR_METRIC),
            ("mean_squared_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("mean_squared_log_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("median_absolute_error", metadata._UNBOUNDED_ERROR_METRIC),
            ("mean_absolute_percentage_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("median_absolute_percentage_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("r2_score", metadata._SCORE_MAX_1),
            ("root_mean_squared_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("root_mean_squared_log_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("root_mean_squared_percentage_error", metadata._NONNEGATIVE_ERROR_METRIC),
            ("symmetric_bias", metadata._UNBOUNDED_METRIC),
            ("symmetric_mean_absolute_percentage_error", metadata._ERROR_RATE_METRIC),
            ("tracking_signal", metadata._UNBOUNDED_ERROR_METRIC),
        ]
    )
    def test_inspect(self, metric: str, expected: MetricMetadata) -> None:
        result = inspect(metric)
        self.assertEqual(expected, result)

    def test_inspect_unknown(self) -> None:
        with self.assertRaises(ValueError):
            _ = inspect("mango")
