# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

import pandas as pd
from kats.utils.dataframe_utils import rename_columns_by_prefix


class TestRenameColumnsByPrefix(TestCase):
    def test_basic_rename(self) -> None:
        df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "test_metric_count": [100, 110, 120],
                "control_metric_count": [90, 95, 100],
            }
        )
        prefix_map = {
            "test_metric_": "numerator_test",
            "control_metric_": "numerator_control",
        }

        result = rename_columns_by_prefix(df, prefix_map)

        self.assertEqual(
            list(result.columns),
            ["time", "numerator_test", "numerator_control"],
        )
        self.assertEqual(result["numerator_test"].tolist(), [100, 110, 120])

    def test_time_column_preserved(self) -> None:
        df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "metric_value": [10, 20, 30],
            }
        )
        prefix_map = {"metric_": "renamed_metric"}

        result = rename_columns_by_prefix(df, prefix_map)

        self.assertIn("time", result.columns)
        self.assertEqual(result["time"].tolist(), [1, 2, 3])

    def test_error_when_prefix_matches_zero_columns(self) -> None:
        df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "column1": [10, 20, 30],
            }
        )
        prefix_map = {"nonexistent_": "renamed"}

        with self.assertRaises(ValueError) as context:
            rename_columns_by_prefix(df, prefix_map)

        self.assertIn("does not match any column", str(context.exception))

    def test_error_when_prefix_matches_multiple_columns(self) -> None:
        df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "test_metric1": [10, 20, 30],
                "test_metric2": [40, 50, 60],
            }
        )
        prefix_map = {"test_": "renamed"}

        with self.assertRaises(ValueError) as context:
            rename_columns_by_prefix(df, prefix_map)

        self.assertIn("matches multiple columns", str(context.exception))
