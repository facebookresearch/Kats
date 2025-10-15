# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from typing import Dict

import pandas as pd


def rename_columns_by_prefix(
    df: pd.DataFrame,
    prefix_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Renames columns in the dataframe based on provided prefix mappings.

    This utility function maps columns from the input dataframe to standardized names
    by matching column names against prefixes. Each column must match exactly one prefix.

    Args:
        df: Input pandas DataFrame containing time series data
        prefix_map: Dictionary mapping column prefixes to desired column names.
                   Example: {"test_": "value_test", "control_": "value_control"}

    Returns:
        pandas.DataFrame: A copy of the input dataframe with renamed columns.
                         The "time" column is preserved unchanged.

    Raises:
        ValueError: If a column matches multiple prefixes or doesn't match any prefix
                   (excluding the "time" column which is always preserved).

    Example:
        >>> df = pd.DataFrame({
        ...     "time": [1, 2, 3],
        ...     "test_metric_count": [100, 110, 120],
        ...     "control_metric_count": [90, 95, 100],
        ... })
        >>> prefix_map = {
        ...     "test_metric_": "numerator_test",
        ...     "control_metric_": "numerator_control",
        ... }
        >>> renamed_df = rename_columns_by_prefix(df, prefix_map)
        >>> list(renamed_df.columns)
        ['time', 'numerator_test', 'numerator_control']
    """
    rename_mapping = {}

    for column in df.columns:
        # Skip the time column
        if column == "time":
            continue

        # Find all prefixes that match this column
        matching_prefixes = [
            prefix for prefix in prefix_map.keys() if column.startswith(prefix)
        ]

        if len(matching_prefixes) > 1:
            raise ValueError(
                f"Column '{column}' matches multiple prefixes: {matching_prefixes}. "
                f"Each column must match exactly one prefix from: {list(prefix_map.keys())}"
            )

        if len(matching_prefixes) == 0:
            raise ValueError(
                f"Column '{column}' does not match any prefix from the provided mapping. "
                f"Expected prefixes: {list(prefix_map.keys())}"
            )

        # Exactly one match - add to rename mapping
        new_name = prefix_map[matching_prefixes[0]]
        rename_mapping[column] = new_name

    return df.rename(columns=rename_mapping, copy=True)
