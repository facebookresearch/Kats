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
    time_col_name: str = "time",
) -> pd.DataFrame:
    """
    Renames columns in the dataframe based on provided prefix mappings.
    Each prefix in prefix_map must match exactly one column in df.
    No column can be matched by more than one prefix.
    Columns that do not match any prefix are left unchanged.
    The time_col_name column is always preserved.

    Args:
        df: Input pandas DataFrame.
        prefix_map: Dictionary mapping column prefixes to desired column names.

    Returns:
        pandas.DataFrame: A copy of the input dataframe with renamed columns.

    Raises:
        ValueError: If a prefix matches zero or multiple columns,
                    or if a column is matched by multiple prefixes.

    Example:
        >>> df = pd.DataFrame({
        ...     "time": [1, 2, 3],
        ...     "test_metric_count": [100, 110, 120],
        ...     "control_metric_count": [90, 95, 100],
        ...     "other_column": [1, 2, 3],
        ... })
        >>> prefix_map = {
        ...     "test_metric_": "numerator_test",
        ...     "control_metric_": "numerator_control",
        ... }
        >>> renamed_df = rename_columns_by_prefix(df, prefix_map)
        >>> list(renamed_df.columns)
        ['time', 'numerator_test', 'numerator_control', 'other_column']

    Example (error if multiple prefixes match the same column):
        >>> df = pd.DataFrame({
        ...     "test_metric_count": [1, 2, 3],
        ...     "time": [0, 1, 2]
        ... })
        >>> prefix_map = {
        ...     "test_": "foo",
        ...     "test_metric_": "bar"
        ... }
        >>> # This will raise a ValueError:
        >>> renamed_df = rename_columns_by_prefix(df, prefix_map)
        ValueError: Column 'test_metric_count' is matched by multiple prefixes: 'test_' and 'test_metric_'. Each column can only be matched by one prefix.
    """
    rename_mapping = {}
    matched_columns = {}

    for prefix, new_name in prefix_map.items():
        # Find all columns that match this prefix
        matched = [
            col for col in df.columns if col != time_col_name and col.startswith(prefix)
        ]
        if len(matched) == 0:
            raise ValueError(
                f"Prefix '{prefix}' does not match any column in the DataFrame."
            )
        if len(matched) > 1:
            raise ValueError(
                f"Prefix '{prefix}' matches multiple columns: {matched}. "
                "Each prefix must match exactly one column."
            )
        matched_col = matched[0]
        if matched_col in matched_columns:
            raise ValueError(
                f"Column '{matched_col}' is matched by multiple prefixes: "
                f"'{matched_columns[matched_col]}' and '{prefix}'. "
                "Each column can only be matched by one prefix."
            )
        rename_mapping[matched_col] = new_name
        matched_columns[matched_col] = prefix

    return df.rename(columns=rename_mapping, copy=True)
