# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base I/O code for time series data in Kats

This is a base implementation to load datasets for test and evaluation/benchmarking
purposes. We currently support the following data sets:

1. air passengers data
2. m3 meta data
3. Peyton manning data
4. retail sales data
5. yosemite temps data
6. multi ts data
7. mean change detection test data
8. multivariate anomaly simulated data
"""

import io
import os
import pkgutil
import sys
from typing import overload, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import pandas as pd
from kats.consts import TimeSeriesData


def load_data(file_name: str, reset_columns: bool = False) -> pd.DataFrame:
    """load data for tests and tutorial notebooks"""
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    assert data_object is not None
    df = pd.read_csv(io.BytesIO(data_object), encoding="utf8")
    if reset_columns:
        df.columns = ["time", "y"]
    return df


@overload
def load_air_passengers(return_ts: Literal[True]) -> TimeSeriesData:
    ...


@overload
def load_air_passengers(return_ts: Literal[False] = ...) -> pd.DataFrame:
    ...


def load_air_passengers(return_ts: bool = True) -> Union[pd.DataFrame, TimeSeriesData]:
    """Load and return air passengers time series dataset

    ==============================
    Length                     144
    Granularity              daily
    Label                     none

    Args:
        return_ts: return class:`kats.consts.TimeSeriesData` by default
                   return `pandas.DataFrame` otherwise

    Returns:
        data: class:`kats.consts.TimeSeriesData`
        file_name: `str`, the physical path of air passengers data set
        descr: `str`, full description of this data set

    Examples
    >>> from kats.data import load_air_passengers
    >>> air_passengers_ts = load_air_passengers()
    >>> print(air_passengers_ts)
              time    y
        0   1949-01-01  112
        1   1949-02-01  118
        2   1949-03-01  132
        3   1949-04-01  129
        4   1949-05-01  121
        ..         ...  ...
        139 1960-08-01  606
        140 1960-09-01  508
        141 1960-10-01  461
        142 1960-11-01  390
        143 1960-12-01  432

        [144 rows x 2 columns]
    """
    df = load_data("air_passengers.csv")
    df.columns = ["time", "y"]

    if return_ts:
        return TimeSeriesData(df)
    else:
        return df
