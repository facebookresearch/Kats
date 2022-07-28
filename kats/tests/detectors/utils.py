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

from typing import List, Tuple

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from scipy.special import expit  # @manual


def gen_no_trend_data_ndim(time: pd.Series, ndim: int = 1) -> TimeSeriesData:
    np.random.seed(20)
    n_days = len(time)
    data = np.ones((n_days, ndim)) * np.random.randint(1000, size=(1, ndim))
    no_trend_data = pd.DataFrame(data)
    no_trend_data["time"] = time

    return TimeSeriesData(no_trend_data)


def gen_trend_data_ndim(
    time: pd.Series,
    seasonality: float = 0.00,
    change_smoothness: float = 5.0,
    ndim: int = 1,
) -> Tuple[TimeSeriesData, List[float]]:
    np.random.seed(20)

    n_days = len(time)
    ix = np.array([np.arange(n_days) for i in range(ndim)])
    initial = np.random.randint(9000.0, 10000.0, size=(ndim, 1))
    trend_change = -np.random.randint(60, size=(ndim, 1))
    trend = np.random.randint(2.0, 6.0, size=(ndim, 1))
    noise = np.array([1e-3] * ndim).reshape((ndim, 1))
    t_change = np.random.randint(int(0.4 * n_days), int(0.7 * n_days), size=(ndim, 1))

    data = (
        (initial + trend * ix + trend_change * (ix - t_change) * expit((ix - t_change)))
        * (1 - seasonality * (ix % 7 >= 5))
        * np.array(
            [np.cumprod(1 + noise[i] * np.random.randn(n_days)) for i in range(ndim)]
        )
    )

    trend_data = pd.DataFrame(data.T)
    trend_data["time"] = time

    t_change = [t_change[i][0] for i in range(len(t_change))]

    return TimeSeriesData(trend_data), t_change
