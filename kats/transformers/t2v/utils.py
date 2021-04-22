#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def Normalize(
    timeseries: np.ndarray,
) -> np.ndarray:
    """
    A function for normalizing a timeseries with its max and min values.

    :Parameters:
    timeseries: np.ndarray
        A single timeseries data in the form of an array.
    """
    _min = np.min(timeseries)
    _max = np.max(timeseries)

    timeseries = (timeseries - _min) / (_max - _min)

    return timeseries


def Standardize(
    timeseries: np.ndarray,
) -> np.ndarray:
    """
    A function for normalizing a timeseries with its mean and standard
    deviation.

    :Parameters:
    timeseries: np.ndarray
        A single timeseries data in the form of an array.
    """
    mean = np.mean(timeseries)
    std = np.std(timeseries)

    timeseries = (timeseries - mean) / std

    return timeseries
