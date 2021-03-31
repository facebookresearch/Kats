#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Dict, Optional

from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)


BASE_MODELS = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "sarima": sarima.SARIMAModel,
    "prophet": prophet.ProphetModel,
    "linear": linear_model.LinearModel,
    "quadratic": quadratic_model.QuadraticModel,
    "theta": theta.ThetaModel,
}

import logging

import numpy as np
import pandas as pd


def calc_mape(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Calculate MAPE = average(abs((truth-predictions)/truth))
    """
    base = np.abs((truth - predictions) / truth)
    # filter out np.inf
    base = base[base < np.inf]
    return np.mean(base)


def calc_mae(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Calculate MAE = average(abs(truth-predictions))
    """
    return np.average(np.abs(truth - predictions))


class BaseTHModel:
    """
    Base class for temporal hierarhical models.

    The object stores the information of base model. We allow users to pass model info (i.e., model_name and model_params),
    or to pass residuals and forecasts of a trained model directly.

    :Parameters:
    level: int
        level of the base model, should be a positive integer.
    model_name: Optional[str]=None
        model_name of forecast model
    model_params: Optional[object]=None
        model_params of forecast model
    residuals: Optional[np.ndarray]=None
        residuals of forecast model (necessary if both model_name and model_params are None.)
    fcsts: Optional[np.ndarray]=None
        forecasts of forecast model (necessary if both model_name and model_params are None.)
    """

    def __init__(
        self,
        level: int,
        model_name: Optional[str] = None,
        model_params: Optional[object] = None,
        residuals: Optional[np.ndarray] = None,
        fcsts: Optional[np.ndarray] = None,
    ) -> None:

        if not isinstance(level, int) or level < 1:
            msg = f"Level should be a positive integer but receive {level}."
            logging.error(msg)
            raise ValueError(msg)

        if (residuals is None or residuals.size == 0) or (
            fcsts is None or fcsts.size == 0
        ):
            # when residuals or fcsts are missing
            if (not model_name) or (not model_params):
                msg = "model_name and model_params are needed when residuals or fcsts is missing."
                logging.error(msg)
                raise ValueError(msg)
            if model_name not in BASE_MODELS:
                msg = f"model_name {model_name} is not supported!"
                logging.error(msg)
                raise ValueError(msg)

        self.level = level
        self.model_name = model_name
        self.model_params = model_params
        self.residuals = residuals
        self.fcsts = fcsts

    def __str__(self):
        return "BaseTHModel"


class GetAggregateTS:
    """
    Class for aggregating time series to different levels.

    :Parameters:
    data: TimeSeriesData
        Time series to be aggregated.

    """

    def __init__(self, data: TimeSeriesData) -> None:

        if not data.is_univariate():
            msg = f"Only support univariate time series, but get {type(data.value)}."
            logging.error(msg)
            raise ValueError(msg)

        self.data = TimeSeriesData(data.to_dataframe().copy())

    def _aggregate_single(self, ts, k):
        if k == 1:
            return ts
        if k > len(ts):
            msg = f"Level {k} should be less than the length of training time series (len(TS)={len(ts)})!"
            logging.error(msg)
            raise ValueError(msg)
        n = len(ts)
        m = (n // k) * k

        value = pd.Series(ts.value.values[-m:].reshape(-1, k).sum(axis=1))
        time = pd.Series(ts.time.values[(n - m + k - 1) : n : k])
        return TimeSeriesData(time=time, value=value)

    def aggregate(self, levels: List[int]) -> Dict[int, TimeSeriesData]:
        """
        Function for aggregating time series.

        :Parameters:
        levels:List[int]
            List of levels the time series to be aggregated for.
        :Returns:
        Dict[int, TimeSeriesData]
            Dictionary of aggregated time series for each level.
        """
        if not isinstance(levels, list):
            msg = f"Parameter 'levels' should be a list but receive {type(levels)}."
            logging.error(msg)
            raise ValueError(msg)
        for k in levels:
            if not isinstance(k, int) or k < 1:
                msg = f"Level should be a positive int, but receive {k}."
                logging.error(msg)
                raise ValueError(msg)
        return {k: self._aggregate_single(self.data, k) for k in levels}

    def __str__(self):
        return "GetAggregateTS"
