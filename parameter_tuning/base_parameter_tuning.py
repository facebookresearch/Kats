#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import infrastrategy.kats.models.model as m
import numpy as np
import pandas as pd
import psutil
from infrastrategy.kats.consts import Params, TimeSeriesData
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import TimeSeriesSplit


class BaseParameterTuning:
    def __init__(self):
        self.logger = logging.getLogger()

    @staticmethod
    def timeseries_cross_validate(
        model_class: m.Model,
        model_param_class: Params,
        params: Dict,
        time_series: Union[TimeSeriesData, pd.DataFrame],
        n_splits=5,
        metric: Callable = mse,
    ) -> Tuple[Dict, float, List[float]]:
        logging.info(
            "Worker {} for params {}".format(psutil.Process().cpu_num(), params)
        )
        if isinstance(time_series, pd.DataFrame):
            if len(time_series.columns) == 2:
                if np.all(time_series.columns == ["time", "y"]):
                    pass
                elif np.all(time_series.columns == ["ds", "y"]):
                    time_series = time_series.rename(columns={"ds": "time"})
            elif len(time_series.columns) == 1:
                time_series = time_series.reset_index().rename(
                    columns={"index": "time", "0": "y"}
                )
        elif isinstance(time_series, TimeSeriesData):
            time_series = pd.DataFrame(
                {"time": pd.to_datetime(time_series.time),
                "y": time_series.value}
            )
        tscv = TimeSeriesSplit(n_splits=n_splits)
        measure_list = []
        model_params = model_param_class(**params)
        for train_indices, test_indices in tscv.split(time_series):
            ts_data = TimeSeriesData(time_series.iloc[train_indices])
            model = model_class(ts_data, model_params)
            try:
                model.fit()
                predicted = model.predict(
                    steps=test_indices.shape[0],
                    freq=pd.infer_freq(time_series.time)
                )
                if time_series.time[test_indices[0]] != predicted.iloc[0].time:
                    raise ValueError(
                        "Test sets initial timestamp and prediction's initial"
                        " time step do not match."
                    )
                if time_series.time[test_indices[-1]] != predicted.iloc[-1].time:
                    raise ValueError(
                        "Test sets last timestamp and prediction's"
                        " last time step do not match."
                    )
                measure_list.append(
                    metric(time_series.iloc[test_indices]["y"], predicted.fcst)
                )
            except Exception as ex:
                logging.exception(ex)
                measure_list.append(np.inf)
        return params, np.mean(measure_list), measure_list

    @staticmethod
    def dict_product(d: Dict[str, list]) -> Dict[str, Any]:
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))
