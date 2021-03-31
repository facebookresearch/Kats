#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from multiprocessing import Pool
from typing import Callable, Dict, Union

import infrastrategy.kats.models.model as m
import pandas as pd
from infrastrategy.kats.consts import Params, TimeSeriesData
from infrastrategy.kats.parameter_tuning.base_parameter_tuning import (
    BaseParameterTuning,
)
from sklearn.metrics import mean_squared_error as mse


class GridSearch(BaseParameterTuning):
    def __init__(self):
        super().__init__()

    @staticmethod
    def search(
        model_class: m.Model,
        model_param_class: Params,
        time_series: Union[TimeSeriesData, pd.DataFrame],
        param_dims: Dict = None,
        return_result_for_all: bool = False,
        n_splits: int = 5,
        metric: Callable = mse,
    ) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            param_score_list = []
            with Pool() as pool:
                param_score_list = pool.starmap(
                    GridSearch.timeseries_cross_validate,
                    [
                        (
                            model_class,
                            model_param_class,
                            params,
                            time_series,
                            n_splits,
                            metric,
                        )
                        for params in GridSearch.dict_product(param_dims)
                    ],
                )
            logging.debug(
                "The list of parameters and their" "scores:" + str(param_score_list)
            )
            if return_result_for_all:
                return (min(param_score_list, key=lambda x: x[1]), param_score_list)
            else:
                return min(param_score_list, key=lambda x: x[1])
