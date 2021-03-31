#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Ensemble models with weighted average individual models
# Assume we have k base models, after we make forecasts with each individual
# model, we learn the weights for each individual model based on corresponding
# back testing results, i.e., model with better performance should have higher
# weight.
import logging
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import infrastrategy.kats.models.model as mm
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models.ensemble import ensemble
from infrastrategy.kats.models.ensemble.ensemble import BASE_MODELS, EnsembleParams
from infrastrategy.kats.utils.backtesters import BackTesterSimple


class WeightedAvgEnsemble(ensemble.BaseEnsemble):
    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def _backtester_single(
        self,
        params,
        model_class,
        alpha=0.2,
        train_percentage=80,
        test_percentage=20,
        err_method="mape",
    ):
        bt = BackTesterSimple(
            [err_method],
            self.data,
            params,
            train_percentage,
            test_percentage,
            model_class
        )
        bt.run_backtest()
        return bt.get_error_value(err_method)

    def _backtester_all(self, err_method="mape"):
        num_process = min(len(BASE_MODELS.keys()), (cpu_count() - 1) // 2)
        pool = Pool(processes=(num_process), maxtasksperchild=1000)
        backtesters = {}
        for model in self.params.models:
            backtesters[model.model_name] = pool.apply_async(
                self._backtester_single,
                args=(model.model_params, BASE_MODELS[model.model_name.lower()]),
                kwds={"err_method": err_method},
            )
        pool.close()
        pool.join()
        self.errors = {model: res.get() for model, res in backtesters.items()}
        original_weights = {
            model: 1 / (err + sys.float_info.epsilon)
            for model, err in self.errors.items()
        }
        self.weights = {
            model: err / sum(original_weights.values())
            for model, err in original_weights.items()
        }
        return self.weights

    def predict(self, steps, **kwargs):
        self.freq = kwargs.get("freq", "D")
        err_method = kwargs.get("err_method", "mape")
        # calculate the weights
        self._backtester_all(err_method=err_method)

        # fit model with all available time series
        pred_dict = self._predict_all(steps, **kwargs)

        fcst_all = pd.concat(
            [x.fcst.reset_index(drop=True) for x in pred_dict.values()], axis=1
        )
        fcst_all.columns = pred_dict.keys()
        self.fcst_weighted = fcst_all.dot(np.array(list(self.weights.values())))

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        dates = dates[dates != last_date]
        self.fcst_dates = dates.to_pydatetime()
        self.dates = dates[dates != last_date]

        self.fcst_df = pd.DataFrame({"time": self.dates, "fcst": self.fcst_weighted})

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        logging.info("Generating chart for forecast result from Ensemble.")
        mm.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "Weighted Average Ensemble"
