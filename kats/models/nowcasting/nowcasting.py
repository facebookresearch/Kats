#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import kats.models.model as m
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.nowcasting.feature_extraction import LAG, ROC
from kats.models.nowcasting.model_io import (
    serialize_for_zippy,
    deserialize_from_zippy,
)
from sklearn.ensemble import GradientBoostingRegressor


class NowcastingParams(Params):
    # will add more parameters to this class
    def __init__(self, step=1, **kwargs) -> None:
        super().__init__()
        self.step = step
        logging.debug(f"Initialized QuadraticModel with parameters: step:{step}")

    def validate_params(self):
        logging.warning("Method validate_params() is not implemented.")
        pass


class NowcastingModel(m.Model):
    def __init__(self, data: TimeSeriesData, params: NowcastingParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.df = data.to_dataframe()
        self.step = params.step

    def feature_extraction(self) -> None:
        """
        Example of output
        +----+---------------------+---------+------------+-----------+-----------+-----------+-----------+----------+
        |    | time                |       y |     ROC_10 |    ROC_15 |    ROC_20 |    ROC_25 |    ROC_30 |   LAG_10 |
        |----+---------------------+---------+------------+-----------+-----------+-----------+-----------+----------|
        | 30 | 2020-02-05 00:00:00 | 7234.93 | -0.278597  | -0.266019 | -0.265405 | -0.26516  | -0.261438 |  10018   |
        | 31 | 2020-02-06 00:00:00 | 7272.51 | -0.275543  | -0.271799 | -0.261365 | -0.261355 | -0.263012 |  10029   |
        | 32 | 2020-02-07 00:00:00 | 7280.06 | -0.0407474 | -0.27387  | -0.26061  | -0.261068 | -0.260461 |  10038.6 |
        +----+---------------------+---------+------------+-----------+-----------+-----------+-----------+----------+

        """
        feature_names = []
        for n in [10, 15, 20, 25, 30]:
            self.df = ROC(self.df, n)
            feature_names.append("ROC_" + str(n))
        for n in [10, 15, 20, 25, 30]:
            self.df = LAG(self.df, n)
            feature_names.append("LAG_" + str(n))
        # filterout + - inf, nan
        self.df = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(1)]
        self.feature_names = feature_names

    def label_extraction(self) -> None:
        # negative shift creates NaN as label in last few rows
        self.df["label"] = LAG(self.data.to_dataframe(), -self.step)[
            "LAG_-" + str(self.step)
        ]

    ###################### module 1: for offline training ######################
    # fit() used to fit model
    def fit(self) -> None:
        logging.debug(
            "Call fit() with parameters: " "step:{step}".format(step=self.params.step)
        )

        # gradient boosted tree
        train_index = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(1)].index
        X_train, y_train = (
            self.df[self.feature_names].loc[train_index],
            self.df["label"].loc[train_index],
        )
        reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)
        self.model = reg

    # save_model: save sklearn model as str
    def save_model(self) -> bytes:
        return serialize_for_zippy(self.model)

    ###################### module 2: for online prediction ######################
    def predict(self, model=None, df=None, **kwargs):
        """
        Different from other algorithms
        Nowcasting forecasts 1 time unit only
        in order to keep precision
        Multiple step forecast is doable
        If model or df are overwritten in the call it
        won't use the internal ones.
        """
        logging.debug(
            "Call predict() with parameters. "
            "Forecast 1 step only, kwargs:{kwargs}".format(kwargs=kwargs)
        )
        if model:
            self.model = model
        if df is not None:
            if "y" in self.df.columns:
                prediction = self.model.predict(df[self.feature_names])
            else:
                raise Exception("External df as input has wrong format")
        else:
            prediction = self.model.predict(self.df[-self.step :][self.feature_names])
        return prediction

    # load model: load model_as_str and decode into a model for use
    def load_model(self, model_as_bytes: bytes) -> None:
        self.model = deserialize_from_zippy(model_as_bytes)

    def plot(self):
        pass

    def __str__(self):
        return "Nowcasting"
