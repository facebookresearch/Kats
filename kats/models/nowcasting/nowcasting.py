# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Nowcasting is the basic model for short-term forecasting.

This modules contains class NowcastingParams, which is the class parameter
and class NowcastingModel, which is the model.

  Typical usage example:

  nr = NowcastingModel(data = data, params = NowcastingParams(step = 10))
  nr.feature_extraction()
  nr.label_extraction()
  nr.fit()
  output = nr.predict()
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, List

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
    """The class for Nowcasting Parameters.

    Takes parameters for class NowcastingModel.

    Attributes:
        step: An integer indicating how many steps ahead we are forecasting. Default is 1.
    """

    def __init__(self, step: int = 1, **kwargs) -> None:
        super().__init__()
        self.step = step
        logging.debug(f"Initialized QuadraticModel with parameters: step:{step}")

    def validate_params(self):
        """Raises: NotImplementedError("Subclasses should implement this!")."""

        logging.warning("Method validate_params() is not implemented.")
        raise NotImplementedError("Subclasses should implement this!")


class NowcastingModel(m.Model):
    """The class for Nowcasting Model.

    This class performs data processing and short term prediction, for time series
    based on machine learning methodology.

    Attributes:
        TimeSeriesData: Time Series Data Source.
        NowcastingParams: parameters for Nowcasting.
    """

    def __init__(
        self,
        data: TimeSeriesData,
        params: NowcastingParams,
        model: Any = None,
        feature_names: List[str] = [],
    ) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.df = data.to_dataframe()
        self.step = params.step
        self.model = model
        self.feature_names = feature_names

    def feature_extraction(self) -> None:
        """Extracts features for time series data.

        Example of output:
        .. list-table:: Title
        :widths: 10 50 25 25 25
        :header-rows: 1

        * - index
            - time
            - y
            - ROC_10
            - ROC_15
        * - 30
            - 2020-02-05 00:00:00
            - 7234.93
            - -0.278597
            - -0.266019
        * - 31
            - 2020-02-06 00:00:00
            - 7272.51
            - -0.275543
            - -0.271799
        """

        feature_names = []

        for n in [10, 15, 20, 25, 30]:
            self.df = ROC(self.df, n)
            feature_names.append("ROC_" + str(n))
        for n in [10, 15, 20, 25, 30]:
            self.df = LAG(self.df, n)
            feature_names.append("LAG_" + str(n))
        self.df = self.df[
            ~self.df.isin([np.nan, np.inf, -np.inf]).any(1)
        ]  # filterout + - inf, nan
        self.feature_names = feature_names

    def label_extraction(self) -> None:
        """Extracts labels from time seires data."""

        self.df["label"] = LAG(self.data.to_dataframe(), -self.step)[
            "LAG_-" + str(self.step)
        ]

    ###################### module 1: for offline training ######################

    def fit(self) -> None:
        """Fits model."""

        logging.debug(
            "Call fit() with parameters: " "step:{step}".format(step=self.step)
        )
        train_index = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(1)].index
        X_train, y_train = (
            self.df[self.feature_names].loc[train_index],
            self.df["label"].loc[train_index],
        )
        # We use gradient boosted tree for general ML cases
        reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)
        self.model = reg

    def save_model(self) -> bytes:
        """Saves sklearn model as bytes."""

        return serialize_for_zippy(self.model)

    ###################### module 2: for online prediction ######################
    def predict(self, model=None, df=None, **kwargs):
        """Predicts the time series in the future.

        Nowcasting forecasts at the time unit of step ahead.
        This is in order to keep precision and different from usual algorithms.
        If model or df are overwritten in the function, it won't use the internal ones.

        Args:
            model: An external sklearn model.
            df: An external dataset.

        Returns:
            A float variable, the forecast at future step.
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

    def load_model(self, model_as_bytes: bytes) -> None:
        """Loads model_as_str and decodes into the class NowcastingModel.

        Args:
            model_as_bytes: a binary variable, indicating whether to read as bytes.
        """

        self.model = deserialize_from_zippy(model_as_bytes)

    def plot(self):
        """Raises: NotImplementedError("Subclasses should implement this!")。"""

        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        """Returns the name as Nowcasting,"""

        return "Nowcasting"
