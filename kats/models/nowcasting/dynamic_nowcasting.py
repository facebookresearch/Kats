# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic Nowcasting is a basic model for short-term forecasting.

This modules contains class NowcastingParams, which is the class parameter
and class NowcastingModel, which is the model.

  Typical usage example:

  nr = DyanamicNowcastingModel(data = data, params = NowcastingParams(step = 10))
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
from kats.models.nowcasting.feature_extraction import LAG, ROC, MA, MOM
from kats.models.nowcasting.model_io import (
    serialize_for_zippy,
    deserialize_from_zippy,
)
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer


def poly(df, n):
    """
    Takes the column x from the dataframe df and takes
    the value from x to the power n
    """
    poly = pd.Series(df.x ** n, name="poly_" + str(n))
    df = df.join(poly)
    return df


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
        poly_model: Any = None,
        feature_names: List[str] = [],
        poly_feature_names: List[str] = [],
        scaler: Any = None,
        label_scaler: Any = None,
        y_train_season_obj: Any = None,
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
        self.poly_model = poly_model
        self.df_poly = data.to_dataframe()
        self.poly_feature_names = poly_feature_names
        self.df_nowcasting = data.to_dataframe()
        self.scaler = scaler
        self.label_scaler = label_scaler
        self.y_train_season_obj = y_train_season_obj

    def feature_extraction(self) -> None:
        """
        Extarct features for training
        """
        # Add the hour, minute, and x column to the data
        self.df_poly["hour"] = self.df_poly["time"].apply(lambda y: y.hour)
        self.df_poly["minute"] = self.df_poly["time"].apply(lambda y: y.minute)
        self.df_poly["x"] = self.df_poly["hour"] * 60 + self.df_poly["minute"]

        # Empty list to hold the feature names
        poly_feature_names = []

        # Add the poly columns to the df_poly
        for degree in [0, 1, 2, 3, 4, 5]:
            self.df_poly = poly(self.df_poly, degree)
            poly_feature_names.append("poly_" + str(degree))

        # filterout + - inf, nan
        self.df_poly = self.df_poly[
            ~self.df_poly.isin([np.nan, np.inf, -np.inf]).any(1)
        ]

        # Save the poly feature name
        self.poly_feature_names = poly_feature_names
        feature_names = []

        #########################################################################################
        train_index_poly = self.df_poly[
            ~self.df_poly.isin([np.nan, np.inf, -np.inf]).any(1)
        ].index
        X_train_poly, y_train_poly = (
            self.df_poly[self.poly_feature_names].loc[train_index_poly],
            self.df_poly["y"].loc[train_index_poly],
        )

        # Build the Polynomial Regression Model
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_poly, y_train_poly)
        self.poly_model = lin_reg
        y_train_season = lin_reg.predict(X_train_poly)
        self.y_train_season_obj = y_train_season
        # print(self.df_poly)
        # print(X_train_poly)
        # print(self.y_train_season_obj)
        #########################################################################################

        # If we have a 0, switch to basic nowcasting!
        if np.sum(self.df["y"].isin([0.0])) > 0:
            for n in [10, 15, 20, 25, 30]:
                self.df = MOM(self.df, n)
                feature_names.append("MOM_" + str(n))
            for n in [10, 15, 20, 25, 30]:
                self.df = LAG(self.df, n)
                feature_names.append("LAG_" + str(n))
            self.df = self.df[
                ~self.df.isin([np.nan, np.inf, -np.inf]).any(1)
            ]  # filterout + - inf, nan
            self.feature_names = feature_names

        else:
            for n in [10, 15, 20, 25, 30]:
                self.df = ROC(self.df, n)
                feature_names.append("ROC_" + str(n))
            for n in [10, 15, 20, 25, 30]:
                self.df = MOM(self.df, n)
                feature_names.append("MOM_" + str(n))
            for n in [1, 2, 3, 4, 5, 20]:
                self.df = LAG(self.df, n)
                feature_names.append("LAG_" + str(n))
            for n in [10, 20, 30, 40]:
                self.df = MA(self.df, n)
                feature_names.append("MA_" + str(n))
            self.df = self.df[
                ~self.df.isin([np.nan, np.inf, -np.inf]).any(1)
            ]  # filterout + - inf, nan
            self.feature_names = feature_names

    def label_extraction(self) -> None:
        """Extracts labels from time seires data."""

        # self.df["label"] = LAG(self.data.to_dataframe(), -self.step)[
        #    "LAG_-" + str(self.step)
        # ]
        self.df["label"] = self.df["y"]

    ###################### module 1: for offline training ######################

    def fit(self) -> None:
        """Fits model."""

        logging.debug(
            "Call fit() with parameters: " "step:{step}".format(step=self.step)
        )

        if np.sum(self.df["y"].isin([0.0])) == 0:
            n = 1
            train_index = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(1)].index

            X_train = self.df[self.feature_names].loc[train_index]
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = min_max_scaler.fit_transform(X_train)
            self.scaler = min_max_scaler

            n = 1
            y_train = (
                self.df["label"].loc[train_index] - self.y_train_season_obj[train_index]
            ).diff(-n)[:-n]
            pt = PowerTransformer()
            y_train = pt.fit_transform(np.array(y_train).reshape(-1, 1))
            self.label_scaler = pt

            X_train = X_train[:-n]

            # reg = GradientBoostingRegressor()
            reg = RandomForestRegressor()
            # reg =linear_model.Lasso(alpha=10)
            reg.fit(X_train, y_train)
            self.model = reg

        else:
            n = 1
            train_index = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(1)].index

            X_train = self.df[self.feature_names].loc[train_index]
            # min_max_scaler = preprocessing.MinMaxScaler()
            # X_train = min_max_scaler.fit_transform(X_train)
            # self.scaler = min_max_scaler

            n = 1
            y_train = (
                self.df["label"].loc[train_index] - self.y_train_season_obj[train_index]
            ).diff(-n)[:-n]
            X_train = X_train[:-n]

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            self.model = reg

    def save_model(self) -> bytes:
        """Saves sklearn model as bytes."""

        return serialize_for_zippy(self.model)

    ###################### module 2: for online prediction ######################
    def predict(self, **kwargs):
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
        if np.sum(self.df["y"].isin([0.0])) == 0:
            X_test = self.df[-self.step :][self.feature_names]
            X_test = self.scaler.transform(X_test)
            y_predict = self.model.predict(X_test)
            y = self.label_scaler.inverse_transform(y_predict.reshape(-1, 1))
            poly_now = self.y_train_season_obj[-1]
            first_occ = np.where(self.y_train_season_obj == poly_now)
            polynext = self.y_train_season_obj[first_occ[0][0] + self.step]
            now = self.df["y"][-self.step :]
            return (now - poly_now) - y.ravel() + polynext
        else:
            X_test = self.df[-self.step :][self.feature_names]
            y_predict = self.model.predict(X_test)
            poly_now = self.y_train_season_obj[-1]
            first_occ = np.where(self.y_train_season_obj == poly_now)
            polynext = self.y_train_season_obj[first_occ[0][0] + self.step]
            now = self.df["y"][-self.step :]
            # print(y_predict)
            return (now - poly_now) - y_predict + polynext

    def predict_polyfit(self, model=None, df=None, **kwargs):
        poly_now = self.y_train_season_obj[-1]
        first_occ = np.where(self.y_train_season_obj == poly_now)
        polynext = self.y_train_season_obj[first_occ[0][0] + self.step]
        return polynext

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
