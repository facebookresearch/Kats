#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
#
# This file defines tests for the Backtester classes

import statistics
import unittest
from datetime import timedelta
from io import BytesIO

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    quadratic_model,
    sarima,
)
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.models.ensemble.ensemble import BaseModelParams, EnsembleParams
from kats.models.ensemble.median_ensemble import MedianEnsembleModel
from kats.models.ensemble.weighted_avg_ensemble import WeightedAvgEnsemble
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
from kats.models.linear_model import LinearModel, LinearModelParams
from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.quadratic_model import (
    QuadraticModel,
    QuadraticModelParams,
)
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.models.stlf import STLFModel, STLFParams
from kats.models.theta import ThetaModel, ThetaParams
from kats.utils.backtesters import (
    BackTesterExpandingWindow,
    BackTesterFixedWindow,
    BackTesterRollingWindow,
    BackTesterSimple,
    CrossValidation,
    _return_fold_offsets as return_fold_offsets,
)
from manifold.clients.python import ManifoldClient

ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]
TIMESTEPS = 36  # Timesteps for test data
FREQUENCY = "MS"  # Frequency for model
PERCENTAGE = 75  # Percentage of train data
EXPANDING_WINDOW_START = 50
EXPANDING_WINDOW_STEPS = 3
ROLLING_WINDOW_TRAIN = 50
ROLLING_WINDOW_STEPS = 3
FIXED_WINDOW_TRAIN_PERCENTAGE = 50
FIXED_WINDOW_PERCENTAGE = 25
SLOPE = 5  # Slope for linear model
INTERCEPT = 3  # Intercept for linear model
SKIPPED_MODELS_LINEAR = {  # Skipping some models for linear test
    "arima",
    "sarima",
    "holtwinters",
    "prophet",
    "median_ensemble",
    "weighted_ensemble",
    "stlf",
    "theta",
}
FLOAT_ROUNDING_PARAM = 3  # Number of decimal places to round low floats to 0
CV_NUM_FOLDS = 3
RUN_FAST = True
MODELS_TO_SKIP = {"prophet", "stlf"}
DATA_FILE = "kats/kats/data/air_passengers.csv"
BUCKET = "kats_dev_ds_no_pii"
FILE_PATH = "flat/air"
ALL_MODELS = {  # Mapping model_name (string) -> model_class, model_params
    "arima": (ARIMAModel, ARIMAParams(p=1, d=0, q=0)),
    "sarima": (
        SARIMAModel,
        SARIMAParams(
            p=1,
            d=0,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
        ),
    ),
    "linear": (LinearModel, LinearModelParams()),
    "quadatric": (QuadraticModel, QuadraticModelParams()),
    "holtwinters": (HoltWintersModel, HoltWintersParams()),
    "prophet": (ProphetModel, ProphetParams()),
    "stlf": (STLFModel, STLFParams(m=12, method="theta")),
    "theta": (ThetaModel, ThetaParams()),
    "median_ensemble": (
        MedianEnsembleModel,
        EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=0, q=0)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    sarima.SARIMAParams(
                        p=1,
                        d=0,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        ),
    ),
    "weighted_ensemble": (
        WeightedAvgEnsemble,
        EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=0, q=0)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    sarima.SARIMAParams(
                        p=1,
                        d=0,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        ),
    ),
}

# Read Data File
try:
    DATA = pd.read_csv("kats/kats/data/air_passengers.csv")
except FileNotFoundError:
    with ManifoldClient.get_client(BUCKET) as client:
        stream = BytesIO()
        client.sync_get(FILE_PATH, stream, timeout=timedelta(minutes=5))
        stream.seek(0)
        DATA = pd.read_csv(stream, encoding="utf-8")


def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2))


def mse(pred, truth):
    return ((np.abs(truth - pred)) ** 2).mean()


def mape(pred, truth):
    return np.mean(np.abs((truth - pred) / truth))


def smape(pred, truth):
    return ((abs(truth - pred) / (truth + pred)).sum()) * (2.0 / truth.size)


def mae(pred, truth):
    return (np.abs(truth - pred)).mean()


def mase(training_inputs, pred, truth):
    naive_error = np.abs(np.diff(training_inputs)).sum() / (
        training_inputs.shape[0] - 1
    )
    return ((np.abs(truth - pred)).mean()) / naive_error


error_funcs = {
    "mape": mape,
    "smape": smape,
    "mae": mae,
    "mase": mase,
    "mse": mse,
    "rmse": rmse,
}


class SimpleBackTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)
        train_data = DATA.iloc[: len(DATA) - TIMESTEPS]
        test_data = DATA.tail(TIMESTEPS)
        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.backtester_results = {}

        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the backtester for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Training the model
            model_class, model_params = model_attributes
            temp_model = model_class(
                data=TimeSeriesData(train_data), params=model_params
            )
            temp_model.fit()
            # Getting model predictions
            temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)
            # Creating and running backtester for this given model class
            temp_backtester = BackTesterSimple(
                ALL_ERRORS,
                TSData,
                model_params,
                PERCENTAGE,
                (100 - PERCENTAGE),
                model_class,
            )
            temp_backtester.run_backtest()

            # Using model predictions from temp_model to calculate true errors
            true_errors = {}
            pred = np.array(temp_fcst["fcst"])
            truth = np.array(test_data["y"])
            train = np.array(train_data["y"])
            for error, func in error_funcs.items():
                if error == "mase":
                    true_errors[error] = func(train, pred, truth)
                else:
                    true_errors[error] = func(pred, truth)
            cls.backtester_results[model_name] = (temp_backtester, true_errors)

        # Linear Test
        # Maps model_name (string) -> backtester_object
        cls.backtester_linear_results = {}
        # Creating artificial linear data
        fake_data = DATA.copy()
        for i, _ in fake_data.iterrows():
            new_val = SLOPE * i + INTERCEPT
            fake_data.at[i, "y"] = new_val

        # Iterating through all model types and creating backtester
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some models that model linear data well
            if model_name in SKIPPED_MODELS_LINEAR:
                continue
            model_class, model_params = model_attributes
            temp_backtester = BackTesterSimple(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                PERCENTAGE,
                (100 - PERCENTAGE),
                model_class,
            )
            temp_backtester.run_backtest()
            cls.backtester_linear_results[model_name] = temp_backtester

    # Tests backtester error values against "true" error values
    def test_error_values(self):
        for _, test_results in self.backtester_results.items():
            backtester, error_results = test_results
            for error_name, value in error_results.items():
                self.assertEqual(
                    round(value, FLOAT_ROUNDING_PARAM),
                    round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
                )

    # Ensures backtester errors are all 0 for models trained on artificially
    # linear data
    def test_linear_values(self):
        for _, backtester_linear in self.backtester_linear_results.items():
            for _, value in backtester_linear.errors.items():
                self.assertEqual(round(value, FLOAT_ROUNDING_PARAM), 0.0)


###############################################################################


class ExpandingWindowBackTesterTest(SimpleBackTesterTest):
    @classmethod
    def setUpClass(cls):
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)

        train_folds, test_folds = cls.create_folds(
            cls,
            DATA,
            EXPANDING_WINDOW_START / 100.0 * len(DATA),
            PERCENTAGE / 100.0 * len(DATA),
            EXPANDING_WINDOW_STEPS,
            (100 - PERCENTAGE) / 100.0 * len(DATA),
        )

        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.backtester_results = {}

        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the backtester for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping prophet model if need to run tests faster
            if RUN_FAST:
                if model_name in MODELS_TO_SKIP:
                    continue
            # Training the model
            model_class, model_params = model_attributes
            # Dict to store errors
            true_errors = {}
            for i in range(0, len(train_folds)):
                train_fold = train_folds[i]
                test_fold = test_folds[i]
                temp_model = model_class(
                    data=TimeSeriesData(train_fold), params=model_params
                )
                temp_model.fit()
                # Getting model predictions
                temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

                # Using model predictions from temp_model to calculate true errors
                pred = np.array(temp_fcst["fcst"])
                truth = np.array(test_fold["y"])
                train = np.array(train_fold["y"])
                for error, func in error_funcs.items():
                    if error == "mase":
                        err = func(train, pred, truth)
                    else:
                        err = func(pred, truth)
                    if error in true_errors:
                        true_errors[error].append(err)
                    else:
                        true_errors[error] = [err]
            for error_name, values in true_errors.items():
                true_errors[error_name] = statistics.mean(values)

            # Creating and running backtester for this given model class
            temp_backtester = BackTesterExpandingWindow(
                ALL_ERRORS,
                TSData,
                model_params,
                EXPANDING_WINDOW_START,
                PERCENTAGE,
                (100 - PERCENTAGE),
                EXPANDING_WINDOW_STEPS,
                model_class,
                multi=False,
            )
            temp_backtester.run_backtest()
            cls.backtester_results[model_name] = (temp_backtester, true_errors)

        # Linear Test
        # Maps model_name (string) -> backtester_object
        cls.backtester_linear_results = {}
        # Creating artificial linear data
        fake_data = DATA.copy()
        for i, _ in fake_data.iterrows():
            new_val = SLOPE * i + INTERCEPT
            fake_data.at[i, "y"] = new_val

        # Iterating through all model types and creating backtester
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some models that model linear data well
            if model_name in SKIPPED_MODELS_LINEAR:
                continue
            model_class, model_params = model_attributes
            temp_backtester = BackTesterExpandingWindow(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                EXPANDING_WINDOW_START,
                PERCENTAGE,
                (100 - PERCENTAGE),
                EXPANDING_WINDOW_STEPS,
                model_class,
                multi=False,
            )
            temp_backtester.run_backtest()
            cls.backtester_linear_results[model_name] = temp_backtester

    def create_folds(
        self, data, start_train_size, end_train_size, num_folds, test_size
    ):
        train_folds = []
        test_folds = []
        offsets = return_fold_offsets(
            int(start_train_size), int(end_train_size), num_folds
        )

        for offset in offsets:
            train_folds.append(data.iloc[: int(start_train_size + offset)])
            test_folds.append(
                data.iloc[
                    int(start_train_size + offset) : int(
                        start_train_size + offset + test_size
                    )
                ]
            )
        return train_folds, test_folds


###############################################################################


class RollingWindowBackTesterTest(SimpleBackTesterTest):
    @classmethod
    def setUpClass(cls):
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)

        train_folds, test_folds = cls.create_folds(
            cls,
            DATA,
            ROLLING_WINDOW_TRAIN / 100.0 * len(DATA),
            (100 - PERCENTAGE) / 100.0 * len(DATA),
            ROLLING_WINDOW_STEPS,
        )

        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.backtester_results = {}

        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the backtester for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping prophet model if need to run tests faster
            if RUN_FAST:
                if model_name in MODELS_TO_SKIP:
                    continue
            # Training the model
            model_class, model_params = model_attributes
            # Dict to store errors
            true_errors = {}
            for i in range(0, len(train_folds)):
                train_fold = train_folds[i]
                test_fold = test_folds[i]
                temp_model = model_class(
                    data=TimeSeriesData(train_fold), params=model_params
                )
                temp_model.fit()
                # Getting model predictions
                temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

                # Using model predictions from temp_model to calculate true errors
                pred = np.array(temp_fcst["fcst"])
                truth = np.array(test_fold["y"])
                train = np.array(train_fold["y"])
                for error, func in error_funcs.items():
                    if error == "mase":
                        err = func(train, pred, truth)
                    else:
                        err = func(pred, truth)
                    if error in true_errors:
                        true_errors[error].append(err)
                    else:
                        true_errors[error] = [err]
            for error_name, values in true_errors.items():
                true_errors[error_name] = statistics.mean(values)

            # Creating and running backtester for this given model class
            temp_backtester = BackTesterRollingWindow(
                ALL_ERRORS,
                TSData,
                model_params,
                ROLLING_WINDOW_TRAIN,
                (100 - PERCENTAGE),
                ROLLING_WINDOW_STEPS,
                model_class,
                multi=False,
            )
            temp_backtester.run_backtest()
            cls.backtester_results[model_name] = (temp_backtester, true_errors)

        # Linear Test
        # Maps model_name (string) -> backtester_object
        cls.backtester_linear_results = {}
        # Creating artificial linear data
        fake_data = DATA.copy()
        for i, _ in fake_data.iterrows():
            new_val = SLOPE * i + INTERCEPT
            fake_data.at[i, "y"] = new_val

        # Iterating through all model types and creating backtester
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some models that model linear data well
            if model_name in SKIPPED_MODELS_LINEAR:
                continue
            model_class, model_params = model_attributes
            temp_backtester = BackTesterRollingWindow(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                ROLLING_WINDOW_TRAIN,
                (100 - PERCENTAGE),
                ROLLING_WINDOW_STEPS,
                model_class,
                multi=False,
            )
            temp_backtester.run_backtest()
            cls.backtester_linear_results[model_name] = temp_backtester

    def create_folds(self, data, train_size, test_size, num_folds):
        offsets = return_fold_offsets(
            0, int(len(data) - train_size - test_size), num_folds
        )
        train_folds = []
        test_folds = []

        for offset in offsets:
            train_folds.append(data.iloc[offset : int(offset + train_size)])
            test_folds.append(
                data.iloc[
                    int(offset + train_size) : int(offset + train_size + test_size)
                ]
            )
        return train_folds, test_folds


###############################################################################


class FixedWindowBackTesterTest(SimpleBackTesterTest):
    @classmethod
    def setUpClass(cls):
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)
        offset = int(FIXED_WINDOW_PERCENTAGE / 100.0 * len(DATA))
        train_folds, test_folds = cls.create_folds(
            cls,
            DATA,
            FIXED_WINDOW_TRAIN_PERCENTAGE / 100.0 * len(DATA),
            (100 - PERCENTAGE) / 100.0 * len(DATA),
            offset,
        )

        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.backtester_results = {}

        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the backtester for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Training the model
            model_class, model_params = model_attributes
            # Dict to store errors
            true_errors = {}
            for i in range(0, len(train_folds)):
                train_fold = train_folds[i]
                test_fold = test_folds[i]
                temp_model = model_class(
                    data=TimeSeriesData(train_fold), params=model_params
                )
                temp_model.fit()
                # Getting model predictions
                temp_fcst = temp_model.predict(steps=TIMESTEPS + offset, freq=FREQUENCY)

                # Using model predictions from temp_model to calculate true errors
                pred = np.array(temp_fcst["fcst"])[offset:]
                truth = np.array(test_fold["y"])
                train = np.array(train_fold["y"])
                for error, func in error_funcs.items():
                    if error == "mase":
                        err = func(train, pred, truth)
                    else:
                        err = func(pred, truth)
                    if error in true_errors:
                        true_errors[error].append(err)
                    else:
                        true_errors[error] = [err]
            for error_name, values in true_errors.items():
                true_errors[error_name] = statistics.mean(values)

            # Creating and running backtester for this given model class
            temp_backtester = BackTesterFixedWindow(
                ALL_ERRORS,
                TSData,
                model_params,
                FIXED_WINDOW_TRAIN_PERCENTAGE,
                (100 - PERCENTAGE),
                FIXED_WINDOW_PERCENTAGE,
                model_class,
            )
            temp_backtester.run_backtest()
            cls.backtester_results[model_name] = (temp_backtester, true_errors)

        # Linear Test
        # Maps model_name (string) -> backtester_object
        cls.backtester_linear_results = {}
        # Creating artificial linear data
        fake_data = DATA.copy()
        for i, _ in fake_data.iterrows():
            new_val = SLOPE * i + INTERCEPT
            fake_data.at[i, "y"] = new_val

        # Iterating through all model types and creating backtester
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some models that model linear data well
            if model_name in SKIPPED_MODELS_LINEAR:
                continue
            model_class, model_params = model_attributes
            temp_backtester = BackTesterFixedWindow(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                FIXED_WINDOW_TRAIN_PERCENTAGE,
                (100 - PERCENTAGE),
                FIXED_WINDOW_PERCENTAGE,
                model_class,
            )
            temp_backtester.run_backtest()
            cls.backtester_linear_results[model_name] = temp_backtester

    def create_folds(self, data, train_size, test_size, window_size):
        train_folds = [data.iloc[0 : int(train_size)]]
        test_folds = [
            data.iloc[
                int(train_size + window_size) : int(
                    train_size + window_size + test_size
                )
            ]
        ]
        return train_folds, test_folds


###############################################################################


class CrossValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # noqa C901
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)

        expanding_train_folds, expanding_test_folds = cls.create_folds(
            cls,
            DATA,
            EXPANDING_WINDOW_START / 100 * len(DATA),
            CV_NUM_FOLDS,
            (100 - PERCENTAGE) / 100.0 * len(DATA),
            True,
        )

        rolling_train_folds, rolling_test_folds = cls.create_folds(
            cls,
            DATA,
            EXPANDING_WINDOW_START / 100 * len(DATA),
            CV_NUM_FOLDS,
            (100 - PERCENTAGE) / 100.0 * len(DATA),
            False,
        )

        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.expanding_cv_results = {}
        cls.rolling_cv_results = {}

        # For Expanding Window CV:
        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the CV for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some larger models for the test
            if "ensemble" in model_name:
                continue
            # Training the model
            model_class, model_params = model_attributes
            # Dict to store errors
            true_errors = {}
            for i in range(0, len(expanding_train_folds)):
                train_fold = expanding_train_folds[i]
                test_fold = expanding_test_folds[i]
                temp_model = model_class(
                    data=TimeSeriesData(train_fold), params=model_params
                )
                temp_model.fit()
                # Getting model predictions
                temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

                # Using model predictions from temp_model to calculate true errors
                pred = np.array(temp_fcst["fcst"])
                truth = np.array(test_fold["y"])
                train = np.array(train_fold["y"])
                for error, func in error_funcs.items():
                    if error == "mase":
                        err = func(train, pred, truth)
                    else:
                        err = func(pred, truth)
                    if error in true_errors:
                        true_errors[error].append(err)
                    else:
                        true_errors[error] = [err]
            for error_name, values in true_errors.items():
                true_errors[error_name] = statistics.mean(values)

            # Creating and running CV Object for this given model class
            temp_cv = CrossValidation(
                ALL_ERRORS,
                TSData,
                model_params,
                EXPANDING_WINDOW_START,
                100 - PERCENTAGE,
                CV_NUM_FOLDS,
                model_class,
                multi=False,
            )
            temp_cv.run_cv()
            cls.expanding_cv_results[model_name] = (temp_cv, true_errors)

        # For Rolling Window CV:
        # Iterating through each model type and doing following:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the CV for that given model class
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some larger models for the test
            if "ensemble" in model_name:
                continue
            # Training the models
            model_class, model_params = model_attributes
            # Dict to store errors
            true_errors = {}
            for i in range(0, len(rolling_train_folds)):
                train_fold = rolling_train_folds[i]
                test_fold = rolling_test_folds[i]
                temp_model = model_class(
                    data=TimeSeriesData(train_fold), params=model_params
                )
                temp_model.fit()
                # Getting model predictions
                temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

                # Using model predictions from temp_model to calculate true errors
                pred = np.array(temp_fcst["fcst"])
                truth = np.array(test_fold["y"])
                train = np.array(train_fold["y"])
                for error, func in error_funcs.items():
                    if error == "mase":
                        err = func(train, pred, truth)
                    else:
                        err = func(pred, truth)
                    if error in true_errors:
                        true_errors[error].append(err)
                    else:
                        true_errors[error] = [err]
            for error_name, values in true_errors.items():
                true_errors[error_name] = statistics.mean(values)

            # Creating and running CV Object for this given model class
            # Using EXPANDING_WINDOW_START
            temp_cv = CrossValidation(
                ALL_ERRORS,
                TSData,
                model_params,
                ROLLING_WINDOW_TRAIN,
                100 - PERCENTAGE,
                CV_NUM_FOLDS,
                model_class,
                rolling_window=True,
                multi=False,
            )
            temp_cv.run_cv()
            cls.rolling_cv_results[model_name] = (temp_cv, true_errors)

        # Linear Test
        # Maps model_name (string) -> backtester_object
        cls.expanding_cv_linear_results = {}
        cls.rolling_cv_linear_results = {}
        # Creating artificial linear data
        fake_data = DATA.copy()
        for i, _ in fake_data.iterrows():
            new_val = SLOPE * i + INTERCEPT
            fake_data.at[i, "y"] = new_val

        # Iterating through all model types and creating CV objects
        for model_name, model_attributes in ALL_MODELS.items():
            # Skipping some models that model linear data well
            if not model_name == "linear":
                continue
            model_class, model_params = model_attributes
            temp_expanding_cv = CrossValidation(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                EXPANDING_WINDOW_START,
                100 - PERCENTAGE,
                CV_NUM_FOLDS,
                model_class,
                multi=False,
            )
            temp_rolling_cv = CrossValidation(
                ALL_ERRORS,
                TimeSeriesData(fake_data),
                model_params,
                ROLLING_WINDOW_TRAIN,
                100 - PERCENTAGE,
                CV_NUM_FOLDS,
                model_class,
                rolling_window=True,
                multi=False,
            )
            temp_expanding_cv.run_cv()
            temp_rolling_cv.run_cv()
            cls.expanding_cv_linear_results[model_name] = temp_expanding_cv
            cls.rolling_cv_linear_results[model_name] = temp_rolling_cv

    def create_folds(self, data, train_size, num_folds, test_size, expanding):
        train_folds = []
        test_folds = []

        if expanding:
            end_train_size = len(data) - test_size
            window_size = (end_train_size - train_size) / (num_folds - 1)
            for i in range(0, num_folds):
                end_train_range = int((i * window_size) + train_size)
                train_folds.append(data.iloc[:end_train_range])
                test_folds.append(
                    data.iloc[end_train_range : end_train_range + int(test_size)]
                )
            return train_folds, test_folds

        fold_size = (len(data) - train_size - test_size) / (num_folds - 1)

        for i in range(0, num_folds):
            offset = int(i * fold_size)
            train_folds.append(data.iloc[offset : int(offset + train_size)])
            test_folds.append(
                data.iloc[
                    int(offset + train_size) : int(offset + train_size + test_size)
                ]
            )
        return train_folds, test_folds

    # Tests backtester error values against "true" error values
    def test_error_values_expanding(self):
        for _, test_results in self.expanding_cv_results.items():
            cv, error_results = test_results
            for error_name, value in error_results.items():
                self.assertEqual(
                    round(value, FLOAT_ROUNDING_PARAM),
                    round(cv.errors[error_name], FLOAT_ROUNDING_PARAM),
                )

    # Ensures backtester errors are all 0 for models trained on artificially
    # linear data
    def test_linear_values_expanding(self):
        for _, cv_linear in self.expanding_cv_linear_results.items():
            for _, value in cv_linear.errors.items():
                self.assertEqual(round(value, FLOAT_ROUNDING_PARAM), 0.0)

    # Tests backtester error values against "true" error values
    def test_error_values_rolling(self):
        for _, test_results in self.rolling_cv_results.items():
            cv, error_results = test_results
            for error_name, value in error_results.items():
                self.assertEqual(
                    round(value, FLOAT_ROUNDING_PARAM),
                    round(cv.errors[error_name], FLOAT_ROUNDING_PARAM),
                )

    # Ensures backtester errors are all 0 for models trained on artificially
    # linear data
    def test_linear_values_rolling(self):
        for _, cv_linear in self.rolling_cv_linear_results.items():
            for _, value in cv_linear.errors.items():
                self.assertEqual(round(value, FLOAT_ROUNDING_PARAM), 0.0)


###############################################################################


class OneStepForecastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Error accuracy Test
        # Setting up data
        DATA.columns = ["time", "y"]
        TSData = TimeSeriesData(DATA)
        train_data = DATA.iloc[: len(DATA) - 1]
        test_data = DATA.tail(1)
        # Maps model_name (string) -> Tuple(backtester_object, true_errors dict)
        # true_errors dict Maps error_name (string) -> error_value (float)
        cls.backtester_results = {}

        # Doing following for Holt Winters model:
        #   1. Training the model
        #   2. Getting model predictions and calculating the true errors
        #      using the local class functions
        #   3. Creating the backtester for that given model class
        model_name = "holtwinters"
        model_attributes = ALL_MODELS[model_name]
        # Training the model
        model_class, model_params = model_attributes
        temp_model = model_class(data=TimeSeriesData(train_data), params=model_params)
        temp_model.fit()
        # Getting model predictions
        temp_fcst = temp_model.predict(steps=1, freq=FREQUENCY)
        # Creating and running backtester for this given model class
        one_step_pct = (1 / len(DATA)) * 100
        temp_backtester = BackTesterSimple(
            ALL_ERRORS,
            TSData,
            model_params,
            (100 - one_step_pct),
            one_step_pct,
            model_class,
            freq=FREQUENCY,
        )
        temp_backtester.run_backtest()
        # Using model predictions from temp_model to calculate true errors
        true_errors = {}
        pred = np.array(temp_fcst["fcst"])
        truth = np.array(test_data["y"])
        train = np.array(train_data["y"])
        for error, func in error_funcs.items():
            if error == "mase":
                true_errors[error] = func(train, pred, truth)
            else:
                true_errors[error] = func(pred, truth)
        cls.backtester_results[model_name] = (temp_backtester, true_errors)

    # Tests backtester error values against "true" error values
    def test_error_values(self):
        for _, test_results in self.backtester_results.items():
            backtester, error_results = test_results
            for error_name, value in error_results.items():
                self.assertEqual(
                    round(value, FLOAT_ROUNDING_PARAM),
                    round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
                )
