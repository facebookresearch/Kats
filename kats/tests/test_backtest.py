# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file defines tests for the Backtester classes

import io
import os
import pkgutil
import statistics
import unittest
import unittest.mock as mock
from typing import List, Tuple

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.tests.test_backtester_dummy_data import (
    PROPHET_EMPTY_DUMMY_DATA,
    PROPHET_0_108_FCST_DUMMY_DATA,
    PROPHET_0_72_FCST_DUMMY_DATA,
    PROPHET_0_90_FCST_DUMMY_DATA,
    PROPHET_18_90_FCST_DUMMY_DATA,
    PROPHET_36_108_FCST_DUMMY_DATA,
    PROPHET_0_72_GAP_36_FCST_DUMMY_DATA,
)
from kats.utils.backtesters import (
    BackTesterExpandingWindow,
    BackTesterFixedWindow,
    BackTesterRollingWindow,
    BackTesterSimple,
    CrossValidation,
    _return_fold_offsets as return_fold_offsets,
)

# Constants
ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]  # Errors to test
TIMESTEPS = 36  # Timesteps for test data
FREQUENCY = "MS"  # Frequency for model
PERCENTAGE = 75  # Percentage of train data
EXPANDING_WINDOW_START = 50  # Expanding window start training percentage
EXPANDING_WINDOW_STEPS = 3  # Expanding window number of steps
ROLLING_WINDOW_TRAIN = 50  # Rolling window start training percentage
ROLLING_WINDOW_STEPS = 3  # Rolling window number of steps
FIXED_WINDOW_TRAIN_PERCENTAGE = 50  # Fixed window ahead training percentage
FIXED_WINDOW_PERCENTAGE = 25  # Fixed window ahead window percentage
FLOAT_ROUNDING_PARAM = 3  # Number of decimal places to round low floats to 0
CV_NUM_FOLDS = 3  # Number of folds for cross validation


"""
Following are ground truth error functions
"""


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


error_funcs = {  # Maps error name to function that calculates error
    "mape": mape,
    "smape": smape,
    "mae": mae,
    "mase": mase,
    "mse": mse,
    "rmse": rmse,
}


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class SimpleBackTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up data
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        cls.TSData = TimeSeriesData(DATA)
        cls.train_data = cls.TSData[: len(cls.TSData) - TIMESTEPS]
        cls.test_data = TimeSeriesData(DATA.tail(TIMESTEPS))

    def prophet_predict_side_effect(self, *args, **kwargs):
        if (
            len(kwargs) > 1
            and kwargs["steps"] == TIMESTEPS
            and kwargs["freq"] == FREQUENCY
        ):
            return PROPHET_0_108_FCST_DUMMY_DATA
        else:
            return PROPHET_EMPTY_DUMMY_DATA

    def test_error_values(self) -> None:
        """
        Testing process consists of the following:
          1. Backtest with mocked model
          2. Ensure backtester used correct train test splits
          3. Train (mocked) model locally and extract errors
          4. Comparing errors to backtester results
        """

        # Mock model results
        model_class = mock.MagicMock()
        model_class().predict.side_effect = self.prophet_predict_side_effect
        model_params = mock.MagicMock()

        # Creating and running backtester for this given model class
        bt = BackTesterSimple(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=model_params,
            train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            model_class=model_class,
        )
        bt.run_backtest()

        # Testing that backtester initialized model with correct train/test split
        model_class.assert_called_with(data=self.train_data, params=model_params)

        # Training local model and getting predictions
        temp_model = model_class(data=self.train_data, params=model_params)
        temp_model.fit()
        temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

        # Using model predictions from local model to calculate true errors
        true_errors = {}
        pred = np.array(temp_fcst["fcst"])
        truth = np.array(self.test_data.value)
        train = np.array(self.train_data.value)
        for error, func in error_funcs.items():
            if error == "mase":
                true_errors[error] = func(train, pred, truth)
            else:
                true_errors[error] = func(pred, truth)
        ground_truth_errors = (bt, true_errors)

        # Comparing local model errors to backtester errors
        backtester, error_results = ground_truth_errors
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
            )


class ExpandingWindowBackTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up data
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        cls.TSData = TimeSeriesData(DATA)

        cls.train_folds, cls.test_folds = cls.create_folds(
            cls,
            data=DATA,
            start_train_size=EXPANDING_WINDOW_START / 100.0 * len(DATA),
            end_train_size=PERCENTAGE / 100.0 * len(DATA),
            test_size=(100 - PERCENTAGE) / 100.0 * len(DATA),
            num_folds=EXPANDING_WINDOW_STEPS,
        )

        # Mock model results
        cls.model_class = mock.MagicMock()
        cls.model_class().predict.side_effect = [
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_0_90_FCST_DUMMY_DATA,
            PROPHET_0_108_FCST_DUMMY_DATA,
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_0_90_FCST_DUMMY_DATA,
            PROPHET_0_108_FCST_DUMMY_DATA,
        ]  # Once for backtester once for local model
        cls.model_params = mock.MagicMock()

    def test_error_values(self) -> None:
        """
        Testing process consists of the following:
          1. Backtest with mocked model
          2. Ensure backtester used correct train test splits
          3. Train (mocked) model locally and extract errors
          4. Comparing errors to backtester results
        """

        # Create and run backtester
        bt = BackTesterExpandingWindow(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            start_train_percentage=EXPANDING_WINDOW_START,
            end_train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            expanding_steps=EXPANDING_WINDOW_STEPS,
            model_class=self.model_class,
            multi=False,
        )
        bt.run_backtest()

        # Test model initialization
        self.assertEqual(self.model_class.call_count, 4)
        self.model_class.assert_has_calls(
            [
                mock.call(
                    data=TimeSeriesData(self.train_folds[0]), params=self.model_params
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    data=TimeSeriesData(self.train_folds[1]), params=self.model_params
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    data=TimeSeriesData(self.train_folds[2]), params=self.model_params
                ),
            ]
        )

        # Test model predict
        # self.assertEqual(self.model_class().predict.call_count, 3)
        self.model_class().assert_has_calls(
            [
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
            ]
        )

        # Training local model and getting predictions
        true_errors = {}  # Dict to store errors
        for i in range(0, len(self.train_folds)):
            train_fold = self.train_folds[i]
            test_fold = self.test_folds[i]
            temp_model = self.model_class(
                data=TimeSeriesData(train_fold), params=self.model_params
            )
            temp_model.fit()

            # Getting model predictions
            temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

            # Using model predictions from local_model to calculate true errors
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

        # Calculating errors
        for error_name, values in true_errors.items():
            true_errors[error_name] = statistics.mean(values)
        ground_truth_errors = (bt, true_errors)

        # # Test that local model errors equal backtester errors
        backtester, error_results = ground_truth_errors
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
            )

    def test_one_step_forecast(self) -> None:
        """
        Tests that if expanding steps is one, the folds returned are the same
        as BacktesterSimple.
        """

        # Calculate folds from Expanding Window Backtester
        expanding_backtester = BackTesterExpandingWindow(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            start_train_percentage=PERCENTAGE,
            end_train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            expanding_steps=1,
            model_class=self.model_class,
            multi=False,
        )
        one_step_folds_expanding = expanding_backtester._create_train_test_splits()

        # Calculate folds from Simple Backtester
        simple_backtester = BackTesterSimple(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            model_class=self.model_class,
        )
        folds_simple = simple_backtester._create_train_test_splits()

        # Test that folds are equivalent
        self.assertEqual(one_step_folds_expanding, folds_simple)

    def create_folds(
        self,
        data: pd.DataFrame,
        start_train_size: float,
        end_train_size: float,
        test_size: float,
        num_folds: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Ground truth fold creation
        """

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


class RollingWindowBackTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up data
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        cls.TSData = TimeSeriesData(DATA)

        cls.train_folds, cls.test_folds = cls.create_folds(
            cls,
            data=DATA,
            train_size=ROLLING_WINDOW_TRAIN / 100.0 * len(DATA),
            test_size=(100 - PERCENTAGE) / 100.0 * len(DATA),
            num_folds=ROLLING_WINDOW_STEPS,
        )

        # Mock model results
        cls.model_class = mock.MagicMock()
        cls.model_class().predict.side_effect = [
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_18_90_FCST_DUMMY_DATA,
            PROPHET_36_108_FCST_DUMMY_DATA,
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_18_90_FCST_DUMMY_DATA,
            PROPHET_36_108_FCST_DUMMY_DATA,
        ]  # Once for backtester once for local model
        cls.model_params = mock.MagicMock()

    def test_error_values(self) -> None:
        """
        Testing process consists of the following:
          1. Backtest with mocked model
          2. Ensure backtester used correct train test splits
          3. Train (mocked) model locally and extract errors
          4. Comparing errors to backtester results
        """

        # Create and run backtester
        bt = BackTesterRollingWindow(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=ROLLING_WINDOW_TRAIN,
            test_percentage=(100 - PERCENTAGE),
            sliding_steps=ROLLING_WINDOW_STEPS,
            model_class=self.model_class,
            multi=False,
        )
        bt.run_backtest()

        # Test model initialization
        self.assertEqual(self.model_class.call_count, 4)
        self.model_class.assert_has_calls(
            [
                mock.call(
                    data=TimeSeriesData(self.train_folds[0]), params=self.model_params
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    data=TimeSeriesData(self.train_folds[1]), params=self.model_params
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    data=TimeSeriesData(self.train_folds[2]), params=self.model_params
                ),
            ]
        )

        # Test model predict
        # self.assertEqual(self.model_class().predict.call_count, 3)
        self.model_class().assert_has_calls(
            [
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
            ]
        )

        # Training local model and getting predictions
        true_errors = {}  # Dict to store errors
        for i in range(0, len(self.train_folds)):
            train_fold = self.train_folds[i]
            test_fold = self.test_folds[i]
            temp_model = self.model_class(
                data=TimeSeriesData(train_fold), params=self.model_params
            )
            temp_model.fit()

            # Getting model predictions
            temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

            # Using model predictions from local_model to calculate true errors
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

        # Calculating errors
        for error_name, values in true_errors.items():
            true_errors[error_name] = statistics.mean(values)
        ground_truth_errors = (bt, true_errors)

        # # Test that local model errors equal backtester errors
        backtester, error_results = ground_truth_errors
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
            )

    def test_one_step_forecast(self) -> None:
        """
        Tests that if sliding steps is one, the folds returned are the same
        as BacktesterSimple.
        """

        # Calculate folds from Rolling Window Backtester
        rolling_backtester = BackTesterRollingWindow(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            sliding_steps=1,
            model_class=self.model_class,
            multi=False,
        )
        one_step_folds_rolling = rolling_backtester._create_train_test_splits()

        # Calculate folds from Simple Backtester
        simple_backtester = BackTesterSimple(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            model_class=self.model_class,
        )
        folds_simple = simple_backtester._create_train_test_splits()

        # Test that folds are equivalent
        self.assertEqual(one_step_folds_rolling, folds_simple)

    def create_folds(
        self,
        data: pd.DataFrame,
        train_size: float,
        test_size: float,
        num_folds: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Ground truth fold creation
        """

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


class FixedWindowBackTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up data
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        cls.TSData = TimeSeriesData(DATA)

        # Creating folds
        cls.train_data = cls.TSData[: len(cls.TSData) - (TIMESTEPS * 2)]
        cls.test_data = TimeSeriesData(DATA.tail(TIMESTEPS))

    def prophet_predict_side_effect(self, *args, **kwargs):
        if (
            len(kwargs) > 1
            and kwargs["steps"] == TIMESTEPS * 2
            and kwargs["freq"] == FREQUENCY
        ):
            return PROPHET_0_72_GAP_36_FCST_DUMMY_DATA
        else:
            return PROPHET_EMPTY_DUMMY_DATA

    def test_error_values(self) -> None:
        """
        Testing process consists of the following:
          1. Backtest with mocked model
          2. Ensure backtester used correct train test splits
          3. Train (mocked) model locally and extract errors
          4. Comparing errors to backtester results
        """

        # Mock model results
        model_class = mock.MagicMock()
        model_class().predict.side_effect = self.prophet_predict_side_effect
        model_params = mock.MagicMock()

        # Creating and running backtester for this given model class
        bt = BackTesterFixedWindow(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=model_params,
            train_percentage=FIXED_WINDOW_TRAIN_PERCENTAGE,
            test_percentage=(100 - PERCENTAGE),
            window_percentage=FIXED_WINDOW_PERCENTAGE,
            model_class=model_class,
        )
        bt.run_backtest()

        # Testing that backtester initialized model with correct train/test split
        model_class.assert_called_with(data=self.train_data, params=model_params)

        # Training local model and getting predictions
        temp_model = model_class(data=self.train_data, params=model_params)
        temp_model.fit()
        temp_fcst = temp_model.predict(steps=TIMESTEPS * 2, freq=FREQUENCY)[TIMESTEPS:]

        # Using model predictions from local model to calculate true errors
        true_errors = {}
        pred = np.array(temp_fcst["fcst"])
        truth = np.array(self.test_data.value)
        train = np.array(self.train_data.value)
        for error, func in error_funcs.items():
            if error == "mase":
                true_errors[error] = func(train, pred, truth)
            else:
                true_errors[error] = func(pred, truth)
        ground_truth_errors = (bt, true_errors)

        # Test that local model errors equal backtester errors
        backtester, error_results = ground_truth_errors
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(backtester.errors[error_name], FLOAT_ROUNDING_PARAM),
            )


class CrossValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setting up data
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        cls.TSData = TimeSeriesData(DATA)

        # Mock model and params
        cls.model_class = mock.MagicMock()
        cls.model_params = mock.MagicMock()

    def test_error_values_expanding(self) -> None:
        """
        Tests Expanding Window Cross Validation

        Testing process consists of the following:
          1. Creates folds from the data.
          2. Trains local model on folds.
          3. Calculates local model predictions and errors.
          4. Runs CV class on the data.
          5. Tests CV initialization and model prediction
          6. Compares CV errors to local model errors.
        """

        # Create folds
        DATA = self.TSData.to_dataframe()
        expanding_train_folds, expanding_test_folds = self.create_folds(
            data=DATA,
            train_size=EXPANDING_WINDOW_START / 100 * len(DATA),
            num_folds=CV_NUM_FOLDS,
            test_size=(100 - PERCENTAGE) / 100.0 * len(DATA),
            expanding=True,
        )

        # Create data structures to store CV results
        expanding_cv_results = {}
        true_errors = {}

        # Mock model results
        self.model_class().predict.side_effect = [
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_0_90_FCST_DUMMY_DATA,
            PROPHET_0_108_FCST_DUMMY_DATA,
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_0_90_FCST_DUMMY_DATA,
            PROPHET_0_108_FCST_DUMMY_DATA,
        ]  # Once for ground truth once for CV

        # Iterate through folds and train model to produce ground truth
        for i in range(0, len(expanding_train_folds)):
            train_fold = expanding_train_folds[i]
            test_fold = expanding_test_folds[i]
            temp_model = self.model_class(
                # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]` for
                #  1st param but got `Tuple[int, int]`.
                data=TimeSeriesData(train_fold),
                params=self.model_params,
            )
            temp_model.fit()
            # Getting model predictions
            temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

            # Using model predictions from temp_model to calculate true errors
            pred = np.array(temp_fcst["fcst"])
            # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param
            #  but got `typing_extensions.Literal['y']`.
            truth = np.array(test_fold["y"])
            # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param
            #  but got `typing_extensions.Literal['y']`.
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

        # Calculate average error across folds
        for error_name, values in true_errors.items():
            true_errors[error_name] = statistics.mean(values)

        # Creating and running CV Object for this given model class
        temp_cv = CrossValidation(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=EXPANDING_WINDOW_START,
            test_percentage=(100 - PERCENTAGE),
            num_folds=CV_NUM_FOLDS,
            model_class=self.model_class,
            multi=False,
        )
        temp_cv.run_cv()
        expanding_cv_results = (temp_cv, true_errors)

        # Test CV initialization
        self.model_class.assert_has_calls(
            [
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(expanding_train_folds[0]),
                    params=self.model_params,
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(expanding_train_folds[1]),
                    params=self.model_params,
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(expanding_train_folds[2]),
                    params=self.model_params,
                ),
            ]
        )

        # Testing CV predict
        # self.assertEqual(self.model_class().predict.call_count, 6)
        self.model_class().assert_has_calls(
            [
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
            ]
        )

        # Test that CV errors equal model prediction errors
        cv, error_results = expanding_cv_results
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(cv.errors[error_name], FLOAT_ROUNDING_PARAM),
            )

    # Tests backtester error values against "true" error values
    def test_error_values_rolling(self) -> None:
        """
        Tests Rolling Window Cross Validation

        Testing process consists of the following:
          1. Creates folds from the data.
          2. Trains local model on folds.
          3. Calculates local model predictions and errors.
          4. Runs CV class on the data.
          5. Tests CV initialization and model prediction
          6. Compares CV errors to local model errors.
        """

        # Create folds
        DATA = self.TSData.to_dataframe()
        rolling_train_folds, rolling_test_folds = self.create_folds(
            data=DATA,
            train_size=EXPANDING_WINDOW_START / 100 * len(DATA),
            num_folds=CV_NUM_FOLDS,
            test_size=(100 - PERCENTAGE) / 100.0 * len(DATA),
            expanding=False,
        )

        # Create data structures to store CV results
        rolling_cv_results = {}
        true_errors = {}

        # Mock model results
        self.model_class().predict.side_effect = [
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_18_90_FCST_DUMMY_DATA,
            PROPHET_36_108_FCST_DUMMY_DATA,
            PROPHET_0_72_FCST_DUMMY_DATA,
            PROPHET_18_90_FCST_DUMMY_DATA,
            PROPHET_36_108_FCST_DUMMY_DATA,
        ]  # Once for ground truth once for CV

        # Iterate through folds and train model to produce ground truth
        for i in range(0, len(rolling_train_folds)):
            train_fold = rolling_train_folds[i]
            test_fold = rolling_test_folds[i]
            temp_model = self.model_class(
                # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]` for
                #  1st param but got `Tuple[int, int]`.
                data=TimeSeriesData(train_fold),
                params=self.model_params,
            )
            temp_model.fit()
            # Getting model predictions
            temp_fcst = temp_model.predict(steps=TIMESTEPS, freq=FREQUENCY)

            # Using model predictions from temp_model to calculate true errors
            pred = np.array(temp_fcst["fcst"])
            # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param
            #  but got `typing_extensions.Literal['y']`.
            truth = np.array(test_fold["y"])
            # pyre-fixme[6]: Expected `typing_extensions.Literal[0]` for 1st param
            #  but got `typing_extensions.Literal['y']`.
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

        # Calculate average error across folds
        for error_name, values in true_errors.items():
            true_errors[error_name] = statistics.mean(values)

        # Creating and running CV Object for this given model class
        temp_cv = CrossValidation(
            error_methods=ALL_ERRORS,
            data=self.TSData,
            params=self.model_params,
            train_percentage=ROLLING_WINDOW_TRAIN,
            test_percentage=(100 - PERCENTAGE),
            num_folds=CV_NUM_FOLDS,
            model_class=self.model_class,
            rolling_window=True,
            multi=False,
        )
        temp_cv.run_cv()
        rolling_cv_results = (temp_cv, true_errors)

        # Test CV initialization
        # self.assertEqual(self.model_class.call_count, 7)
        self.model_class.assert_has_calls(
            [
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(rolling_train_folds[0]),
                    params=self.model_params,
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(rolling_train_folds[1]),
                    params=self.model_params,
                ),
                mock.call().fit(),
                mock.call().predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call(
                    # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]`
                    #  for 1st param but got `Tuple[int, int]`.
                    data=TimeSeriesData(rolling_train_folds[2]),
                    params=self.model_params,
                ),
            ]
        )

        # Testing CV predict
        # self.assertEqual(self.model_class().predict.call_count, 6)
        self.model_class().assert_has_calls(
            [
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
                mock.call.fit(),
                mock.call.predict(steps=TIMESTEPS, freq=FREQUENCY),
            ]
        )

        # Test that CV errors equal model prediction errors
        cv, error_results = rolling_cv_results
        for error_name, value in error_results.items():
            self.assertEqual(
                round(value, FLOAT_ROUNDING_PARAM),
                round(cv.errors[error_name], FLOAT_ROUNDING_PARAM),
            )

    def create_folds(
        self,
        data: pd.DataFrame,
        train_size: float,
        num_folds: int,
        test_size: float,
        expanding: bool,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        train_folds = []
        test_folds = []
        """
        Ground truth fold creation
        """

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
