#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file defines the BackTester classes for Kats

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from infrastrategy.kats.consts import Params, TimeSeriesData


# Constant to indicate error types supported
ALLOWED_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


# Returns absolute size corresponding to percentage of array of length size
def get_percent_size(size, percent):
    val = np.floor(size * percent / 100)
    return int(val)


# Returns approximately even length fold offsets for a given range
def return_fold_offsets(start, end, num_folds):
    offsets = [0]
    splits = np.array_split(range(end - start), num_folds - 1)
    for split in splits:
        prev_offset = offsets[-1]
        offsets.append(split.size + prev_offset)
    return offsets


"""Abstract Class with general backtesting methods

This class defines the parent functions for various backtesting methods.

Required args/attributes:
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model
    multi: boolean. Flag to use multiprocessing
    offset: int. Gap between train/test datasets (default 0)

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    error_funcs: dict. Dictionary (string -> function) mapping error name to
                       function that calculates it
    freq: str. Frequency of pandas dataframe (inferred)
    raw_errors: List of numpy.arrays storing raw errors (truth - predicted)
"""


class BackTesterParent(ABC):
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        model_class,
        multi: bool,
        offset=0,
        **kwargs
    ):
        self.error_methods = error_methods
        self.data = data
        self.model_class = model_class
        self.params = params
        self.multi = multi
        self.offset = offset

        self.results = []
        self.errors = {}
        self.size = len(data.time)
        self.error_funcs = {
            "mape": self._calc_mape,
            "smape": self._calc_smape,
            "mae": self._calc_mae,
            "mase": self._calc_mase,
            "mse": self._calc_mse,
            "rmse": self._calc_rmse,
        }

        if self.size <= 0:
            logging.error("self.size <= 0")
            logging.error("self.size: {0}".format(self.size))
            raise ValueError("Passing an empty time series")

        for error in self.error_methods:
            if error not in ALLOWED_ERRORS:
                logging.error("Invalid error type passed")
                logging.error("error name: {0}".format(error))
                raise ValueError("Unsupported error type")
            self.errors[error] = 0.0

        # Handling frequency
        if "freq" in kwargs:
            self.freq = kwargs["freq"]
        else:
            logging.info("Inferring frequency")
            self.freq = pd.infer_freq(self.data.time)

        self.raw_errors = []

        logging.info("Instantiated BackTester")
        if kwargs:
            logging.info(
                "Additional arguments: {0}".format(
                    (", ".join(["{}={!r}".format(k, v) for k, v in kwargs.items()]))
                )
            )
        logging.info("Model type: {0}".format(self.model_class))
        logging.info("Error metrics: {0}".format(self.error_methods))

        super().__init__()

    # Calculates all errors in error_methods and stores them in errors dict
    def calc_error(self) -> float:
        logging.info("Calculating Errors")
        if len(self.results) <= 0:
            logging.error("Empty forecast")
            raise ValueError("No results from forecast")

        # Storing total length of predictions for weighting fold errors
        total_fold_length = sum(result[1].size for result in self.results)

        for result in self.results:
            if len(result) != 4:
                logging.error("Invalid result: {0}".format(result))
                raise ValueError("Invalid result")
            training_inputs, truth, _, predictions = result

            # Storing raw errors
            self.raw_errors.append(truth - predictions)

            if training_inputs.size <= 0:
                logging.error("No training data provided ")
                raise ValueError("Not enough training data")

            if predictions.size <= 0:
                logging.error("No predictions provided")
                raise ValueError("Not enough predictions")

            if truth.size <= 0:
                logging.error("No ground truth data provided")
                raise ValueError("Not enough ground truth data")

            if predictions.size != truth.size:
                logging.error("Unequal amount of labels and predictions")
                raise ValueError("Incorrect dimensionality of predictions & labels")

            diffs = np.abs(truth - predictions)

            for error_type in self.error_methods:
                # Weighting the errors by the relative fold length if
                # predictions are of different sizes
                self.errors[error_type] = (
                    self.errors[error_type]
                    + (
                        self.error_funcs[error_type](
                            training_inputs, predictions, truth, diffs
                        )
                    )
                    * float(len(truth))
                    / total_fold_length
                )

    def _calc_mape(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MAPE")
        return np.mean(np.abs((truth - predictions) / truth))

    def _calc_smape(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating SMAPE")
        return ((abs(truth - predictions) / (truth + predictions)).sum()) * (
            2.0 / truth.size
        )

    def _calc_mae(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MAE")
        return diffs.mean()

    # MASE
    # mean(|actual - forecast| / naiveError), where
    # naiveError = 1/ (n-1) sigma^n_[i=2](|actual_[i] - actual_[i-1]|)
    def _calc_mase(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MASE")
        naive_error = np.abs(np.diff(training_inputs)).sum() / (
            training_inputs.shape[0] - 1
        )
        return diffs.mean() / naive_error

    def _calc_mse(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating MSE")
        return ((diffs) ** 2).mean()

    def _calc_rmse(
        self,
        training_inputs: np.array,
        predictions: np.array,
        truth: np.array,
        diffs: np.array,
    ) -> float:
        logging.info("Calculating RMSE")
        return np.sqrt(self._calc_mse(training_inputs, predictions, truth, diffs))

    # Trains model, evaluates it, and stores results in results list
    def _create_model(
        self,
        training_data_indices: Tuple[int, int],
        testing_data_indices: Tuple[int, int],
    ):
        training_data_start, training_data_end = training_data_indices
        testing_data_start, testing_data_end = testing_data_indices
        logging.info("Creating TimeSeries train test objects for split")
        logging.info(
            "Train split of {0}, {1}".format(training_data_start, training_data_end)
        )
        logging.info(
            "Test split of {0}, {1}".format(testing_data_start, testing_data_end)
        )

        if (
            training_data_start < 0
            or training_data_start > self.size
            or training_data_end < 0
            or training_data_end > self.size
        ):
            logging.error(
                "Train Split of {0}, {1} was invalid".format(
                    training_data_start, training_data_end
                )
            )
            raise ValueError("Invalid training data indices in split")

        if (
            testing_data_start < 0
            or testing_data_start > self.size
            or testing_data_end < 0
            or testing_data_end > self.size
        ):
            logging.error(
                "Test Split of {0}, {1} was invalid".format(
                    testing_data_start, testing_data_end
                )
            )
            raise ValueError("Invalid testing data indices in split")

        training_data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[training_data_start:training_data_end],
                    "y": self.data.value[training_data_start:training_data_end],
                }
            )
        )

        testing_data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[testing_data_start:testing_data_end],
                    "y": self.data.value[testing_data_start:testing_data_end],
                }
            )
        )
        if training_data.value.size <= 0:
            logging.error("No training data provided ")
            raise ValueError("Not enough training data")

        if testing_data.value.size <= 0:
            logging.error("No testing data provided ")
            raise ValueError("Not enough testing data")

        logging.info("Training model")
        train_model = self.model_class(data=training_data, params=self.params)
        train_model.fit()

        logging.info("Making forecast prediction")
        fcst = train_model.predict(
            steps=testing_data.value.size + self.offset, freq=self.freq
        )
        train_data_only = np.array(training_data.value)
        truth = np.array(testing_data.value)
        predictions = np.array(fcst["fcst"])
        if self.offset:
            predictions = predictions[self.offset :]

        if not self.multi:
            self.results.append((train_data_only, truth, train_model, predictions))
        else:
            return (train_data_only, truth, train_model, predictions)

    def _build_and_train_models(
        self, splits: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
    ):
        training_splits, testing_splits = splits

        num_splits = len(training_splits)
        if not self.multi:
            for train_split, test_split in zip(training_splits, testing_splits):
                self._create_model(train_split, test_split)
        else:
            pool = mp.Pool(processes=num_splits)
            futures = [
                pool.apply_async(self._create_model, args=(train_split, test_split))
                for train_split, test_split in zip(training_splits, testing_splits)
            ]
            self.results = [fut.get() for fut in futures]
            pool.close()

    def run_backtest(self):
        self._build_and_train_models(self._create_train_test_splits())
        self.calc_error()

    def get_error_value(self, error_name):
        if error_name in self.errors:
            return self.errors[error_name]
        else:
            logging.error("Invalid error name: {0}".format(error_name))
            raise ValueError("Invalid error name")

    @abstractmethod
    def _create_train_test_splits(self):
        pass


"""Class to execute basic Train/Test Split backtest

This class defines the functions to execute a simple backtest

Required args/attributes:
    train_percentage: float. Percentage of data used for training
    test_percentage: float. Percentage of data used for testing
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    error_funcs: dict. Dictionary (string -> function) mapping error name to
                       function that calculates it
    freq: str. Frequency of pandas dataframe (inferred)
    raw_errors: List of numpy.arrays storing raw errors (predicted - truth)
"""


class BackTesterSimple(BackTesterParent):
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        model_class,
        **kwargs
    ):
        logging.info("Initializing train/test percentages")
        if train_percentage <= 0:
            logging.error("Non positive training percentage")
            raise ValueError("Invalid training percentage")
        elif train_percentage > 100:
            logging.error("Too large training percentage")
            raise ValueError("Invalid training percentage")
        self.train_percentage = train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage

        logging.info("Calling parent class constructor")
        super().__init__(error_methods, data, params, model_class, False, **kwargs)

    def _create_train_test_splits(self):
        logging.info("Creating train test splits")
        train_size = get_percent_size(self.size, self.train_percentage)
        test_size = get_percent_size(self.size, self.test_percentage)

        if train_size <= 0 or train_size >= self.size:
            logging.error("Invalid training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            raise ValueError("Incorrect training size")

        if test_size <= 0 or test_size >= self.size:
            logging.error("Invalid testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect testing size")

        if train_size + test_size > self.size:
            logging.error("Training and Testing sizes too big")
            logging.error("Training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            logging.error("Testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect training and testing sizes")

        return [(0, train_size)], [(train_size, train_size + test_size)]


###############################################################################


"""Class to execute Expanding Window backtest

This class defines the functions to execute an expanding window backtest
More info here:
https://our.internmc.facebook.com/intern/wiki/Forecaster/User_Guide/Backtests/#expanding-window-backtes

Required args/attributes:
    start_train_percentage: float. Initial training window
    end_train_percentage: float. Final training window
    test_percentage: float. Percentage of data used for testing
    expanding_steps: int. Number of expanding steps to take (# of folds)
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model
    multi: boolean (optional). Flag to use multiprocessing. Default True

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    error_funcs: dict. Dictionary (string -> function) mapping error name to
                       function that calculates it
    freq: str. Frequency of pandas dataframe (inferred)
    raw_errors: List of numpy.arrays storing raw errors (predicted - truth)
"""


class BackTesterExpandingWindow(BackTesterParent):
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        start_train_percentage: float,
        end_train_percentage: float,
        test_percentage: float,
        expanding_steps: int,
        model_class,
        multi=True,
        **kwargs
    ):
        logging.info("Initializing train/test percentages")
        if start_train_percentage <= 0:
            logging.error("Non positive start training percentage")
            raise ValueError("Invalid start training percentage")
        elif start_train_percentage > 100:
            logging.error("Too large start training percentage")
            raise ValueError("Invalid end training percentage")
        self.start_train_percentage = start_train_percentage
        if end_train_percentage <= 0:
            logging.error("Non positive end training percentage")
            raise ValueError("Invalid start training percentage")
        elif end_train_percentage > 100:
            logging.error("Too large end training percentage")
            raise ValueError("Invalid end training percentage")
        elif end_train_percentage < self.start_train_percentage:
            logging.error("Ending Training % < Start Training %")
            raise ValueError("Start Training percentage must be less than End")
        self.end_train_percentage = end_train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage
        if expanding_steps < 0:
            logging.error("Non positive expanding steps")
            raise ValueError("Invalid expanding steps")
        self.expanding_steps = expanding_steps

        logging.info("Calling parent class constructor")
        super().__init__(error_methods, data, params, model_class, multi, **kwargs)

    def _create_train_test_splits(self):
        logging.info("Creating train test splits")
        start_train_size = get_percent_size(self.size, self.start_train_percentage)
        end_train_size = get_percent_size(self.size, self.end_train_percentage)
        test_size = get_percent_size(self.size, self.test_percentage)

        if start_train_size <= 0 or start_train_size >= self.size:
            logging.error(
                "Invalid starting training size: {0}".format(start_train_size)
            )
            logging.error(
                "Start Training Percentage: {0}".format(self.start_train_percentage)
            )
            raise ValueError("Incorrect starting training size")

        if end_train_size <= 0 or end_train_size >= self.size:
            logging.error("Invalid ending training size: {0}".format(end_train_size))
            logging.error(
                "End Training Percentage: {0}".format(self.end_train_percentage)
            )
            logging.error(
                "End Training Percentage: {0}".format(self.end_train_percentage)
            )
            raise ValueError("Incorrect starting training size")

        if test_size <= 0 or test_size >= self.size:
            logging.error("Invalid testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect testing size")

        if end_train_size + test_size > self.size:
            logging.error("Training and Testing sizes too big")
            logging.error("End Training size: {0}".format(end_train_size))
            logging.error(
                "End Training Percentage: {0}".format(self.end_train_percentage)
            )
            logging.error("Testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect training and testing sizes")

        # Handling edge case where only 1 fold is needed (same as BackTesterSimple)
        if self.expanding_steps == 1:
            return (
                [(0, start_train_size)],
                [(start_train_size, start_train_size + test_size)],
            )

        train_splits = []
        test_splits = []
        offsets = return_fold_offsets(
            start_train_size, end_train_size, self.expanding_steps
        )
        for offset in offsets:
            train_splits.append((0, int(start_train_size + offset)))
            test_splits.append(
                (
                    int(start_train_size + offset),
                    int(start_train_size + offset + test_size),
                )
            )
        return train_splits, test_splits


###############################################################################


"""Class to execute Rolling Window backtest

This class defines the functions to execute a rolling window backtest
More info here:
https://our.internmc.facebook.com/intern/wiki/Forecaster/User_Guide/Backtests/#expanding-window-backtes

Required args/attributes:
    train_percentage: float. Percentage of data used for training
    test_percentage: float. Percentage of data used for testing
    sliding_steps: float. Number of rolling steps to take (# of folds)
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model
    multi: boolean (optional). Flag to use multiprocessing. Default True

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    error_funcs: dict. Dictionary (string -> function) mapping error name to
                       function that calculates it
    freq: str. Frequency of pandas dataframe (inferred)
    raw_errors: List of numpy.arrays storing raw errors (predicted - truth)
"""


class BackTesterRollingWindow(BackTesterParent):
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        sliding_steps: int,
        model_class,
        multi=True,
        **kwargs
    ):
        logging.info("Initializing train/test percentages")
        if train_percentage <= 0:
            logging.error("Non positive training percentage")
            raise ValueError("Invalid training percentage")
        elif train_percentage > 100:
            logging.error("Too large training percentage")
            raise ValueError("Invalid training percentage")
        self.train_percentage = train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage
        if sliding_steps < 0:
            logging.error("Non positive sliding steps")
            raise ValueError("Invalid sliding steps")
        self.sliding_steps = sliding_steps

        logging.info("Calling parent class constructor")
        super().__init__(error_methods, data, params, model_class, multi, **kwargs)

    def _create_train_test_splits(self):
        logging.info("Creating train test splits")
        train_size = get_percent_size(self.size, self.train_percentage)
        test_size = get_percent_size(self.size, self.test_percentage)

        if train_size <= 0 or train_size >= self.size:
            logging.error("Invalid training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            raise ValueError("Incorrect training size")

        if test_size <= 0 or test_size >= self.size:
            logging.error("Invalid testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect testing size")

        if train_size + test_size > self.size:
            logging.error("Training and Testing sizes too big")
            logging.error("Training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            logging.error("Testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect training and testing sizes")

        # Handling edge case where only 1 fold is needed (same as BackTesterSimple)
        if self.sliding_steps == 1:
            return [(0, train_size)], [(train_size, train_size + test_size)]

        train_splits = []
        test_splits = []
        offsets = return_fold_offsets(
            0, self.size - train_size - test_size, self.sliding_steps
        )
        for offset in offsets:
            train_splits.append((offset, int(offset + train_size)))
            test_splits.append(
                (int(offset + train_size), int(offset + train_size + test_size))
            )
        return train_splits, test_splits


###############################################################################


"""Class to execute Fixed Window Ahead backtest

This class defines the functions to execute a fixed window ahead backtest.
Purpose of this backtest is to focus on the long range forecasting ability
of the model. Fixed window is defined as the gap between the training and
testing sets.

Required args/attributes:
    train_percentage: float. Percentage of data used for training
    test_percentage: float. Percentage of data used for testing
    window_percentage: int. Percentage of data used for fixed window
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    error_funcs: dict. Dictionary (string -> function) mapping error name to
                       function that calculates it
    freq: str. Frequency of pandas dataframe (inferred)
    raw_errors: List of numpy.arrays storing raw errors (predicted - truth)
"""


class BackTesterFixedWindow(BackTesterParent):
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        window_percentage: int,
        model_class,
        **kwargs
    ):
        logging.info("Initializing train/test percentages")
        if train_percentage <= 0:
            logging.error("Non positive training percentage")
            raise ValueError("Invalid training percentage")
        elif train_percentage > 100:
            logging.error("Too large training percentage")
            raise ValueError("Invalid training percentage")
        self.train_percentage = train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage
        if window_percentage < 0:
            logging.error("Non positive window percentage")
            raise ValueError("Invalid window percentage")
        elif window_percentage > 100:
            logging.error("Too large window percentage")
            raise ValueError("Invalid window percentage")
        self.window_percentage = window_percentage

        offset = get_percent_size(len(data.time), self.window_percentage)

        logging.info("Calling parent class constructor")
        super().__init__(
            error_methods, data, params, model_class, False, offset, **kwargs
        )

    def _create_train_test_splits(self):
        logging.info("Creating train test splits")
        train_size = get_percent_size(self.size, self.train_percentage)
        test_size = get_percent_size(self.size, self.test_percentage)
        window_size = get_percent_size(self.size, self.window_percentage)

        if train_size <= 0 or train_size >= self.size:
            logging.error("Invalid training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            raise ValueError("Incorrect training size")

        if test_size <= 0 or test_size >= self.size:
            logging.error("Invalid testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect testing size")

        if train_size + test_size + window_size > self.size:
            logging.error("Combo of Training, Testing, & Window sizes too big")
            logging.error("Training size: {0}".format(train_size))
            logging.error("Training Percentage: {0}".format(self.train_percentage))
            logging.error("Testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            logging.error("Window size: {0}".format(window_size))
            logging.error("Window Percentage: {0}".format(self.window_percentage))
            raise ValueError("Incorrect training, testing, & window sizes")

        train_splits = [(0, int(train_size))]
        test_splits = [
            (int(train_size + window_size), int(train_size + window_size + test_size))
        ]
        return train_splits, test_splits


###############################################################################


"""Class to initiate Time Series Cross Validation

Required args/attributes:
    train_percentage: float. Initial training window
    test_percentage: float. Percentage of data used for testing
    num_folds: int. Number of folds
    error_methods: list. List of strings indicating which errors to calculate
                         (see ALLOWED_ERRORS for exhaustive list)
    data: TimeSeriesData. KATS custom TimeSeriesData object
    params: Params. Parameters to train model
    model_class: Untyped. Defines type of model
    rolling_window (optional). Flag to use rolling window method. Default False
    multi: boolean (optional). Flag to use multiprocessing. Default True

Public Attributes:
    results: list. List of tuples (training_data, testing_data, trained_model,
                    forecast_predictions) storing forecast results
    errors: dict. Dictionary (string -> float) mapping the error type to value
    size: int. Number of datapoints
    raw_errors: List of numpy.arrays storing raw errors (predicted - truth)
    backtester: Backtesting Object that does training & evaluation
"""


class CrossValidation:
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        num_folds: int,
        model_class,
        rolling_window=False,
        multi=True,
    ):
        logging.info("Initializing and validating parameter values")
        if train_percentage <= 0:
            logging.error("Non positive training percentage")
            raise ValueError("Invalid training percentage")
        elif train_percentage > 100:
            logging.error("Too large training percentage")
            raise ValueError("Invalid training percentage")
        self.train_percentage = train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage
        if num_folds < 0:
            logging.error("Non positive number of folds")
            raise ValueError("Invalid number of folds")
        self.num_folds = num_folds

        self.size = len(data.time)
        if self.size <= 0:
            logging.error("self.size <= 0")
            logging.error("self.size: {0}".format(self.size))
            raise ValueError("Passing an empty time series")

        self.results = []
        self.errors = {}
        self.raw_errors = []

        if not rolling_window:
            self.backtester = BackTesterExpandingWindow(
                error_methods,
                data,
                params,
                self.train_percentage,
                100 - self.test_percentage,
                self.test_percentage,
                self.num_folds,
                model_class,
                multi=multi,
            )
        else:
            self.backtester = BackTesterRollingWindow(
                error_methods,
                data,
                params,
                self.train_percentage,
                self.test_percentage,
                self.num_folds,
                model_class,
                multi=multi,
            )

    # Run cross validation
    def run_cv(self):
        logging.info("Running training and evaluation")
        self.backtester.run_backtest()
        self.results = self.backtester.results
        self.errors = self.backtester.errors
        self.raw_errors = self.backtester.raw_errors
        logging.info("Finished")

    def get_error_value(self, error_name):
        if error_name in self.errors:
            return self.errors[error_name]
        else:
            logging.error("Invalid error name: {0}".format(error_name))
            raise ValueError("Invalid error name")
