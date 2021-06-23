# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the BackTester classes for Kats.

Kats supports multiple types of backtesters, including:
  - :class:`BackTesterSimple` (basic train & test backtesting).
  - :class:`BackTesterFixedWindow` (discontinuous train & test data).
  - :class:`BackTesterExpandingWindow` (increasing train window size over
    multiple iterations).
  - :class:`BackTesterRollingWindow` (sliding train & test windows over
    multiple iterations).

This module also supports :class:`CrossValidation` with both expanding and
rolling windows.

For more information, check out the Kats tutorial notebook on backtesting!
"""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData


# Constant to indicate error types supported
ALLOWED_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


class BackTesterParent(ABC):
    """
    This class defines the parent functions for various backtesting methods.

    Attributes:
        error_methods: List of strings indicating which errors to calculate
          (see `ALLOWED_ERRORS` for exhaustive list).
        data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
        params: Parameters to train model with.
        model_class: Defines the model type to use for backtesting.
        multi: Boolean flag to use multiprocessing (if set to True).
        offset: Gap between train/test datasets (default 0).
        results: List of tuples `(training_data, testing_data, trained_model,
          forecast_predictions)` storing forecast results.
        errors: Dictionary mapping the error type to value.
        size: An integer for the total number of datapoints.
        error_funcs: Dictionary mapping error name to the
          function that calculates it.
        freq: A string representing the (inferred) frequency of the
          `pandas.DataFrame`.

    Raises:
      ValueError: The time series is empty or an invalid error type was passed.
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        model_class: Type,
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

    def calc_error(self) -> Optional[float]:
        """
        Calculates all errors in `self.error_methods` and stores them in the
        errors dict.

        Returns:
          The error value. None if the error value does not exist.
        """

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
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates MAPE error."""

        logging.info("Calculating MAPE")
        return np.mean(np.abs((truth - predictions) / truth))

    def _calc_smape(
        self,
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates SMAPE error."""

        logging.info("Calculating SMAPE")
        return ((abs(truth - predictions) / (truth + predictions)).sum()) * (
            2.0 / truth.size
        )

    def _calc_mae(
        self,
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates MAE error."""

        logging.info("Calculating MAE")
        return diffs.mean()

    def _calc_mase(
        self,
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates MASE error.

        mean(|actual - forecast| / naiveError), where
        naiveError = 1/ (n-1) sigma^n_[i=2](|actual_[i] - actual_[i-1]|)
        """

        logging.info("Calculating MASE")
        naive_error = np.abs(np.diff(training_inputs)).sum() / (
            training_inputs.shape[0] - 1
        )
        return diffs.mean() / naive_error

    def _calc_mse(
        self,
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates MSE error."""

        logging.info("Calculating MSE")
        return ((diffs) ** 2).mean()

    def _calc_rmse(
        self,
        training_inputs: np.ndarray,
        predictions: np.ndarray,
        truth: np.ndarray,
        diffs: np.ndarray,
    ) -> float:
        """Calculates RMSE error."""

        logging.info("Calculating RMSE")
        return np.sqrt(self._calc_mse(training_inputs, predictions, truth, diffs))

    def _create_model(
        self,
        training_data_indices: Tuple[int, int],
        testing_data_indices: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Type, np.ndarray]]:
        """
        Trains model, evaluates it, and stores results in results list.
        """

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
    ) -> None:
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

    def run_backtest(self) -> None:
        """Executes backtest."""

        self._build_and_train_models(self._create_train_test_splits())
        self.calc_error()

    def get_error_value(self, error_name: str) -> float:
        """Gets requested error value.

        Args:
          error_name: A string of the error whose value should be returned.

        Returns:
          A float of the eror value.

        Raises:
          ValueError: The error name is invalid.
        """

        if error_name in self.errors:
            return self.errors[error_name]
        else:
            logging.error("Invalid error name: {0}".format(error_name))
            raise ValueError("Invalid error name")

    @abstractmethod
    def _create_train_test_splits(self):
        pass


class BackTesterSimple(BackTesterParent):
    """Defines the functions to execute a simple train/test backtest.

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      error_methods: List of strings indicating which errors to calculate
        (see `ALLOWED_ERRORS` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
      error_funcs: Dictionary mapping error name to the
        function that calculates it.
      freq: A string representing the (inferred) frequency of the
        `pandas.DataFrame`.
      raw_errors: List storing raw errors (truth - predicted).


    Raises:
      ValueError: Invalid train and/or test params passed. Or the time series
        is empty.

    Sample Usage:
      >>> df = pd.read_csv("kats/data/air_passengers.csv")
      >>> ts = TimeSeriesData(df=df)
      >>> params = ARIMAParams(p=1, d=1, q=1)
      >>> all_errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]
      >>> backtester = BackTesterSimple(
            error_methods=all_errors,
            data=ts,
            params=params,
            train_percentage=75,
            test_percentage=25,
            model_class=ARIMAModel,
          )
      >>> backtester.run_backtest()
      >>> mape = backtester.get_error_value("mape") # Retrieve MAPE error
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        model_class: Type,
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

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        train_size = _get_percent_size(self.size, self.train_percentage)
        test_size = _get_percent_size(self.size, self.test_percentage)

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


class BackTesterExpandingWindow(BackTesterParent):
    """Defines functions to execute an expanding window backtest.

    An expanding window backtest conducts a backtest over multiple iterations,
    wherein each iteration, the size of the training dataset increases by a
    fixed amount, while the test dataset "slides" forward to accommodate.
    Iterations continue until the complete data set is used to either train
    or test in the final interation.

    For more information, check out the Kats tutorial notebooks!

    Attributes:
      start_train_percentage: A float for the initial percentage of data used
        for training.
      end_train_percentage: A float for the final percentage of data used for
        training.
      test_percentage: A float for the percentage of data used for testing.
      expanding_steps: An integer for the number of expanding steps (i.e.
        number of folds).
      error_methods: List of strings indicating which errors to calculate
        (see `ALLOWED_ERRORS` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      multi: A boolean flag to toggle multiprocessing (default True).
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
      error_funcs: Dictionary mapping error name to
        function that calculates it.
      freq: A string representing the (inferred) frequency of the
        `pandas.DataFrame`.
      raw_errors: List storing raw errors (truth - predicted).

    Raises:
      ValueError: One or more of the train, test, or expanding steps params
        were invalid. Or the time series is empty.

    Sample Usage:
      >>> df = pd.read_csv("kats/data/air_passengers.csv")
      >>> ts = TimeSeriesData(df=df)
      >>> params = ARIMAParams(p=1, d=1, q=1)
      >>> all_errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]
      >>> backtester = BackTesterExpandingWindow(
            error_methods=all_errors,
            data=ts,
            params=paramsparams,
            start_train_percentage=50,
            end_train_percentage=75,
            test_percentage=25,
            expanding_steps=3,
            model_class=ARIMAModel,
          )
      >>> backtester.run_backtest()
      >>> mape = backtester.get_error_value("mape") # Retrieve MAPE error
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        start_train_percentage: float,
        end_train_percentage: float,
        test_percentage: float,
        expanding_steps: int,
        model_class: Type,
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

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        start_train_size = _get_percent_size(self.size, self.start_train_percentage)
        end_train_size = _get_percent_size(self.size, self.end_train_percentage)
        test_size = _get_percent_size(self.size, self.test_percentage)

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
        offsets = _return_fold_offsets(
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


class BackTesterRollingWindow(BackTesterParent):
    """Defines functions to execute a rolling window backtest.

    An rolling window backtest conducts a backtest over multiple iterations,
    wherein each iteration, the start location of the training dataset moves
    forward by a fixed amount, while the test dataset "slides" forward to
    accommodate. Iterations continue until the end of the test set meets the
    end of the full data set.

    For more information, check out the Kats tutorial notebooks!

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      sliding_steps: An integer for the number of rolling steps (i.e.
        number of folds).
      error_methods: List of strings indicating which errors to calculate
        (see `ALLOWED_ERRORS` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      multi: A boolean flag to toggle multiprocessing (default True).
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
      error_funcs: Dictionary mapping error name to the
        function that calculates it.
      freq: A string representing the (inferred) frequency of the
        `pandas.DataFrame`.
      raw_errors: List storing raw errors (truth - predicted).

    Raises:
      ValueError: One or more of the train, test, or sliding steps params
        were invalid. Or the time series is empty.

    Sample Usage:
      >>> df = pd.read_csv("kats/data/air_passengers.csv")
      >>> ts = TimeSeriesData(df=df)
      >>> params = ARIMAParams(p=1, d=1, q=1)
      >>> all_errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]
      >>> backtester = BackTesterExpandingWindow(
            error_methods=all_errors,
            data=ts,
            params=paramsparams,
            train_percentage=50,
            test_percentage=25,
            expanding_steps=3,
            model_class=ARIMAModel,
          )
      >>> backtester.run_backtest()
      >>> mape = backtester.get_error_value("mape") # Retrieve MAPE error
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        sliding_steps: int,
        model_class: Type,
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

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        train_size = _get_percent_size(self.size, self.train_percentage)
        test_size = _get_percent_size(self.size, self.test_percentage)

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
        offsets = _return_fold_offsets(
            0, self.size - train_size - test_size, self.sliding_steps
        )
        for offset in offsets:
            train_splits.append((offset, int(offset + train_size)))
            test_splits.append(
                (int(offset + train_size), int(offset + train_size + test_size))
            )
        return train_splits, test_splits


class BackTesterFixedWindow(BackTesterParent):
    """Defines functions to execute a fixed window ahead backtest.

    A fixed window ahead backtest is similar to a standard (i.e. simple)
    backtest, with the caveat that there is a gap between the train and test
    data sets. The purpose of this type backtest is to focus on the long range
    forecasting ability of the model.

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      window_percentage: A float for the percentage of data used for the
        fixed window.
      error_methods: List of strings indicating which errors to calculate
        (see `ALLOWED_ERRORS` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
      error_funcs: Dictionary mapping error name to the
        function that calculates it.
      freq: A string representing the (inferred) frequency of the
        `pandas.DataFrame`.
      raw_errors: List storing raw errors (truth - predicted).

    Raises:
      ValueError: One or more of the train, test, or fixed window params were
        invalid. Or the time series is empty.

    Sample Usage:
      >>> df = pd.read_csv("kats/data/air_passengers.csv")
      >>> ts = TimeSeriesData(df=df)
      >>> params = ARIMAParams(p=1, d=1, q=1)
      >>> all_errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]
      >>> backtester = BackTesterFixedWindow(
            error_methods=all_errors,
            data=ts,
            params=paramsparams,
            train_percentage=50,
            test_percentage=25,
            window_percentage=25,
            model_class=ARIMAModel,
          )
      >>> backtester.run_backtest()
      >>> mape = backtester.get_error_value("mape") # Retrieve MAPE error
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        window_percentage: int,
        model_class: Type,
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

        offset = _get_percent_size(len(data.time), self.window_percentage)

        logging.info("Calling parent class constructor")
        super().__init__(
            error_methods, data, params, model_class, False, offset, **kwargs
        )

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        train_size = _get_percent_size(self.size, self.train_percentage)
        test_size = _get_percent_size(self.size, self.test_percentage)
        window_size = _get_percent_size(self.size, self.window_percentage)

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


class CrossValidation:
    """Defines class to execute cross validation.

    Cross validation is a useful technique to use multiple folds of the
    training and testing data to help optimize the performance of the
    model (e.g. hyperparameter tuning). For more info on cross validation, see
    https://en.wikipedia.org/wiki/Cross-validation_(statistics)

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      num_folds: An integer for the number of folds to use.
      error_methods: List of strings indicating which errors to calculate
        (see `ALLOWED_ERRORS` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      rolling_window: A boolean flag to use the rolling window method instead
        of the expanding window method (default False).
      multi: A boolean flag to toggle multiprocessing (default True).
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
      raw_errors: List storing raw errors (truth - predicted).

    Raises:
      ValueError: One or more of the train, test, or num_folds params
        were invalid. Or the time series is empty.

    Sample Usage:
      >>> df = pd.read_csv("kats/data/air_passengers.csv")
      >>> ts = TimeSeriesData(df=df)
      >>> params = ARIMAParams(p=1, d=1, q=1)
      >>> all_errors = ["mape", "smape", "mae", "mase", "mse", "rmse"]
      >>> cv = CrossValidation(
            error_methods=all_errors,
            data=ts,
            params=paramsparams,
            train_percentage=50,
            test_percentage=25,
            num_folds=3,
            model_class=ARIMAModel,
            rolling_window=True
          )
      >>> backtester.run_backtest()
      >>> mape = cv.get_error_value("mape") # Retrieve MAPE error
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        num_folds: int,
        model_class: Type,
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
            self._backtester = BackTesterExpandingWindow(
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
            self._backtester = BackTesterRollingWindow(
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
    def run_cv(self) -> None:
        """Runs the cross validation."""

        logging.info("Running training and evaluation")
        self._backtester.run_backtest()
        self.results = self._backtester.results
        self.errors = self._backtester.errors
        self.raw_errors = self._backtester.raw_errors
        logging.info("Finished")

    def get_error_value(self, error_name: str) -> float:
        """Gets requested error value.

        Args:
          error_name: A string of the error whose value should be returned.

        Returns:
          A float of the eror value.

        Raises:
          ValueError: The error name is invalid.
        """

        if error_name in self.errors:
            return self.errors[error_name]
        else:
            logging.error("Invalid error name: {0}".format(error_name))
            raise ValueError("Invalid error name")


def _get_percent_size(size: int, percent: float) -> int:
    """
    Returns absolute size corresponding to percentage of array of length size.
    """

    val = np.floor(size * percent / 100)
    return int(val)


def _return_fold_offsets(start: int, end: int, num_folds: int) -> List[int]:
    """
    Returns approximately even length fold offsets for a given range.
    """

    offsets = [0]
    splits = np.array_split(range(end - start), num_folds - 1)
    for split in splits:
        prev_offset = offsets[-1]
        offsets.append(split.size + prev_offset)
    return offsets
