# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # pragma: no cover


import numpy as np
import pandas as pd
from kats.consts import _log_error, Params, TimeSeriesData
from kats.metrics.metrics import core_metric, CoreMetric

from kats.utils.datapartition import DataPartitionBase

if TYPE_CHECKING:
    from kats.models.model import Model


DataPartition = Union[
    List[TimeSeriesData],
    Dict[Union[int, str], TimeSeriesData],
    TimeSeriesData,
]


@dataclass
class BacktesterResult:
    # all raw evaluation results
    raw_errors: Optional[List[pd.DataFrame]]
    # error metrics of each split
    fold_errors: List[Optional[Dict[str, List[float]]]]
    # summary error metrics
    summary_errors: Optional[Dict[str, float]]


class Forecaster(Protocol):
    def __call__(self, train: DataPartition, test: DataPartition) -> pd.DataFrame:
        ...  # pragma: no cover

    """
    Function of fitting a forecasting model with `train` and evaluate the fitted model on `test`.
    Return a pd.DataFrame containing the information for computing evaluation metrics.
    """


class Scorer(Protocol):
    def __call__(self, result: pd.DataFrame) -> Dict[str, float]:
        ...  # pragma: no cover

    """Function for calculating evaluation metrics based on `result`.
    """


def _check_max_core(max_core: Optional[int]) -> int:
    """Helper function for validating core number for multi-processing."""

    total_cores = cpu_count()
    if isinstance(max_core, int) and max_core > 0 and max_core < total_cores:
        core_num = max_core
    else:
        core_num = max((total_cores - 1) // 2, 1)
        logging.warning(
            f"Input `max_core` = {max_core} is invalid, setting `max_core` = {core_num} instead."
        )
    return core_num


def _get_scorer(
    scorer: Union[str, List[str], CoreMetric]
) -> Optional[Callable[[pd.DataFrame], Dict[str, float]]]:
    """Helper function for validating `scorer`."""

    if isinstance(scorer, str):
        scorer = [scorer]
    if isinstance(scorer, list):
        methods = []
        for error in scorer:
            try:
                methods.append((error, core_metric(error)))
            except Exception as e:
                msg = f"Unsupported error function {error} with error message {e}."
                _log_error(msg)
        # define scorer function
        # pyre-fixme Incompatible return type [7]: Expected `Optional[typing.Callable[[DataFrame], Dict[str, float]]]` but got `Union[Metric, MultiOutputMetric, WeightedMetric]`.
        def calc_error(result: pd.DataFrame) -> Dict[str, float]:
            errors = {}
            for name, func in methods:
                errors[name] = func(result["y"].values, result["fcst"].values)
            return errors

        return calc_error

    elif callable(scorer):
        # pyre-fixme Incompatible return type [7]: Expected `Optional[typing.Callable[[DataFrame], Dict[str, float]]]` but got `Union[Metric, MultiOutputMetric, WeightedMetric]`.
        return scorer

    msg = "Input `scorer` is invalid."
    _log_error(msg)


def kats_units_forecaster(
    params: Params,
    # pyre-fixme
    model_class: Type,
    train: TimeSeriesData,
    test: TimeSeriesData,
) -> Optional[pd.DataFrame]:
    """
    Forecaster function for uni-time series Kats forecaasting model.
    """
    try:
        model = model_class(data=train, params=params)
        model.fit()
        fcst = model.predict(steps=len(test))
        fcst["y"] = test.value.values
        return fcst
    except Exception as e:
        msg = f"Fail to fit model with `model_class` = {model_class} and `params` = {params} with error message: {e}"
        logging.warning(msg)
    return None


class GenericBacktester(ABC):
    """
    This class defines the module for backtesting for generic forecasting model.

    Attributes:
        datapartition: a datapartition object defining the data partition logic of backtesting.
        forecaster: a callable object following protcol `Forecaster`, used for fitting data and generating forecasts.
        scorer: a strategy to evaluate the performance of forecasting model. Can be a or a list of strings representing the error metrics,
                or a callable following `Scorer` protocol.
        summarizer: a string for the method summarizing error metrics. Can be 'average' (i.e., averaged over all splits) or 'weighted' (i.e., weighted averaged by the size of test size). Default is 'average'.
        raw_errors: a boolean for whether returning raw data on the test sets.
        fold_errors: a boolean for whether returning the error metrics on the test sets.
        multi: a boolean for whether using multi-processing.
        max_core: a integer for the number of cores used by multi-processing. Default is None, which sets `max_core = max((all_available_core-1)//2-1, 1)`.
        error_score: a float for the substitude of error score if it is np.nan. Default is np.nan.

    Sample Usage:
        >>> # Define the forecaster
        >>> fcster = partial(kats_units_forecaster, params=ProphetParams(), model_class=ProphetModel())
        >>> # Define data partition logict
        >>> dp = SimpleDataPartition(train_frac = 0.9)
        >>> # Initiate backtester object
        >>> gbt = GenericBacktester(datapartition = dp, scorer = ['smape','mape'], forecaster = fcster)
        >>> ts = TimeSeriesData(pd.read_csv("kats/data/air_passengers.csv"))
        >>> gbt.run_backtester(ts)
    """

    def __init__(
        self,
        datapartition: DataPartitionBase,
        forecaster: Forecaster,
        scorer: Union[str, CoreMetric, List[Union[str, CoreMetric]], Scorer],
        summarizer: str = "average",
        raw_errors: bool = True,
        fold_errors: bool = True,
        multi: bool = True,
        max_core: Optional[int] = None,
        error_score: float = np.nan,
    ) -> None:

        self.datapartition: DataPartitionBase = datapartition
        # pyre-fixme
        self.scorer: Callable[[pd.DataFrame], Dict[str, float]] = _get_scorer(scorer)
        self.forecaster = forecaster

        self.raw_errors: bool = raw_errors
        self.fold_errors: bool = fold_errors
        self.summarizer: str = summarizer
        self.backtest_result: Optional[BacktesterResult] = None

        self.multi: bool = multi
        self.max_core: int = _check_max_core(max_core) if self.multi else 1
        self.error_score: float = error_score

    def run_backtester(self, data: DataPartition) -> None:
        train_test_data = self.datapartition.split(data)
        logging.info("Successfully finish data partitioning!")

        if self.multi:
            pool = Pool(self.max_core)
            all_res = pool.starmap(self._fit_and_scorer, train_test_data)
            pool.close()
            pool.join()
        else:
            all_res = []
            for train, test in train_test_data:
                tmp_all_res = self._fit_and_scorer(train, test)
                all_res.append(tmp_all_res)
        logging.info("Successfully finish evaluating models on all partitions.")
        self.backtest_result = self._summarize(all_res)
        return None

    def _fit_and_scorer(
        self, train: DataPartition, test: DataPartition
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        try:
            res = self.forecaster(train=train, test=test)
            score = self.scorer(res)
            return res, score
        except Exception as e:
            logging.info(
                f"Failed to fit model and get evaluation with error message: {e}."
            )
            return None, None

    def _get_fold_errors(
        self, raw_fold_errors: List[Optional[Dict[str, float]]]
    ) -> Tuple[List[str], List[Dict[str, float]]]:

        valid_fold_errors = [t for t in raw_fold_errors if t is not None]
        # get error metric names
        if not valid_fold_errors:
            msg = "No valid evaluation available, i.e., every evaluation failed."
            _log_error(msg)

        error_metrics = list(valid_fold_errors[0].keys())
        base_error_score = {em: self.error_score for em in error_metrics}
        fold_errors = []
        for fd in raw_fold_errors:
            if fd is None:
                fold_errors.append(base_error_score)
            else:
                fold_errors.append(fd)
        return error_metrics, fold_errors

    def _summarize(
        self, results: List[Tuple[pd.DataFrame, Dict[str, float]]]
    ) -> BacktesterResult:

        if not results:
            _log_error("Fail to get evaluation results!")

        raw_errors = [res[0] for res in results]
        raw_fold_errors = [res[1] for res in results]
        # get error metric names
        # pyre-fixme
        error_metrics, fold_errors = self._get_fold_errors(raw_fold_errors)

        # aggregate error metrics from each fold
        summary_errors = {}
        num_fold = len(fold_errors)
        weight = np.ones(num_fold)
        if self.summarizer == "weighted":
            weight = np.array([len(t) for t in raw_errors])
            weight = weight / np.sum(weight) * num_fold
        for em in error_metrics:
            summary_errors[em] = np.nanmean(
                np.array([t[em] for t in fold_errors]) * weight
            )

        backtest_res = BacktesterResult(
            raw_errors=raw_errors if self.raw_errors else None,
            # pyre-fixme
            fold_errors=fold_errors if self.fold_errors else None,
            summary_errors=summary_errors,
        )

        return backtest_res

    def get_errors(self) -> Optional[BacktesterResult]:
        if self.backtest_result is not None:
            return self.backtest_result
        msg = "Please execute function `run_backtester()` first."
        _log_error(msg)


class KatsSimpleBacktester(GenericBacktester):
    """
    Backtester module for single time series Kats forecasting models.

    Attributes:
        datapartition: a datapartition object defining the data partition logic of backtesting.
        scorer: a strategy to evaluate the performance of forecasting model. Can be a or a list of strings representing the error metrics,
                or a callable following `Scorer` protocol.
        model_params: a parameter object for the parameters of the target forecasting model.
        model_class: the model class for the target forecasting model.
        summarizer: a string for the method summarizing error metrics. Can be 'average' (i.e., averaged over all splits) or 'weighted' (i.e., weighted averaged by the size of test size). Default is 'average'.
        raw_errors: a boolean for whether returning raw data on the test sets.
        fold_errors: a boolean for whether returning the error metrics on the test sets.
        multi: a boolean for whether using multi-processing.
        max_core: a integer for the number of cores used by multi-processing. Default is None, which sets `max_core = max((all_available_core-1)//2-1, 1)`.
        error_score: a float for the substitude of error score if it is np.nan. Default is np.nan.

    Sample Usage:
        >>> # Define data partition logict
        >>> dp = SimpleDataPartition(train_frac = 0.9)
        >>> # Initiate backtester object
        >>> ksbt = KatsSimpleBacktester(datapartition = dp, scorer = ['smape','mape'], model_params = ProphetParams(), model_class = ProphetModel)
        >>> ts = TimeSeriesData(pd.read_csv("kats/data/air_passengers.csv"))
        >>> ksbt.run_backtester(ts)

    """

    def __init__(
        self,
        datapartition: DataPartitionBase,
        scorer: Union[str, CoreMetric, List[Union[str, CoreMetric]], Scorer],
        model_params: Params,
        # pyre-fixme
        model_class: Type,
        summarizer: str = "average",
        raw_errors: bool = True,
        fold_errors: bool = True,
        multi: bool = True,
        max_core: Optional[int] = None,
        error_score: float = np.nan,
    ) -> None:
        fcster = partial(
            kats_units_forecaster, params=model_params, model_class=model_class
        )
        super(KatsSimpleBacktester, self).__init__(
            datapartition=datapartition,
            scorer=scorer,
            # pyre-fixme
            forecaster=fcster,
            summarizer=summarizer,
            raw_errors=raw_errors,
            fold_errors=fold_errors,
            multi=multi,
            max_core=max_core,
            error_score=error_score,
        )


class BackTesterParent(ABC):
    """
    This class defines the parent functions for various backtesting methods.

    Attributes:
        error_methods: List of strings indicating which errors to calculate
          (see `kats.metrics` for exhaustive list).
        data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
        params: Parameters to train model with.
        model_class: Defines the model type to use for backtesting.
        multi: Boolean flag to use multiprocessing (if set to True).
        offset: Gap between train/test datasets (default 0).
        results: List of tuples `(training_data, testing_data, trained_model,
          forecast_predictions)` storing forecast results.
        errors: Dictionary mapping the error type to value.
        size: An integer for the total number of datapoints.
        freq: A string representing the (inferred) frequency of the
          `pandas.DataFrame`.

    Raises:
      ValueError: The time series is empty or an invalid error type was passed.
    """

    error_methods: List[Tuple[str, CoreMetric]]
    data: TimeSeriesData
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    model_class: Type
    params: Params
    multi: bool
    offset: int
    results: List[Tuple[np.ndarray, np.ndarray, "Model[Params]", np.ndarray]]
    errors: Dict[str, float]
    size: int
    freq: Optional[str]
    raw_errors: List[np.ndarray]

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        multi: bool,
        offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.size = size = len(data.time)
        if not size:
            msg = "Passed an empty time series"
            logging.error(msg)
            raise ValueError(msg)

        self.data = data
        self.model_class = model_class
        self.params = params
        self.multi = multi
        self.offset = offset

        self.results = []
        # Handling frequency
        if "freq" in kwargs:
            self.freq = kwargs["freq"]
        else:
            logging.info("Inferring frequency")
            self.freq = pd.infer_freq(self.data.time)

        self.raw_errors = []

        methods = []
        errors = {}
        for error in error_methods:
            try:
                methods.append((error, core_metric(error)))
                errors[error] = 0.0
            except ValueError:
                msg = f"Unsupported error function {error}"
                logging.error(msg)
                raise ValueError(msg)
        self.errors = errors
        self.error_methods = methods

        logging.info("Instantiated BackTester")
        if kwargs:
            logging.info(
                "Additional arguments: {0}".format(
                    (", ".join(["{}={!r}".format(k, v) for k, v in kwargs.items()]))
                )
            )
        logging.info("Model type: {0}".format(self.model_class))
        logging.info("Error metrics: {0}".format(error_methods))

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

            for name, method in self.error_methods:
                # Weighting the errors by the relative fold length if
                # predictions are of different sizes
                weight = float(len(truth)) / total_fold_length
                self.errors[name] += weight * method(truth, predictions)

    def _create_model(
        self,
        training_data_indices: Tuple[int, int],
        testing_data_indices: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, "Model[Params]", np.ndarray]]:
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
            self.results = results = []
            for fut in futures:
                result = fut.get()
                assert result is not None
                results.append(result)
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
    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        raise NotImplementedError()


class BackTesterSimple(BackTesterParent):
    """Defines the functions to execute a simple train/test backtest.

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      error_methods: List of strings indicating which errors to calculate
        (see `kats.metrics` for exhaustive list).
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
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        **kwargs: Any,
    ) -> None:
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
        train_size = _get_absolute_size(self.size, self.train_percentage)
        test_size = _get_absolute_size(self.size, self.test_percentage)

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


class BackTesterRollingOrigin(BackTesterParent):
    """Defines functions to execute an rolling origin backtest.

    A rolling forecast origin backtest conducts a backtest over multiple
    iterations, wherein each iteration, the "forecasting origin"
    (the location separating training and testing datasets) "slides" forward
    by a fixed amount.

    Hereby, the size of the training dataset is usually increased at each
    iteration, while the test dataset "slides" forward to accommodate.
    However, the size of the training dataset can alternatively be held
    constant, in which case at each iteration the start location of the
    training dataset moves forward by the same amount as the "forecast origin".
    Iterations continue until the complete data set is used to either train
    or test in the final interation.

    This procedure is also known in literature as a rolling origin evaluation
    with a continuously increasing in-sample size (train set) and a constant
    out-sample size (test set).
    For more information, check out the Kats tutorial notebooks!

    Attributes:
      start_train_percentage: A float for the initial percentage of data used
        for training. (The train percentage at the end will be 100 -
        test_percentage)
      test_percentage: A float for the percentage of data used for testing.
        (The test set is taken at sliding positions from start_train_percentage
         up to the end of the dataset - only the last fold is at the very end.)
      expanding_steps: An integer for the number of expanding steps (i.e.
        number of folds).
      error_methods: List of strings indicating which errors to calculate
        (see `kats.metrics` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      constant_train_size: A bool defining if the training data size should be
        constant instead of expanding it at each iteration (default False).
      multi: A boolean flag to toggle multiprocessing (default True).
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
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
      >>> backtester = BackTesterRollingOrigin(
            error_methods=all_errors,
            data=ts,
            params=params,
            start_train_percentage=50,
            test_percentage=25,
            expanding_steps=3,
            model_class=ARIMAModel,
            constant_train_size=False,
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
        test_percentage: float,
        expanding_steps: int,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        constant_train_size: bool = False,
        multi: bool = True,
        **kwargs: Any,
    ) -> None:
        logging.info("Initializing train/test percentages")
        if start_train_percentage <= 0:
            logging.error("Non positive start training percentage")
            raise ValueError("Invalid start training percentage")
        elif start_train_percentage > 100:
            logging.error("Too large start training percentage")
            raise ValueError("Invalid end training percentage")
        self.start_train_percentage = start_train_percentage
        if test_percentage <= 0:
            logging.error("Non positive test percentage")
            raise ValueError("Invalid test percentage")
        elif test_percentage > 100:
            logging.error("Too large test percentage")
            raise ValueError("Invalid test percentage")
        self.test_percentage = test_percentage
        if start_train_percentage + test_percentage > 100:
            logging.error("Too large combined train and test percentage")
            raise ValueError(  # noqa
                "Invalid training and testing percentage combination"
            )
        elif start_train_percentage + test_percentage == 100:
            if expanding_steps > 1:
                logging.error(
                    "Too large combined train and test percentage for "
                    "%s expanding steps",
                    expanding_steps,
                )
                raise ValueError(
                    "Invalid training and testing percentage combination "
                    f"given for {expanding_steps} expanding steps"
                )
        if expanding_steps < 0:
            logging.error("Non positive expanding steps")
            raise ValueError("Invalid expanding steps")
        self.expanding_steps = expanding_steps
        self.constant_train_size = constant_train_size

        logging.info("Calling parent class constructor")
        super().__init__(error_methods, data, params, model_class, multi, **kwargs)

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        start_train_size = _get_absolute_size(self.size, self.start_train_percentage)
        test_size = _get_absolute_size(self.size, self.test_percentage)

        if start_train_size <= 0 or start_train_size >= self.size:
            logging.error(
                "Invalid starting training size: {0}".format(start_train_size)
            )
            logging.error(
                "Start Training Percentage: {0}".format(self.start_train_percentage)
            )
            raise ValueError("Incorrect starting training size")

        if test_size <= 0 or test_size >= self.size:
            logging.error("Invalid testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect testing size")

        if start_train_size + test_size > self.size:
            if start_train_size + test_size > self.size:
                logging.error("Training and Testing sizes too big")
                logging.error("Start Training size: {0}".format(start_train_size))
                logging.error(
                    "Start Training Percentage: {0}".format(self.start_train_percentage)
                )
            logging.error("Testing size: {0}".format(test_size))
            logging.error("Testing Percentage: {0}".format(self.test_percentage))
            raise ValueError("Incorrect training and testing sizes")
        elif start_train_size + test_size == self.size:
            if self.expanding_steps > 1:
                logging.error(
                    "Training and Testing sizes too big " "for multiple steps"
                )
                logging.error("Start Training size: {0}".format(start_train_size))
                logging.error(
                    "Start Training Percentage: {0}".format(self.start_train_percentage)
                )
                logging.error("Testing size: {0}".format(test_size))
                logging.error("Testing Percentage: {0}".format(self.test_percentage))
                logging.error("Expanding steps: {}".format(self.expanding_steps))
                raise ValueError(
                    "Incorrect training and testing sizes " "for multiple steps"
                )

        # Handling edge case where only 1 fold is needed (same as BackTesterSimple)
        if self.expanding_steps == 1:
            return (
                [(0, start_train_size)],
                [(start_train_size, start_train_size + test_size)],
            )

        train_splits = []
        test_splits = []
        offsets = _return_fold_offsets(
            0, self.size - start_train_size - test_size, self.expanding_steps
        )
        for offset in offsets:
            skip_size = 0
            if self.constant_train_size:
                skip_size = offset
            train_splits.append((skip_size, int(start_train_size + offset)))
            test_splits.append(
                (
                    int(start_train_size + offset),
                    int(start_train_size + offset + test_size),
                )
            )
        return train_splits, test_splits


class BackTesterExpandingWindow(BackTesterRollingOrigin):
    """Defines functions to exectute an expanding window backtest.

    This class will be deprecated soon. Please use `BackTesterRollingOrigin`
    with param `constant_train_size = True`.
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
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        multi: bool = True,
        **kwargs: Any,
    ) -> None:
        logging.info(
            "BackTesterExpandingWindow will be deprecated. Please use the "
            "updated API found in BackTesterRollingOrigin."
        )
        super().__init__(
            error_methods=error_methods,
            data=data,
            params=params,
            start_train_percentage=start_train_percentage,
            test_percentage=test_percentage,
            expanding_steps=expanding_steps,
            model_class=model_class,
            multi=multi,
            constant_train_size=False,
            **kwargs,
        )


class BackTesterRollingWindow(BackTesterRollingOrigin):
    """Defines functions to execute a rolling window backtest.

    This class will be deprecated soon. Please use `BackTesterRollingOrigin`
    with param `constant_train_size = False`.
    """

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        sliding_steps: int,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        multi: bool = True,
        **kwargs: Any,
    ) -> None:
        logging.info(
            "BackTesterRollingWindow will be deprecated. Please use the "
            "updated API found in BackTesterRollingOrigin."
        )
        super().__init__(
            error_methods=error_methods,
            data=data,
            params=params,
            start_train_percentage=train_percentage,
            test_percentage=test_percentage,
            expanding_steps=sliding_steps,
            model_class=model_class,
            multi=multi,
            constant_train_size=True,
            **kwargs,
        )


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
        (see `kats.metrics` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      results: List of tuples `(training_data, testing_data, trained_model,
        forecast_predictions)` storing forecast results.
      errors: Dictionary mapping the error type to value.
      size: An integer for the total number of datapoints.
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
            params=params,
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
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        **kwargs: Any,
    ) -> None:
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

        offset = _get_absolute_size(len(data.time), self.window_percentage)

        logging.info("Calling parent class constructor")
        super().__init__(
            error_methods, data, params, model_class, False, offset, **kwargs
        )

    def _create_train_test_splits(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Creates train/test folds for the backtest."""

        logging.info("Creating train test splits")
        train_size = _get_absolute_size(self.size, self.train_percentage)
        test_size = _get_absolute_size(self.size, self.test_percentage)
        window_size = _get_absolute_size(self.size, self.window_percentage)

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
    """Defines class to execute time series cross validation.

    Cross validation is a useful technique to use multiple folds of the
    training and testing data to help optimize the performance of the
    model (e.g. hyperparameter tuning). For more info on cross validation, see
    https://en.wikipedia.org/wiki/Cross-validation_(statistics)

    This procedure is also known in literature as a rolling origin evaluation.

    Attributes:
      train_percentage: A float for the percentage of data used for training.
      test_percentage: A float for the percentage of data used for testing.
      num_folds: An integer for the number of folds to use.
      error_methods: List of strings indicating which errors to calculate
        (see `kats.metrics` for exhaustive list).
      data: :class:`kats.consts.TimeSeriesData` object to perform backtest on.
      params: Parameters to train model with.
      model_class: Defines the model type to use for backtesting.
      constant_train_size: A boolean flag to keep the train set size constant,
          sliding it forward with each iteration instead of expanding the
          train set with each iteration (default False).
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
            params=params,
            train_percentage=50,
            test_percentage=25,
            num_folds=3,
            model_class=ARIMAModel,
            rolling_window=True
          )
      >>> cv.run_cv()
      >>> mape = cv.get_error_value("mape") # Retrieve MAPE error
    """

    size: int
    results: List[Tuple[np.ndarray, np.ndarray, "Model[Params]", np.ndarray]]
    num_folds: int
    errors: Dict[str, float]
    raw_errors: List[np.ndarray]
    _backtester: BackTesterParent

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        num_folds: int,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        model_class: Type,
        constant_train_size: bool = False,
        multi: bool = True,
    ) -> None:
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

        if not constant_train_size:
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


def _get_absolute_size(size: int, percent: float) -> int:
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
