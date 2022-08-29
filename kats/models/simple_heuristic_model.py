# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""The Simple Heuristic model

Simple Heuristic model is a model that applies simple rules like mean or percentiles on historical data to get prediction.
"""
import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model

MODELS = ["last", "mean", "median", "percentile"]


class SimpleHeuristicModelParams(Params):
    """Parameter class for Simple Heuristic model.

    Attributes:
        method: a method from MODELS that will be used on historical data
                "last" method will use the last values as prediction
                "mean" method will use the mean of historical data as prediction
                "median" method will use median of historical data as prediction
                "percentile" method will use percentile and alpha of historical data as prediction
        quantile: quantile to compute, which must be a number between 0 and 100 inclusive.
    """

    def __init__(self, method: str = "last", quantile: int = 95) -> None:
        super().__init__()
        self.method = method
        self.quantile = quantile
        logging.debug(
            f"Initialized SimpleHeuristicModel parameters with method: {self.method}"
        )

    def validate_params(self) -> None:
        "Validate SimpleHeuristicModel Model Parameters"

        if self.method not in MODELS:
            msg = f"Only support {', '.join(MODELS)}"
            logging.error(msg)
            raise ValueError(msg)

        if not isinstance(self.quantile, int):
            msg = "Percentile needs to be a number from 0 to 100"
            logging.error(msg)
            raise ValueError(msg)

        if self.quantile < 0 or self.quantile > 100:
            msg = "Percentile needs to be a number from 0 to 100"
            logging.error(msg)
            raise ValueError(msg)


class SimpleHeuristicModel(Model[SimpleHeuristicModelParams]):
    """Model class for SimpleHeuristicModel Model.

    This class provides the fit, predict and plot methods for the Simple Heuristic Model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input time series data as `TimeSeriesData`
        params: the parameter class defined with `SimpleHeuristicModelParams`
    """

    model: Callable[[np.ndarray], np.ndarray]
    include_history: bool = False
    dates: Optional[pd.DatetimeIndex] = None
    y_fcst: Optional[np.ndarray] = None
    y_fcst_lower: Optional[np.ndarray] = None
    y_fcst_upper: Optional[np.ndarray] = None
    fcst_df: pd.DataFrame = pd.DataFrame(data=None)
    freq: Optional[str] = None

    def __init__(
        self, data: TimeSeriesData, params: SimpleHeuristicModelParams
    ) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        self.model = self._calc_last
        self.y_fcst = None
        self.y_fcst_lower = None
        self.y_fcst_upper = None
        self.fcst_df = pd.DataFrame(data=None)

    def _calc_last(self, fitted_values: np.ndarray) -> np.ndarray:
        """subtracts last 'steps' data points from self.data.value

        Args:
            fitted_values: numpy ndarray with shape (len(self.data.value)//steps, steps) from self.data.value
        """
        return fitted_values[-1, :]

    def _calc_mean(self, fitted_values: np.ndarray) -> np.ndarray:
        """calculates mean from self.data.value
           returns ndarray with shape (steps,)

        Args:
            fitted_values: numpy ndarray with shape (len(self.data.value)//steps, steps) from self.data.value
        """
        return np.mean(fitted_values, 0)

    def _calc_median(self, fitted_values: np.ndarray) -> np.ndarray:
        """calculates median from self.data.value
           returns ndarray with shape (steps,)

        Args:
            fitted_values: numpy ndarray with shape (len(self.data.value)//steps, steps) from self.data.value
        """
        return np.median(fitted_values, 0)

    def _calc_percentile(self, fitted_values: np.ndarray) -> np.ndarray:
        """calculates percentile from self.data.value
           returns ndarray with shape (steps,)

        Args:
            fitted_values: numpy ndarray with shape (len(self.data.value)//steps, steps) from self.data.value
        """
        return np.percentile(fitted_values, self.params.quantile, 0)

    def fit(self) -> None:
        "fit Simple Heuristic Model."
        logging.debug(
            "Call fit() with parameters: "
            "method:{method}".format(method=self.params.method)
        )

        if self.params.method == "last":
            self.model = self._calc_last
        elif self.params.method == "mean":
            self.model = self._calc_mean
        elif self.params.method == "median":
            self.model = self._calc_median
        else:
            assert self.params.method == "percentile"
            self.model = self._calc_percentile

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, *args: Any, include_history: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """predict with fitted Simple Heuristic Model.

        Args:
            steps: the steps or length of the prediction horizon
            include_history: whether to include the historical data in the prediction

        Returns:
            The predicted dataframe with the following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        logging.debug(f"Call predict() with parameters. Steps:{steps}")
        if self.model is None:
            raise ValueError("fit must be called before predict.")

        # TO DO: infer more robust frequency
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        if self.freq is None:
            logging.info("Could not infer freq. Defaulting to daily 'D'")

        self.include_history = include_history

        # pyre-fixme[16]: `Optional` has no attribute `value`.
        fitted_len = (len(self.data.value) // steps) * steps
        fitted_values_s = self.data.value[-fitted_len:]
        fitted_values = np.reshape(np.asarray(fitted_values_s), [-1, steps])

        self.y_fcst = self.model(fitted_values)
        self.y_fcst_lower = self.y_fcst
        self.y_fcst_upper = self.y_fcst

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]

        if self.include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))
            self.y_fcst = np.concatenate((np.asarray(self.data.value), self.y_fcst))
            self.y_fcst_lower = np.concatenate(
                (np.asarray(self.data.value), self.y_fcst_lower)
            )
            self.y_fcst_upper = np.concatenate(
                (np.asarray(self.data.value), self.y_fcst_upper)
            )

        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            },
            copy=False,
        )
        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def __str__(self) -> str:
        return "Simple Heuristic Model"
