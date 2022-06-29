# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Forecasting with simple linear regression model
#
# In the simplest case, the regression model explores a linear relationship
# between the forecast variable `y` (observed time series) and a single
# predictor variable `x` (time).

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from statsmodels.sandbox.regression.predstd import wls_prediction_std


class LinearModelParams(Params):
    """Parameter class for Linear model.

    This is the parameter class for the linear model.

    Attributes:
        alpha: The alpha level for the confidence interval. The default alpha = 0.05 returns a 95% confidence interval
    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha
        logging.debug(
            "Initialized LinearModel parameters. " "alpha:{alpha}".format(alpha=alpha)
        )

    def validate_params(self) -> None:
        """Validate Linear Model Parameters

        Since the linear model does not require key parameters to be defined this is not required for this class
        """
        logging.info("Method validate_params() is not implemented.")
        pass


class LinearModel(Model[LinearModelParams]):
    """Model class for Linear Model.

    This class provides the fit, predict and plot methods for the Linear Model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input time series data as `TimeSeriesData`
        params: the parameter class defined with `LinearModelParams`
    """

    def __init__(self, data: TimeSeriesData, params: LinearModelParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.model: Optional[sm.OLS] = None
        self.fcst_df = pd.DataFrame(data=None)
        self.freq: Optional[str] = None
        self._X_future: Optional[List[int]] = None
        self.past_length: int = len(data.time)
        self.dates: Optional[pd.DatetimeIndex] = None
        self.y_fcst: Optional[Union[pd.Series, np.ndarray]] = None
        self.sdev: Optional[Union[np.ndarray, float]] = None
        self.y_fcst_lower: Optional[Union[pd.Series, np.ndarray, float]] = None
        self.y_fcst_upper: Optional[Union[pd.Series, np.ndarray, float]] = None

    def fit(self) -> None:
        """fit Linear Model."""
        logging.debug(
            "Call fit() with parameters: "
            "alpha:{alpha}".format(alpha=self.params.alpha)
        )

        # prepare X and y for linear model
        _X = list(range(self.past_length))
        X = sm.add_constant(_X)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        y = self.data.value
        lm = sm.OLS(y, X)
        self.model = lm.fit()

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """predict with fitted linear model.

        Args:
            steps: the steps or length of the prediction horizon
            include_history: whether to include the historical data in the prediction

        Returns:
            The predicted dataframe with the following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.include_history = include_history

        if include_history:
            self._X_future = list(range(0, self.past_length + steps))
        else:
            self._X_future = list(range(self.past_length, self.past_length + steps))

        X_fcst = sm.add_constant(self._X_future)
        if self.model is None:
            raise ValueError("Call fit() before predict()")
        else:
            y_fcst = self.model.predict(X_fcst)

        self.sdev, self.y_fcst_lower, self.y_fcst_upper = wls_prediction_std(
            self.model, exog=X_fcst, alpha=self.params.alpha
        )

        self.y_fcst = pd.Series(y_fcst, copy=False)
        self.y_fcst_lower = pd.Series(self.y_fcst_lower, copy=False)
        self.y_fcst_upper = pd.Series(self.y_fcst_upper, copy=False)

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]

        if include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))

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
        return "Linear Model"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """get default parameter search space for Linear model."""
        return [
            {
                "name": "alpha",
                "type": "choice",
                "value_type": "float",
                "values": [0.01, 0.05, 0.1, 0.25],
                "is_ordered": True,
            },
        ]
