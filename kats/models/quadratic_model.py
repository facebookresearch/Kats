# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Forecasting with quadratic model

The quadratic (non-linear) regression model explores a linear relationship
between the forecast variable `y` (observed time series) and predictor
variables `x` and `x^2`, where `x` is the time.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from statsmodels.sandbox.regression.predstd import wls_prediction_std


class QuadraticModelParams(Params):
    """Parameter class for Quadratic model.

    This is the parameter class for the quadratic model.
    Attributes:
        alpha: The alpha level for the confidence interval. The default alpha =
        0.05 returns a 95% confidence interval
    """

    def __init__(self, alpha: float = 0.05, **kwargs: Any) -> None:
        super().__init__()
        self.alpha = alpha
        logging.debug(
            "Initialized QuadraticModel parameters. "
            "alpha:{alpha}".format(alpha=alpha)
        )

    def validate_params(self) -> None:
        """Validate Quadratic Model Parameters

        Since the quadratic model does not require key parameters to be defined
        this is not required for this class
        """
        logging.info("Method validate_params() is not implemented.")
        pass


class QuadraticModel(Model[QuadraticModelParams]):
    """Model class for Quadratic Model.

    This class provides the fit, predict and plot methods for the Quadratic Model

    Attributes:
        data: the input time series data as :class:`kats.consts.TimeSeriesData`
        params: the parameter class defined with `QuadraticModelParams`
    """

    past_length: Optional[int] = None
    model: Optional[Model[QuadraticModelParams]] = None
    freq: Optional[str] = None
    _X_future: Optional[List[int]] = None
    sdev: Optional[float] = None
    dates: Optional[pd.DatetimeIndex] = None
    y_fcst: Optional[pd.Series] = None
    y_fcst_lower: Optional[pd.Series] = None
    y_fcst_upper: Optional[pd.Series] = None
    fcst_df: Optional[pd.DataFrame] = None

    def __init__(self, data: TimeSeriesData, params: QuadraticModelParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self) -> None:
        """fit Quadratic Model."""
        logging.debug(
            "Call fit() with parameters: "
            "alpha:{alpha}".format(alpha=self.params.alpha)
        )

        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.past_length = len(self.data.time)
        _X = list(range(self.past_length))
        _X_quad = np.column_stack([_X, np.power(_X, 2)])
        X_quad = sm.add_constant(_X_quad)

        # pyre-fixme[16]: `Optional` has no attribute `value`.
        y = self.data.value
        quad_model = sm.OLS(y, X_quad)

        self.model = quad_model.fit()

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """predict with fitted quadratic model.

        Args:
            steps: the steps or length of the prediction horizon
            include_history: whether to include the historical data in the prediction

        Returns:
            The predicted dataframe with the following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        past_length = self.past_length
        if past_length is None:
            raise ValueError("Call fit() before predict().")
        model = self.model
        assert model is not None
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `Optional` has no attribute `time`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.include_history = include_history

        if include_history:
            _X_future = list(range(0, past_length + steps))
        else:
            _X_future = list(range(past_length, past_length + steps))
        self._X_future = _X_future
        _X_fcst = np.column_stack([_X_future, np.power(_X_future, 2)])
        X_fcst = sm.add_constant(_X_fcst)
        y_fcst = model.predict(X_fcst)
        sdev, y_fcst_lower, y_fcst_upper = wls_prediction_std(
            self.model, exog=X_fcst, alpha=self.params.alpha
        )
        self.sdev = sdev
        self.y_fcst = pd.Series(y_fcst, copy=False)
        self.y_fcst_lower = pd.Series(y_fcst_lower, copy=False)
        self.y_fcst_upper = pd.Series(y_fcst_upper, copy=False)

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]

        if include_history:
            self.dates = np.concatenate((pd.to_datetime(self.data.time), self.dates))

        self.fcst_df = fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            },
            copy=False,
        )
        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return fcst_df

    def __str__(self) -> str:
        return "Quadratic"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """get default parameter search space for Quadratic model."""
        return [
            {
                "name": "alpha",
                "type": "choice",
                "value_type": "float",
                "values": [0.01, 0.05, 0.1, 0.25],
                "is_ordered": True,
            },
        ]
