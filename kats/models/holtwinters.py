# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""The Holt Winters model is a time series forecast model that applies exponential smoothing three times, it serves as the extension of the simple exponential smoothing forecast model.

More details about the different exponential smoothing model can be found here:
https://en.wikipedia.org/wiki/Exponential_smoothing
In this module we adopt the Holt Winters model implementation from the statsmodels package, full details can be found as follows:
https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
We rewrite the corresponding API to accommodate the Kats development style
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.emp_confidence_int import EmpConfidenceInt
from kats.utils.parameter_tuning_utils import (
    get_default_holtwinters_parameter_search_space,
)
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing as HoltWinters,
    HoltWintersResults,
)


class HoltWintersParams(Params):
    """Parameter class for the HoltWinters model

    Not all parameters from the statsmodels API have been implemented here, the full list of the parameter can be found:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html

    Attributes:
        trend: Optional; A string that specifies the type of trend component. Can be 'add' and 'mul' or 'additive' and 'multiplicative'. Default is 'add'
        damped: Optional; A boolean indicates whether the trend should be damped or not. Default is False
        seasonal: Optional; A string that specifies the type of seasonal component Can be 'add' and 'mul' or 'additive' and 'multiplicative'. Default is None
        seasonal_periods: Optional; An integer that specifies the period for the seasonal component, e.g. 4 for quarterly data and 7 for weekly seasonality of daily data. Default is None
    """

    trend: Optional[str]
    damped: Optional[bool]
    seasonal: Optional[str]
    seasonal_periods: Optional[int]

    __slots__ = ["trend", "damped", "seasonal", "seasonal_periods"]

    def __init__(
        self,
        trend: str = "add",
        damped: bool = False,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.validate_params()
        logging.debug(
            "Initialized HoltWintersParams with parameters. "
            "trend:{trend},\
            damped:{damped},\
            seasonal:{seasonal},\
            seasonal_periods{seasonal_periods}".format(
                trend=trend,
                damped=damped,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        )

    def validate_params(self) -> None:
        """Validate the types and values of the input parameters
        Args:
            None

        Returns:
            None
        """

        if self.trend not in ["add", "mul", "additive", "multiplicative", None]:
            msg = "trend parameter is not valid!\
                         use 'add' or 'mul' instead!"
            logging.error(msg)
            raise ValueError(msg)

        if self.seasonal not in ["add", "mul", "additive", "multiplicative", None]:
            msg = "seasonal parameter is not valid!\
                         use 'add' or 'mul' instead!"
            logging.error(msg)
            raise ValueError(msg)


class HoltWintersModel(Model[HoltWintersParams]):
    """Model class for the HoltWinters model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input historical time series data from TimeSeriesData
        params: The HoltWinters model parameters from HoltWintersParams
    """

    model: Optional[HoltWintersResults] = None
    freq: Optional[str] = None
    dates: Optional[pd.DatetimeIndex] = None
    alpha: Optional[float] = None
    y_fcst: Optional[pd.Series] = None
    fcst_df: Optional[pd.DataFrame] = None

    def __init__(self, data: TimeSeriesData, params: HoltWintersParams) -> None:

        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs: Any) -> None:
        """Fit the model with the specified input parameters"""

        logging.debug("Call fit() with parameters:{kwargs}".format(kwargs=kwargs))
        holtwinters = HoltWinters(
            # pyre-fixme[16]: `Optional` has no attribute `value`.
            self.data.value,
            trend=self.params.trend,
            damped=self.params.damped,
            seasonal=self.params.seasonal,
            seasonal_periods=self.params.seasonal_periods,
        )
        self.model = holtwinters.fit()
        logging.info("Fitted HoltWinters.")

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """Predict with fitted HoltWinters model

        If the alpha keyword argument is specified, an empirical confidence interval is computed through a K-fold cross validation and a linear regression model, the forecast outcome will include a confidence interval there; otherwise no confidence interval is included in the final forecast. Please refer to the 'emp_confidence_int' module for full detailed implementation of the empirical confidence interval computation

        Args:
            steps:
            include_history:
        Returns:
            A pd.DataFrame with the forecast and confidence interval (if empirical confidence interval calculation is triggered)
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        if "freq" not in kwargs:
            # pyre-fixme[16]: `Optional` has no attribute `time`.
            self.freq = pd.infer_freq(self.data.time)
        else:
            self.freq = kwargs["freq"]
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods
        self.include_history = include_history

        if "alpha" in kwargs:
            self.alpha = alpha = kwargs["alpha"]
            # build empirical CI
            error_methods = kwargs.get("error_methods", ["mape"])
            train_percentage = kwargs.get("train_percentage", 70)
            test_percentage = kwargs.get("test_percentage", 10)
            # pyre-fixme[6]: For 1st param expected `Sized` but got
            #  `Optional[TimeSeriesData]`.
            sliding_steps = kwargs.get("sliding_steps", len(self.data) // 5)
            multi = kwargs.get("multi", True)
            eci = EmpConfidenceInt(
                error_methods=error_methods,
                # pyre-fixme[6]: For 2nd param expected `TimeSeriesData` but got
                #  `Optional[TimeSeriesData]`.
                data=self.data,
                params=self.params,
                train_percentage=train_percentage,
                test_percentage=test_percentage,
                sliding_steps=sliding_steps,
                model_class=HoltWintersModel,
                confidence_level=1 - alpha,
                multi=False,
            )
            logging.debug(
                "Use EmpConfidenceInt for CI with parameters: error_methods = "
                f"{error_methods}, train_percentage = {train_percentage}, "
                f"test_percentage = {test_percentage}, sliding_steps = "
                f"{sliding_steps}, confidence_level = {1-alpha}, multi={multi}."
            )
            fcst = eci.get_eci(steps=steps)
            self.y_fcst = fcst["fcst"]
        else:
            fcst = model.forecast(steps)
            self.y_fcst = fcst
            fcst = pd.DataFrame({"time": self.dates, "fcst": fcst}, copy=False)
        logging.info("Generated forecast data from Holt-Winters model.")

        if include_history:
            history_fcst = model.predict(start=0, end=len(self.data.time) - 1)
            self.fcst_df = fcst = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "time": self.data.time,
                            "fcst": history_fcst,
                        },
                        copy=False,
                    ),
                    fcst,
                ],
                copy=False,
            )
        else:
            self.fcst_df = fcst

        return fcst

    def __str__(self) -> str:
        return "HoltWinters"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default HoltWinters parameter search space.

        Args:
            None

        Returns:
            A dictionary with the default HoltWinters parameter search space
        """

        return get_default_holtwinters_parameter_search_space()
