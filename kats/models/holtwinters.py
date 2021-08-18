# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Holt Winters model is a time series forecast model that applies exponential smoothing three times, it serves as the extension of the simple exponential smoothing forecast model.

More details about the different exponential smoothing model can be found here:
https://en.wikipedia.org/wiki/Exponential_smoothing
In this module we adopt the Holt Winters model implementation from the statsmodels package, full details can be found as follows:
https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
We rewrite the corresponding API to accommodate the Kats development style
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Any

import kats.models.model as m
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.emp_confidence_int import EmpConfidenceInt
from kats.utils.parameter_tuning_utils import (
    get_default_holtwinters_parameter_search_space,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWinters


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

    __slots__ = ["trend", "damped", "seasonal", "seasonal_periods"]

    def __init__(
        self,
        trend: str = "add",
        damped: bool = False,
        # pyre-fixme[9]: seasonal has type `str`; used as `None`.
        seasonal: str = None,
        # pyre-fixme[9]: seasonal_periods has type `int`; used as `None`.
        seasonal_periods: int = None,
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

    def validate_params(self):
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


class HoltWintersModel(m.Model):
    """Model class for the HoltWinters model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input historical time series data from TimeSeriesData
        params: The HoltWinters model parameters from HoltWintersParams
    """

    def __init__(self, data: TimeSeriesData, params: HoltWintersParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs) -> None:
        """Fit the model with the specified input parameters"""

        logging.debug("Call fit() with parameters:{kwargs}".format(kwargs=kwargs))
        holtwinters = HoltWinters(
            self.data.value,
            trend=self.params.trend,
            damped=self.params.damped,
            seasonal=self.params.seasonal,
            seasonal_periods=self.params.seasonal_periods,
        )
        # pyre-fixme[16]: `HoltWintersModel` has no attribute `model`.
        self.model = holtwinters.fit()
        logging.info("Fitted HoltWinters.")

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, **kwargs
    ) -> pd.DataFrame:
        """Predict with fitted HoltWinters model

        If the alpha keyword argument is specified, an empirical confidence interval is computed through a K-fold cross validation and a linear regression model, the forecast outcome will include a confidence interval there; otherwise no confidence interval is included in the final forecast. Please refer to the 'emp_confidence_int' module for full detailed implementation of the empirical confidence interval computation

        Args:
            steps:
            include_history:
        Returns:
            A pd.DataFrame with the forecast and confidence interval (if empirical confidence interval calculation is triggered)
        """

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        if "freq" not in kwargs:
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `freq`.
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `data`.
            self.freq = pd.infer_freq(self.data.time)
        else:
            self.freq = kwargs["freq"]
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        # pyre-fixme[16]: `HoltWintersModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]  # Return correct number of periods
        # pyre-fixme[16]: `HoltWintersModel` has no attribute `include_history`.
        self.include_history = include_history

        if "alpha" in kwargs:
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `alpha`.
            self.alpha = kwargs["alpha"]
            # build empirical CI
            error_methods = kwargs.get("error_methods", ["mape"])
            train_percentage = kwargs.get("train_percentage", 70)
            test_percentage = kwargs.get("test_percentage", 10)
            sliding_steps = kwargs.get("sliding_steps", len(self.data) // 5)
            multi = kwargs.get("multi", True)
            eci = EmpConfidenceInt(
                error_methods=error_methods,
                data=self.data,
                params=self.params,
                train_percentage=train_percentage,
                test_percentage=test_percentage,
                sliding_steps=sliding_steps,
                model_class=HoltWintersModel,
                confidence_level=1 - self.alpha,
                multi=False,
            )
            logging.debug(
                f"""Use EmpConfidenceInt for CI with parameters: error_methods = {error_methods}, train_percentage = {train_percentage},
                    test_percentage = {test_percentage}, sliding_steps = {sliding_steps}, confidence_level = {1-self.alpha}, multi={multi}."""
            )
            fcst = eci.get_eci(steps=steps)
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `y_fcst`.
            self.y_fcst = fcst["fcst"]
        else:
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `model`.
            fcst = self.model.forecast(steps)
            self.y_fcst = fcst
            fcst = pd.DataFrame({"time": self.dates, "fcst": fcst})
        logging.info("Generated forecast data from Holt-Winters model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        if include_history:
            history_fcst = self.model.predict(start=0, end=len(self.data.time))
            # pyre-fixme[16]: `HoltWintersModel` has no attribute `fcst_df`.
            self.fcst_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "time": self.data.time,
                            "fcst": history_fcst,
                        }
                    ),
                    fcst,
                ]
            )
        else:
            self.fcst_df = fcst

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """Plot forecast results from the HoltWinters model"""

        logging.info("Generating chart for forecast result from arima model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "HoltWinters"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default HoltWinters parameter search space.

        Args:
            None

        Returns:
            A dictionary with the default HoltWinters parameter search space
        """

        return get_default_holtwinters_parameter_search_space()
