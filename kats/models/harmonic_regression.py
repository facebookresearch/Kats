# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from scipy import optimize


@dataclass
class HarmonicRegressionParams(Params):
    period: float
    fourier_order: int

    def __post_init__(self) -> None:
        if self.period <= 0:
            msg = f"The provided period must be greater than zero. Value: {self.period}"
            logging.error(msg)
            raise ValueError(msg)

        if self.fourier_order <= 0:
            msg = (
                "The provided fourier order must be greater than zero. "
                f"Value: {self.fourier_order}"
            )
            logging.error(msg)
            raise ValueError(msg)


class HarmonicRegressionModel(Model[Optional[np.ndarray]]):
    params: Optional[np.ndarray] = None
    harms: Optional[np.ndarray] = None

    def __init__(self, data: TimeSeriesData, params: HarmonicRegressionParams) -> None:
        super().__init__(data, params)
        # pyre-fixme[16]: `Optional` has no attribute `is_univariate`.
        if not self.data.is_univariate():
            msg = "The provided time series data is not univariate."
            logging.error(msg)
            raise ValueError(msg)
        self.regression_params = params

    def setup_data(self) -> None:
        pass

    def validate_inputs(self) -> None:
        pass

    def fit(self) -> None:
        """Fits harmonic regression to the time series.
        See fit_harmonics for details.
        """
        self.params, self.harms = self.fit_harmonics(
            self.regression_params.period, self.regression_params.fourier_order
        )

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, dates: pd.Series, *_args: Optional[Any], **_kwargs: Optional[Any]
    ) -> pd.DataFrame:
        """Predicts with harmonic regression values.
         Call fit before calling this function.
        Parameters
        ----------
        dates: dates to compute the predictions for
        Returns
        -------
        Pandas DataFrame with the dates (time) and the
        forecast values (fcst)
        """
        if self.params is None or self.harms is None:
            msg = "Call fit before predict."
            logging.error(msg)
            raise ValueError(msg)
        harmonics = HarmonicRegressionModel.fourier_series(
            dates, self.regression_params.period, self.regression_params.fourier_order
        )
        result = np.sum(self.params * harmonics, axis=1)
        return pd.DataFrame({"time": dates, "fcst": result.tolist()}, copy=False)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = "Time Series v.s Fitted Harmonics",
        **kwargs: Optional[Any],
    ) -> plt.Axes:
        """Demeans the time series, fits the harmonics,
            returns the plot and error metrics.
        Parameters
        ----------
        Returns
            Plot of the original time series and the fitted harmonics
            Dataframe with mean square error and absolute error
        """

        if self.params is None or self.harms is None:
            msg = "Call fit before plot."
            logging.error(msg)
            raise ValueError(msg)

        fitted_harmonics = np.sum(self.params * self.harms, axis=1)

        if ax is None:
            if figsize is None:
                figsize = (20, 6)
            _, ax = plt.subplots(figsize=figsize)

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # pyre-fixme[16]: `Optional` has no attribute `time`.
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        (line1,) = ax.plot(self.data.time, self.data.value, label="Original signal")
        (line2,) = ax.plot(self.data.time, fitted_harmonics, label="Fitted harmonics")
        ax.legend(handles=[line1, line2])

        return ax

    @staticmethod
    def fourier_series(
        dates: pd.Series, period: float, series_order: int
    ) -> np.ndarray:
        """Provides Fourier series components with the specified frequency
        and order. The starting time is always the epoch.
        Parameters
        ----------
        dates: pd.Series containing timestamps.
        period: Number of hours of the period.
        series_order: Number of components.
        Returns
        -------
        Matrix with seasonality features.
        """
        # convert to days since epoch
        t = (
            np.array((dates - datetime(1970, 1, 1)).dt.total_seconds().astype(np.float))
            / 3600.0
        )
        return np.column_stack(
            [
                fun((2.0 * (i + 1) * np.pi * t / period))
                for i in range(series_order)
                for fun in (np.sin, np.cos)
            ]
        )

    @staticmethod
    def make_harm_eval(
        harmonics: np.ndarray,
    ) -> Callable[..., np.ndarray]:
        """Defines evaluation function for the optimizer
        Parameters
        ----------
        harmonics: the harmonics to fit
        Returns
        -------
        The evaluation function for the optimizer
        """

        def harm_eval(step: np.ndarray, *params: Real) -> np.ndarray:
            params_ = np.array(params)
            mul = np.multiply(harmonics[step.astype(int)], params_)
            return np.sum(mul, axis=1)

        return harm_eval

    def fit_harmonics(
        self, period: float, fourier_order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs harmonic regression.
        Harmonic regression fits cosines
        amplitude*cos(freq*t + phase). Using double angle identity formulas,
        we have:
        beta1*cos(freq*t) + beta2*sin(freq*t). Thus, we can fit two coefficients,
        which will take care of the amplitude and the phase. If we generate the
        raw cos(freq*t) and sin(freq*t) for each freq we want to have,
        it becomes a linear regression. Since we ignore
        intercept, we demean the time series before fitting.

        Since the regression takes care of the phase, we can pick time 0 wherever
        we want, we just have to use the same for training, test, validation,
        and prediction. We pick that as the epoch; so when we generate
        the raw cos and sin values for the test set,
        and apply the parameters from the training, it will have the right phase.

        Parameters
        ----------
        period: float; seasonality in hours; e.g. 24 for daily
        fourier_order: int; number of harmonics for the given frequency
        harms: externally computed harmonics
        Returns:
            params: coefficients
            harms: feature matrix the generated raw cos and sin;
            for each fourier_order, there is one cos-sin pair.
            Number of colums: fourier_order*2
        """
        # pyre-fixme[16]: `Optional` has no attribute `value`.
        time_series = self.data.value

        harms = HarmonicRegressionModel.fourier_series(
            # pyre-fixme[16]: `Optional` has no attribute `time`.
            self.data.time,
            period,
            fourier_order,
        )

        steps = np.array(list(range(0, len(time_series.index))))
        demeaned = time_series - time_series.mean()

        params, params_covariance = optimize.curve_fit(
            HarmonicRegressionModel.make_harm_eval(harms),
            steps,
            demeaned.tolist(),
            # This is to set the number of params via the number of initial values
            p0=[0.001] * (fourier_order * 2),
        )
        return params, harms
