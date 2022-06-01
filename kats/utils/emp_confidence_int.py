# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Empirical Confidence (Prediction) Interval

This is an empirical way to estimate the prediction interval for any forecasting models
The high level idea is to estimate the empirical error distributions from a specific
forecasting model, and use linear regression model to fit the standard error (S.E.) with
the time horizon, under the assumption that longer horizon has larger S.E.
"""

import logging
from typing import List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.utils.backtesters import BackTesterRollingWindow
from scipy import stats


FigSize = Tuple[int, int]


class EmpConfidenceInt:
    """ "class for empirical confidence interval

    The steps are listed as follows:
    1. Run K-fold CV for a given model and data,
       each fold contains h (horizon) time ahead
    2. For each horizon, calculate the Std of K error terms (S.E)
    3. Fit linear model: S.E. ~ Horizon
    4. Estimate the S.E. for each horizon for the true future
    5. Lower/Upper = Point Estimate -/+ Z_Score * S.E.

    Attributes:
        error_method: list of strings indicating which errors to calculate
            we currently support "mape", "smape", "mae", "mase", "mse", "rmse"
        data: the time series data in `TimeSeriesData` format
        params: the Kats model parameter object
        train_percentage: percentage of data used for training
        test_percentage: percentage of data used for testing
        sliding_steps: number of rolling steps to take (# of folds)
        model_class: the Kats model class
        multi: flag to use multiprocessing, the default is True
        confidence_level: the confidence level for the prediction interval
    """

    freq: Optional[str] = None
    dates: Optional[pd.DatetimeIndex] = None
    predicted: Optional[np.ndarray] = None
    coefs: Optional[np.ndarray] = None
    df: Optional[pd.DataFrame] = None
    SE: Optional[pd.DataFrame] = None

    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        params: Params,
        train_percentage: float,
        test_percentage: float,
        sliding_steps: int,
        # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
        model_class: Type,
        multi: bool = True,
        confidence_level: float = 0.8,
    ) -> None:
        self.error_methods = error_methods
        self.data = data
        self.params = params
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
        self.model_class = model_class
        self.multi = multi
        self.confidence_level = confidence_level
        logging.debug(
            "Initialized Empirical CI calculation with parameters. "
            "error_methods:{error_methods},"
            "data:{data},"
            "params:{params},"
            "train_percentage:{train_percentage},"
            "test_percentage:{test_percentage},"
            "sliding_steps:{sliding_steps},"
            "model_class:{model_class},"
            "multi:{multi},"
            "confidence_level:{confidence_level}".format(
                error_methods=error_methods,
                data=data,
                params=params,
                train_percentage=train_percentage,
                test_percentage=test_percentage,
                sliding_steps=sliding_steps,
                model_class=model_class,
                multi=multi,
                confidence_level=confidence_level,
            )
        )

    def run_cv(self) -> None:
        """Running the cross validation process

        run cv with given model and data
        get errors for each horizon as follows
            | horizon | error |
            |    1    |  12.3 |
            ...
        then calculate the std for each horizon
            | horizon |  std  |
            |    1    |  1.2  |
        """

        logging.info("Creating backtester object.")
        backtester = BackTesterRollingWindow(
            self.error_methods,
            self.data,
            self.params,
            self.train_percentage,
            self.test_percentage,
            self.sliding_steps,
            self.model_class,
            self.multi,
        )

        logging.info("Run backtesting.")
        backtester.run_backtest()
        self.SE = (
            pd.DataFrame(backtester.raw_errors, copy=False).transpose().std(axis=1)
        )

    def get_lr(self) -> None:
        """Fit linear regression model

        Fit linear regression model for
        std ~ horizon
        return the fitted model
        """
        y = self.SE
        if y is None:
            raise ValueError("Must call run_cv() before get_lr().")
        X = pd.DataFrame(
            {
                "horizon": np.arange(1, len(y) + 1),
                "const": np.ones(len(y), dtype=int),
            },
            copy=False,
        )
        logging.info("Fit the OLS model and get the coefs.")
        self.coefs = np.linalg.lstsq(X, y)[0]

    def get_eci(self, steps: int, freq: str = "D") -> pd.DataFrame:
        """Get empirical prediction interval

        Args:
            steps: the length of forecasting horizon.
            freq: the frequency.

        Returns:
            The dataframe of forecasted values with prediction intervals
        """

        self.freq = freq
        # get dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.run_cv()
        self.get_lr()

        # run model with all data
        m = self.model_class(self.data, self.params)
        m.fit()
        self.predicted = predicted = m.predict(steps, freq=self.freq)

        # get margin of error
        horizons = np.arange(1, steps + 1)
        coefs = self.coefs
        assert coefs is not None  # set by get_lr above
        me = stats.norm.ppf(self.confidence_level) * (horizons * coefs[0] + coefs[1])

        self.df = df = pd.DataFrame(
            {
                "time": predicted.time,
                "fcst": predicted.fcst,
                "fcst_lower": predicted.fcst - me,
                "fcst_upper": predicted.fcst + me,
            },
            copy=False,
        )
        return df

    def diagnose(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[FigSize] = None,
        linecolor: str = "b",
        linelabel: str = "Fitted",
        secolor: str = "r",
        selabel: str = "Empirical S.E.",
        legend: bool = True,
    ) -> plt.Axes:
        """Diagnose the linear model fit for SE.

        Plot the OLS fit: SE ~ Horizon

        Args:
            ax: the axes to use. If None, create one.
            figsize: if creating Axes, the figsize to use. If None, use (10, 6).
            linecolor: the line color to use.
            linelabel: the legend label to use for the line plot.
            secolor: the SE color to use.
            selabel: the legend label to use for the SE plot.
            legend: if True, add the legend.

        Returns:
            The matplotlib Axes.
        """
        se = self.SE
        coefs = self.coefs
        if coefs is None:
            raise ValueError("Must call get_lr() before diagnose().")
        assert se is not None

        if ax is None:
            if figsize is None:
                figsize = (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        x = np.arange(1, len(se) + 1)
        ax.scatter(x, se, label=selabel, color=secolor)
        predictions = coefs[0] * np.array(x) + coefs[1]
        ax.plot(x, predictions, linecolor, label=linelabel)
        if legend:
            ax.legend(loc=2)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("SE")
        ax.set_title("Diagnosis plot for SE ~ Horizon")
        fig.tight_layout()
        return ax

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[FigSize] = None,
        linecolor: str = "k",
        fcstcolor: str = "#4267B2",
        intervalcolor: str = "#4267B2",
        intervalalpha: float = 0.2,
        modelcolor: str = "r",
        modelalpha: float = 0.2,
        grid: bool = True,
    ) -> plt.Axes:
        """Make plot for model fitting with new uncertainty intervals.

        Args:
            ax: the axes to use. If None, create one.
            figsize: if creating Axes, the figsize to use. If None, use (10, 6).
            linecolor: the line color to use.
            fcstcolor: the line color to use for the forecast.
            intervalcolor: the fill color to use for the forecast interval.
            intervalalpha: the alpha to use for the forecast interval.
            modelcolor: the fill color to use for the model prediction interval.
            modelalpha: the alpha to use for the model prediction interval.
            grid: if True, plot gridlnes.

        Returns:
            The matplotlib Axes.
        """
        predicted = self.predicted
        if predicted is None:
            raise ValueError("Must call get_eci() before plot().")
        dates = self.dates
        df = self.df
        assert dates is not None and df is not None

        if ax is None:
            if figsize is None:
                figsize = (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        ax.plot(pd.to_datetime(self.data.time), self.data.value, linecolor)
        fcst_dates = dates.to_pydatetime()
        ax.plot(fcst_dates, df.fcst, ls="-", c=fcstcolor)

        ax.fill_between(
            fcst_dates,
            df.fcst_lower,
            df.fcst_upper,
            color=intervalcolor,
            alpha=intervalalpha,
        )

        # if there are default CI from the model, plot as well
        if predicted.shape[1] == 4:
            ax.fill_between(
                fcst_dates,
                predicted.fcst_lower,
                predicted.fcst_upper,
                color=modelcolor,
                alpha=modelalpha,
            )

        if grid:
            ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel(xlabel="time")
        ax.set_ylabel(ylabel="y")
        fig.tight_layout()
        return ax
