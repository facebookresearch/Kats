#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import Params, TimeSeriesData
from infrastrategy.kats.utils.backtesters import BackTesterRollingWindow
from scipy import stats


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


class EmpConfidenceInt:
    """"class for empirical confidence interval

    Steps:
    1. Run K-fold CV for a given model and data,
       each fold contains h (horizon) time ahead
    2. For each horizon, calculate the Std of K error terms (S.E)
    3. Fit linear model: S.E. ~ Horizon
    4. Estimate the S.E. for each horizon for the true future
    5. Lower/Upper = Point Estimate -/+ Z_Score * S.E.
    """

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
        confidence_level: float = 0.8,
        **kwargs
    ):
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

    def run_cv(self):
        """
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
        self.SE = pd.DataFrame(backtester.raw_errors).transpose().std(axis=1)

    def get_lr(self,):
        """
        Fit linear regression model for
        std ~ horizon
        return the fitted model
        """
        X = pd.DataFrame(
            {
                "horizon": [x + 1 for x in range(len(self.SE))],
                "const": [1] * len(self.SE),
            }
        )
        y = self.SE
        logging.info("Fit the OLS model and get the coefs.")
        self.coefs = np.linalg.lstsq(X, y)[0]

    def get_eci(self, steps, **kwargs):
        """
        get empirical confidence interval
        """
        self.freq = kwargs.get("freq", "D")
        # get dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods

        self.run_cv()
        self.get_lr()

        # run model with all data
        m = self.model_class(self.data, self.params)
        m.fit()
        self.predicted = m.predict(steps, freq=self.freq)

        # get margin of error
        horizons = np.array([x + 1 for x in range(steps)])
        me = stats.norm.ppf(self.confidence_level) * (
            horizons * self.coefs[0] + self.coefs[1]
        )

        self.df = pd.DataFrame(
            {
                "time": self.predicted.time,
                "fcst": self.predicted.fcst,
                "fcst_lower": self.predicted.fcst - me,
                "fcst_upper": self.predicted.fcst + me,
            }
        )
        return self.df

    def diagnose(self):
        """
        Diagnose the linear model fit for SE
        Plot the OLS fit: SE ~ Horizon
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        x = [x + 1 for x in range(len(self.SE))]
        ax.scatter(x, self.SE, label="Empirical S.E.", color="r")
        predictions = self.coefs[0] * np.array(x) + self.coefs[1]
        ax.plot(x, predictions, "b", label="Fitted")
        ax.legend(loc=2)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("SE")
        ax.set_title("Diagnosis plot for SE ~ Horizon")
        fig.tight_layout()

    def plot(self):
        """
        Make plot for model fitting with new uncertainty intervals
        """
        logging.info("Generating chart for forecast result with emp. conf. intervals.")
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.plot(pd.to_datetime(self.data.time), self.data.value, "k")
        fcst_dates = self.dates.to_pydatetime()
        ax.plot(fcst_dates, self.df.fcst, ls="-", c="#4267B2")

        ax.fill_between(
            fcst_dates,
            self.df.fcst_lower,
            self.df.fcst_upper,
            color="#4267B2",
            alpha=0.2,
        )

        # if there are default CI from the model, plot as well
        if self.predicted.shape[1] == 4:
            ax.fill_between(
                fcst_dates,
                self.predicted.fcst_lower,
                self.predicted.fcst_upper,
                color="r",
                alpha=0.2,
            )

        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel(xlabel="time")
        ax.set_ylabel(ylabel="y")
        fig.tight_layout()
