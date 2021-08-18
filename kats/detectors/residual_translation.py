# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Detectors based on predictors, basically work as follows:
calculate the residual (i.e., difference between predicted and current value),
and translate it into a false-alarm probability by how large it is.
This is often done by assuming that residuals are distributed normally.
In practice, the residuals are often non-normal (sometimes even being
asymmetric). This module “learns” the distribution of the residual (using kernel
density estimation), and outputs a false-alarm probability based on it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KernelDensity


class KDEResidualTranslator:
    _kde: Optional[KernelDensity] = None

    def __init__(
        self, ignore_below_frac: float = 0, ignore_above_frac: float = 1
    ) -> None:
        """
        Translates residuals (difference between outcome and prediction)
        to false-alarm probability using kernel density estimation
        on the residuals.

        Args:
            ignore_below_frac: Lower quantile to ignore during training
                (makes the translator more robust to outliers); default 0.
            ignore_above_frac: Upper quantile to ignore during training
                (makes the translator more robust to outliers); default 1.

        Examples:

        .. code-block:: py

            # ts_data is some data
            y = ts_data
            # We create a prediction by a rolling window of length 7
            yhat = pd.DataFrame({
                "value": self._y.value.rolling(7).mean().shift(1),
                "time": self._y.time})
            yhat = TimeSeriesData(yhat)

            trn = KDEResidualTranslator()

            # We can transform outcomes to probabilities using y and y_hat
            trn = trn.fit(y=y, yhat=yhat)
            proba = trn.predict(y)

            # We can transform outcomes to probabilities using residuals
            residual = self._y - self._yhat
            trn = trn.fit(residual=residual)
            proba = trn.predict(residual=residual)
        """
        if ignore_below_frac < 0 or ignore_above_frac > 1:
            raise ValueError("Illegal ignore fractions")
        if ignore_below_frac > ignore_above_frac:
            raise ValueError("Illegal ignore fractions")
        self._ignore_below_frac = ignore_below_frac
        self._ignore_above_frac = ignore_above_frac

    def fit(
        self,
        y: Optional[TimeSeriesData] = None,
        yhat: Optional[TimeSeriesData] = None,
        yhat_lower: Optional[TimeSeriesData] = None,
        yhat_upper: Optional[TimeSeriesData] = None,
        residual: Optional[TimeSeriesData] = None,
    ) -> KDEResidualTranslator:
        """
        Fits a dataframe to a model of the residuals.

        Args:
            df: A pandas DataFrame containg the following columns:

                1. Either
                    a. `residual`, or
                    b. `y` and `yhat` with optionally both `yhat_lower` and
                       `yhat_upper`
                2. At most one of `ds` and `ts`
        """
        residual = self._get_residual(y, yhat, yhat_lower, yhat_upper, residual)

        value = residual.value
        mask = value > value.quantile(self._ignore_below_frac)
        # pyre-fixme[6]: Expected `DataFrame` for 1st param but got
        #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
        mask &= value < value.quantile(self._ignore_above_frac)
        value = value[mask]

        kde = KernelDensity(bandwidth=10.0, kernel="gaussian")
        extent = value.quantile(0.95) - value.quantile(0.05)
        params = {
            "kernel": ["gaussian"],
            "bandwidth": np.linspace(extent / 1000, extent / 10, 1000),
        }
        search = RandomizedSearchCV(
            kde, params, random_state=0, scoring=lambda k, x: k.score_samples(x).sum()
        )
        best_params = search.fit(value.to_frame()).best_params_
        kde = KernelDensity(**best_params)
        kde.fit(value.to_frame())
        self._kde = kde
        return self

    @property
    def kde_(self) -> Optional[KernelDensity]:
        """
        Returns:
            KernelDensity object fitted to the residuals.
        """
        return self._kde

    def predict_proba(
        self,
        y: Optional[TimeSeriesData] = None,
        yhat: Optional[TimeSeriesData] = None,
        yhat_lower: Optional[TimeSeriesData] = None,
        yhat_upper: Optional[TimeSeriesData] = None,
        residual: Optional[TimeSeriesData] = None,
    ) -> TimeSeriesData:
        """
        Predicts the probability of a residual

        Args:
            df: A pandas DataFrame containg the following columns:

                1. Either
                    a. `residual`, or
                    b. `y` and `yhat` with optionally both `yhat_lower` and
                        `yhat_upper`
                2. At most one of `ds` and `ts`

        Returns:
            A series where there is a probability corresponding to
                each instance (row) in the input.
        """
        proba = self.predict_log_proba(y, yhat, yhat_lower, yhat_upper, residual)
        proba.value = np.exp(proba.value)
        return proba

    def predict_log_proba(
        self,
        y: Optional[TimeSeriesData] = None,
        yhat: Optional[TimeSeriesData] = None,
        yhat_lower: Optional[TimeSeriesData] = None,
        yhat_upper: Optional[TimeSeriesData] = None,
        residual: Optional[TimeSeriesData] = None,
    ) -> TimeSeriesData:
        """
        Predicts the natural-log probability of a residual

        Args:
            df: A pandas DataFrame containg the following columns:

                1. Either
                    a. `residual`, or
                    b. `y` and `yhat` with optionally both `yhat_lower` and
                        `yhat_upper`
                2. At most one of `ds` and `ts`

        Returns:
            A series where there is a probability corresponding to
                each instance (row) in the input.
        """
        residual = self._get_residual(y, yhat, yhat_lower, yhat_upper, residual)
        for _ in range(30):
            print(type(residual))

        log_proba = pd.DataFrame(
            {
                # pyre-fixme[16]: `KDEResidualTranslator` has no attribute `_kde`.
                "value": self._kde.score_samples(residual.value.to_frame()),
                "time": residual.time,
            }
        )

        return TimeSeriesData(log_proba)

    def _get_residual(
        self,
        y: Optional[TimeSeriesData],
        yhat: Optional[TimeSeriesData],
        yhat_lower: Optional[TimeSeriesData],
        yhat_upper: Optional[TimeSeriesData],
        residual: Optional[TimeSeriesData],
    ) -> TimeSeriesData:
        if yhat is not None:
            if y is None:
                raise ValueError("Must include y if supplying yhat")
            if residual is not None:
                raise ValueError("Must not include residuals if supplying yhat")
            residual = y - yhat
            if (yhat_lower is not None) != (yhat_upper is not None):
                raise ValueError(
                    "Must supply either both yhat_lower and yhat_upper" "or neither"
                )
            if yhat_lower is not None:
                assert yhat_upper is not None
                assert yhat_lower is not None
                residual /= yhat_upper - yhat_lower
        elif residual is not None:
            if any(c is not None for c in [y, yhat, yhat_lower, yhat_upper]):
                raise ValueError(
                    "Must not include y, yhat, yhat_lower, yhat_upper"
                    "if supplying residuals"
                )
        else:
            raise ValueError("Must supply y and yhat or residual")

        nulls = residual.value.isnull()
        residual.value = residual.value[~nulls]
        residual.time = residual.time[~nulls]
        return residual
