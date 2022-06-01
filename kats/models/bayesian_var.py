# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
 Bayesian estimation of Vector Autoregressive Model using
 Minnesota prior on the coefficient matrix. This version is
 useful for regularization when they are too many coefficients
 to be estimated.

 Implementation inspired by the following two articles/papers:
    https://www.mathworks.com/help/econ/normalbvarm.html#mw_4a1ab118-9ef3-4380-8c5a-12b848254117
    http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf (page 5)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import kats.models.model as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import _log_error, Params, TimeSeriesData
from numpy.linalg import inv
from scipy.linalg import block_diag


@dataclass
class BayesianVARParams(Params):
    """Parameter class for Bayesian VAR model

    Attributes:
        p: Historical lag to use
        Below parameters are hyperparameters in the covariance matrix for
        coefficient prior. See page 5 in
        http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf for more details.
        phi_0: tightness on the variance of the first lag
        phi_1: relative tightness of other variables
        phi_2: relative tightness of the exogenous variables
        phi_3: decay with lag is parameterized as lag^phi_3
    """

    p: int = 5
    phi_0: float = 0.02
    phi_1: float = 0.25
    phi_2: float = 20
    phi_3: float = 3

    def validate_params(self) -> None:
        if self.p <= 0:
            raise ValueError(f"Lag order must be positive, but got {self.p}")
        if self.phi_0 <= 0:
            raise ValueError(f"phi_0 must be positive, but got {self.phi_0}")
        if self.phi_1 <= 0 or self.phi_1 > 1:
            raise ValueError(
                f"phi_1 must be positive and at most 1, but got {self.phi_1}. "
                "See page 5 of http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf"
                "for more details."
            )
        if self.phi_2 <= 0:
            raise ValueError(f"phi_2 must be positive, but got {self.phi_2}")
        if self.phi_3 <= 0:
            raise ValueError(f"phi_3 must be positive, but got {self.phi_3}")


class BayesianVAR(m.Model[BayesianVARParams]):
    """
    Model class for bayesian VAR

    This class provides fit, predict, and plot methods for bayesian VAR model

    Attributes:
        data: the input time series data as `TimeSeriesData`
        params: the parameter class defined with `BayesianVARParams`
    """

    sigma_ols: Optional[np.ndarray] = None
    v_posterior: Optional[np.ndarray] = None
    mu_posterior: Optional[np.ndarray] = None
    resid: Optional[pd.DataFrame] = None
    forecast: Optional[Dict[str, TimeSeriesData]] = None
    forecast_max_time: Optional[datetime] = None
    data: TimeSeriesData
    time_freq: str
    X: np.ndarray
    Y: np.ndarray
    m: int
    T: int
    r: int
    p: int
    phi_0: float
    phi_1: float
    phi_2: float
    phi_3: float
    N: int
    num_mu_coefficients: int
    fitted: bool = False
    forecast_vals: Optional[List[np.ndarray]] = None

    def __init__(self, data: TimeSeriesData, params: BayesianVARParams) -> None:
        if data.is_univariate():
            msg = "Bayesian VAR Model only accepts multivariate time series."
            raise _log_error(msg)

        self.data = data
        self.time_freq = BayesianVAR._check_get_freq(data)
        self.X, self.Y = BayesianVAR._convert_timeseries_np(data)
        assert (
            self.X.shape[1] == self.Y.shape[1]
        ), "Expected same amount of data on time axis for X and Y"

        self.m, self.T = self.Y.shape
        self.r = self.X.shape[0]
        self.p = params.p
        self.phi_0 = params.phi_0
        self.phi_1 = params.phi_1
        self.phi_2 = params.phi_2
        self.phi_3 = params.phi_3
        self.N = (self.m * self.p) + self.r + 1
        self.num_mu_coefficients = self.m * self.N

        logging.info(f"Initializing Bayesian VAR model with: {self}")

    def __str__(self) -> str:
        return (
            f"BayesianVAR(p={self.p}, m={self.m}, r={self.r}, T={self.T}, "
            f"N={self.N}, phi_0={self.phi_0}, phi_1={self.phi_1}, "
            f"phi_2={self.phi_2}, phi_3={self.phi_3})"
        )

    @staticmethod
    def _check_get_freq(data: TimeSeriesData) -> str:
        """Checks for consistent time frequency."""
        freq = pd.infer_freq(data.time)
        if freq is None:
            raise ValueError(
                "Unable to infer time series frequency. Please check for "
                "missing or duplicate times or irregularly-spaced times."
            )

        return freq

    @staticmethod
    def _convert_timeseries_np(
        timeseries: TimeSeriesData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data_df = timeseries.to_dataframe()
        Y = data_df.drop(columns=[timeseries.time_col_name]).to_numpy().T
        X = np.expand_dims(pd.RangeIndex(0, len(timeseries)), axis=0)
        return X, Y

    def _get_training_residuals(self) -> pd.DataFrame:
        residuals = []

        # create dataframe with each column corresponding to the residual
        p = self.p
        T = self.T
        times = self.data.time.iloc[p:T]
        self.forecast_vals = forecast_vals = []
        if times.empty:
            logging.info(
                "Performing one-step ahead forecasting on history from step "
                f"{self.p} to {self.T-1} (t={times[0]} to {times[-1]}) inclusive."
            )
        for t in range(p, T):
            point_pred = self._evaluate_point_t(self.X, self.Y, t)
            forecast_vals.append(point_pred)
            residuals.append(self.Y[:, t] - point_pred)
        df_resid = pd.DataFrame(
            residuals, index=times, columns=self.data.value.columns, copy=False
        )

        return df_resid

    def fit(self) -> None:
        """Fit Bayesian VAR model"""

        self.sigma_ols = self._compute_sigma_ols()

        mu_prior = np.zeros((self.m, self.N))
        for i in range(self.m):
            mu_prior[i, self.p * i] = 1
        mu_prior = mu_prior.flatten()

        v_prior = self._construct_v_prior()

        Z_sig_Z_sum = 0
        Z_sig_y_sum = 0

        num_mu = self.num_mu_coefficients
        for t in range(self.p, self.T):
            Z_t = self._construct_Zt(
                self.X, self.Y, t
            )  # shape: m x [m * (m * p + r + 1)]

            z_sum_term = (
                Z_t.T @ inv(self.sigma_ols)
            ) @ Z_t  # shape: [m * (m * p + r + 1)] x [m * (m * p + r + 1)]
            y_sum_term = (Z_t.T @ inv(self.sigma_ols)) @ self.Y[
                :, t
            ]  # shape: [m * (m * p + r + 1)] x 1

            assert (
                num_mu,
                num_mu,
            ) == z_sum_term.shape, (
                f"Expected {(num_mu, num_mu)}, got {z_sum_term.shape}"
            )
            assert (
                num_mu,
            ) == y_sum_term.shape, f"Expected {(num_mu,)}, got {y_sum_term.shape}"

            Z_sig_Z_sum += z_sum_term
            Z_sig_y_sum += y_sum_term

        v_posterior = inv(
            inv(v_prior) + Z_sig_Z_sum
        )  # shape: [m * (m * p + r + 1)] x [m * (m * p + r + 1)]
        self.v_posterior = v_posterior
        assert (
            num_mu,
            num_mu,
        ) == v_posterior.shape, f"Expected {(num_mu, num_mu)}, got {v_posterior.shape}"

        mu_posterior = v_posterior @ (
            inv(v_prior) @ mu_prior + Z_sig_y_sum
        )  # shape: [m * (m * p + r + 1)] x 1
        self.mu_posterior = mu_posterior
        assert (
            num_mu,
        ) == mu_posterior.shape, f"Expected {(num_mu,)}, got {mu_posterior.shape}"
        self.resid = self._get_training_residuals()
        self.fitted = True

    def _construct_z(self, X: np.ndarray, Y: np.ndarray, t: int) -> np.ndarray:
        assert t >= self.p, f"Need t={t} >= p={self.p}."
        assert self.r == X.shape[0]
        assert self.m == Y.shape[0]

        new_yt = np.fliplr(Y[:, t - self.p : t]).flatten()
        z = np.concatenate(
            [new_yt, X[:, t].T, np.array([1])], axis=0
        )  # shape: [(m * p + r + 1) x 1]

        assert (self.N,) == z.shape, f"Expected {(self.N,)} but got {z.shape}"

        return z

    def _construct_Zt(self, X: np.ndarray, Y: np.ndarray, t: int) -> np.ndarray:
        z = self._construct_z(X, Y, t)
        Z_t = block_diag(*([z] * self.m))

        assert (
            self.m,
            self.num_mu_coefficients,
        ) == Z_t.shape, (
            f"Expected {(self.m, self.num_mu_coefficients)}, got {Z_t.shape}"
        )

        return Z_t  # shape: m x [m * (m * p + m + 1)]

    def _construct_X_OLS(self) -> np.ndarray:
        X_OLS = np.zeros((self.N, self.T - self.p))

        for t in range(self.p, self.T):
            X_OLS[:, t - self.p] = self._construct_z(
                self.X, self.Y, t
            )  # X_OLS ignores first p values

        return X_OLS

    def _compute_sigma_ols(self) -> np.ndarray:
        Y_suffix = self.Y[:, self.p :]
        X_OLS = self._construct_X_OLS()

        beta_ols = (Y_suffix @ X_OLS.T) @ inv(X_OLS @ X_OLS.T)
        sse = (Y_suffix - beta_ols @ X_OLS) @ (
            Y_suffix - beta_ols @ X_OLS
        ).T  # should produce [m x m] matrix

        assert (
            self.m,
            self.m,
        ) == sse.shape, f"Expected {(self.m, self.m)}, but got {sse.shape}"
        assert self.T > (self.m * self.p) + 1

        return sse / float(self.T - (self.m * self.p) - 1)

    def _sigma_ijl(
        self,
        i: int,
        j: Optional[int],
        lag: Optional[int],
        variance: np.ndarray,
        is_exogenous: bool,
    ) -> float:
        """
        Taken from page 5 of http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf
        """

        def h(x: float) -> float:
            return x**self.phi_3

        if i == j:
            assert lag is not None
            return self.phi_0 / h(lag)
        elif is_exogenous:
            return self.phi_0 * self.phi_2
        else:  # endogenous variable j
            assert lag is not None
            return self.phi_0 * (self.phi_1 / h(lag)) * (variance[j] / variance[i])

    def _construct_v_prior(self) -> np.ndarray:
        num_mu = self.num_mu_coefficients
        cov = np.zeros((num_mu, num_mu))

        variance = np.var(self.Y, axis=1)

        element_ind = 0

        for i in range(self.m):
            for j in range(self.m):  # iterate through the m classes of lagged variables
                for lag in range(1, self.p + 1):  # iterate through the lags
                    cov[element_ind][element_ind] = self._sigma_ijl(
                        i, j, lag, variance, is_exogenous=False
                    )
                    element_ind += 1

            for _ex in range(self.r):  # exogenous variables
                cov[element_ind][element_ind] = self._sigma_ijl(
                    i, None, None, variance, is_exogenous=True
                )
                element_ind += 1

            # constant term of 1
            cov[element_ind][element_ind] = self._sigma_ijl(
                i, None, None, variance, is_exogenous=True
            )
            element_ind += 1

        assert (
            element_ind == num_mu
        ), f"Final element: {element_ind}, expected: {num_mu}"

        return cov  # shape: [m * (m * p + r + 1)] x [m * (m * p + r + 1)] matrix

    def _evaluate_point_t(
        self, X_new: np.ndarray, Y_new: np.ndarray, t: int
    ) -> np.ndarray:
        assert t >= self.p, f"Need t={t} > p={self.p}."

        Z_t = self._construct_Zt(X_new, Y_new, t)
        point_prediction = Z_t @ self.mu_posterior  # shape [m x 1]

        assert (self.m,) == point_prediction.shape

        return point_prediction

    def _look_ahead_step(self, X_ahead: np.ndarray, Y_curr: np.ndarray) -> np.ndarray:
        # Y_curr has one less element than X_ahead
        assert Y_curr.shape[1] + 1 == X_ahead.shape[1]
        t_ahead = X_ahead.shape[1] - 1  # -1 for 0-indexed array

        Z_t = self._construct_Zt(X_ahead, Y_curr, t_ahead)
        look_ahead_pred = Z_t @ self.mu_posterior  # shape [m x 1]

        assert (self.m,) == look_ahead_pred.shape

        return look_ahead_pred

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, verbose: bool = False
    ) -> Dict[str, TimeSeriesData]:
        """Predict with the fitted VAR model.

        Args:
            steps: Number of time steps to forecast
            include_history: return fitted values also

        Returns:
            Dictionary of predicted results for each metric. Each metric result
            has following columns: `time`, `fcst`, `fcst_lower`, and `fcst_upper`
            Note confidence intervals of forecast are not yet implemented.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before predict().")
        if not steps and not include_history:
            raise ValueError(
                "Forecast produced no values. Please set steps > 0 or "
                "include_history=True."
            )

        X_ahead = self.X
        Y_curr = self.Y
        T = self.T

        if include_history:
            times = self.data.time.iloc[self.p : T].tolist()
            forecast_vals = self.forecast_vals
            assert forecast_vals is not None
        else:
            times = []
            forecast_vals = []

        if steps:
            # future forecasting -- X_ahead is one time step ahead of Y_curr
            ahead_times = pd.date_range(
                start=self.data.time.iloc[-1], periods=steps + 1, freq=self.time_freq
            )[1:]
            logging.info(
                f"Performing future forecasting from step {T} to {T+steps-1} ("
                f"t={ahead_times[0]} to t={ahead_times[-1]}) inclusive."
            )
            assert len(ahead_times) == steps

            ahead_time = X_ahead[np.newaxis, :, -1]
            for step, time in zip(range(T, T + steps), ahead_times):
                X_ahead = np.concatenate([X_ahead, ahead_time + step], axis=1)
                look_ahead_pred = self._look_ahead_step(X_ahead, Y_curr)

                if verbose:
                    logging.info(
                        f"Performing future forecasting at t={time}, step={step}."
                    )

                forecast_vals.append(look_ahead_pred)

                Y_curr = np.concatenate(
                    [Y_curr, look_ahead_pred[:, np.newaxis]], axis=1
                )

            times += ahead_times

        forecast_length = len(times)

        assert forecast_length == len(
            forecast_vals
        ), f"{forecast_length} != {len(forecast_vals)}"

        self.forecast = indiv_forecasts = {}
        self.forecast_max_time = times[-1]

        logging.warning(
            "Upper and lower confidence intervals of forecast not yet implemented "
            "for Bayesian VAR model."
        )

        for i, c in enumerate(self.data.value.columns.tolist()):
            c_forecast = pd.DataFrame(
                {
                    "time": times,
                    "fcst": [forecast_vals[f_t][i] for f_t in range(forecast_length)],
                    "fcst_lower": [-1] * forecast_length,
                    "fcst_upper": [-1] * forecast_length,
                },
                copy=False,
            )
            indiv_forecasts[c] = TimeSeriesData(c_forecast)

        return indiv_forecasts

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = "Input Timeseries & Forecast",
        ls: str = "--",
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot forecasted results from Bayesian VAR model"""
        forecast = self.forecast
        if forecast is None:
            raise ValueError("Must call predict() before plot()")
        data = self.data

        if ax is None:
            if figsize is None:
                figsize = (20, 6)
            _, ax = plt.subplots(figsize=figsize)

        ax.set_title(title)

        for i, c in enumerate(data.value.columns):
            color = f"C{i}"
            ax.plot(data.time, data.value[c], c=color)
            ax.plot(forecast[c].time, forecast[c].value, ls, c=color)

        return ax

    @property
    def sigma_u(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.sigma_ols,
            index=self.data.value.columns,
            columns=self.data.value.columns,
            copy=False,
        )

    @property
    def k_ar(self) -> int:
        return self.p
