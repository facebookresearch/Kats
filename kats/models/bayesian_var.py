# Copyright (c) Facebook, Inc. and its affiliates.
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
from datetime import datetime, timedelta
from typing import Dict, Tuple

import kats.models.model as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from numpy.linalg import inv  # @manual
from scipy.linalg import block_diag  # @manual


@dataclass
class BayesianVARParams(Params):
    """Parameter class for Bayesian VAR model

    Attributes:
        p: Historical lag to use
        Below parameters are hyperparameters in the covariance matrix for coefficient prior.
        See page 5 in http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf for more details.
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
        assert self.p > 0, f"Lag order should be positive, but got {self.p}"
        assert self.phi_0 > 0
        assert self.phi_1 > 0
        assert self.phi_2 > 0
        assert self.phi_3 > 0
        assert self.phi_1 <= 1, (
            "Parameter phi_1 should generally be <= 1. "
            "See page 5 of http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf"
            "for more details."
        )


class BayesianVAR(m.Model):
    """
    Model class for bayesian VAR

    This class provides fit, predict, and plot methods for bayesian VAR model

    Attributes:
        data: the input time series data as `TimeSeriesData`
        params: the parameter class defined with `BayesianVARParams`
    """

    def __init__(self, data: TimeSeriesData, params: BayesianVARParams) -> None:
        # Ignore the input time column and re-index to 0...T
        copy_data = data.to_dataframe()
        self.start_date = copy_data.time[0]
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        copy_data.time = pd.RangeIndex(0, len(copy_data))
        copy_data = TimeSeriesData(copy_data)

        self.time_freq = BayesianVAR._check_get_freq(
            copy_data
        )  # check for consistent frequency
        self.data = copy_data

        self.X, self.Y = BayesianVAR._convert_timeseries_np(copy_data)
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

        self.fitted = False

        logging.info(
            "Initializing Bayesian VAR model with: "
            f"BVAR(p={self.p}, m={self.m}, r={self.r}, T={self.T}, N={self.N}, "
            f"phi_0={self.phi_0}, phi_1={self.phi_1}, "
            f"phi_2={self.phi_2}, phi_3={self.phi_3})"
        )

    @staticmethod
    def _check_get_freq(data) -> None:
        time_diff = data.time.diff().dropna()
        diff_unique = time_diff.unique()

        assert len(diff_unique) == 1, (
            f"Frequency of metrics is not constant: {diff_unique} "
            "Please check for missing or duplicate values"
        )

        return diff_unique.item()

    @staticmethod
    def _convert_timeseries_np(
        timeseries: TimeSeriesData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data_df = timeseries.to_dataframe()
        Y = data_df.drop(columns=["time"]).to_numpy().T

        m, T = Y.shape
        X = np.expand_dims(pd.RangeIndex(0, len(timeseries)), axis=0)

        return X, Y

    def _get_training_residuals(self):
        times = []
        residuals = []

        logging.info(
            f"Performing one-step ahead forecasting on history on from t={self.p} to t={self.T-1}."
        )
        # create dataframe with each column corresponding to the residual
        for t in range(self.p, self.T):
            point_pred = self._evaluate_point_t(self.X, self.Y, t)
            time = self.X[:, t].item()
            times.append(time)
            residuals.append(self.Y[:, t] - point_pred)
        times_new = [self.start_date + timedelta(days=x) for x in times]
        df_resid = pd.DataFrame(
            residuals, index=times_new, columns=self.data.value.columns
        )

        return df_resid

    def fit(self) -> None:
        """Fit Bayesian VAR model"""

        # pyre-fixme[16]: `BayesianVAR` has no attribute `sigma_ols`.
        self.sigma_ols = self._compute_sigma_ols()

        mu_prior = np.zeros((self.m, self.N))
        for i in range(self.m):
            mu_prior[i, self.p * i] = 1
        mu_prior = mu_prior.flatten()

        v_prior = self._construct_v_prior()

        Z_sig_Z_sum = 0
        Z_sig_y_sum = 0

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
                self.num_mu_coefficients,
                self.num_mu_coefficients,
            ) == z_sum_term.shape, f"Expected {(self.num_mu_coefficients, self.num_mu_coefficients)}, got {z_sum_term.shape}"
            assert (
                self.num_mu_coefficients,
            ) == y_sum_term.shape, (
                f"Expected {(self.num_mu_coefficients,)}, got {y_sum_term.shape}"
            )

            Z_sig_Z_sum += z_sum_term
            Z_sig_y_sum += y_sum_term

        # pyre-fixme[16]: `BayesianVAR` has no attribute `v_posterior`.
        self.v_posterior = inv(
            inv(v_prior) + Z_sig_Z_sum
        )  # shape: [m * (m * p + r + 1)] x [m * (m * p + r + 1)]
        assert (
            self.num_mu_coefficients,
            self.num_mu_coefficients,
        ) == self.v_posterior.shape, f"Expected {(self.num_mu_coefficients, self.num_mu_coefficients)}, got {self.v_posterior.shape}"

        # pyre-fixme[16]: `BayesianVAR` has no attribute `mu_posterior`.
        self.mu_posterior = self.v_posterior @ (
            inv(v_prior) @ mu_prior + Z_sig_y_sum
        )  # shape: [m * (m * p + r + 1)] x 1
        assert (
            self.num_mu_coefficients,
        ) == self.mu_posterior.shape, (
            f"Expected {(self.num_mu_coefficients,)}, got {self.mu_posterior.shape}"
        )
        # pyre-fixme[16]: `BayesianVAR` has no attribute `resid`.
        self.resid = self._get_training_residuals()
        self.fitted = True

    def _construct_z(self, X, Y, t: int) -> np.ndarray:
        assert t >= self.p, f"Need t={t} >= p={self.p}."
        assert self.r == X.shape[0]
        assert self.m == Y.shape[0]

        new_yt = np.fliplr(Y[:, t - self.p : t]).flatten()
        z = np.concatenate(
            [new_yt, X[:, t].T, np.array([1])], axis=0
        )  # shape: [(m * p + r + 1) x 1]

        assert (self.N,) == z.shape, f"Expected {(self.N,)} but got {z.shape}"

        return z

    def _construct_Zt(self, X, Y, t: int) -> np.ndarray:
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

    def _sigma_ijl(self, i, j, lag, variance, is_exogenous) -> float:
        """
        Taken from page 5 of http://apps.eui.eu/Personal/Canova/Articles/ch10.pdf
        """

        def h(x):
            return x ** self.phi_3

        if i == j:
            return self.phi_0 / h(lag)
        elif is_exogenous:
            return self.phi_0 * self.phi_2
        else:  # endogenous variable j
            return self.phi_0 * (self.phi_1 / h(lag)) * (variance[j] / variance[i])

    def _construct_v_prior(self) -> np.ndarray:
        cov = np.zeros((self.num_mu_coefficients, self.num_mu_coefficients))

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
            element_ind == self.num_mu_coefficients
        ), f"Final element: {element_ind}, expected: {self.num_mu_coefficients}"

        return cov  # shape: [m * (m * p + r + 1)] x [m * (m * p + r + 1)] matrix

    def _evaluate_point_t(self, X_new, Y_new, t) -> np.ndarray:
        assert t >= self.p, f"Need t={t} > p={self.p}."

        Z_t = self._construct_Zt(X_new, Y_new, t)
        # pyre-fixme[16]: `BayesianVAR` has no attribute `mu_posterior`.
        point_prediction = Z_t @ self.mu_posterior  # shape [m x 1]

        assert (self.m,) == point_prediction.shape

        return point_prediction

    def _look_ahead_step(
        self, X_ahead, Y_curr
    ) -> np.ndarray:  # Y_curr has one less element than X_ahead
        assert Y_curr.shape[1] + 1 == X_ahead.shape[1]
        t_ahead = X_ahead.shape[1] - 1  # -1 for 0-indexed array

        Z_t = self._construct_Zt(X_ahead, Y_curr, t_ahead)
        # pyre-fixme[16]: `BayesianVAR` has no attribute `mu_posterior`.
        look_ahead_pred = Z_t @ self.mu_posterior  # shape [m x 1]

        assert (self.m,) == look_ahead_pred.shape

        return look_ahead_pred

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history=False, verbose=False
    ) -> Dict[str, TimeSeriesData]:
        """Predict with the fitted VAR model

        Args:
            steps: Number of time steps to forecast
            include_history: return fitted values also

        Returns:
            Disctionary of predicted results for each metric. Each metric result
            has following columns: `time`, `fcst`, `fcst_lower`, and `fcst_upper`
            Note confidence intervals of forecast are not yet implemented.
        """

        assert self.fitted, "Please fit the model before forecasting with .fit()"

        times = []
        forecast_vals = []

        if include_history:
            logging.info(
                f"Performing one-step ahead forecasting on history on from t={self.p} to t={self.T-1}."
            )

            for t in range(self.p, self.T):
                point_pred = self._evaluate_point_t(self.X, self.Y, t)
                time = self.X[:, t].item()

                if verbose:
                    logging.info(
                        f"Performing one-step ahead forecasting with history on t={time}."
                    )

                times.append(time)
                forecast_vals.append(point_pred)

        # future forecasting -- X_ahead is one time step ahead of Y_curr
        X_ahead = self.X
        Y_curr = self.Y

        logging.info(
            f"Performing future forecasting from t={self.T} to t={self.T+steps-1}."
        )

        for _t in range(self.T, self.T + steps):
            ahead_time = X_ahead[np.newaxis, :, -1] + self.time_freq
            X_ahead = np.concatenate([X_ahead, ahead_time], axis=1)
            look_ahead_pred = self._look_ahead_step(X_ahead, Y_curr)
            time = ahead_time.item()

            if verbose:
                logging.info(f"Performing future forecasting with t={time}.")

            times.append(time)
            forecast_vals.append(look_ahead_pred)

            Y_curr = np.concatenate([Y_curr, look_ahead_pred[:, np.newaxis]], axis=1)

        assert (
            len(times) > 0
        ), "Forecast produced no values. Please set steps > 0 or include_history=True."

        indiv_forecasts = {}
        forecast_length = len(times)

        logging.warning(
            "Upper and lower confidence intervals of forecast not yet implemented for Bayesian VAR model."
        )
        times_new = [self.start_date + timedelta(days=x) for x in times]

        for i, c in enumerate(self.data.value.columns):
            c_forecast = pd.DataFrame(
                {
                    "time": times_new,
                    "fcst": [forecast_vals[f_t][i] for f_t in range(forecast_length)],
                    "fcst_lower": [-1] * forecast_length,
                    "fcst_upper": [-1] * forecast_length,
                }
            )
            indiv_forecasts[c] = TimeSeriesData(c_forecast)

        # pyre-fixme[16]: `BayesianVAR` has no attribute `forecast`.
        self.forecast = indiv_forecasts
        # pyre-fixme[16]: `BayesianVAR` has no attribute `forecast_max_time`.
        self.forecast_max_time = max(times_new)

        return self.forecast

    # pyre-fixme[14]: `plot` overrides method defined in `Model` inconsistently.
    # pyre-fixme[40]: Non-static method `plot` cannot override a static method
    #  defined in `m.Model`.
    def plot(self) -> None:
        """Plot forecasted results from Bayesian VAR model"""
        assert hasattr(self, "forecast"), "Please run .predict() before plotting"

        plt.figure(figsize=(20, 6))
        plt.title("Input Timeseries & Forecast")

        for i, c in enumerate(self.data.value.columns):
            color = f"C{i}"
            plt.plot(self.data.time, self.data.value[c], c=color)
            # pyre-fixme[16]: `BayesianVAR` has no attribute `forecast`.
            plt.plot(self.forecast[c].time, self.forecast[c].value, "--", c=color)

    @property
    def sigma_u(self) -> pd.DataFrame:
        return pd.DataFrame(
            # pyre-fixme[16]: `BayesianVAR` has no attribute `sigma_ols`.
            self.sigma_ols,
            index=self.data.value.columns,
            columns=self.data.value.columns,
        )

    @property
    def k_ar(self) -> int:
        return self.p
