#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Dict, Any

from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detector import Detector

MIN_POINTS = 10
LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)


class BayesOnlineChangePoint(Detector):
    """
    Given an univariate time series, this class
    performs changepoint detection, i.e. it tells
    us when the time series shows a change. This is online,
    which means it gives the best estimate based on a
    lookehead number of time steps (which is the lag).

    This faithfully implements the algorithm in
    Adams & McKay, 2007. "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

    The basic idea is to see whether the new values are
    improbable, when compared to a bayesian predictive model,
    built from the previous observations.

    The parameters are:
    data: This is univariate time series data. We require more
    than 10 points, otherwise it is not very meaningful to define
    changepoints.

    lag: This specifies, how many time steps we will look ahead to
    determine the change. There is a tradeoff in setting this parameter.
    A small lag means we can detect a change really fast, which is important
    in many applications. However, this also means we will make more
    mistakes/have lower confidence since we might mistake a spike for change.

    debug: This is a boolean. If set to true, this shows additional plots.
    Currently, it shows a plot of the predicted mean and variance, after
    lag steps, and the predictive probability of the next point. If the
    results are unusual, the user should set it to true in order to
    debug.

    """
    def __init__(self, data: TimeSeriesData, lag: int = 10,
                 debug: bool = False):
        self.data = data
        self.data_values = data.value.values
        self.T = data.value.shape[0]
        self.lag = lag
        self.threshold = None
        self.debug = debug

    def detector(self, model: Any, threshold: float = 0.5,
                 changepoint_prior: float = 0.01) -> Dict[str, Any]:

        self.threshold = threshold
        self.rt_posterior = self._find_posterior(model, changepoint_prior)
        return self._construct_output(self.threshold, lag=self.lag)

    def _find_posterior(self, model: Any, changepoint_prior: float) -> np.ndarray:
        """
        This calculates the posterior distribution over changepoints.
        The steps here are the same as the algorithm described in
        Adams & McKay, 2007. https://arxiv.org/abs/0710.3742
        """
        #P(r_t|x_t)
        rt_posterior = np.zeros((self.T, self.T))

        # initialize first step
        # P(r_0=1) = 1
        rt_posterior[0, 0] = 1.
        model.update_sufficient_stats(x=self.data_values[0])
        # To avoid growing a large dynamic list, we construct a large
        # array and grow the array backwards from the end.
        # This is conceptually equivalent to array, which we insert/append
        # to the beginning - but avoids reallocating memory.
        message = np.zeros(self.T)
        m_ptr = -1

        # set up arrays for debugging
        self.pred_mean_arr = np.zeros((self.T, self.T))
        self.pred_var_arr = np.zeros((self.T, self.T))
        self.next_pred_prob = np.zeros((self.T, self.T))

        # Calculate the log priors once outside the for-loop.
        log_cp_prior = np.log(changepoint_prior)
        log_om_cp_prior = np.log(1. - changepoint_prior)

        # from the second step onwards
        for i in range(1, self.T):
            this_pt = self.data_values[i]

            #P(x_t | r_t-1, x_t^r)
            # this arr has a size of t, each element says what is the predictive prob.
            # of a point, it the current streak began at t
            # Step 3 of paper
            pred_arr = model.pred_prob(t=i, x=this_pt)

            # record the mean/variance/prob for debugging
            if self.debug:
                pred_mean = model.pred_mean(t=i, x=this_pt)
                pred_var = model.pred_var(t=i, x=this_pt)
                self.pred_mean_arr[i, 0 : i] = pred_mean
                self.pred_var_arr[i, 0 : i] = pred_var
                self.next_pred_prob[i, 0 : i] = pred_arr

            # calculate prob that this is a changepoint, i.e. r_t = 0
            # step 5 of paper
            # this is elementwise multiplication of pred and message
            log_change_point_prob = np.logaddexp.reduce(
                pred_arr + message[self.T + m_ptr: self.T] + log_cp_prior
            )

            # step 4
            # log_growth_prob = pred_arr + message + np.log(1. - changepoint_prior)
            message[self.T + m_ptr: self.T] = pred_arr + message[self.T + m_ptr: self.T] + log_om_cp_prior

            #P(r_t, x_1:t)
            # log_joint_prob = np.append(log_change_point_prob, log_growth_prob)
            m_ptr -= 1
            message[self.T + m_ptr] = log_change_point_prob

            #calculate evidence, step 6
            #(P(x_1:t))
            # log_evidence = logsumexp(log_joint_prob)
            #
            # We use two facts here to make this more efficient:
            #
            #    (1) log(e^(x_1+c) + ... + e^(x_n+c))
            #            = log(e^c . (e^(x_1) + ... + e^(x_n)))
            #            = c + log(e^(x_1) + ... + e^(x_n))
            #
            #    (2) log(e^x_1 + e^x_2 + ... + e^x_n)                        [Associativity of logsumexp]
            #            = log(e^x_1 + e^(log(e^x_2 + ... + e^x_n)))
            #
            # In particular, we rewrite:
            #
            #    (5)   logaddexp_vec(pred_arr + message + log_cp_prior)
            #    (4+6) logaddexp_vec(append(log_change_point_prob, pred_arr + message + log_om_cp_prior))
            #
            # to
            #
            #    M = logaddexp_vector(pred_arr + message) + log_cp_prior     (using (1))
            #    logaddexp_binary(                                           (using (2))
            #        log_change_point_prob,
            #        M - log_cp_prior + log_om_cp_prior                      (using (1))
            #     )
            #
            # In this way, we avoid up to T expensive log and exp calls by avoiding
            # the repeated calculation of logaddexp_vector(pred_arr + message)
            # while adding in only a single binary (not T length) logsumexp
            # call in return and some fast addition and multiplications.
            log_evidence = np.logaddexp(
                log_change_point_prob,
                log_change_point_prob - log_cp_prior + log_om_cp_prior
            )

            # step 7
            # log_posterior = log_joint_prob - log_evidence
            log_posterior = message[self.T + m_ptr: self.T] - log_evidence
            rt_posterior[i, 0 : (i + 1)] = np.exp(log_posterior)

            # step 8
            model.update_sufficient_stats(x=this_pt)

            # pass the joint as a message to next step
            # message = log_joint_prob
            # Message is now passed implicitly - as we set it directly above.

        return rt_posterior

    def plot(self, threshold: float = None, lag: int = None):

        if threshold is None:
            threshold = self.threshold

        if lag is None:
            lag = self.lag

        #do some work to define the changepoints
        cp_output = self._construct_output(threshold=threshold, lag=lag)
        change_points = cp_output['change_points']
        y_min_cpplot = np.min(self.data.value.values)
        y_max_cpplot = np.max(self.data.value.values)

        sns.set()

        # Plot the time series
        plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(211)

        ax1.plot(self.data.time.values, self.data.value.values, 'r-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Values')

        # plot change points on the time series
        ax1.vlines(x=change_points, ymin=y_min_cpplot, ymax=y_max_cpplot,
                  colors='b', linestyles='dashed')

        # if in debugging mode, plot the mean and variance as well
        if self.debug:
            x_debug = list(range(1, self.T - self.lag))
            y_debug_mean = self.pred_mean_arr[lag + 1:self.T, lag]
            y_debug_uv = (self.pred_mean_arr[lag + 1:self.T, lag]
                          + self.pred_var_arr[lag + 1:self.T, lag])

            y_debug_lv = (self.pred_mean_arr[lag + 1:self.T, lag]
                          - self.pred_var_arr[lag + 1:self.T, lag])

            ax1.plot(x_debug, y_debug_mean, 'k-')
            ax1.plot(x_debug, y_debug_uv, 'k--')
            ax1.plot(x_debug, y_debug_lv, 'k--')

        ax2 = plt.subplot(212, sharex=ax1)

        cp_plot_x = list(range(0, self.T - self.lag))
        cp_plot_y = np.copy(self.rt_posterior[self.lag:self.T, self.lag])
        # handle the fact that first point is not a changepoint
        cp_plot_y[0] = 0.

        ax2.plot(cp_plot_x, cp_plot_y)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Changepoint Probability')

        # if debugging, we also want to show the predictive probabities
        if self.debug:
            plt.figure(figsize=(10, 4))
            plt.plot(
                list(range(1, self.T - self.lag)),
                self.next_pred_prob[lag + 1:self.T, lag],
                'k-'
            )
            plt.xlabel('Time')
            plt.ylabel('Log Prob. Density Function')
            plt.title('Debugging: Predicted Probabilities')

    def _construct_output(self, threshold: float, lag: int) -> Dict[str, Any]:
        # till lag, prob = 0, so prepend array with zeros
        change_prob = np.hstack(
            (self.rt_posterior[self.lag:self.T, self.lag],
             np.zeros(self.lag))
        )

        change_points = np.where(change_prob > threshold)[0]
        # handle the fact that the first point is not a changepoint
        if len(change_points) > 0 and change_points[0] == 0:
            change_points = np.delete(change_points, 0)
        output = {
            'change_prob': change_prob,
            'change_points': change_points
        }

        return output

    def adjust_parameters(self, threshold: float, lag: int) -> Dict[str, Any]:
        """
        if the preset parameters are not giving the desired result,
        the user can adjust the parameters. Since the algorithm
        calculates changepoints for all lags, we can see how
        changepoints look like for other lag/threshold
        """
        cp_output = self._construct_output(threshold=threshold, lag=lag)
        self.plot(threshold=threshold, lag=lag)

        return cp_output


def check_data(data: TimeSeriesData):
    if data.value.shape[0] < MIN_POINTS:
        raise ValueError(f"""
            Data must have {MIN_POINTS} points,
            it only has {data.shape[0]} points
            """)

    # For now, we only support univariate time series
    if not isinstance(data.value, pd.Series):
        msg = "Only support univariate time series, but get {type}.".format(
            type=type(data.value)
        )
        raise ValueError(msg)


class NormalKnownPrec(object):

    def __init__(self, data: TimeSeriesData = None, empirical: bool = True,
                 mean_prior: float = None, mean_prec_prior: float = None,
                 known_prec: float = None):
        """
        This model is the Normal-Normal model, with known precision
        It is specified in terms of precision for convenience.
        """
        # \mu \sim N(\mu0, \frac{1}{\lambda0})
        # x \sim N(\mu,\frac{1}{\lambda})

        # hyper parameters for mean and precision
        self.mu_0 = mean_prior
        self.lambda_0 = mean_prec_prior
        self.lambda_val = known_prec

        # For efficiency, we simulate a dynamically growing list with
        # insertions at the start, by a fixed size array with a pointer
        # where we grow the array from the end of the array. This
        # makes insertions constant time and means we can use
        # vectorized computation throughout.
        self._maxT = len(data)
        self._mean_arr_num = np.zeros(self._maxT)
        self._mean_arr = np.zeros(self._maxT)
        self._prec_arr = np.zeros(self._maxT)
        self._std_arr = np.zeros(self._maxT)
        self._ptr = 0

        # if priors are going to be decided empirically,
        # we ignore these settings above
        # Also, we need to pass on the data in this case
        if empirical:
            check_data(data)
            self._find_empirical_prior(data)

    def _find_empirical_prior(self, data: TimeSeriesData):
        """
        if priors are not defined, we take an empirical Bayes
        approach and define the priors from the data
        """

        data_arr = data.value

        # best guess of mu0 is data mean
        self.mu_0 = data_arr.mean()

        # variance of the mean: \lambda_0 = \frac{N}{\sigma^2}
        self.lambda_0 = 1. / data_arr.var()

        # to find the variance of the data we just look at small
        # enough windows such that the mean won't change between
        window_size = 10
        var_arr = data_arr.rolling(window_size).var()[window_size - 1 :]

        self.lambda_val = 1. / var_arr.mean()

        # print("mu_0:", self.mu_0 )
        # print("lambda_0:", self.lambda_0)
        # print("lambda_val:", self.lambda_val)

    @staticmethod
    def _norm_logpdf(x, mean, std):
        """
        Hardcoded version of scipy.norm.logpdf.
        This is hardcoded because scipy version is slow due to checks +
        uses log(pdf(...)) - which wastefully computes exp(..) and log(...).
        """

        return -np.log(std) - LOG_SQRT2PI - 0.5 * ((x - mean) / std)**2

    def pred_prob(self, t: int, x: float) -> np.ndarray:
        """
        t is the time, x is the new data point
        We will give log predictive probabilities for
        changepoints that started at times from 0 to t

        This posterior predictive is from
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        equation 36
        """

        pred_arr = self._norm_logpdf(
            x,
            self._mean_arr[self._maxT + self._ptr : self._maxT + self._ptr + t],
            self._std_arr[self._maxT + self._ptr : self._maxT + self._ptr + t]
        )
        return pred_arr

    def pred_mean(self, t: int, x: float) -> np.ndarray:
        return self._mean_arr[self._maxT + self._ptr : self._maxT + self._ptr + t]

    def pred_var(self, t: int, x: float) -> np.ndarray:
        return self._std_arr[self._maxT + self._ptr : self._maxT + self._ptr + t]

    def update_sufficient_stats(self, x: float):
        """
        We will store the sufficient stats for
        a streak starting at times 0, 1, ....t

        This is eqn 29 and 30 in Kevin Murphy's note
        """
        # \lambda = \lambda_0 + n * \lambda
        # hence, online, at each step: lambda[i] = lambda[i-1] + 1* lambda

        # for numerator of the mean.
        # n*\bar{x}*\lambda + \mu_0 * \lambda_0

        # So, online we add x*\lambda to the numerator from the previous step

        # I think we can do it online, but I will need to think more
        # for now we'll just keep track of the sum

        # update the precision array
        self._prec_arr[self._maxT + self._ptr : self._maxT] += self.lambda_val

        # update the numerator of the mean array
        self._mean_arr_num[self._maxT + self._ptr : self._maxT] += x * self.lambda_val

        # Grow list (backwards from the end of the array for efficiency)
        self._ptr -= 1

        self._prec_arr[self._ptr] = self.lambda_0 + 1. * self.lambda_val

        self._std_arr[self._maxT + self._ptr : self._maxT] = np.sqrt(
            1. / self._prec_arr[self._maxT + self._ptr : self._maxT]
            + 1. / self.lambda_val
        )

        self._mean_arr_num[self._ptr] = (x * self.lambda_val + self.mu_0 * self.lambda_0)

        # update the mean array itself
        self._mean_arr[self._maxT + self._ptr : self._maxT] = (
            self._mean_arr_num[self._maxT + self._ptr : self._maxT]
            / self._prec_arr[self._maxT + self._ptr : self._maxT]
        )
