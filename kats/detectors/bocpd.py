# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains classes and functions used for implementing
the Bayesian Online Changepoint Detection algorithm.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from kats.consts import TimeSeriesChangePoint, TimeSeriesData, SearchMethodEnum
from kats.detectors.detector import Detector
from scipy.special import logsumexp  # @manual

# pyre-ignore[21]: Could not find name `invgamma` in `scipy.stats`.
# pyre-ignore[21]: Could not find name `nbinom` in `scipy.stats`.
from scipy.stats import invgamma, linregress, norm, nbinom  # @manual

try:
    import kats.utils.time_series_parameter_tuning as tpt

    _no_ax = False
except ImportError:
    _no_ax = True

_MIN_POINTS = 10
_LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)


class BOCPDModelType(Enum):
    """Bayesian Online Change Point Detection model type.

    Describes the type of predictive model used by the
    BOCPD algorithm.
    """

    NORMAL_KNOWN_MODEL = 1
    TREND_CHANGE_MODEL = 2
    POISSON_PROCESS_MODEL = 3


class BOCPDMetadata:
    """Metadata for the BOCPD model.

    This gives information about
    the type of detector, the name of the time series and
    the model used for detection.

    Attributes:
        model: The kind of predictive model used.
        ts_name: string, name of the time series for which the detector is
            is being run.
    """

    def __init__(self, model: BOCPDModelType, ts_name: Optional[str] = None):
        self._detector_type = BOCPDetector
        self._model = model
        self._ts_name = ts_name

    @property
    def detector_type(self):
        return self._detector_type

    @property
    def model(self):
        return self._model

    @property
    def ts_name(self):
        return self._ts_name


@dataclass
class BOCPDModelParameters(ABC):
    """Data class containing data for predictive models used in BOCPD.

    Particular predictive models derive from this class.

    Attributes:
        prior_choice: list of changepoint probability priors
            over which we will search hyperparameters
        cp_prior: default prior for probability of changepoint.
        search_method: string, representing the search method
            for the hyperparameter tuning library. Allowed values
            are 'random' and 'gridsearch'.
    """

    data: Optional[TimeSeriesData] = None

    prior_choice: Dict[str, List[float]] = field(
        default_factory=lambda: {"cp_prior": [0.001, 0.002, 0.005, 0.01, 0.02]}
    )

    cp_prior: float = 0.1
    search_method: str = "random"

    def set_prior(self, param_dict: Dict[str, float]):
        """Setter method, which sets the value of the parameters.

        Currently, this sets the value of the prior probability of changepoint.

        Args:
            param_dict: dictionary of the form {param_name: param_value}.

        Returns:
            None.
        """

        if "cp_prior" in param_dict:
            self.cp_prior = param_dict["cp_prior"]


@dataclass
class NormalKnownParameters(BOCPDModelParameters):
    """Data class containing the parameters for Normal  predictive model.

    This assumes that the data comes from a normal distribution with known
    precision.

    Attributes:
        empirical: Boolean, should we derive the prior empirically. When
            this is true, the mean_prior, mean_prec_prior and known_prec
            are derived from the data, and don't need to be specified.
        mean_prior: float, mean of the prior normal distribution.
        mean_prec_prior: float, precision of the prior normal distribution.
        known_prec: float, known precision of the data.
        known_prec_multiplier: float, a multiplier of the known precision.
            This is a variable, that is used in the hyperparameter search,
            to multiply with the known_prec value.
        prior_choice: List of parameters to search, for hyperparameter tuning.
    """

    empirical: bool = True
    mean_prior: Optional[float] = None
    mean_prec_prior: Optional[float] = None
    known_prec: Optional[float] = None
    known_prec_multiplier: float = 1.0

    prior_choice: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "known_prec_multiplier": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cp_prior": [0.001, 0.002, 0.005, 0.01, 0.02],
        }
    )

    def set_prior(self, param_dict: Dict[str, float]):
        """Sets priors

        Sets the value of the prior based on the
        parameter dictionary passed.

        Args:
            param_dict: Dictionary of parameters required for
                setting the prior value.

        Returns:
            None.
        """

        if "known_prec_multiplier" in param_dict:
            self.known_prec_multiplier = param_dict["known_prec_multiplier"]
        if "cp_prior" in param_dict:
            self.cp_prior = param_dict["cp_prior"]


@dataclass
class TrendChangeParameters(BOCPDModelParameters):
    """Parameters for the trend change predictive model.

    This model assumes that the data is generated from a Bayesian
    linear model.

    Attributes:
        mu_prior: array, mean of the normal priors on the slope and intercept
        num_likelihood_samples: int, number of samples generated, to calculate
            the posterior.
        num_points_prior: int,
        readjust_sigma_prior: Boolean, whether we should readjust the Inv. Gamma
        prior for the variance, based on the data.
        plot_regression_prior: Boolean, plot prior. set as False, unless trying to
            debug.
    """

    mu_prior: Optional[np.ndarray] = None
    num_likelihood_samples: int = 100
    num_points_prior: int = _MIN_POINTS
    readjust_sigma_prior: bool = False
    plot_regression_prior: bool = False


@dataclass
class PoissonModelParameters(BOCPDModelParameters):
    """Parameters for the Poisson predictive model.

    Here, the data is generated from a Poisson distribution.

    Attributes:
        alpha_prior: prior value of the alpha value of the Gamma prior.
        beta_prior: prior value of the beta value of the Gamma prior.
    """

    alpha_prior: float = 1.0
    beta_prior: float = 0.05


class BOCPDetector(Detector):
    """Bayesian Online Changepoint Detection.

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

    Attrbutes:
        data: TimeSeriesData, data on which we will run the BOCPD algorithm.
    """

    def __init__(self, data: TimeSeriesData) -> None:
        self.data = data

        self.models: Dict[BOCPDModelType, Type[_PredictiveModel]] = {
            BOCPDModelType.NORMAL_KNOWN_MODEL: _NormalKnownPrec,
            BOCPDModelType.TREND_CHANGE_MODEL: _BayesianLinReg,
            BOCPDModelType.POISSON_PROCESS_MODEL: _PoissonProcessModel,
        }
        self.parameter_type: Dict[BOCPDModelType, Type[BOCPDModelParameters]] = {
            BOCPDModelType.NORMAL_KNOWN_MODEL: NormalKnownParameters,
            BOCPDModelType.TREND_CHANGE_MODEL: TrendChangeParameters,
            BOCPDModelType.POISSON_PROCESS_MODEL: PoissonModelParameters,
        }
        self.available_models = self.models.keys()

        self.change_prob = {}
        self._run_length_prob = {}
        self.detected_flag = False

        assert (
            self.models.keys() == self.parameter_type.keys()
        ), f"Expected equivalent models in .models and .parameter_types, but got {self.models.keys()} and {self.parameter_type.keys()}"

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(
        self,
        model: BOCPDModelType = BOCPDModelType.NORMAL_KNOWN_MODEL,
        model_parameters: Union[None, BOCPDModelParameters] = None,
        lag: int = 10,
        choose_priors: bool = True,
        changepoint_prior: float = 0.01,
        threshold: float = 0.5,
        debug: bool = False,
        agg_cp: bool = True,
    ) -> List[Tuple[TimeSeriesChangePoint, BOCPDMetadata]]:
        """The main detector method.

        This function runs the BOCPD detector
        and returns the list of changepoints, along with some metadata

        Args:
            model: This specifies the probabilistic model, that generates
                the data within each segment. The user can input several
                model types depending on the behavior of the time series.
                Currently allowed models are:
                NORMAL_KNOWN_MODEL: Normal model with variance known. Use
                this to find level shifts in normally distributed data.
                TREND_CHANGE_MODEL : This model assumes each segment is
                generated from ordinary linear regression. Use this model
                to understand changes in slope, or trend in time series.
                POISSON_PROCESS_MODEL: This assumes a poisson generative model.
                Use this for count data, where most of the values are close
                to zero.

            model_parameters: Model Parameters correspond to specific parameters
                for a specific model. They are defined in the
                NormalKnownParameters, TrendChangeParameters,
                PoissonModelParameters classes.

            lag: integer referring to the lag in reporting the changepoint. We
                report the changepoint after seeing "lag" number of data points.
                Higher lag gives greater certainty that this is indeed a changepoint.
                Lower lag will detect the changepoint faster. This is the tradeoff.

            choose_priors: If True, then hyperparameter tuning library (HPT) is used
                to choose the best priors which maximizes the posterior predictive

            changepoint_prior: This is a Bayesian algorithm. Hence, this parameter
                specifies the prior belief on the probability
                that a given point is a changepoint. For example,
                if you believe 10% of your data will be a changepoint,
                you can set this to 0.1.

            threshold: We report the probability of observing the changepoint
                at each instant. The actual changepoints are obtained by
                denoting the points above this threshold to be a changepoint.

            debug: This surfaces additional information, such as the plots of
                predicted means and variances, which allows the user to see
                debug why changepoints were not properly detected.

            agg_cp: It is tested and believed that by aggregating run-length
                posterior, we may have a stronger signal for changepoint
                detection. When setting this parameter as True, posterior
                will be the aggregation of run-length posterior by fetching
                maximum values diagonally.

        Returns:
             List[Tuple[TimeSeriesChangePoint, BOCPDMetadata]]: Each element in this
             list is a changepoint, an object of TimeSeriesChangepoint class. The start_time
             gives the time that the change was detected. The metadata contains data about
             the name of the time series (useful when multiple time series are run simultaneously),
             and the predictive model used.
        """

        assert (
            model in self.available_models
        ), f"Requested model {model} not currently supported. Please choose one from: {self.available_models}"

        if model_parameters is None:
            model_parameters = self.parameter_type[model]()

        assert isinstance(
            model_parameters, self.parameter_type[model]
        ), f"Expected parameter type {self.parameter_type[model]}, but got {model_parameters}"

        if choose_priors:
            changepoint_prior, model_parameters = self._choose_priors(
                model, model_parameters
            )

        if getattr(model_parameters, "data", 0) is None:
            model_parameters.data = self.data

        logging.debug(f"Newest model parameters: {model_parameters}")

        if not self.data.is_univariate() and not self.models[model].is_multivariate():
            msg = "Model {model.name} support univariate time series, but get {type}.".format(
                model=model, type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

        # parameters_dict = dataclasses.asdict(model_parameters)
        # pyre-fixme[45]: Cannot instantiate abstract class `_PredictiveModel` with `__init__`, `is_multivariate`, `pred_mean` and 4 additional abstract methods.Pyre
        underlying_model = self.models[model](
            data=self.data, parameters=model_parameters
        )
        underlying_model.setup()

        logging.debug(f"Creating detector with lag {lag} and debug option {debug}.")
        bocpd = _BayesOnlineChangePoint(
            data=self.data, lag=lag, debug=debug, agg_cp=agg_cp
        )

        logging.debug(
            f"Running .detector() with model {underlying_model}, threshold {threshold}, changepoint prior {changepoint_prior}."
        )
        detector_results_all = bocpd.detector(
            model=underlying_model,
            threshold=threshold,
            changepoint_prior=changepoint_prior,
        )

        self.detected_flag = True

        change_points = []
        for ts_name, detector_results in detector_results_all.items():
            change_indices = detector_results["change_points"]
            change_probs = detector_results["change_prob"]
            self.change_prob[ts_name] = change_probs
            self._run_length_prob[ts_name] = detector_results["run_length_prob"]

            logging.debug(
                f"Obtained {len(change_indices)} change points from underlying model in ts={ts_name}."
            )

            for cp_index in change_indices:
                cp_time = self.data.time.values[cp_index]
                cp = TimeSeriesChangePoint(
                    start_time=cp_time,
                    end_time=cp_time,
                    confidence=change_probs[cp_index],
                )
                bocpd_metadata = BOCPDMetadata(model=model, ts_name=ts_name)
                change_points.append((cp, bocpd_metadata))

            logging.debug(
                f"Returning {len(change_points)} change points to client in ts={ts_name}."
            )

        return change_points

    def plot(
        self,
        change_points: List[Tuple[TimeSeriesChangePoint, BOCPDMetadata]],
        ts_names: Optional[List[str]] = None,
    ) -> None:
        """Plots the change points, along with the time series.

        Use this function to visualize the results of the changepoint detection.

        Args:
            change_points: List of changepoints, which are the return value of the detector() function.
            ts_names: List of names of the time series, useful in case multiple time series are used.

        Returns:
            None.
        """
        # TODO note: Once  D23226664 lands, replace this with self.data.time_col_name
        time_col_name = "time"

        # Group changepoints together
        change_points_per_ts = self.group_changepoints_by_timeseries(change_points)
        ts_names = ts_names or list(change_points_per_ts.keys())

        data_df = self.data.to_dataframe()

        for ts_name in ts_names:
            ts_changepoints = change_points_per_ts[ts_name]

            plt.plot(data_df[time_col_name].values, data_df[ts_name].values)

            logging.info(
                f"Plotting {len(ts_changepoints)} change points for {ts_name}."
            )
            if len(ts_changepoints) == 0:
                logging.warning("No change points detected!")

            for change in ts_changepoints:
                plt.axvline(x=change[0].start_time, color="red")

            plt.show()

    def _choose_priors(
        self, model: BOCPDModelType, params: BOCPDModelParameters
    ) -> Tuple[Any, BOCPDModelParameters]:
        """Chooses priors which are defined by the model parameters.

        Chooses priors which are defined by the model parameters.
        All BOCPDModelParameters classes have a changepoint prior to iterate on.
        Other parameters can be added to specific models.
        This function runs a parameter search using the hyperparameter tuning library
        to get the best hyperparameters.

        Args:
            model: Type of predictive model.
            params: Parameters class, containing list of values of the parameters
            on which to run hyperparameter tuning.

        Returns:
            best_cp_prior: best value of the prior on the changepoint probabilities.
            params: parameter dictionary, where the selected values are set.
        """
        if _no_ax:
            raise RuntimeWarning("choose_priors requires ax-platform be installed")
        # test these changepoint_priors
        param_dict = params.prior_choice

        # which parameter seaching method are we using
        search_method = params.search_method

        # pick search iterations and method based on definition
        if search_method == "random":
            search_N, SearchMethod = 3, SearchMethodEnum.RANDOM_SEARCH_UNIFORM
        elif search_method == "gridsearch":
            search_N, SearchMethod = 1, SearchMethodEnum.GRID_SEARCH
        else:
            raise Exception(
                f"Search method has to be in random or gridsearch but it is {search_method}!"
            )

        # construct the custom parameters for the HPT library
        custom_parameters = [
            {
                "name": k,
                "type": "choice",
                "values": v,
                "value_type": "float",
                "is_ordered": False,
            }
            for k, v in param_dict.items()
        ]

        eval_fn = self._get_eval_function(model, params)

        # Use the HPT library
        seed_value = 100

        ts_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=custom_parameters,
            selected_search_method=SearchMethod,
            seed=seed_value,
        )

        for _ in range(search_N):
            ts_tuner.generate_evaluate_new_parameter_values(
                evaluation_function=eval_fn, arm_count=4
            )

        scores_df = ts_tuner.list_parameter_value_scores()

        scores_df = scores_df.sort_values(by="mean", ascending=False)

        best_params = scores_df.parameters.values[0]

        params.set_prior(best_params)

        best_cp_prior = best_params["cp_prior"]

        return best_cp_prior, params

    def _get_eval_function(
        self, model: BOCPDModelType, model_parameters: BOCPDModelParameters
    ):
        """
        generates the objective function evaluated by hyperparameter
        tuning library for choosing the priors
        """

        def eval_fn(params_to_eval: Dict[str, float]) -> float:
            changepoint_prior = params_to_eval["cp_prior"]
            model_parameters.set_prior(params_to_eval)
            logging.debug(model_parameters)
            logging.debug(params_to_eval)
            # pyre-fixme[45]: Cannot instantiate abstract class `_PredictiveModel` with `__init__`, `is_multivariate`, `pred_mean` and 4 additional abstract methods.Pyre
            underlying_model = self.models[model](
                data=self.data, parameters=model_parameters
            )
            change_point = _BayesOnlineChangePoint(data=self.data, lag=3, debug=False)
            change_point.detector(
                model=underlying_model,
                changepoint_prior=changepoint_prior,
                threshold=0.4,
            )
            post_pred = np.mean(change_point.get_posterior_predictive())

            return post_pred

        return eval_fn

    def group_changepoints_by_timeseries(
        self, change_points: List[Tuple[TimeSeriesChangePoint, BOCPDMetadata]]
    ) -> Dict[str, List[Tuple[TimeSeriesChangePoint, BOCPDMetadata]]]:
        """Helper function to group changepoints by time series.

        For multivariate inputs, all changepoints are output in
        a list and the time series they correspond to is referenced
        in the metadata. This function is a helper function to
        group these changepoints by time series.

        Args:
            change_points: List of changepoints, with metadata containing the time
                series names. This is the return value of the detector() method.

        Returns:
            Dictionary, with time series names, and their corresponding changepoints.
        """

        if self.data.is_univariate():
            data_df = self.data.to_dataframe()
            ts_names = [x for x in data_df.columns if x != "time"]
        else:
            # Multivariate
            ts_names = self.data.value.columns

        change_points_per_ts = {}
        for ts_name in ts_names:
            change_points_per_ts[ts_name] = []
        for cp in change_points:
            change_points_per_ts[cp[1].ts_name].append(cp)

        return dict(change_points_per_ts)

    def get_change_prob(self) -> Dict[str, np.ndarray]:
        """Returns the probability of being a changepoint.

        Args:
            None.

        Returns:
            For every point in the time series. The return
            type is a dict, with the name of the timeseries
            as the key, and the value is an array of probabilities
            of the same length as the timeseries data.
        """

        if not self.detected_flag:
            raise ValueError("detector needs to be run before getting prob")
        return self.change_prob

    def get_run_length_matrix(self) -> Dict[str, np.ndarray]:
        """Returns the entire run-time posterior.
        Args:
            None.

        Returns:
            The return type is a dict, with the name of the timeseries
            as the key, and the value is an array of probabilities
            of the same length as the timeseries data.
        """

        if not self.detected_flag:
            raise ValueError("detector needs to be run before getting prob")

        return self._run_length_prob


class _BayesOnlineChangePoint(Detector):
    """The underlying implementation of the BOCPD algorithm.

    This is called by the class BayesianOnlineChangepoint. The user should
    call the top level class, and not this one.

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

    Attributes::
    data: This is univariate time series data. We require more
        than 10 points, otherwise it is not very meaningful to define
        changepoints.

    T: number of values in the time series data.

    lag: This specifies, how many time steps we will look ahead to
        determine the change. There is a tradeoff in setting this parameter.
        A small lag means we can detect a change really fast, which is important
        in many applications. However, this also means we will make more
        mistakes/have lower confidence since we might mistake a spike for change.

    threshold: Threshold between 0 and 1. Probability values above this threshold
        will be denoted as changepoint.

    debug: This is a boolean. If set to true, this shows additional plots.
        Currently, it shows a plot of the predicted mean and variance, after
        lag steps, and the predictive probability of the next point. If the
        results are unusual, the user should set it to true in order to
        debug.

    agg_cp: It is tested and believed that by aggregating run-length
        posterior, we may have a stronger signal for changepoint
        detection. When setting this parameter as True, posterior
        will be the aggregation of run-length posterior by fetching
        maximum values diagonally.
    """

    rt_posterior: Optional[np.ndarray] = None
    pred_mean_arr: Optional[np.ndarray] = None
    pred_std_arr: Optional[np.ndarray] = None
    next_pred_prob: Optional[np.ndarray] = None

    def __init__(
        self,
        data: TimeSeriesData,
        lag: int = 10,
        debug: bool = False,
        agg_cp: bool = False,
    ):
        self.data = data
        self.T = data.value.shape[0]
        self.lag = lag
        self.threshold = None
        self.debug = debug
        self.agg_cp = agg_cp
        # We use tensors for all data throughout; if the data is univariate
        # then the last dimension is trivial. In this way, we standardise
        # the same calculation throughout with fewer additional checks
        # for univariate and bivariate data.
        if not data.is_univariate():
            self._ts_slice = slice(None)
            self.P = data.value.shape[1]  # Number of time series
            self._ts_names = self.data.value.columns
            self.data_values = data.value.values
        else:
            self.P = 1
            self._ts_slice = 0
            data_df = self.data.to_dataframe()
            self._ts_names = [x for x in data_df.columns if x != "time"]

            self.data_values = np.expand_dims(data.value.values, axis=1)

        self.posterior_predictive = 0.0
        self._posterior_shape = (self.T, self.T, self.P)
        self._message_shape = (self.T, self.P)

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(
        self,
        model: Any,
        threshold: Union[float, np.ndarray] = 0.5,
        changepoint_prior: Union[float, np.ndarray] = 0.01,
    ) -> Dict[str, Any]:
        """Runs the actual BOCPD detection algorithm.

        Args:
            model: Predictive Model for BOCPD
            threshold: values between 0 and 1, array since this can be specified
                separately for each time series.
            changepoint_prior: array, each element between 0 and 1. Each element
                specifies the prior probability of observing a changepoint
                in each time series.

        Returns:
            Dictionary, with key as the name of the time series, and value containing
            list of change points and their probabilities.
        """

        self.threshold = threshold
        if isinstance(self.threshold, float):
            self.threshold = np.repeat(threshold, self.P)
        if isinstance(changepoint_prior, float):
            changepoint_prior = np.repeat(changepoint_prior, self.P)
        self.rt_posterior = self._find_posterior(model, changepoint_prior)
        # pyre-fixme[6]: Expected `ndarray` for 1st param but got `Union[float,
        #  np.ndarray]`.
        return self._construct_output(self.threshold, lag=self.lag)

    def get_posterior_predictive(self):
        """Returns the posterior predictive.

        This is  sum_{t=1}^T P(x_{t+1}|x_{1:t})

        Args:
            None.

        Returns:
            Array of predicted log probabilities for the next point.
        """

        return self.posterior_predictive

    def _find_posterior(self, model: Any, changepoint_prior: np.ndarray) -> np.ndarray:
        """
        This calculates the posterior distribution over changepoints.
        The steps here are the same as the algorithm described in
        Adams & McKay, 2007. https://arxiv.org/abs/0710.3742
        """

        # P(r_t|x_t)
        rt_posterior = np.zeros(self._posterior_shape)

        # initialize first step
        # P(r_0=1) = 1
        rt_posterior[0, 0] = 1.0
        model.update_sufficient_stats(x=self.data_values[0, self._ts_slice])
        # To avoid growing a large dynamic list, we construct a large
        # array and grow the array backwards from the end.
        # This is conceptually equivalent to array, which we insert/append
        # to the beginning - but avoids reallocating memory.
        message = np.zeros(self._message_shape)
        m_ptr = -1

        # set up arrays for debugging
        self.pred_mean_arr = np.zeros(self._posterior_shape)
        self.pred_std_arr = np.zeros(self._posterior_shape)
        self.next_pred_prob = np.zeros(self._posterior_shape)

        # Calculate the log priors once outside the for-loop.
        log_cp_prior = np.log(changepoint_prior)
        log_om_cp_prior = np.log(1.0 - changepoint_prior)

        self.posterior_predictive = 0.0
        log_posterior = 0.0

        # from the second step onwards
        for i in range(1, self.T):
            this_pt = self.data_values[i, self._ts_slice]

            # P(x_t | r_t-1, x_t^r)
            # this arr has a size of t, each element says what is the predictive prob.
            # of a point, it the current streak began at t
            # Step 3 of paper
            pred_arr = model.pred_prob(t=i, x=this_pt)

            # Step 9 posterior predictive
            if i > 1:
                self.posterior_predictive += logsumexp(pred_arr + log_posterior)

            # record the mean/variance/prob for debugging
            if self.debug:
                pred_mean = model.pred_mean(t=i, x=this_pt)
                pred_std = model.pred_std(t=i, x=this_pt)
                # pyre-fixme[16]: `Optional` has no attribute `__setitem__`.
                self.pred_mean_arr[i, 0:i, self._ts_slice] = pred_mean
                self.pred_std_arr[i, 0:i, self._ts_slice] = pred_std
                self.next_pred_prob[i, 0:i, self._ts_slice] = pred_arr

            # calculate prob that this is a changepoint, i.e. r_t = 0
            # step 5 of paper
            # this is elementwise multiplication of pred and message
            log_change_point_prob = np.logaddexp.reduce(
                pred_arr
                + message[self.T + m_ptr : self.T, self._ts_slice]
                + log_cp_prior,
                axis=0,
            )

            # step 4
            # log_growth_prob = pred_arr + message + np.log(1.0 - changepoint_prior)
            message[self.T + m_ptr : self.T, self._ts_slice] = (
                pred_arr
                + message[self.T + m_ptr : self.T, self._ts_slice]
                + log_om_cp_prior
            )

            # P(r_t, x_1:t)
            # log_joint_prob = np.append(log_change_point_prob, log_growth_prob)
            m_ptr -= 1
            message[self.T + m_ptr, self._ts_slice] = log_change_point_prob

            # calculate evidence, step 6
            # (P(x_1:t))
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
                log_change_point_prob - log_cp_prior + log_om_cp_prior,
            )

            # step 7
            # log_posterior = log_joint_prob - log_evidence
            log_posterior = (
                message[self.T + m_ptr : self.T, self._ts_slice] - log_evidence
            )
            rt_posterior[i, 0 : (i + 1), self._ts_slice] = np.exp(log_posterior)

            # step 8
            model.update_sufficient_stats(x=this_pt)

            # pass the joint as a message to next step
            # message = log_joint_prob
            # Message is now passed implicitly - as we set it directly above.

        return rt_posterior

    def plot(
        self,
        threshold: Optional[Union[float, np.ndarray]] = None,
        lag: Optional[int] = None,
        ts_names: Optional[List[str]] = None,
    ):
        """Plots the changepoints along with the timeseries.

        Args:
            threshold: between 0 and 1. probability values above the threshold will be
                determined to be changepoints.
            lag: lags to use. If None, use the lags this was initialized with.
            ts_names: list of names of the time series. Useful when there are multiple
                time series.

        Returns:
            None.
        """
        if threshold is None:
            threshold = self.threshold

        if lag is None:
            lag = self.lag

        # do some work to define the changepoints
        cp_outputs = self._construct_output(threshold=threshold, lag=lag)
        if ts_names is None:
            ts_names = self._ts_names

        for ts_ix, ts_name in enumerate(ts_names):
            cp_output = cp_outputs[ts_name]
            change_points = cp_output["change_points"]
            ts_values = self.data.value[ts_name].values
            y_min_cpplot = np.min(ts_values)
            y_max_cpplot = np.max(ts_values)

            # Plot the time series
            plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(211)

            ax1.plot(list(range(self.T)), ts_values, "r-")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Values")

            # plot change points on the time series
            ax1.vlines(
                x=change_points,
                ymin=y_min_cpplot,
                ymax=y_max_cpplot,
                colors="b",
                linestyles="dashed",
            )

            # if in debugging mode, plot the mean and variance as well
            if self.debug:
                x_debug = list(range(lag + 1, self.T))
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                y_debug_mean = self.pred_mean_arr[lag + 1 : self.T, lag, ts_ix]
                y_debug_uv = (
                    self.pred_mean_arr[lag + 1 : self.T, lag, ts_ix]
                    + self.pred_std_arr[lag + 1 : self.T, lag, ts_ix]
                )

                y_debug_lv = (
                    self.pred_mean_arr[lag + 1 : self.T, lag, ts_ix]
                    - self.pred_std_arr[lag + 1 : self.T, lag, ts_ix]
                )

                ax1.plot(x_debug, y_debug_mean, "k-")
                ax1.plot(x_debug, y_debug_uv, "k--")
                ax1.plot(x_debug, y_debug_lv, "k--")

            ax2 = plt.subplot(212, sharex=ax1)

            cp_plot_x = list(range(0, self.T - lag))
            cp_plot_y = np.copy(self.rt_posterior[lag : self.T, lag, ts_ix])
            # handle the fact that first point is not a changepoint
            cp_plot_y[0] = 0.0

            ax2.plot(cp_plot_x, cp_plot_y)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Changepoint Probability")

            # if debugging, we also want to show the predictive probabities
            if self.debug:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    list(range(lag + 1, self.T)),
                    self.next_pred_prob[lag + 1 : self.T, lag, ts_ix],
                    "k-",
                )
                plt.xlabel("Time")
                plt.ylabel("Log Prob. Density Function")
                plt.title("Debugging: Predicted Probabilities")

    def _calc_agg_cppprob(self, t: int) -> np.ndarray:
        rt_posterior = self.rt_posterior
        assert rt_posterior is not None
        run_length_pos = rt_posterior[:, :, t]
        np.fill_diagonal(run_length_pos, 0.0)
        change_prob = np.zeros(self.T)
        for i in range(self.T):
            change_prob[i] = np.max(run_length_pos[i:, : (self.T - i)].diagonal())
        return change_prob

    def _construct_output(self, threshold: np.ndarray, lag: int) -> Dict[str, Any]:
        output = {}
        rt_posterior = self.rt_posterior
        assert rt_posterior is not None

        for t, t_name in enumerate(self._ts_names):
            if not self.agg_cp:
                # till lag, prob = 0, so prepend array with zeros
                change_prob = np.hstack(
                    (rt_posterior[lag : self.T, lag, t], np.zeros(lag))
                )
                # handle the fact that the first point is not a changepoint
                change_prob[0] = 0.0
            elif self.agg_cp:
                change_prob = self._calc_agg_cppprob(t)

            change_points = np.where(change_prob > threshold[t])[0]
            output[t_name] = {
                # pyre-fixme[61]: `change_prob` may not be initialized here.
                "change_prob": change_prob,
                "change_points": change_points,
                "run_length_prob": rt_posterior[:, :, t],
            }

        return output

    def adjust_parameters(self, threshold: np.ndarray, lag: int) -> Dict[str, Any]:
        """Adjust the parameters.

        If the preset parameters are not giving the desired result,
        the user can adjust the parameters. Since the algorithm
        calculates changepoints for all lags, we can see how
        changepoints look like for other lag/threshold.

        Args:
            threshold: between 0 and 1. Probabilities above threshold are
                considered to be changepoints.
            lag: lag at which changepoints are calculated.

        Returns:
            cp_output: Dictionary with changepoint list and probabilities.
        """

        cp_output = self._construct_output(threshold=threshold, lag=lag)
        self.plot(threshold=threshold, lag=lag)

        return cp_output


def check_data(data: TimeSeriesData):
    """Small helper function to check if the data is in the appropriate format.

    Currently, this only checks if we have enough data points to run the
    algorithm meaningfully.

    Args:
        data: TimeSeriesData object, on which to run the algorithm.

    Returns:
        None.
    """

    if data.value.shape[0] < _MIN_POINTS:
        raise ValueError(
            f"""
            Data must have {_MIN_POINTS} points,
            it only has {data.value.shape[0]} points
            """
        )


class _PredictiveModel(ABC):
    """Abstract class for BOCPD Predictive models.

    This is an abstract class. All Predictive models
    for BOCPD derive from this class.

    Attributes:
        data: TimeSeriesdata object we are modeling.
        parameters: Parameter class, which contains BOCPD model parameters.
    """

    @abstractmethod
    def __init__(self, data: TimeSeriesData, parameters: BOCPDModelParameters) -> None:
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def pred_prob(self, t: int, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def pred_mean(self, t: int, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def pred_std(self, t: int, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def update_sufficient_stats(self, x: float) -> None:
        pass

    @staticmethod
    @abstractmethod
    def is_multivariate() -> bool:
        pass


class _NormalKnownPrec(_PredictiveModel):
    """Predictive model where data comes from a Normal distribution.

    This model is the Normal-Normal model, with known precision
    It is specified in terms of precision for convenience.
    It assumes that the data is generated from a normal distribution with
    known precision.
    The prior on the mean of the normal, is a normal distribution.

    Attributes:
        data: The Timeseriesdata object, for which the algorithm is run.
        parameters: Parameters specifying the prior.
    """

    def __init__(self, data: TimeSeriesData, parameters: NormalKnownParameters):

        # \mu \sim N(\mu0, \frac{1}{\lambda0})
        # x \sim N(\mu,\frac{1}{\lambda})

        empirical = parameters.empirical
        mean_prior = parameters.mean_prior
        mean_prec_prior = parameters.mean_prec_prior
        known_prec = parameters.known_prec
        self.parameters = parameters
        self._maxT = len(data)

        # hyper parameters for mean and precision
        self.mu_0 = mean_prior
        self.lambda_0 = mean_prec_prior
        self.lambda_val = known_prec
        if data.is_univariate():
            self._data_shape = self._maxT
        else:
            # Multivariate
            self.P = data.value.values.shape[1]
            # If the user didn't specify the priors as multivariate
            # then we assume the same prior(s) over all time series.
            if self.mu_0 is not None and isinstance(self.mu_0, float):
                self.mu_0 = np.repeat(self.mu_0, self.P)
            if self.mu_0 is not None and isinstance(self.lambda_0, float):
                self.lambda_0 = np.repeat(self.lambda_0, self.P)
            if self.mu_0 is not None and isinstance(self.lambda_val, float):
                self.lambda_val = np.repeat(self.lambda_val, self.P)
            self._data_shape = (self._maxT, self.P)

        # For efficiency, we simulate a dynamically growing list with
        # insertions at the start, by a fixed size array with a pointer
        # where we grow the array from the end of the array. This
        # makes insertions constant time and means we can use
        # vectorized computation throughout.
        self._mean_arr_num = np.zeros(self._data_shape)
        self._std_arr = np.zeros(self._data_shape)
        self._ptr = 0

        # if priors are going to be decided empirically,
        # we ignore these settings above
        # Also, we need to pass on the data in this case
        if empirical:
            check_data(data)
            self._find_empirical_prior(data)

        if (
            self.lambda_0 is not None
            and self.lambda_val is not None
            and self.mu_0 is not None
        ):
            # We set these here to avoid recomputing the linear expression
            # throughout + avoid unnecessarily zeroing the memory etc.
            self._mean_arr = np.repeat(
                np.expand_dims(self.mu_0 * self.lambda_0, axis=0), self._maxT, axis=0
            )
            self._prec_arr = np.repeat(
                np.expand_dims(self.lambda_0, axis=0), self._maxT, axis=0
            )
        else:
            raise ValueError("Priors for NormalKnownPrec should not be None.")

    def setup(self):
        # everything is already set up in __init__!
        pass

    def _find_empirical_prior(self, data: TimeSeriesData):
        """
        if priors are not defined, we take an empirical Bayes
        approach and define the priors from the data
        """

        data_arr = data.value

        # best guess of mu0 is data mean
        if data.is_univariate():
            self.mu_0 = data_arr.mean(axis=0)
        else:
            self.mu_0 = data_arr.mean(axis=0).values

        # variance of the mean: \lambda_0 = \frac{N}{\sigma^2}
        if data.is_univariate():
            self.lambda_0 = 1.0 / data_arr.var(axis=0)
        else:
            self.lambda_0 = 1.0 / data_arr.var(axis=0).values

        # to find the variance of the data we just look at small
        # enough windows such that the mean won't change between
        window_size = 10
        var_arr = data_arr.rolling(window_size).var()[window_size - 1 :]

        if data.is_univariate():
            self.lambda_val = self.parameters.known_prec_multiplier / var_arr.mean()
        else:
            self.lambda_val = (
                self.parameters.known_prec_multiplier / var_arr.mean().values
            )

        logging.debug("Empirical Prior: mu_0:", self.mu_0)
        logging.debug("Empirical Prior: lambda_0:", self.lambda_0)
        logging.debug("Empirical Prior: lambda_val:", self.lambda_val)

    @staticmethod
    def _norm_logpdf(x, mean, std):
        """
        Hardcoded version of scipy.norm.logpdf.
        This is hardcoded because scipy version is slow due to checks +
        uses log(pdf(...)) - which wastefully computes exp(..) and log(...).
        """

        return -np.log(std) - _LOG_SQRT2PI - 0.5 * ((x - mean) / std) ** 2

    def pred_prob(self, t: int, x: float) -> np.ndarray:
        """Returns log predictive probabilities.

        We will give log predictive probabilities for
        changepoints that started at times from 0 to t.

        This posterior predictive is from
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        equation 36.

        Args:
            t is the time,
            x is the new data point

        Returns:
            pred_arr: Array with predicted log probabilities for each starting point.
        """

        pred_arr = self._norm_logpdf(
            x,
            self._mean_arr[self._maxT + self._ptr : self._maxT + self._ptr + t],
            self._std_arr[self._maxT + self._ptr : self._maxT + self._ptr + t],
        )
        return pred_arr

    def pred_mean(self, t: int, x: float) -> np.ndarray:
        return self._mean_arr[self._maxT + self._ptr : self._maxT + self._ptr + t]

    def pred_std(self, t: int, x: float) -> np.ndarray:
        return self._std_arr[self._maxT + self._ptr : self._maxT + self._ptr + t]

    def update_sufficient_stats(self, x: float) -> None:
        """Updates sufficient statistics with new data.

        We will store the sufficient stats for
        a streak starting at times 0, 1, ....t.

        This is eqn 29 and 30 in Kevin Murphy's note:
                https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Args:
            x: The new data point.

        Returns:
            None.
        """

        # \lambda = \lambda_0 + n * \lambda
        # hence, online, at each step: lambda[i] = lambda[i-1] + 1* lambda

        # for numerator of the mean.
        # n*\bar{x}*\lambda + \mu_0 * \lambda_0

        # So, online we add x*\lambda to the numerator from the previous step

        # I think we can do it online, but I will need to think more
        # for now we'll just keep track of the sum

        # Grow list (backwards from the end of the array for efficiency)
        self._ptr -= 1

        # update the precision array
        self._prec_arr[self._maxT + self._ptr : self._maxT] += self.lambda_val

        # update the numerator of the mean array
        self._mean_arr_num[self._maxT + self._ptr : self._maxT] += x * self.lambda_val

        # This is now handled by initializing the array with this value.
        # self._prec_arr[self._ptr] = self.lambda_0 + 1. * self.lambda_val

        self._std_arr[self._maxT + self._ptr : self._maxT] = np.sqrt(
            1.0 / self._prec_arr[self._maxT + self._ptr : self._maxT]
            + 1.0 / self.lambda_val
        )

        # This is now handled by initializing the array with self.mu_0 * self.lambda_0
        # self._mean_arr_num[self._ptr] = (x * self.lambda_val + self.mu_0 * self.lambda_0)

        # update the mean array itself
        self._mean_arr[self._maxT + self._ptr : self._maxT] = (
            self._mean_arr_num[self._maxT + self._ptr : self._maxT]
            / self._prec_arr[self._maxT + self._ptr : self._maxT]
        )

    @staticmethod
    def is_multivariate():
        return True


class _BayesianLinReg(_PredictiveModel):
    """Predictive model for BOCPD where data comes from linear model.

    Defines the predictive model, where we assume that the data points
    come from a Bayesian Linear model, where the values are regressed
    against time.
    We use a conjugate prior, where we impose an Inverse gamma prior on
    sigma^2 and normal prior on the conditional distribution of beta
    p(beta|sigma^2)
    See https://en.wikipedia.org/wiki/Bayesian_linear_regression
    for the calculations.

    Attributes:
        data: TimeSeriesData object, on which algorithm is run
        parameters: Specifying all the priors.
    """

    mu_prior: Optional[np.ndarray] = None
    prior_regression_numpoints: Optional[int] = None

    def __init__(
        self,
        data: TimeSeriesData,
        parameters: TrendChangeParameters,
    ):
        mu_prior = parameters.mu_prior
        num_likelihood_samples = parameters.num_likelihood_samples
        num_points_prior = parameters.num_points_prior
        readjust_sigma_prior = parameters.readjust_sigma_prior
        plot_regression_prior = parameters.plot_regression_prior
        self.parameters = parameters
        self.data = data

        logging.info(
            f"Initializing bayesian linear regression with data {data}, "
            f"mu_prior {mu_prior}, {num_likelihood_samples} likelihood samples, "
            f"{num_points_prior} points to run basic linear regression with, "
            f"sigma prior adjustment {readjust_sigma_prior}, "
            f"and plot prior regression {plot_regression_prior}"
        )
        self._x = None
        self._y = None
        self.t = 0

        # Random numbers I tried out to make the sigma_squared values really large
        self.a_0 = 0.1  # TODO find better priors?
        self.b_0 = 200  # TODO

        self.all_time = np.array(range(data.time.shape[0]))
        self.all_vals = data.value

        self.lambda_prior = 2e-7 * np.identity(2)

        self.num_likelihood_samples = num_likelihood_samples
        self.min_sum_samples = (
            math.sqrt(self.num_likelihood_samples) / 10000
        )  # TODO: Hack for getting around probabilities of 0 -- cap it at some minimum

        self._mean_arr = {}
        self._std_arr = {}

    def setup(self) -> None:
        """Sets up the regression, by calculating the priors.

        Args:
            None.

        Returns:
            None.
        """
        data = self.data
        mu_prior = self.parameters.mu_prior
        num_points_prior = self.parameters.num_points_prior
        readjust_sigma_prior = self.parameters.readjust_sigma_prior
        plot_regression_prior = self.parameters.plot_regression_prior

        # Set up linear regression prior
        if mu_prior is None:
            if data is not None:
                self.prior_regression_numpoints = num_points_prior

                time = self.all_time[: self.prior_regression_numpoints]
                vals = self.all_vals[: self.prior_regression_numpoints]

                logging.info("Running basic linear regression.")

                # Compute basic linear regression
                slope, intercept, r_value, p_value, std_err = linregress(time, vals)
                self.mu_prior = mu_prior = np.array(
                    [intercept, slope]
                )  # Set up mu_prior

                if readjust_sigma_prior:
                    logging.info("Readjusting the prior for Inv-Gamma for sigma^2.")
                    # these values are the mean/variance of sigma^2: Inv-Gamma(*,*)
                    sigma_squared_distribution_mean = (
                        _BayesianLinReg._residual_variance(time, vals, intercept, slope)
                    )
                    sigma_squared_distribution_variance = 1000  # TODO: we don't really know what the variance of sigma^2: Inv-Gamma(a, b) should be

                    # The following values are computed from https://reference.wolfram.com/language/ref/InverseGammaDistribution.html
                    # We want to match the mean of Inv-Gamma(a, b) to the sigma^2 mean (called mu), and variances together too (called var).
                    # We obtain mu = b / (a-1) and var = b^2 / ((a-2) * (a-1)^2) and then we simply solve for a and b.
                    self.a_0 = 2.0 + (
                        sigma_squared_distribution_mean
                        / sigma_squared_distribution_variance
                    )
                    self.b_0 = sigma_squared_distribution_mean * (self.a_0 - 1)
            else:
                self.mu_prior = mu_prior = np.zeros(2)
                logging.warning("No data provided -- reverting to default mu_prior.")
        else:
            self.mu_prior = mu_prior

        logging.info(f"Obtained mu_prior: {self.mu_prior}")
        logging.info(f"Obtained a_0, b_0 values of {self.a_0}, {self.b_0}")

        if plot_regression_prior:
            intercept, slope = tuple(mu_prior)
            _BayesianLinReg._plot_regression(
                self.all_time, self.all_vals, intercept, slope
            )

    @staticmethod
    def _plot_regression(x, y, intercept, slope):
        plt.plot(x, y, ".")
        plt.plot(x, intercept + slope * x, "-")
        plt.show()

    @staticmethod
    def _residual_variance(x, y, intercept, slope):
        n = len(x)
        assert n == len(y)
        x = np.array(x)
        y = np.array(y)

        predictions = intercept + slope * x
        residuals = predictions - y

        return np.sum(np.square(residuals)) / (n - 2)

    @staticmethod
    def _sample_bayesian_linreg(mu_n, lambda_n, a_n, b_n, num_samples):

        # this is to make sure the results are consistent
        # and tests don't break randomly
        seed_value = 100
        np.random.seed(seed_value)

        sample_sigma_squared = invgamma.rvs(a_n, scale=b_n, size=1)

        # Sample a beta value from Normal(mu_n, sigma^2 * inv(lambda_n))
        assert (
            len(mu_n.shape) == 1
        ), f"Expected 1 dimensional mu_n, but got {mu_n.shape}"

        all_beta_samples = np.random.multivariate_normal(
            mu_n, sample_sigma_squared * np.linalg.inv(lambda_n), size=num_samples
        )

        return all_beta_samples, sample_sigma_squared

    @staticmethod
    def _compute_bayesian_likelihood(beta, sigma_squared, x, val):
        prediction = np.matmul(beta, x)
        bayesian_likelihoods = norm.pdf(
            val, loc=prediction, scale=np.sqrt(sigma_squared)
        )

        return bayesian_likelihoods, prediction

    @staticmethod
    def _sample_likelihood(mu_n, lambda_n, a_n, b_n, x, val, num_samples):
        (
            all_sample_betas,
            sample_sigma_squared,
        ) = _BayesianLinReg._sample_bayesian_linreg(
            mu_n, lambda_n, a_n, b_n, num_samples
        )

        bayesian_likelihoods, prediction = _BayesianLinReg._compute_bayesian_likelihood(
            all_sample_betas, sample_sigma_squared, x, val
        )

        return bayesian_likelihoods, prediction, sample_sigma_squared

    def pred_prob(self, t, x) -> np.ndarray:
        """Predictive probability of a new data point

        Args:
            t: time
            x: the new data point

        Returns:
            pred_arr: Array with log predictive probabilities for each starting point.
        """

        # TODO: use better priors
        def log_post_pred(y, t, rl):
            N = self._x.shape[0]

            x_arr = self._x[N - rl - 1 : N, :]
            y_arr = self._y[N - rl - 1 : N].reshape(-1, 1)

            xtx = np.matmul(x_arr.transpose(), x_arr)  # computes X^T X
            xty = np.squeeze(np.matmul(x_arr.transpose(), y_arr))  # computes X^T Y
            yty = np.matmul(y_arr.transpose(), y_arr)  # computes Y^T Y

            # Bayesian learning update

            lambda_n = xtx + self.lambda_prior
            mu_n = np.matmul(
                np.linalg.inv(lambda_n),
                np.squeeze(np.matmul(self.lambda_prior, self.mu_prior) + xty),
            )

            a_n = self.a_0 + t / 2
            mu_prec_prior = np.matmul(
                np.matmul(self.mu_prior.transpose(), self.lambda_prior), self.mu_prior
            )
            mu_prec_n = np.matmul(np.matmul(mu_n.transpose(), lambda_n), mu_n)
            b_n = self.b_0 + 1 / 2 * (yty + mu_prec_prior - mu_prec_n)

            if a_n < 0 or b_n < 0:
                logging.info(
                    f"""
                    Got nonpositive parameters for Inv-Gamma: {a_n}, {b_n}.
                    Likely, integer overflow -- maybe scale down the data?
                    """
                )
            # cannot allow this to fail arbitrarily, so falling back to prior
            if a_n < 0:
                a_n = self.a_0
            if b_n < 0:
                b_n = self.b_0

            # Compute likelihood of new point x under new Bayesian parameters

            x_new = np.array([1.0, t]).reshape(2, -1)

            (
                indiv_likelihoods,
                prediction,
                var_pred,
            ) = _BayesianLinReg._sample_likelihood(
                mu_n, lambda_n, a_n, b_n, x_new, y, self.num_likelihood_samples
            )

            likelihoods = np.sum(indiv_likelihoods)
            likelihoods = max(likelihoods, self.min_sum_samples)
            avg_likelihood = likelihoods / self.num_likelihood_samples

            mean_prediction = np.mean(prediction)
            std_prediction = np.sqrt(var_pred)

            self._mean_arr[t].append(mean_prediction)
            self._std_arr[t].append(std_prediction)

            return np.log(avg_likelihood)

        if t % 50 == 1:  # put 1 because then t=1 will show up
            logging.info(f"Running Bayesian Linear Regression with t={t}.")

        # initialize empty mean and std deviation arrays
        self._mean_arr[t] = []
        self._std_arr[t] = []

        pred_arr = np.array([log_post_pred(y=x, t=t, rl=rl) for rl in range(t)])

        return pred_arr

    # pyre-fixme[15]: `pred_mean` overrides method defined in `_PredictiveModel`
    #  inconsistently.
    def pred_mean(self, t: int, x: float) -> float:
        """Predicted mean at the next time point.

        Args:
            t: time.
            x: the new data point.

        Returns:
            meant_arr[t]: mean value predicted at the next data point.
        """

        return self._mean_arr[t]

    # pyre-fixme[15]: `pred_std` overrides method defined in `_PredictiveModel`
    #  inconsistently.
    def pred_std(self, t: int, x: float) -> float:
        """
        predicted standard deviation at the next time point.
        Args:
            t: time.
            x: the new data point.

        Returns:
            std_arr[t]: predicted std. dev at the next point.
        """

        return self._std_arr[t]

    def update_sufficient_stats(self, x: float) -> None:
        """Updates sufficient statistics.

        Updates the sufficient statistics for posterior calculation,
        based on the new data point.

        Args:
            x: the new data point.

        Returns:
            None.
        """

        current_t = self.t

        if self._x is None:
            self._x = np.array([1.0, current_t]).reshape(-1, 2)
        else:
            new_x = np.array([1.0, current_t]).reshape(-1, 2)
            self._x = np.vstack([self._x, new_x])

        self.t += 1

        if self._y is None:
            self._y = np.array([x])
        else:
            self._y = np.append(self._y, np.array([x]))

    @staticmethod
    def is_multivariate():
        # This class hasn't been confirmed / checked / tested
        # we assume NO for now.
        return False


class _PoissonProcessModel(_PredictiveModel):
    """BOCPD Predictive model, where data comes from Poisson.

    Predictive model, which assumes that the data
    comes from a Poisson distribution. We use a
    gamma distribution as a prior on the poisson rate parameter.

    Attributes:
        data: TimeSeriesData object, on which algorithm is run.
        parameters: Specifying all the priors.
    """

    def __init__(self, data: TimeSeriesData, parameters: PoissonModelParameters):
        self.data = data

        self.gamma_alpha = (
            parameters.alpha_prior
        )  # prior for rate lambda ~ Gamma(alpha, beta)
        self.gamma_beta = parameters.beta_prior

        self.parameters = parameters

        self._events = []
        self._p = {}
        self._n = {}
        self._mean_arr = {}
        self._std_arr = {}

        self._t = 0

    def setup(self):
        # everything is already set up in __init__!
        pass

    def pred_prob(self, t, x):  # predict the probability that time t, we have value x
        """Predictive log probability of a new data point.

        Args:
            t: time.
            x: the new data point.

        Returns:
            probs: array of log probabilities, for each starting point.
        """

        probs = nbinom.logpmf(x, self._n[t], self._p[t])
        return probs

    def pred_mean(self, t, x):
        """Predicted mean at the next time point.

        Args:
            t: time.
            x: the new data point.

        Returns:
            mean_arr[t]: mean predicted value at the next point.
        """

        return self._mean_arr[t]

    def pred_std(self, t, x):
        """Predicted std dev  at the next time point.

        Args:
            t: time.
            x: the new data point.

        Returns:
            std_arr[t]: std. deviation of the prediction at the next point.
        """

        return self._std_arr[t]

    def update_sufficient_stats(self, x):
        """Updates sufficient statistics.

        Updates the sufficient statistics for posterior calculation,
        based on the new data point.

        Args:
            x: the new data point.

        Returns:
            None.
        """

        new_n = []
        new_p = []
        new_mean_arr = []
        new_std_arr = []

        self._t += 1
        self._events.insert(0, x)

        num_events_before = 0
        for t in range(
            1, self._t + 1
        ):  # t is the number of previous events we consider to adjust poisson rate
            num_events_before += self._events[t - 1]

            # adjust our posterior distribution
            # these values are calculated from matching the mean and std deviation of the negative binomial
            # to the equation on page 4 of http://people.stat.sc.edu/Hitchcock/stat535slidesday18.pdf
            n = self.gamma_alpha + num_events_before
            p = (t + self.gamma_beta) / (t + 1 + self.gamma_beta)

            new_n.append(n)
            new_p.append(p)

            new_mean_arr.append(nbinom.mean(n, p))  # the mean is n * (1-p) / p
            new_std_arr.append(
                nbinom.std(n, p)
            )  # the std deviation is np.sqrt(n * (1-p)) / p

        self._n[self._t] = new_n
        self._p[self._t] = new_p
        self._mean_arr[self._t] = new_mean_arr
        self._std_arr[self._t] = new_std_arr

    @staticmethod
    def is_multivariate():
        # This class hasn't been confirmed / checked / tested
        # we assume NO for now.
        return False
