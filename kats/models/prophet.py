# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    from fbprophet import Prophet

    _no_prophet = False
except ImportError:
    _no_prophet = True
    Prophet = Dict[str, Any]  # for Pyre

import numpy as np
from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import get_default_prophet_parameter_search_space


def _error_msg(msg: str) -> None:
    logging.error(msg)
    raise ValueError(msg)


class ProphetParams(Params):
    """Parameter class for Prophet model

    This is the parameter class for prophet model, it contains all necessary
    parameters as definied in Prophet implementation:
    https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py

    Attributes:
        growth: String 'linear' or 'logistic' to specify a linear or logistic
            trend.
        changepoints: List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
        n_changepoints: Number of potential changepoints to include. Not used
            if input `changepoints` is supplied. If `changepoints` is not supplied,
            then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
        changepoint_range: Proportion of history in which trend changepoints will
            be estimated. Defaults to 0.8 for the first 80%. Not used if
            `changepoints` is specified.
        yearly_seasonality: Fit yearly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        weekly_seasonality: Fit weekly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        daily_seasonality: Fit daily seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        holidays: pd.DataFrame with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also
            optionally can have a column prior_scale specifying the prior scale for
            that holiday.
        seasonality_mode: 'additive' (default) or 'multiplicative'.
        seasonality_prior_scale: Parameter modulating the strength of the
            seasonality model. Larger values allow the model to fit larger seasonal
            fluctuations, smaller values dampen the seasonality. Can be specified
            for individual seasonalities using add_seasonality.
        holidays_prior_scale: Parameter modulating the strength of the holiday
            components model, unless overridden in the holidays input.
        changepoint_prior_scale: Parameter modulating the flexibility of the
            automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
        mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
            with the specified number of MCMC samples. If 0, will do MAP
            estimation.
        interval_width: Float, width of the uncertainty intervals provided
            for the forecast. If mcmc_samples=0, this will be only the uncertainty
            in the trend using the MAP estimate of the extrapolated generative
            model. If mcmc.samples>0, this will be integrated over all model
            parameters, which will include uncertainty in seasonality.
        uncertainty_samples: Number of simulated draws used to estimate
            uncertainty intervals. Settings this value to 0 or False will disable
            uncertainty estimation and speed up the calculation.
        cap: A boolean or a float representing the capacity, provided for logistic growth. When `cap` is a float, the capacity will be automatically set the value.
            When `cap=False`, no capacity will be provided. When `cap=True`, the model expects `data` to contain the capacity information. Default is None, which is equivalent to False.
        floor: A boolean or a float representing the floor. When `floor` is a float, the floor will be automatically set the value.
            When `floor=False`, no floor will be provided. When `floor=True`, the model expects `data` to contain the floor information. Default is None, which is equivalent to False.
        custom_seasonlities: customized seasonalities, dict with keys
            "name", "period", "fourier_order"
        extra_regressors: A list of dictionary representing the additional regressors. each regressor is a dictionary with required key "name"
        ( and optional keys "prior_scale" and "mode"). Default is None, which means no additional regressors.
    """

    growth: str
    changepoints: Optional[List[float]]
    n_changepoints: int
    changepoint_range: float
    yearly_seasonality: str
    weekly_seasonality: str
    daily_seasonality: str
    holidays: Optional[pd.DataFrame]
    seasonality_mode: str
    seasonality_prior_scale: float
    holidays_prior_scale: float
    changepoint_prior_scale: float
    mcmc_samples: int
    interval_width: float
    uncertainty_samples: int
    cap: Union[bool, float, None] = None
    floor: Union[bool, float, None] = None
    custom_seasonalities: List[Dict[str, Any]]
    extra_regressors: List[Dict[str, Any]]

    def __init__(
        self,
        growth: str = "linear",
        changepoints: Optional[List[float]] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: str = "auto",
        weekly_seasonality: str = "auto",
        daily_seasonality: str = "auto",
        holidays: Optional[pd.DataFrame] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000,
        cap: Union[bool, float, int, None] = None,
        floor: Union[bool, float, int, None] = None,
        custom_seasonalities: Optional[List[Dict[str, Any]]] = None,
        extra_regressors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if _no_prophet:
            raise RuntimeError("requires fbprophet to be installed")
        super().__init__()
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.cap = cap if cap is not None else False
        self.floor = floor if floor is not None else False
        self.custom_seasonalities = (
            [] if custom_seasonalities is None else custom_seasonalities
        )
        self.extra_regressors = [] if extra_regressors is None else extra_regressors
        self._reqd_regressor_names: List[str] = []
        self._reqd_cap_floor_names: List[str] = []
        logging.debug(
            "Initialized Prophet with parameters. "
            f"growth:{growth},"
            f"changepoints:{changepoints},"
            f"n_changepoints:{n_changepoints},"
            f"changepoint_range:{changepoint_range},"
            f"yearly_seasonality:{yearly_seasonality},"
            f"weekly_seasonality:{weekly_seasonality},"
            f"daily_seasonality:{daily_seasonality},"
            f"holidays:{holidays},"
            f"seasonality_mode:{seasonality_mode},"
            f"seasonality_prior_scale:{seasonality_prior_scale},"
            f"holidays_prior_scale:{holidays_prior_scale},"
            f"changepoint_prior_scale:{changepoint_prior_scale},"
            f"mcmc_samples:{mcmc_samples},"
            f"interval_width:{interval_width},"
            f"uncertainty_samples:{uncertainty_samples},"
            f"cap:{cap},"
            f"floor:{floor},"
            f"custom_seasonalities:{custom_seasonalities},"
            f"extra_regressors:{self.extra_regressors}"
        )
        self.validate_params()

    def validate_params(self) -> None:
        """validate Prophet parameters

        This method validates some key parameters including growth rate
        and custom_seasonalities.
        """
        # cap must be given when using logistic growth
        if (self.growth == "logistic") and (self.cap is False):
            msg = "Capacity must be provided for logistic growth"
            logging.error(msg)
            raise ValueError(msg)

        # If custom_seasonalities passed, ensure they contain the required keys.
        reqd_seasonality_keys = ["name", "period", "fourier_order"]
        if not all(
            req_key in seasonality
            for req_key in reqd_seasonality_keys
            for seasonality in self.custom_seasonalities
        ):
            msg = f"Custom seasonality dicts must contain the following keys:\n{reqd_seasonality_keys}"
            logging.error(msg)
            raise ValueError(msg)

        # If extra_regressors passed, ensure they contain the required keys.
        all_regressor_keys = {"name", "prior_scale", "mode"}
        for regressor in self.extra_regressors:
            if not isinstance(regressor, dict):
                msg = f"Elements in `extra_regressor` should be a dictionary but receives {type(regressor)}."
                _error_msg(msg)
            if "name" not in regressor:
                msg = "Extra regressor dicts must contain the following keys: 'name'."
                _error_msg(msg)
            if not set(regressor.keys()).issubset(all_regressor_keys):
                msg = f"Elements in `extra_regressor` should only contain keys in {all_regressor_keys} but receives {regressor.keys()}."
                _error_msg(msg)
        self._reqd_regressor_names = [
            regressor["name"] for regressor in self.extra_regressors
        ]
        # check floor and cap
        if (self.cap is not False) and ("cap" not in self._reqd_cap_floor_names):
            self._reqd_cap_floor_names.append("cap")
        if self.floor is not False and ("floor" not in self._reqd_cap_floor_names):
            self._reqd_cap_floor_names.append("floor")


class ProphetModel(Model[ProphetParams]):
    """Model class for Prophet

    This class provides fit, predict, and plot methods for Prophet model

    Attributes:
        data: the input time series data as in :class:`kats.consts.TimeSeriesData`.
              When `data` represents a multivariate time series, we require the target value column named as `y`.
        params: the parameter class definied with `ProphetParams`
        model: the `Prophet` object representing the prophet model. If `ProphetModel` object is not fitted, then `model` is None. Default is None.
        freq: a string or a `pd.Timedelta` object representing the frequency of time series. If `ProphetModel` object is not fitted, then `freq` is None.
    """

    freq: Union[None, str, pd.Timedelta] = None
    model: Optional[Prophet] = None

    def __init__(self, data: TimeSeriesData, params: ProphetParams) -> None:
        super().__init__(data, params)
        if _no_prophet:
            raise RuntimeError("requires fbprophet to be installed")
        self.data: TimeSeriesData = data
        self._data_params_validation()

    def _data_params_validation(self) -> None:
        """Validate whether `data` contains specified regressors or not."""
        extra_regressor_names = set(self.params._reqd_regressor_names)
        # univariate case
        if self.data.is_univariate():
            if len(extra_regressor_names) != 0:
                msg = (
                    f"Missing data for extra regressors: {self.params._reqd_regressor_names}! "
                    "Please include the missing regressors in `data`."
                )
                raise ValueError(msg)
        # multivariate case
        else:
            value_cols = set(self.data.value.columns)
            if "y" not in value_cols:
                msg = "`data` should contain a column called `y` representing the responsive value."
                raise ValueError(msg)
            if not extra_regressor_names.issubset(value_cols):
                msg = f"`data` should contain all columns listed in {extra_regressor_names}."
                raise ValueError(msg)
        # validate cap
        if (self.params.cap is True) and ("cap" not in self.data.value.columns):
            msg = "`data` should contain a column called `cap` representing the cap when `cap = True`."
            _error_msg(msg)
        # validate floor
        if (self.params.floor is True) and ("floor" not in self.data.value.columns):
            msg = "`data` should contain a column called `floor` representing the floor when `floor = True`."
            _error_msg(msg)

    def _ts_to_df(self) -> pd.DataFrame:
        if self.data.is_univariate():
            # handel corner case: `value` column is not named as `y`.
            df = pd.DataFrame({"ds": self.data.time, "y": self.data.value}, copy=False)
        else:
            df = self.data.to_dataframe()
            df.rename(columns={self.data.time_col_name: "ds"}, inplace=True)

        # add cap
        if not isinstance(self.params.cap, bool):
            df["cap"] = self.params.cap
        # add floor
        if not isinstance(self.params.floor, bool):
            df["floor"] = self.params.floor

        col_names = (
            self.params._reqd_regressor_names
            + ["y", "ds"]
            + self.params._reqd_cap_floor_names
        )
        return df[col_names]

    def _future_validation(
        self, steps: int, future: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        non_future = future is None
        if future is None:
            # pyre-fixme
            future = self.model.make_future_dataframe(
                periods=steps, freq=self.freq, include_history=self.include_history
            )
        if "ds" not in future.columns:
            msg = "`future` should be specified and `future` should contain a column named 'ds' representing the timestamps."
            _error_msg(msg)
        if not set(self.params._reqd_regressor_names).issubset(future.columns):
            msg = (
                "`future` should be specified and `future` is missing some regressors!"
            )
            _error_msg(msg)
        if self.params.cap is True and ("cap" not in future.columns):
            msg = "`future` should be specified and `future` should contain a column named 'cap' representing future capacity."
            _error_msg(msg)
        if self.params.floor is True and ("floor" not in future.columns):
            msg = "`future` should be specified and `future` should contain a column named 'floor' representing future floor."
            _error_msg(msg)
        if not isinstance(self.params.floor, bool):
            future["floor"] = self.params.floor
        if not isinstance(self.params.cap, bool):
            future["cap"] = self.params.cap
        if non_future:  # when `future` not generated by helper functions
            if self.include_history:
                future = future.merge(
                    # pyre-fixme
                    self.model.history,
                    on=(
                        ["ds"]
                        + self.params._reqd_regressor_names
                        + self.params._reqd_cap_floor_names
                    ),
                    how="outer",
                )
            else:
                future = future[future.ds > self.data.time.max()]

            reqd_length = steps + int(len(self.data) * self.include_history)
            if len(future) < reqd_length:
                msg = f"Input `future` is not long enough to generate forecasts of {steps} steps."
                _error_msg(msg)
            future.sort_values("ds", inplace=True)
            future = future[:reqd_length]
        return future

    def fit(self, **kwargs: Any) -> None:
        """fit Prophet model

        Args:
            None.

        Returns:
            The fitted prophet model object
        """

        logging.debug(
            "Call fit() with parameters: "
            f"growth:{self.params.growth},"
            f"changepoints:{self.params.changepoints},"
            f"n_changepoints:{self.params.n_changepoints},"
            f"changepoint_range:{self.params.changepoint_range},"
            f"yearly_seasonality:{self.params.yearly_seasonality},"
            f"weekly_seasonality:{self.params.weekly_seasonality},"
            f"daily_seasonality:{self.params.daily_seasonality},"
            f"holidays:{self.params.holidays},"
            f"seasonality_mode:{self.params.seasonality_mode},"
            f"seasonality_prior_scale:{self.params.seasonality_prior_scale},"
            f"holidays_prior_scale:{self.params.holidays_prior_scale},"
            f"changepoint_prior_scale:{self.params.changepoint_prior_scale},"
            f"mcmc_samples:{self.params.mcmc_samples},"
            f"interval_width:{self.params.interval_width},"
            f"uncertainty_samples:{self.params.uncertainty_samples},"
            f"cap:{self.params.cap},"
            f"floor:{self.params.floor},"
            f"custom_seasonalities:{self.params.custom_seasonalities},"
            f"extra_regressors:{self.params.extra_regressors}."
        )

        prophet = Prophet(
            growth=self.params.growth,
            changepoints=self.params.changepoints,
            n_changepoints=self.params.n_changepoints,
            changepoint_range=self.params.changepoint_range,
            yearly_seasonality=self.params.yearly_seasonality,
            weekly_seasonality=self.params.weekly_seasonality,
            daily_seasonality=self.params.daily_seasonality,
            holidays=self.params.holidays,
            seasonality_mode=self.params.seasonality_mode,
            seasonality_prior_scale=self.params.seasonality_prior_scale,
            holidays_prior_scale=self.params.holidays_prior_scale,
            changepoint_prior_scale=self.params.changepoint_prior_scale,
            mcmc_samples=self.params.mcmc_samples,
            interval_width=self.params.interval_width,
            uncertainty_samples=self.params.uncertainty_samples,
        )
        # prepare dataframe for Prophet.fit()
        df = self._ts_to_df()

        # Add any specified custom seasonalities.
        for custom_seasonality in self.params.custom_seasonalities:
            prophet.add_seasonality(**custom_seasonality)

        # Add any extra regressors
        for regressor in self.params.extra_regressors:
            prophet.add_regressor(**regressor)

        try:
            self.model = prophet.fit(df=df)
        except Exception as e:
            logging.error(e)
            logging.error(f"df = {df}")
            raise ValueError(
                f" error_message = {e} and df={df}, raw_ts = {self.data}, {self.params._reqd_cap_floor_names}."
            )
        logging.info("Fitted Prophet model. ")

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self,
        steps: int,
        include_history: bool = False,
        raw: bool = False,
        future: Optional[pd.DataFrame] = None,
        freq: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """predict with fitted Prophet model

        Args:
            steps: The steps or length of prediction horizon.
            include_history: Optional; If include the historical data, default as False.
            raw: Optional; Whether to return the raw forecasts of prophet model, default is False.
            future: Optional; A `pd.DataFrame` object containing necessary information (e.g., extra regressors) to generate forecasts.
                The length of `future` should be no less than `steps` and it should contain a column named `ds` representing the timestamps.
                Default is None.
            freq: Optional; A string representing the frequency of timestamps.
        Returns:
            The predicted dataframe with following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")

        logging.debug(
            "Call predict() with parameters: "
            f"steps:{steps}, include_history:{include_history}, raw:{raw}, future:{future} kwargs:{kwargs}."
        )

        self.freq = freq if freq is not None else self.data.infer_freq_robust()
        self.include_history = include_history
        future = self._future_validation(steps, future)
        fcst = model.predict(future)
        if raw:
            return fcst

        logging.info("Generated forecast data from Prophet model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        self.fcst_df = fcst_df = pd.DataFrame(
            {
                "time": fcst.ds,
                "fcst": fcst.yhat,
                "fcst_lower": fcst.yhat_lower,
                "fcst_upper": fcst.yhat_upper,
            },
            copy=False,
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return fcst_df

    def __str__(self) -> str:
        return "Prophet"

    @staticmethod
    # pyre-fixme[15]: `get_parameter_search_space` overrides method defined in
    #  `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """get default parameter search space for Prophet model"""
        return get_default_prophet_parameter_search_space()


# From now on, the main logics are from github PR https://github.com/facebook/prophet/pull/2186 with some modifications.
def predict_uncertainty(
    prophet_model: Prophet, df: pd.DataFrame, vectorized: bool
) -> pd.DataFrame:
    """Prediction intervals for yhat and trend.

    Args:
        prophet_model: a trained prophet object.
        df: a `pd.dataframe` to generate uncertainty for.
        vectorized: a boolean for whether to use a vectorized method for generating future draws.

    Returns
        a `pd.Dataframe` for uncertainty intervals.
    """
    sim_values = sample_posterior_predictive(prophet_model, df, vectorized)

    lower_p = 100 * (1.0 - prophet_model.interval_width) / 2
    upper_p = 100 * (1.0 + prophet_model.interval_width) / 2

    series = {}

    for key in ["yhat", "trend"]:
        series["{}_lower".format(key)] = prophet_model.percentile(
            sim_values[key], lower_p, axis=0
        )
        series["{}_upper".format(key)] = prophet_model.percentile(
            sim_values[key], upper_p, axis=0
        )

    return pd.DataFrame(series)


def _sample_predictive_trend_vectorized(
    prophet_model: Prophet, df: pd.DataFrame, n_samples: int, iteration: int = 0
) -> np.ndarray:
    """Sample draws of the future trend values. Vectorized version of sample_predictive_trend().

    Args:
        prophet_model: a trained prophet object.
        df: a `pd.dataframe` to generate uncertainty for.
        n_samples: an integer for the number of samples to generate.
        iteration: an integer for the index of model parameters for generating samples.

    Returns:
        A `np.ndarray` object with size (n_samples, len(df)) representing the trend samples.
    """

    deltas = prophet_model.params["delta"][iteration]
    m0 = prophet_model.params["m"][iteration]
    k = prophet_model.params["k"][iteration]
    if prophet_model.growth == "linear":
        expected = prophet_model.piecewise_linear(
            df["t"].values, deltas, k, m0, prophet_model.changepoints_t
        )
    elif prophet_model.growth == "logistic":
        expected = prophet_model.piecewise_logistic(
            df["t"].values,
            df["cap_scaled"].values,
            deltas,
            k,
            m0,
            prophet_model.changepoints_t,
        )
    elif prophet_model.growth == "flat":
        expected = prophet_model.flat_trend(df["t"].values, m0)
    else:
        raise NotImplementedError

    uncertainty = _sample_trend_uncertainty(prophet_model, n_samples, df, iteration)
    return (
        np.tile(expected, (n_samples, 1)) + uncertainty
    ) * prophet_model.y_scale + np.tile(df["floor"].values, (n_samples, 1))


def _sample_trend_uncertainty(
    prophet_model: Prophet,
    n_samples: int,
    df: pd.DataFrame,
    iteration: int = 0,
) -> np.ndarray:
    """Sample draws of future trend changes, vectorizing as much as possible.

    Args:
        prophet_model: a trained prophet object.
        df: a `pd.dataframe` to generate uncertainty for.
        n_samples: an integer for the number of samples to generate.
        iteration: an integer for the index of model parameters for generating samples.

    Returns:
        A `np.ndarray` object with size (n_samples, len(df)) representing the sampels of uncertainties on trend.
    """

    # when there is only historical data
    # given that df is sorted by time, it's last item has the largest date.
    if df["t"].iloc[-1] <= 1:
        # there is no trend uncertainty in historic trends
        uncertainties = np.zeros((n_samples, len(df)))
    else:

        future_df = df.loc[df["t"] > 1]
        n_length = len(future_df)
        hist_len = len(df) - n_length
        # handle 1 length futures by using history
        if n_length > 1:
            single_diff = np.diff(future_df["t"]).mean()
        else:
            single_diff = np.diff(prophet_model.history["t"]).mean()
        change_likelihood = len(prophet_model.changepoints_t) * single_diff
        deltas = prophet_model.params["delta"][iteration]
        m0 = prophet_model.params["m"][iteration]
        k = prophet_model.params["k"][iteration]
        mean_delta = np.mean(np.abs(deltas)) + 1e-8
        if prophet_model.growth == "linear":
            mat = _make_trend_shift_matrix(
                mean_delta, change_likelihood, n_length, n_samples=n_samples
            )
            uncertainties = mat.cumsum(axis=1).cumsum(
                axis=1
            )  # from slope changes to actual values
            uncertainties *= single_diff  # scaled by the actual meaning of the slope
        elif prophet_model.growth == "logistic":
            mat = _make_trend_shift_matrix(
                mean_delta, change_likelihood, n_length, n_samples=n_samples
            )
            uncertainties = _logistic_uncertainty(
                prophet_model=prophet_model,
                mat=mat,
                deltas=deltas,
                k=k,
                m=m0,
                cap=future_df["cap_scaled"].values,
                t_time=future_df["t"].values,
                n_length=n_length,
                single_diff=single_diff,
            )
        elif prophet_model.growth == "flat":
            # no trend uncertainty when there is no growth
            uncertainties = np.zeros((n_samples, n_length))
        else:
            raise NotImplementedError
        # historical part
        if hist_len > 0:
            past_uncertainty = np.zeros((n_samples, hist_len))
            uncertainties = np.concatenate([past_uncertainty, uncertainties], axis=1)
    return uncertainties


def _make_trend_shift_matrix(
    mean_delta: float, likelihood: float, future_length: float, n_samples: int
) -> np.ndarray:
    """
    Creates a matrix of random trend shifts based on historical likelihood and size of shifts.
    Can be used for either linear or logistic trend shifts.
    Each row represents a different sample of a possible future, and each column is a time step into the future.
    """
    # create a bool matrix of where these trend shifts should go
    bool_slope_change = np.random.uniform(size=(n_samples, future_length)) < likelihood
    shift_values = np.random.laplace(0, mean_delta, size=bool_slope_change.shape)
    mat = shift_values * bool_slope_change
    n_mat = np.hstack([np.zeros((len(mat), 1)), mat])[:, :-1]
    mat = (n_mat + mat) / 2
    return mat


def predict(
    prophet_model: Prophet,
    df: Optional[pd.DataFrame] = None,
    vectorized: bool = False,
) -> pd.DataFrame:
    """Predict using the prophet model.
    Args:
        df: a `pd.DataFrame` object with dates and necessary information for predictions.
        vectorized: a boolean for whether to use a vectorized method to compute uncertainty intervals. Default is False.

    Returns:
        A `pd.DataFrame` object for the forecasts.
    """
    if prophet_model.history is None:
        raise Exception("Model has not been fit.")

    if df is None:
        df = prophet_model.history.copy()
    else:
        if df.shape[0] == 0:
            raise ValueError("Dataframe has no rows.")
        df = prophet_model.setup_dataframe(df)

    df["trend"] = prophet_model.predict_trend(df)
    # TODO: the seasoanl part will be computed twice when we need to compute uncertainty. Remove this part if possible.
    seasonal_components = prophet_model.predict_seasonal_components(df)

    if prophet_model.uncertainty_samples:
        intervals = predict_uncertainty(prophet_model, df, vectorized)

    else:
        intervals = None

    # Drop columns except ds, cap, floor, and trend
    cols = ["ds", "trend"]
    if "cap" in df:
        cols.append("cap")
    if prophet_model.logistic_floor:
        cols.append("floor")
    # Add in forecast components
    df2 = pd.concat((df[cols], intervals, seasonal_components), axis=1)
    df2["yhat"] = (
        df2["trend"] * (1 + df2["multiplicative_terms"]) + df2["additive_terms"]
    )
    return df2


def sample_model_vectorized(
    prophet_model: Prophet,
    df: pd.DataFrame,
    seasonal_features: pd.DataFrame,
    iteration: int,
    s_a: np.ndarray,
    s_m: np.ndarray,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Simulate observations from the extrapolated generative model. Vectorized version of sample_model().

    Returns:
        A dictionary (with key ("yhat", "trend")) for the samples of "yhat" and "trend".
    """
    # Get the seasonality and regressor components, which are deterministic per iteration
    beta = prophet_model.params["beta"][iteration]
    Xb_a = (
        np.matmul(seasonal_features.values, beta * s_a.values) * prophet_model.y_scale
    )
    Xb_m = np.matmul(seasonal_features.values, beta * s_m.values)
    # Get the future trend, which is stochastic per iteration
    trends = _sample_predictive_trend_vectorized(
        prophet_model, df, n_samples, iteration
    )

    sigma = prophet_model.params["sigma_obs"][iteration]
    noise_terms = np.random.normal(0, sigma, trends.shape) * prophet_model.y_scale

    return {"yhat": trends * (1 + Xb_m) + Xb_a + noise_terms, "trend": trends}


def sample_posterior_predictive(
    prophet_model: Prophet, df: pd.DataFrame, vectorized: bool
) -> Dict[str, np.ndarray]:
    """Generate posterior samples of a trained Prophet model.

    Args:
        df: a `pd.DataFrame` object with dates and necessary information for predictions.
        vectorized: a boolean for whether to use a vectorized method to generate posterior samples.

    Returns:
        A dictionary with keys ("yhat", "trend") for posterior predictive samples for the "yhat" and "trend".
    """
    n_iterations = prophet_model.params["k"].shape[0]
    samp_per_iter = max(
        1, int(np.ceil(prophet_model.uncertainty_samples / float(n_iterations)))
    )
    # Generate seasonality features once so we can re-use them.
    (
        seasonal_features,
        _,
        component_cols,
        _,
    ) = prophet_model.make_all_seasonality_features(df)
    sim_values = {"yhat": [], "trend": []}
    for i in range(n_iterations):
        if vectorized:
            sims = sample_model_vectorized(
                prophet_model,
                df=df,
                seasonal_features=seasonal_features,
                iteration=i,
                s_a=component_cols["additive_terms"],
                s_m=component_cols["multiplicative_terms"],
                n_samples=samp_per_iter,
            )
            for k in sim_values:
                sim_values[k].append(sims[k])
        else:
            sims = [
                prophet_model.sample_model(
                    df=df,
                    seasonal_features=seasonal_features,
                    iteration=i,
                    s_a=component_cols["additive_terms"],
                    s_m=component_cols["multiplicative_terms"],
                )
                for _ in range(samp_per_iter)
            ]
            for key in sim_values:
                for sim in sims:
                    sim_values[key].append(sim[key].values)
    for k, v in sim_values.items():
        sim_values[k] = np.row_stack(v)
    return cast(Dict[str, np.ndarray], sim_values)


def _make_historical_mat_time(
    deltas: np.ndarray,
    changepoints_t: np.ndarray,
    n_row: int,
    single_diff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a matrix of slope-deltas where these changes occured in training data according to the trained prophet obj
    """
    prev_time = np.arange(0, 1 + single_diff, single_diff)
    idxs = []
    for changepoint in changepoints_t:
        idxs.append(np.where(prev_time > changepoint)[0][0])
    prev_deltas = np.zeros(len(prev_time))
    prev_deltas[idxs] = deltas
    prev_deltas = np.repeat(prev_deltas.reshape(1, -1), n_row, axis=0)
    return prev_deltas, prev_time


def _logistic_uncertainty(
    prophet_model: Prophet,
    mat: np.ndarray,
    deltas: np.ndarray,
    k: float,
    m: float,
    cap: np.ndarray,
    t_time: np.ndarray,
    n_length: int,
    single_diff: float,
) -> np.ndarray:
    """Vectorizes prophet's logistic uncertainty by creating a matrix of future possible trends.
    Args:
        mat: A `np.array` for trend shift matrix returned by _make_trend_shift_matrix()
        deltas: A `np.array` for the size of the trend changes at each changepoint, estimated by the model
        k: A float for initial rate.
        m: A float initial offset.
        cap: A `np.array` of capacities at each t.
        n_length: For each path, the number of future steps to simulate
        single_diff: The difference between each t step in the model context.
    Returns:
        A `np.array` with shape (n_samples, n_length), representing the width of the uncertainty interval (standardized, not on the same scale as the actual data values) around 0.`
    """

    def _ffill(arr: np.ndarray) -> np.ndarray:
        mask = arr == 0
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return arr[np.arange(idx.shape[0])[:, None], idx]

    # for logistic growth we need to evaluate the trend all the way from the start of the train item
    historical_mat, historical_time = _make_historical_mat_time(
        deltas, prophet_model.changepoints_t, len(mat), single_diff
    )
    mat = np.concatenate([historical_mat, mat], axis=1)
    full_t_time = np.concatenate([historical_time, t_time])

    # apply logistic growth logic on the slope changes
    k_cum = np.concatenate(
        (np.ones((mat.shape[0], 1)) * k, np.where(mat, np.cumsum(mat, axis=1) + k, 0)),
        axis=1,
    )
    k_cum_b = _ffill(k_cum)
    gammas = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        x = full_t_time[i] - m - np.sum(gammas[:, :i], axis=1)
        ks = 1 - k_cum_b[:, i] / k_cum_b[:, i + 1]
        gammas[:, i] = x * ks
    # the data before the -n_length is the historical values, which are not needed, so cut the last n_length
    k_t = (mat.cumsum(axis=1) + k)[:, -n_length:]
    m_t = (gammas.cumsum(axis=1) + m)[:, -n_length:]
    sample_trends = cap / (1 + np.exp(-k_t * (t_time - m_t)))
    # remove the mean because we only need width of the uncertainty centered around 0
    # we will add the width to the main forecast - yhat (which is the mean) - later
    return sample_trends - sample_trends.mean(axis=0)
