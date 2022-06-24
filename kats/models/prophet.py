# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd


try:
    from fbprophet import Prophet

    _no_prophet = False
except ImportError:
    _no_prophet = True
    Prophet = Dict[str, Any]  # for Pyre

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
        cap: capacity, provided for logistic growth
        floor: floor, the fcst value must be greater than the specified floor
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
    cap: Optional[float]
    floor: Optional[float]
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
        cap: Optional[float] = None,
        floor: Optional[float] = None,
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
        self.cap = cap
        self.floor = floor
        self.custom_seasonalities = (
            [] if custom_seasonalities is None else custom_seasonalities
        )
        self.extra_regressors = [] if extra_regressors is None else extra_regressors
        self._reqd_regressor_names: List[str] = []
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
        if (self.growth == "logistic") and (self.cap is None):
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

    def _ts_to_df(self) -> pd.DataFrame:
        if self.data.is_univariate():
            # handel corner case: `value` column is not named as `y`.
            df = pd.DataFrame({"ds": self.data.time, "y": self.data.value}, copy=False)
        else:
            df = self.data.to_dataframe()
            df.rename(columns={self.data.time_col_name: "ds"}, inplace=True)

        col_names = self.params._reqd_regressor_names + ["y", "ds"]
        # add "cap" if needed
        if self.params.growth == "logistic":
            df["cap"] = self.params.cap
            col_names.append("cap")
        # add "floor" if needed
        if self.params.floor is not None:
            df["floor"] = self.params.floor
            col_names.append("floor")

        return df[col_names]

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

        self.model = prophet.fit(df=df)
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

        # when extra_regressors are needed
        if len(self.params.extra_regressors) > 0:
            if future is None:
                msg = "`future` should not be None when extra regressors are needed."
                _error_msg(msg)
            elif not set(self.params._reqd_regressor_names).issubset(future.columns):
                msg = "`future` is missing some regressors!"
                _error_msg(msg)
            elif "ds" not in future.columns:
                msg = "`future` should contain a column named 'ds' representing the timestamps."
                _error_msg(msg)
        elif future is None:
            future = model.make_future_dataframe(
                periods=steps, freq=self.freq, include_history=self.include_history
            )

        if include_history:
            future = future.merge(
                # pyre-fixme
                self.model.history,
                on=["ds"] + self.params._reqd_regressor_names,
                how="outer",
            )

        reqd_length = steps + int(len(self.data) * include_history)
        if len(future) < reqd_length:
            msg = f"Input `future` is not long enough to generate forecasts of {steps} steps."
            _error_msg(msg)
        future.sort_values("ds", inplace=True)
        future = future[:reqd_length]

        if self.params.growth == "logistic":
            # assign cap to a new col as Prophet required
            future["cap"] = self.params.cap
        if self.params.floor is not None:
            future["floor"] = self.params.floor

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
