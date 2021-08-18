# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional

import pandas as pd

try:
    from fbprophet import Prophet

    _no_prophet = False
except ImportError:
    _no_prophet = True
    Prophet = dict  # for Pyre

import kats.models.model as m
from kats.consts import Params, TimeSeriesData
from kats.utils.parameter_tuning_utils import (
    get_default_prophet_parameter_search_space,
)


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
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        cap=None,
        floor=None,
        custom_seasonalities: Optional[List[Dict]] = None,
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
        logging.debug(
            "Initialized Prophet with parameters. "
            "growth:{growth},"
            "changepoints:{changepoints},"
            "n_changepoints:{n_changepoints},"
            "changepoint_range:{changepoint_range},"
            "yearly_seasonality:{yearly_seasonality},"
            "weekly_seasonality:{weekly_seasonality},"
            "daily_seasonality:{daily_seasonality},"
            "holidays:{holidays},"
            "seasonality_mode:{seasonality_mode},"
            "seasonality_prior_scale:{seasonality_prior_scale},"
            "holidays_prior_scale:{holidays_prior_scale},"
            "changepoint_prior_scale:{changepoint_prior_scale},"
            "mcmc_samples:{mcmc_samples},"
            "interval_width:{interval_width},"
            "uncertainty_samples:{uncertainty_samples},"
            "cap:{cap},"
            "floor:{floor},"
            "custom_seasonalities:{custom_seasonalities}".format(
                growth=growth,
                changepoints=changepoints,
                n_changepoints=n_changepoints,
                changepoint_range=changepoint_range,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays,
                seasonality_mode=seasonality_mode,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                changepoint_prior_scale=changepoint_prior_scale,
                mcmc_samples=mcmc_samples,
                interval_width=interval_width,
                uncertainty_samples=uncertainty_samples,
                cap=cap,
                floor=floor,
                custom_seasonalities=custom_seasonalities,
            )
        )

    def validate_params(self):
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

        logging.info("Method validate_params() is not fully implemented.")
        pass


class ProphetModel(m.Model):
    """Model class for Prophet

    This class provides fit, predict, and plot methods for Prophet model

    Attributes:
        data: the input time series data as in :class:`kats.consts.TimeSeriesData`
        params: the parameter class definied with `ProphetParams`
    """

    def __init__(self, data: TimeSeriesData, params: ProphetParams) -> None:
        super().__init__(data, params)
        if _no_prophet:
            raise RuntimeError("requires fbprophet to be installed")
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs) -> None:
        """fit Prophet model

        Args:
            None.

        Returns:
            The fitted prophet model object
        """
        # prepare dataframe for Prophet.fit()
        df = pd.DataFrame({"ds": self.data.time, "y": self.data.value})
        logging.debug(
            "Call fit() with parameters: "
            "growth:{growth},"
            "changepoints:{changepoints},"
            "n_changepoints:{n_changepoints},"
            "changepoint_range:{changepoint_range},"
            "yearly_seasonality:{yearly_seasonality},"
            "weekly_seasonality:{weekly_seasonality},"
            "daily_seasonality:{daily_seasonality},"
            "holidays:{holidays},"
            "seasonality_mode:{seasonality_mode},"
            "seasonality_prior_scale:{seasonality_prior_scale},"
            "holidays_prior_scale:{holidays_prior_scale},"
            "changepoint_prior_scale:{changepoint_prior_scale},"
            "mcmc_samples:{mcmc_samples},"
            "interval_width:{interval_width},"
            "uncertainty_samples:{uncertainty_samples},"
            "cap:{cap},"
            "floor:{floor},"
            "custom_seasonalities:{custom_seasonalities}".format(
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
                cap=self.params.cap,
                floor=self.params.floor,
                custom_seasonalities=self.params.custom_seasonalities,
            )
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

        if self.params.growth == "logistic":
            # assign cap to a new col as Prophet required
            df["cap"] = self.params.cap

        # Adding floor if available
        if self.params.floor is not None:
            df["floor"] = self.params.floor

        # Add any specified custom seasonalities.
        for custom_seasonality in self.params.custom_seasonalities:
            prophet.add_seasonality(**custom_seasonality)

        # pyre-fixme[16]: `ProphetModel` has no attribute `model`.
        self.model = prophet.fit(df=df)
        logging.info("Fitted Prophet model. ")

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(self, steps, include_history=False, **kwargs) -> pd.DataFrame:
        """predict with fitted Prophet model

        Args:
            steps: the steps or length of prediction horizon
            include_history: if include the historical data, default as False

        Returns:
            The predicted dataframe with following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `ProphetModel` has no attribute `freq`.
        # pyre-fixme[16]: `ProphetModel` has no attribute `data`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        # pyre-fixme[16]: `ProphetModel` has no attribute `include_history`.
        self.include_history = include_history
        # prepare future for Prophet.predict
        future = kwargs.get("future")
        raw = kwargs.get("raw", False)
        if future is None:
            # pyre-fixme[16]: `ProphetModel` has no attribute `model`.
            # pyre-fixme[16]: `Params` has no attribute `cap`.
            future = self.model.make_future_dataframe(
                periods=steps, freq=self.freq, include_history=self.include_history
            )
            if self.params.growth == "logistic":
                # assign cap to a new col as Prophet required
                future["cap"] = self.params.cap
            if self.params.floor is not None:
                future["floor"] = self.params.floor

        fcst = self.model.predict(future)
        if raw:
            return fcst.tail(steps)

        logging.info("Generated forecast data from Prophet model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        # pyre-fixme[16]: `ProphetModel` has no attribute `fcst_df`.
        self.fcst_df = pd.DataFrame(
            {
                "time": fcst.ds,
                "fcst": fcst.yhat,
                "fcst_lower": fcst.yhat_lower,
                "fcst_upper": fcst.yhat_upper,
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """plot forecasted results from Prophet model"""
        logging.info("Generating chart for forecast result from Prophet model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "Prophet"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """get default parameter search space for Prophet model"""
        # pyre-fixme[7]: Expected `List[Dict[str, object]]` but got `List[Dict[str,
        #  typing.Union[List[typing.Any], bool, str]]]`.
        return get_default_prophet_parameter_search_space()
