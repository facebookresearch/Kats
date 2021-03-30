#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional

import infrastrategy.kats.models.model as m
import pandas as pd
from fbprophet import Prophet
from infrastrategy.kats.consts import Params, TimeSeriesData
from infrastrategy.kats.utils.parameter_tuning_utils import (
    get_default_prophet_parameter_search_space,
)


class ProphetParams(Params):
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
            [] if custom_seasonalities is None
            else custom_seasonalities
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
    def __init__(self, data: TimeSeriesData, params: ProphetParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def fit(self, **kwargs) -> None:
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

        self.model = prophet.fit(df=df)
        logging.info("Fitted Prophet model. ")

    def predict(self, steps, include_history=False, **kwargs) -> pd.DataFrame:
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        self.include_history = include_history
        # prepare future for Prophet.predict
        future = kwargs.get("future")
        raw = kwargs.get("raw", False)
        if future is None:
            future = self.model.make_future_dataframe(
                periods=steps,
                freq=self.freq,
                include_history=self.include_history)
            if self.params.growth == "logistic":
                # assign cap to a new col as Prophet required
                future["cap"] = self.params.cap
            if self.params.floor is not None:
                future["floor"] = self.params.floor

        fcst = self.model.predict(future).tail(steps)
        if raw:
            return fcst

        # if include_history:
        fcst = self.model.predict(future)
        logging.info("Generated forecast data from Prophet model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

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
        logging.info("Generating chart for forecast result from Prophet model.")
        m.Model.plot(self.data, self.fcst_df, include_history=self.include_history)

    def __str__(self):
        return "Prophet"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """
        Move the implementation of get_parameter_search_space() out of prophet
        to avoid version conflict between Ax and fbprophet. Both of them depend
        on eigen but different version, and cause crash in runtime.
        See SEV S201367 for detail.
        """
        return get_default_prophet_parameter_search_space()
