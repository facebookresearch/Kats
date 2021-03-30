#!/usr/bin/env python3
import logging

# Ensemble models with median of individual models
# Assume we have k base models, after we make forecasts with each individual
# model, we take the median from each time point as the final results
import pandas as pd
import infrastrategy.kats.models.model as mm
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models.ensemble import ensemble
from infrastrategy.kats.models.ensemble.ensemble import EnsembleParams


class MedianEnsembleModel(ensemble.BaseEnsemble):
    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def predict(self, steps, **kwargs):
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        self.freq = kwargs.get("freq", "D")
        pred_dict = self._predict_all(steps, **kwargs)

        fcst_all = pd.concat(
            [x.fcst.reset_index(drop=True) for x in pred_dict.values()], axis=1
        )
        fcst_all.columns = pred_dict.keys()
        self.fcst = fcst_all.median(axis=1)

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        dates = dates[dates != last_date]
        self.fcst_dates = dates.to_pydatetime()
        self.dates = dates[dates != last_date]

        self.fcst_df = pd.DataFrame({"time": self.dates, "fcst": self.fcst})

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        logging.info("Generating chart for forecast result from Ensemble.")
        mm.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "Median Ensemble"
