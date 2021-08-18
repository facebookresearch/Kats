# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Median ensembling method

Ensemble models with median of individual models
Assume we have k base models, after we make forecasts with each individual
model, we take the median from each time point as the final results
"""

import logging

import kats.models.model as mm
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.ensemble import ensemble
from kats.models.ensemble.ensemble import EnsembleParams


class MedianEnsembleModel(ensemble.BaseEnsemble):
    """Median ensemble model class

    Attributes:
        data: the input time series data as in :class:`kats.consts.TimeSeriesData`
        params: the model parameter class in Kats
    """

    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def predict(self, steps: int, **kwargs):
        """Predict method of median ensemble model

        Args:
            steps: the length of forecasting horizon

        Returns:
            forecasting results as in pd.DataFrame
        """

        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `MedianEnsembleModel` has no attribute `freq`.
        self.freq = kwargs.get("freq", "D")
        pred_dict = self._predict_all(steps, **kwargs)

        fcst_all = pd.concat(
            [x.fcst.reset_index(drop=True) for x in pred_dict.values()], axis=1
        )
        fcst_all.columns = pred_dict.keys()
        # pyre-fixme[16]: `MedianEnsembleModel` has no attribute `fcst`.
        self.fcst = fcst_all.median(axis=1)

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        dates = dates[dates != last_date]
        # pyre-fixme[16]: `MedianEnsembleModel` has no attribute `fcst_dates`.
        self.fcst_dates = dates.to_pydatetime()
        # pyre-fixme[16]: `MedianEnsembleModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]

        # pyre-fixme[16]: `MedianEnsembleModel` has no attribute `fcst_df`.
        self.fcst_df = pd.DataFrame({"time": self.dates, "fcst": self.fcst})

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def plot(self):
        """Plot method for median ensemble model"""

        logging.info("Generating chart for forecast result from Ensemble.")
        mm.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        """Get default parameter search space for the median ensemble model

        Args:
            None

        Returns:
            Model name as a string
        """
        return "Median Ensemble"
