# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Median ensembling method

Ensemble models with median of individual models
Assume we have k base models, after we make forecasts with each individual
model, we take the median from each time point as the final results
"""

import logging
from typing import cast, List, Optional

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

    freq: Optional[str] = None
    fcst: Optional[pd.DataFrame] = None
    fcst_dates: Optional[pd.DatetimeIndex] = None
    dates: Optional[pd.DatetimeIndex] = None
    fcst_df: Optional[pd.DataFrame] = None

    def __init__(self, data: TimeSeriesData, params: EnsembleParams) -> None:
        self.data = data
        self.params = params
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
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
        # Keep freq in the parameters passed to _predict_all()
        self.freq = freq = kwargs.get("freq", "D")
        pred_dict = self._predict_all(steps, **kwargs)

        fcst_all = pd.concat(
            # pyre-fixme[16]: `Model` has no attribute `fcst`.
            [x.fcst.reset_index(drop=True) for x in pred_dict.values()],
            axis=1,
            copy=False,
        )
        fcst_all.columns = cast(List[str], pred_dict.keys())
        self.fcst = fcst_all.median(axis=1)

        # create future dates
        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)
        dates = dates[dates != last_date]
        self.fcst_dates = dates.to_pydatetime()
        self.dates = dates[dates != last_date]
        self.fcst_df = fcst_df = pd.DataFrame(
            {"time": self.dates, "fcst": self.fcst}, copy=False
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return fcst_df

    def __str__(self) -> str:
        """Get default parameter search space for the median ensemble model

        Args:
            None

        Returns:
            Model name as a string
        """
        return "Median Ensemble"
