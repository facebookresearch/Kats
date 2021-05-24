#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains code to implement the Prophet algorithm
as a Detector Model.
"""

from typing import Optional

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_from_json, model_to_json
from kats.consts import TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import (
    AnomalyResponse,
    ConfidenceBand,
)

PROPHET_TIME_COLUMN = "ds"
PROPHET_VALUE_COLUMN = "y"
PROPHET_YHAT_COLUMN = "yhat"
PROPHET_YHAT_LOWER_COLUMN = "yhat_lower"
PROPHET_YHAT_UPPER_COLUMN = "yhat_upper"


def timeseries_to_prophet_df(ts_data: TimeSeriesData) -> pd.DataFrame:
    """Converts a object of TimeSeriesData to a dataframe, as expected by Prophet.

    Args:
        ts_data: object of class TimeSeriesData.

    Returns:
        pandas DataFrame expected by Prophet.
    """

    if not ts_data.is_univariate():
        raise ValueError("ProphetModel only works with univariate data")

    return pd.DataFrame(
        {
            PROPHET_TIME_COLUMN: ts_data.time,
            PROPHET_VALUE_COLUMN: ts_data.value,
        }
    )


class ProphetDetectorModel(DetectorModel):
    """Prophet based anomaly detection model.

    A Detector Model that does anomaly detection, buy first using the Prophet
    library to forecast the interval for the next point, and comparing this
    to the actually observed data point.

    Attributes:
        strictness_factor: interval_width as required by Prophet.
        uncertainty_samples: Number of samples required by Prophet to
            calculate uncertainty.
        serialized_model: json, representing data from a previously
            serialized model.
    """

    def __init__(
        self,
        strictness_factor: float = 0.8,
        uncertainty_samples: float = 50,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None
            self.strictness_factor = strictness_factor
            self.uncertainty_samples = uncertainty_samples

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        So it can be loaded later.

        Returns:
            json containing information of the model.
        """
        return str.encode(model_to_json(self.model))

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """Trains a model, and returns the anomaly scores.

        Returns the AnomalyResponse, when data is passed to it.

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            AnomalyResponse object. The length of this object is same as data. The score property
            gives the score for anomaly.
        """

        # train on historical, then predict on all data.
        # pyre-fixme[6]: Expected `TimeSeriesData` for 1st param but got
        #  `Optional[TimeSeriesData]`.
        self.fit(data=historical_data, historical_data=None)
        return self.predict(data)

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel` inconsistently.
    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """Used to train a model.

        fit can be called during priming. We train a model using all the data passed in.

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            None.
        """

        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data)
            total_data = historical_data

        # No incremental training. Create a model and train from scratch
        self.model = Prophet(
            interval_width=self.strictness_factor,
            uncertainty_samples=self.uncertainty_samples,
        )

        self.model.fit(timeseries_to_prophet_df(total_data))

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """Predicts anomaly score for future data.

        Predict only expects anomaly score for data. Prophet doesn't need historical_data.

        Args:
            data: TimeSeriesData on which detection is run
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            AnomalyResponse object. The length of this obj.ect is same as data. The score property
            gives the score for anomaly.
        """
        time_df = pd.DataFrame({PROPHET_TIME_COLUMN: data.time})
        predict_df = self.model.predict(time_df)
        zeros = np.zeros(len(data))
        response = AnomalyResponse(
            scores=TimeSeriesData(
                time=data.time,
                value=(data.value - predict_df[PROPHET_YHAT_COLUMN])
                / predict_df[PROPHET_YHAT_COLUMN].abs(),
            ),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=data.time, value=predict_df[PROPHET_YHAT_UPPER_COLUMN]
                ),
                lower=TimeSeriesData(
                    time=data.time, value=predict_df[PROPHET_YHAT_LOWER_COLUMN]
                ),
            ),
            predicted_ts=TimeSeriesData(
                time=data.time, value=predict_df[PROPHET_YHAT_COLUMN]
            ),
            anomaly_magnitude_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
            stat_sig_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
        )
        return response
