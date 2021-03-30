#!/usr/bin/env python3

import json
from typing import Optional

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_from_json, model_to_json
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detector import DetectorModel
from infrastrategy.kats.detectors.detector_consts import (
    AnomalyResponse,
    ConfidenceBand,
)

PROPHET_TIME_COLUMN = "ds"
PROPHET_VALUE_COLUMN = "y"
PROPHET_YHAT_COLUMN = "yhat"
PROPHET_YHAT_LOWER_COLUMN = "yhat_lower"
PROPHET_YHAT_UPPER_COLUMN = "yhat_upper"


def timeseries_to_prophet_df(ts_data: TimeSeriesData) -> pd.DataFrame:
    if not ts_data.is_univariate():
        raise ValueError("ProphetModel only works with univariate data")

    return pd.DataFrame(
        {PROPHET_TIME_COLUMN: ts_data.time, PROPHET_VALUE_COLUMN: ts_data.value}
    )


class ProphetDetectorModel(DetectorModel):
    def __init__(
        self,
        strictness_factor: float = 0.8,
        uncertainty_samples: float = 50,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(json.loads(serialized_model))
        else:
            self.model = None
            self.strictness_factor = strictness_factor
            self.uncertainty_samples = uncertainty_samples

    def serialize(self) -> bytes:
        return str.encode(json.dumps(model_to_json(self.model)))

    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        # train on historical, then predict on all data.
        self.fit(data=historical_data, historical_data=None)
        return self.predict(data)

    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """
        fit can be called during priming. We train a model using all the data passed in.
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

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """
        Predict only expects anomaly score for data. Prophet doesn't need historical_data.
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
