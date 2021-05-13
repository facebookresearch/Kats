#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Slow Drift detector used in One Detection.
For details, see
https://www.internalfb.com/intern/wiki/Anomaly_Detection/Technical_Guide/Slow_Drift_detectors/
"""

from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from data_ai.slow_drift import utils
from data_ai.slow_drift.evaluate_op_stateless import EvaluateOpStateless
from data_ai.slow_drift.model import Model
from data_ai.slow_drift.slow_drift.ttypes import (
    DataPoint,
    ExponentialSmoothingParameters,
    ModelData,
)
from facebook.monitoring.anomaly_structs.ttypes import Anomaly
from kats.consts import TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand


DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.015
DEFAULT_GAMMA = 0.3


def time_series_to_data_points(data: TimeSeriesData) -> List[DataPoint]:
    if not data.is_univariate():
        raise ValueError("Multiple time series not supported for Slow Drift")

    # Additional conversion after division required: otherwise the result is not serializable
    time_copy = (data.time.astype(int) / 10 ** 9).astype(int)
    return [
        DataPoint(timestamp=t, value=v)
        for t, v in zip(time_copy.values, data.value.values)
    ]


class SlowDriftDetectorModel(DetectorModel):
    """
    Designed to find changes to the behaviour of analytical metrics
    that occur of a period of weeks or months, typically too long for Predictive detectors
    to flag changes.

    Contains fit and predict methods
    """

    def __init__(
        self,
        slow_drift_window: int,
        algorithm_version: int,
        seasonality_period: int,
        seasonality_num_points: int,
        n_stdev: float = 1.0,
        ongoing_anomaly: Optional[Anomaly] = None,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model is not None:
            model_data: ModelData = utils._deserialize_model_data(serialized_model)
            self.model = Model(model_data)
        else:
            # pyre-fixme[6]: Expected `str` for 1st param but got `None`.
            self.model = Model.new_model(None, slow_drift_window, algorithm_version)
            self.model.set_parameters(
                ExponentialSmoothingParameters(
                    seasonalityPeriod=seasonality_period,
                    seasonalityNumPoints=seasonality_num_points,
                ),
            )

        self.model.set_ongoing_anomaly(ongoing_anomaly)
        self.n_stdev = n_stdev

    def serialize(self) -> bytes:
        return utils._serialize_model_data(self.model.get_model_data())

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel` inconsistently.
    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """Fit exponential smoothing model to smooth the data"""

        evaluate_op = EvaluateOpStateless(
            ts=time_series_to_data_points(data),
            model=self.model,
            trend_alpha=DEFAULT_ALPHA,
            trend_beta=DEFAULT_BETA,
            trend_gamma=DEFAULT_GAMMA,
        )
        self.model = evaluate_op.train()

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """Return anomalies by detecing spikes in second derivative of smoothed series"""
        evaluate_op = EvaluateOpStateless(
            ts=time_series_to_data_points(data),
            model=self.model,
            trend_alpha=DEFAULT_ALPHA,
            trend_beta=DEFAULT_BETA,
            trend_gamma=DEFAULT_GAMMA,
            n_stdev=self.n_stdev,
        )
        result = evaluate_op.evaluate()
        zeros = np.zeros(len(data))
        return AnomalyResponse(
            scores=TimeSeriesData(
                time=data.time,
                value=pd.Series([d.value for d in result.scoreTimeSeries]),
            ),
            confidence_band=ConfidenceBand(
                lower=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
                upper=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
            ),
            predicted_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
            anomaly_magnitude_ts=TimeSeriesData(
                time=data.time,
                value=pd.Series([d.value for d in result.magnitudeTimeSeries]),
            ),
            stat_sig_ts=TimeSeriesData(time=data.time, value=pd.Series(zeros)),
        )

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """Fits Exponential smoothing model and returns the anomalous drift location as AnomalyResponse"""

        self.fit(data, historical_data)
        return self.predict(data, historical_data)
