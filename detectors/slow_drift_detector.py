#!/usr/bin/env python3

import numpy as np
import pandas as pd

from typing import List
from typing import Optional

from data_ai.slow_drift import utils
from data_ai.slow_drift.model import Model
from data_ai.slow_drift.evaluate_op_stateless import EvaluateOpStateless
from data_ai.slow_drift.slow_drift.ttypes import DataPoint, ExponentialSmoothingParameters, ModelData
from facebook.monitoring.anomaly_structs.ttypes import Anomaly
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detector import DetectorModel
from infrastrategy.kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand


DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.015
DEFAULT_GAMMA = 0.3

def time_series_to_data_points(data: TimeSeriesData) -> List[DataPoint]:
    if not data.is_univariate():
        raise ValueError("Multiple time series not supported for Slow Drift")

    time_copy = data.time.astype(int) / 10**9
    return [DataPoint(timestamp=t, value=v) for t, v in zip(time_copy.values, data.value.values)]


class SlowDriftDetectorModel(DetectorModel):
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
        return utils._serialize_model_data(self.model.get_model_data)

    def fit(self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None) -> None:
        evaluate_op = EvaluateOpStateless(
            ts=time_series_to_data_points(data),
            model=self.model,
            trend_alpha=DEFAULT_ALPHA,
            trend_beta=DEFAULT_BETA,
            trend_gamma=DEFAULT_GAMMA,
        )
        self.model = evaluate_op.train()

    def predict(self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None) -> AnomalyResponse:
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

    def fit_predict(self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None) -> AnomalyResponse:
        self.fit(data, historical_data)
        return self.predict(data, historical_data)
