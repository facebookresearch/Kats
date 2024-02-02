# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from kats.consts import DEFAULT_VALUE_NAME, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse


class StaticThresholdModel(DetectorModel):
    """Static threshold detection model.

    This model optimizes automatically the static thresholds. It implements the (very)
    simple logic of copying of the time series.

    Attributes:
        serialized_model: json, representing data from a previously
            serialized model.
    """

    model: Optional[Prophet]

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        So it can be loaded later.

        Returns:
            json containing information of the model.
        """
        return str.encode(model_to_json(self.model))

    def _zeros_ts(self, data: TimeSeriesData) -> TimeSeriesData:
        return TimeSeriesData(
            time=data.time,
            value=pd.Series(
                np.zeros(len(data)),
                name=data.value.name if data.value.name else DEFAULT_VALUE_NAME,
                copy=False,
            ),
        )

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Copy the TimeSeriesData.

        Returns the AnomalyResponse, when data is passed to it.

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            AnomalyResponse object whose scores is a copy of the time series. The length of this object is same as data.
        """
        output_ts = self._zeros_ts(data)
        output_ts.value = data.value
        return AnomalyResponse(
            scores=output_ts,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=self._zeros_ts(data),
            stat_sig_ts=None,
        )

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> None:
        self.fit_predict(data, historical_data)
        return

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")
