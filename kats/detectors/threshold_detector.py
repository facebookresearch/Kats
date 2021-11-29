# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_from_json, model_to_json
from kats.consts import (
    DEFAULT_VALUE_NAME,
    TimeSeriesData,
)
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse


class StaticThresholdModel(DetectorModel):
    """Static threshold detection model.

    This model optimizes automatically the static thresholds. It implements the (very)
    simple logic of a static threshold.

    Attributes:
        serialized_model: json, representing data from a previously
            serialized model.
    """

    model: Optional[Prophet]

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        upper_threshold: float = 1,
        lower_threshold: float = 0,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

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
            ),
        )

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """Comprare the TimeSeriesData with the upper and lower threshold, and returns the
        0 or 1 based on the value.

        Returns the AnomalyResponse, when data is passed to it.

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            AnomalyResponse object whose scores is a time series of 0 and 1. The length of this object is same as data.
        """
        output_ts = self._zeros_ts(data)
        output_ts.value.loc[data.value > self.upper_threshold] = 1
        output_ts.value.loc[data.value < self.lower_threshold] = 1
        return AnomalyResponse(
            scores=output_ts,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=self._zeros_ts(data),
            stat_sig_ts=None,
        )

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
    ) -> None:
        self.fit_predict(data, historical_data)
        return

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")
