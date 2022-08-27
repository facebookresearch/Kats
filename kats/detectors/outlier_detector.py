# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements the univariate Outlier Detection algorithm as a Detector Model.
"""
import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand
from kats.detectors.outlier import OutlierDetector


class OutlierDetectorModel(DetectorModel):
    """Anamoly detection model based on outlier detector.

    A Detector Model detects outliers for a single time series by decomposing the time series
    and detecting outliers when the values in the residual time series are beyond the
    specified multiplier times the inter quartile range.

    Attributes:
        decomp: 'additive' or 'multiplicative' decomposition
        iqr_mult: iqr_mult * inter quartile range is used to classify outliers
        serialized_model: Optional; json, representing data from a previously serialized model.
    """

    decomp: str
    iqr_mult: float
    model: Optional[OutlierDetector] = None

    def __init__(
        self,
        decomp: str = "additive",
        iqr_mult: float = 3.0,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.decomp = model_dict["decomp"]
            self.iqr_mult = model_dict["iqr_mult"]
        else:
            self.decomp = decomp
            self.iqr_mult = iqr_mult

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        Serialize the model into a json so it can be loaded later.

        Returns:
            json containing information of the model.
        """
        model_dict = {"decomp": self.decomp, "iqr_mult": self.iqr_mult}
        return json.dumps(model_dict).encode("utf-8")

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """Fit OutlierDetector.

        Fit OutlierDetector using both data and historical_data (if provided).

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: Optional; TimeSeriesData corresponding to history. History ends
                exactly where the data begins.

        Returns:
            None.
        """
        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data)
            total_data = historical_data

        self.model = OutlierDetector(
            data=total_data, decomp=self.decomp, iqr_mult=self.iqr_mult
        )
        self.model.detector()

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Get anomaly scores.

        Get anomaly scores only for data.

        Args:
            data: TimeSeriesData on which detection is run
            historical_data: Optional; TimeSeriesData corresponding to history. History ends
                exactly where the data begins.

        Returns:
            AnomalyResponse object. The length of this obj.ect is same as data. The score property
            gives the score for anomaly.
        """
        # When no iterpolate argument is given by default it is taking False
        if "interpolate" not in kwargs:
            interpolate = bool(False)
        else:
            interpolate = bool(kwargs["interpolate"])

        if self.model is None:
            self.fit(data=data, historical_data=historical_data)

        assert self.model is not None
        output_scores_df = self.model.output_scores
        output_detector_remover = self.model.remover(interpolate=interpolate)

        assert output_scores_df is not None
        output_scores_df = output_scores_df[output_scores_df.index >= data.time.min()]

        zeros = pd.DataFrame(np.zeros(shape=output_scores_df.shape), copy=False)
        # all fields other than scores and predicted_ts are left as TimeSeriesData with all zero values
        response = AnomalyResponse(
            scores=TimeSeriesData(
                time=data.time,
                value=output_scores_df,
            ),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(time=data.time, value=zeros),
                lower=TimeSeriesData(time=data.time, value=zeros),
            ),
            predicted_ts=TimeSeriesData(
                time=output_detector_remover.time,
                value=pd.DataFrame(output_detector_remover.value, copy=False),
            ),
            anomaly_magnitude_ts=TimeSeriesData(time=data.time, value=zeros),
            stat_sig_ts=TimeSeriesData(time=data.time, value=zeros),
        )

        return response

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Fit a model and return the anomaly scores.

        Return AnomalyResponse, when data is passed to it.

        Args:
            data: TimeSeriesData on which detection is run.
            historical_data: Optional; TimeSeriesData corresponding to history. History ends
                exactly where the data begins.

        Returns:
            AnomalyResponse object. The length of this object is same as data. The score property
            gives the score for anomaly.
        """
        self.fit(data=data, historical_data=historical_data)

        return self.predict(data=data, **kwargs)
