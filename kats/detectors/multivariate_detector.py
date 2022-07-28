# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements the multivariate Outlier Detection algorithm as a Detector Model.
"""
import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand
from kats.detectors.outlier import (
    MultivariateAnomalyDetector,
    MultivariateAnomalyDetectorType,
)
from kats.models.var import VARParams


class MultivariateAnomalyDetectorModel(DetectorModel):
    """Anamoly detection model based on outlier detector.

    A Detector Model detects outliers for multivariate time series

    Attributes:
        data: Input metrics TimeSeriesData
        params: Parameter class for multivariate VAR/ BVAR model
        training_days: num of days of data to use for initial training.
                    If less than a day, specify as fraction of a day
        model_type: The type of multivariate anomaly detector (currently 'BAYESIAN_VAR' and 'VAR' options available)
        serialized_model: Optional; json, representing data from a previously serialized model.
    """

    params: Params = VARParams(maxlags=2)
    training_days: float
    model_type: MultivariateAnomalyDetectorType = MultivariateAnomalyDetectorType.VAR
    model: Optional[MultivariateAnomalyDetector] = None

    def __init__(
        self,
        params: Params = params,
        training_days: float = 60.0,
        serialized_model: Optional[bytes] = None,
    ) -> None:
        if serialized_model:
            model_dict = json.loads(serialized_model)
            self.params = model_dict["params"]
            self.training_days = model_dict["training_days"]
        else:
            if params:
                self.params = params
            self.training_days = training_days

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        Serialize the model into a json so it can be loaded later.

        Returns:
            json containing information of the model.
        """
        model_dict = {
            "params": self.params,
            "training_days": self.training_days,
            "model_type": self.model_type,
        }

        # model_dict = {"training_days": self.training_days, "model_type": self.model_type}
        return json.dumps(model_dict).encode("utf-8")

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """Fit MultivariateAnomalyDetector.

        Fit MultivariateAnomalyDetector using both data and historical_data (if provided).

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

        self.model = MultivariateAnomalyDetector(
            data=total_data,
            params=self.params,
            training_days=self.training_days,
            model_type=self.model_type,
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
        if self.model is None:
            self.fit(data=data, historical_data=historical_data)

        assert self.model is not None
        output_scores_df = self.model.anomaly_score_df

        assert output_scores_df is not None
        output_scores_df = output_scores_df[output_scores_df.index >= data.time.min()]

        zeros = np.zeros(shape=(data.time.shape[0], output_scores_df.shape[1]))
        padding = np.empty(
            shape=[len(data) - output_scores_df.shape[0], output_scores_df.shape[1]]
        )
        padding[:] = np.NaN
        padding = pd.DataFrame(padding, columns=output_scores_df.columns, copy=False)
        # all fields other than scores are left as TimeSeriesData with all zero values
        response = AnomalyResponse(
            scores=TimeSeriesData(
                time=data.time,
                value=pd.concat(
                    [padding.iloc[:, :-2], output_scores_df.iloc[:, :-2]],
                    ignore_index=True,
                    copy=False,
                ),
            ),
            confidence_band=ConfidenceBand(
                upper=TimeSeriesData(
                    time=data.time, value=pd.DataFrame(zeros, copy=False)
                ),
                lower=TimeSeriesData(
                    time=data.time, value=pd.DataFrame(zeros, copy=False)
                ),
            ),
            predicted_ts=TimeSeriesData(
                time=data.time, value=pd.DataFrame(zeros).iloc[:, :-2]
            ),
            anomaly_magnitude_ts=TimeSeriesData(
                time=data.time,
                value=pd.concat(
                    [padding.iloc[:, -2], output_scores_df.iloc[:, -2]],
                    ignore_index=True,
                    copy=False,
                ),
            ),
            stat_sig_ts=TimeSeriesData(
                time=data.time,
                value=pd.concat(
                    [padding.iloc[:, -1], output_scores_df.iloc[:, -1]],
                    ignore_index=True,
                    copy=False,
                ),
            ),
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

        return self.predict(data=data)
