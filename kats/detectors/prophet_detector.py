# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains code to implement the Prophet algorithm
as a Detector Model.
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    from fbprophet import Prophet
    from fbprophet.serialize import model_from_json, model_to_json

    _no_prophet = False
except ImportError:
    _no_prophet = True

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

MIN_STDEV = 1e-9


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


def deviation_from_predicted_val(
    data: TimeSeriesData,
    predict_df: pd.DataFrame,
    ci_threshold: Optional[float] = None,
    uncertainty_samples: Optional[float] = None,
):
    return (data.value - predict_df[PROPHET_YHAT_COLUMN]) / predict_df[
        PROPHET_YHAT_COLUMN
    ].abs()


def z_score(
    data: TimeSeriesData,
    predict_df: pd.DataFrame,
    ci_threshold: float = 0.8,
    uncertainty_samples: float = 50,
):
    # asymmetric confidence band => points above the prediction use upper bound in calculation, points below the prediction use lower bound

    actual_upper_std = (
        (uncertainty_samples ** 0.5)
        * (predict_df[PROPHET_YHAT_UPPER_COLUMN] - predict_df[PROPHET_YHAT_COLUMN])
        / ci_threshold
    )
    actual_lower_std = (
        (uncertainty_samples ** 0.5)
        * (predict_df[PROPHET_YHAT_COLUMN] - predict_df[PROPHET_YHAT_LOWER_COLUMN])
        / ci_threshold
    )

    # if std is 0, set it to a very small value to prevent division by zero in next step
    upper_std = np.maximum(actual_upper_std, MIN_STDEV)
    lower_std = np.maximum(actual_lower_std, MIN_STDEV)

    upper_score = (
        (data.value > predict_df[PROPHET_YHAT_COLUMN])
        * (data.value - predict_df[PROPHET_YHAT_COLUMN])
        / upper_std
    )
    lower_score = (
        (data.value < predict_df[PROPHET_YHAT_COLUMN])
        * (data.value - predict_df[PROPHET_YHAT_COLUMN])
        / lower_std
    )

    return upper_score + lower_score


class ProphetScoreFunction(Enum):
    deviation_from_predicted_val = "deviation_from_predicted_val"
    z_score = "z_score"


SCORE_FUNC_DICT = {
    ProphetScoreFunction.deviation_from_predicted_val.value: deviation_from_predicted_val,
    ProphetScoreFunction.z_score.value: z_score,
}


class ProphetDetectorModel(DetectorModel):
    """Prophet based anomaly detection model.

    A Detector Model that does anomaly detection, by first using the Prophet
    library to forecast the interval for the next point, and comparing this
    to the actually observed data point.

    Attributes:
        scoring_confidence_interval: interval_width as required by Prophet.
            Confidence interval is used by some scoring strategies to compute
            anomaly scores.
        uncertainty_samples: Number of samples required by Prophet to
            calculate uncertainty.
        serialized_model: json, representing data from a previously
            serialized model.
    """

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        score_func: ProphetScoreFunction = ProphetScoreFunction.deviation_from_predicted_val,
        scoring_confidence_interval: float = 0.8,
        remove_outliers=False,
        outlier_threshold: float = 0.99,
        uncertainty_samples: float = 50,
    ) -> None:
        if _no_prophet:
            raise RuntimeError("requires fbprophet to be installed")
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None

        self.score_func = score_func
        self.scoring_confidence_interval = scoring_confidence_interval
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
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
            interval_width=self.scoring_confidence_interval,
            uncertainty_samples=self.uncertainty_samples,
        )

        data_df = timeseries_to_prophet_df(total_data)

        if self.remove_outliers:
            data_df = self._remove_outliers(data_df, self.outlier_threshold)

        self.model.fit(data_df)

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
                value=SCORE_FUNC_DICT[self.score_func.value](
                    data=data,
                    predict_df=predict_df,
                    ci_threshold=self.model.interval_width,
                    uncertainty_samples=self.uncertainty_samples,
                ),
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

    @staticmethod
    def _remove_outliers(
        ts_df: pd.DataFrame,
        outlier_ci_threshold: float = 0.99,
        uncertainty_samples: float = 50,
    ) -> pd.DataFrame:
        """
        Remove outliers from the time series by fitting a Prophet model to the time series
        and stripping all points that fall outside the confidence interval of the predictions
        of the model.
        """

        ts_dates_df = pd.DataFrame({PROPHET_TIME_COLUMN: ts_df.iloc[:, 0]})

        model = Prophet(
            interval_width=outlier_ci_threshold, uncertainty_samples=uncertainty_samples
        )
        model_pass1 = model.fit(ts_df)

        forecast = model_pass1.predict(ts_dates_df)

        is_outlier = (
            ts_df[PROPHET_VALUE_COLUMN] < forecast[PROPHET_YHAT_LOWER_COLUMN]
        ) | (ts_df[PROPHET_VALUE_COLUMN] > forecast[PROPHET_YHAT_UPPER_COLUMN])

        ts_df = ts_df[~is_outlier]

        return ts_df
