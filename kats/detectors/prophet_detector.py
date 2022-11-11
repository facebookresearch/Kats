# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains code to implement the Prophet algorithm
as a Detector Model.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_from_json, model_to_json
from kats.consts import DEFAULT_VALUE_NAME, TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand
from kats.models.prophet import predict

PROPHET_TIME_COLUMN = "ds"
PROPHET_VALUE_COLUMN = "y"
PROPHET_YHAT_COLUMN = "yhat"
PROPHET_YHAT_LOWER_COLUMN = "yhat_lower"
PROPHET_YHAT_UPPER_COLUMN = "yhat_upper"


MIN_STDEV = 1e-9
PREDICTION_UNCERTAINTY_SAMPLES = 50
OUTLIER_REMOVAL_UNCERTAINTY_SAMPLES = 40


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
        },
        copy=False,
    )


def deviation_from_predicted_val(
    data: TimeSeriesData,
    predict_df: pd.DataFrame,
    ci_threshold: Optional[float] = None,
    uncertainty_samples: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    return (data.value - predict_df[PROPHET_YHAT_COLUMN]) / predict_df[
        PROPHET_YHAT_COLUMN
    ].abs()


def z_score(
    data: TimeSeriesData,
    predict_df: pd.DataFrame,
    ci_threshold: float = 0.8,
    uncertainty_samples: float = PREDICTION_UNCERTAINTY_SAMPLES,
) -> Union[pd.Series, pd.DataFrame]:
    # asymmetric confidence band => points above the prediction use upper bound in calculation, points below the prediction use lower bound

    actual_upper_std = (
        (uncertainty_samples**0.5)
        * (predict_df[PROPHET_YHAT_UPPER_COLUMN] - predict_df[PROPHET_YHAT_COLUMN])
        / ci_threshold
    )
    actual_lower_std = (
        (uncertainty_samples**0.5)
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


SCORE_FUNC_DICT: Dict[str, Any] = {
    ProphetScoreFunction.deviation_from_predicted_val.value: deviation_from_predicted_val,
    ProphetScoreFunction.z_score.value: z_score,
}

DEFAULT_SCORE_FUNCTION: ProphetScoreFunction = (
    ProphetScoreFunction.deviation_from_predicted_val
)
STR_TO_SCORE_FUNC: Dict[str, ProphetScoreFunction] = {  # Used for param tuning
    "deviation_from_predicted_val": ProphetScoreFunction.deviation_from_predicted_val,
    "z_score": ProphetScoreFunction.z_score,
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

    model: Optional[Prophet] = None

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        score_func: Union[str, ProphetScoreFunction] = DEFAULT_SCORE_FUNCTION,
        scoring_confidence_interval: float = 0.8,
        remove_outliers: bool = False,
        outlier_threshold: float = 0.99,
        uncertainty_samples: float = PREDICTION_UNCERTAINTY_SAMPLES,
        outlier_removal_uncertainty_samples: int = OUTLIER_REMOVAL_UNCERTAINTY_SAMPLES,
        vectorize: bool = False,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None

        # We allow score_function to be a str for compatibility with param tuning
        if isinstance(score_func, str):
            if score_func in STR_TO_SCORE_FUNC:
                score_func = STR_TO_SCORE_FUNC[score_func]
            else:
                score_func = DEFAULT_SCORE_FUNCTION
        self.score_func: ProphetScoreFunction = score_func

        self.scoring_confidence_interval = scoring_confidence_interval
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.outlier_removal_uncertainty_samples = outlier_removal_uncertainty_samples

        # To improve runtime performance, we skip the confidence band
        # computation for non-Z score scoring strategy since it will not be
        # used anywhere
        if self.score_func == ProphetScoreFunction.z_score:
            self.uncertainty_samples: float = uncertainty_samples
        else:
            self.uncertainty_samples: float = 0
        self.vectorize = vectorize

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        So it can be loaded later.

        Returns:
            json containing information of the model.
        """
        return str.encode(model_to_json(self.model))

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
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

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
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

        data_df = timeseries_to_prophet_df(total_data)

        if self.remove_outliers:
            data_df = self._remove_outliers(
                data_df,
                self.outlier_threshold,
                uncertainty_samples=self.outlier_removal_uncertainty_samples,
                vectorize=self.vectorize,
            )

        # No incremental training. Create a model and train from scratch
        model = Prophet(
            interval_width=self.scoring_confidence_interval,
            uncertainty_samples=self.uncertainty_samples,
        )
        self.model = model.fit(data_df)

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """Predicts anomaly score for future data.

        Predict only expects anomaly score for data. Prophet doesn't need historical_data.

        Args:
            data: TimeSeriesData on which detection is run
            historical_data: TimeSeriesData corresponding to history. History ends exactly where
                the data begins.

        Returns:
            AnomalyResponse object. The length of this object is same as data. The score property
            gives the score for anomaly.
        """
        model = self.model
        if model is None:
            msg = "Call fit() before predict()."
            logging.error(msg)
            raise ValueError(msg)

        time_df = pd.DataFrame({PROPHET_TIME_COLUMN: data.time}, copy=False)
        model.uncertainty_samples = self.uncertainty_samples
        predict_df = predict(model, time_df, self.vectorize)
        zeros_ts = TimeSeriesData(
            time=data.time, value=pd.Series(np.zeros(len(data)), copy=False)
        )
        predicted_ts = TimeSeriesData(
            time=data.time, value=predict_df[PROPHET_YHAT_COLUMN]
        )

        # If not using z-score, set confidence band equal to prediction
        if model.uncertainty_samples == 0:
            confidence_band = ConfidenceBand(upper=predicted_ts, lower=predicted_ts)
        else:
            confidence_band = ConfidenceBand(
                upper=TimeSeriesData(
                    time=data.time, value=predict_df[PROPHET_YHAT_UPPER_COLUMN]
                ),
                lower=TimeSeriesData(
                    time=data.time, value=predict_df[PROPHET_YHAT_LOWER_COLUMN]
                ),
            )

        response = AnomalyResponse(
            scores=TimeSeriesData(
                time=data.time,
                value=SCORE_FUNC_DICT[self.score_func.value](
                    data=data,
                    predict_df=predict_df,
                    ci_threshold=model.interval_width,
                    uncertainty_samples=model.uncertainty_samples,
                ),
            ),
            confidence_band=confidence_band,
            predicted_ts=predicted_ts,
            anomaly_magnitude_ts=zeros_ts,
            stat_sig_ts=zeros_ts,
        )
        return response

    @staticmethod
    def _remove_outliers(
        ts_df: pd.DataFrame,
        outlier_ci_threshold: float = 0.99,
        uncertainty_samples: float = OUTLIER_REMOVAL_UNCERTAINTY_SAMPLES,
        vectorize: bool = False,
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

        forecast = predict(model_pass1, ts_dates_df, vectorize)

        is_outlier = (
            ts_df[PROPHET_VALUE_COLUMN] < forecast[PROPHET_YHAT_LOWER_COLUMN]
        ) | (ts_df[PROPHET_VALUE_COLUMN] > forecast[PROPHET_YHAT_UPPER_COLUMN])

        ts_df = ts_df[~is_outlier]

        return ts_df


class ProphetTrendDetectorModel(DetectorModel):
    """Prophet based trend detection model."""

    model: Optional[Prophet] = None

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        changepoint_range: float = 1.0,
        weekly_seasonality: str = "auto",
        changepoint_prior_scale: float = 0.01,
    ) -> None:
        if serialized_model:
            self.model = model_from_json(serialized_model)
        else:
            self.model = None

        self.changepoint_range = changepoint_range
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        So it can be loaded later.

        Returns:
            json containing information of the model.
        """
        return str.encode(model_to_json(self.model))

    def _zeros_ts(self, data: pd.DataFrame) -> TimeSeriesData:
        return TimeSeriesData(
            time=data.ds,
            value=pd.Series(
                np.zeros(len(data)),
                name=data.y.name if data.y.name else DEFAULT_VALUE_NAME,
            ),
        )

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        model = Prophet(
            changepoint_range=self.changepoint_range,
            weekly_seasonality=self.weekly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        ts_p = pd.DataFrame({"ds": data.time.values, "y": data.value.values})
        model = model.fit(ts_p)
        self.model = model

        output_ts = self._zeros_ts(ts_p)
        output_ts.value.loc[model.changepoints.index.values] = np.abs(
            np.nanmean(model.params["delta"], axis=0)
        )

        return AnomalyResponse(
            scores=output_ts,
            confidence_band=None,
            predicted_ts=self._zeros_ts(ts_p),
            anomaly_magnitude_ts=self._zeros_ts(ts_p),
            stat_sig_ts=None,
        )

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("fit is not implemented, call fit_predict() instead")

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> AnomalyResponse:
        raise NotImplementedError(
            "predict is not implemented, call fit_predict() instead"
        )
