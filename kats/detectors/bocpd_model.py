# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements the Bayesian Online Changepoint Detection
algorithm as a DetectorModel, to provide a common interface.
"""

import json
from typing import Optional

import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import (
    BOCPDetector,
    BOCPDModelType,
)
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import (
    AnomalyResponse,
    ConfidenceBand,
)


class BocpdDetectorModel(DetectorModel):
    """Implements the Bayesian Online Changepoint Detection as a DetectorModel.

    This provides an unified interface, which is common to all detection algorithms.

    Attributes:
        serialized_model: json containing information about stored model.
        slow_drift: Boolean. True indicates we are trying to detect trend changes.
                    False indicates we are trying to detect level changes.

    Typical Usage:
    level_ts is an instance of TimeSeriesData
    >>> bocpd_detector = BocpdDetectorModel()
    >>> anom = bocpd_detector.fit_predict(data=level_ts)
    """

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        slow_drift: bool = False,
        threshold: Optional[float] = None,
    ):
        if serialized_model is None:
            self.slow_drift = slow_drift
            self.threshold = threshold
        else:
            model_dict = json.loads(serialized_model)
            if "slow_drift" in model_dict:
                self.slow_drift = model_dict["slow_drift"]
            else:
                self.slow_drift = slow_drift
            if "threshold" in model_dict:
                self.threshold = model_dict["threshold"]
            else:
                self.threshold = threshold

    def serialize(self) -> bytes:
        """Returns the serialzed model.

        Args:
            None.

        Returns:
            json containing information about serialized model.
        """

        model_dict = {"slow_drift": self.slow_drift}
        return json.dumps(model_dict).encode("utf-8")

    def _handle_missing_data_extend(
        self, data: TimeSeriesData, historical_data: TimeSeriesData
    ) -> TimeSeriesData:

        # extend() works only when there is no missing data
        # hence, we will interpolate if there is missing data
        # but we will remove the interpolated data when we
        # evaluate, to make sure that the anomaly score is
        # the same length as data
        original_time_list = list(historical_data.time) + list(data.time)

        if historical_data.is_data_missing():
            historical_data = historical_data.interpolate()
        if data.is_data_missing():
            data = data.interpolate()

        historical_data.extend(data)

        # extend has been done, now remove the interpolated data
        data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": [
                        historical_data.time.iloc[i]
                        for i in range(len(historical_data))
                        if historical_data.time.iloc[i] in original_time_list
                    ],
                    "value": [
                        historical_data.value.iloc[i]
                        for i in range(len(historical_data))
                        if historical_data.time.iloc[i] in original_time_list
                    ],
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )

        return data

    # pyre-fixme[14]: `fit_predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """Finds changepoints and returns score.

        Uses the current data and historical data to find the changepoints, and
        returns an AnomalyResponse object, the scores corresponding to probability
        of changepoints.

        Args:
            data: TimeSeriesData object representing the data
            historical_data: TimeSeriesdata object representing the history. Dats
            should start exactly where the historical_data ends.

        Returns:
            AnomalyResponse object, representing the changepoint probabilities. The
            score property contains the changepoint probabilities. The length of
            the object is the same as the length of the data.
        """

        # pyre-fixme[16]: `BocpdDetectorModel` has no attribute `last_N`.
        self.last_N = len(data)

        # if there is historical data
        # we prepend it to data, and run
        # the detector as if we only saw data
        if historical_data is not None:
            data = self._handle_missing_data_extend(data, historical_data)

        bocpd_model = BOCPDetector(data=data)

        if not self.slow_drift:

            if self.threshold is not None:
                _ = bocpd_model.detector(
                    model=BOCPDModelType.NORMAL_KNOWN_MODEL,
                    choose_priors=True,
                    agg_cp=True,
                    threshold=self.threshold,
                )
            else:
                _ = bocpd_model.detector(
                    model=BOCPDModelType.NORMAL_KNOWN_MODEL,
                    choose_priors=True,
                    agg_cp=True,
                )
        else:
            if self.threshold is not None:
                _ = bocpd_model.detector(
                    model=BOCPDModelType.NORMAL_KNOWN_MODEL,
                    choose_priors=True,
                    agg_cp=True,
                    threshold=self.threshold,
                )
            else:
                _ = bocpd_model.detector(
                    model=BOCPDModelType.TREND_CHANGE_MODEL,
                    choose_priors=False,
                    agg_cp=True,
                )

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob = list(change_prob_dict.values())[0]

        # construct the object
        N = len(data)
        default_ts = TimeSeriesData(time=data.time, value=pd.Series(N * [0.0]))
        score_ts = TimeSeriesData(time=data.time, value=pd.Series(change_prob))

        # pyre-fixme[16]: `BocpdDetectorModel` has no attribute `response`.
        self.response = AnomalyResponse(
            scores=score_ts,
            confidence_band=ConfidenceBand(upper=data, lower=data),
            predicted_ts=data,
            anomaly_magnitude_ts=default_ts,
            stat_sig_ts=default_ts,
        )

        return self.response.get_last_n(self.last_N)

    # pyre-fixme[14]: `predict` overrides method defined in `DetectorModel`
    #  inconsistently.
    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")

    # pyre-fixme[14]: `fit` overrides method defined in `DetectorModel` inconsistently.
    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """
        fit can be called during priming. It's a noop for us.
        """
        return
