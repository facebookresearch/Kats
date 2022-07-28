# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements the Bayesian Online Changepoint Detection
algorithm as a DetectorModel, to provide a common interface.
"""

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_consts import AnomalyResponse, ConfidenceBand
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
    ) -> None:
        self.slow_drift: bool = False
        self.threshold: Optional[float] = None
        self.response: Optional[AnomalyResponse] = None
        self.last_N: int = 0
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

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
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

        self.last_N = len(data)

        # if there is historical data
        # we prepend it to data, and run
        # the detector as if we only saw data
        if historical_data is not None:
            historical_data.extend(data, validate=False)
            data = TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": list(historical_data.time),
                        "value": list(historical_data.value),
                    }
                )
            )

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
        default_ts = TimeSeriesData(
            time=data.time, value=pd.Series(N * [0.0], copy=False)
        )
        score_ts = TimeSeriesData(
            time=data.time, value=pd.Series(change_prob, copy=False)
        )

        self.response = AnomalyResponse(
            scores=score_ts,
            confidence_band=ConfidenceBand(upper=data, lower=data),
            predicted_ts=data,
            anomaly_magnitude_ts=default_ts,
            stat_sig_ts=default_ts,
        )

        return self.response.get_last_n(self.last_N)

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """
        fit can be called during priming. It's a noop for us.
        """
        return


class BocpdTrendDetectorModel(DetectorModel):
    def __init__(self, alpha: float = 0.1, beta: float = 0.1) -> None:
        self.alpha = alpha
        self.beta = beta

    def serialize(self) -> bytes:
        """Serialize the model into a json.

        So it can be loaded later.

        Returns:
            json containing information of the model.
        """
        model_dict = {"slow_drift": True}
        return json.dumps(model_dict).encode("utf-8")
        # return str.encode(model_to_json(self.model))

    def _holt_winter_fit(
        self,
        data_ts: TimeSeriesData,
        m: int = 7,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
    ) -> TimeSeriesData:
        exp_smooth = ExponentialSmoothing(
            endog=data_ts.value, trend="add", seasonal="add", seasonal_periods=m
        )
        fit1 = exp_smooth.fit(
            smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma
        )

        level_arr = fit1.level
        trend_arr = fit1.slope
        fit_arr = [x + y for x, y in zip(level_arr, trend_arr)]
        fit_diff = np.diff(fit_arr)
        fit_diff = np.concatenate(([fit_diff[0]], fit_diff))
        trend_ts = TimeSeriesData(
            time=data_ts.time, value=pd.Series(fit_diff, copy=False)
        )
        return trend_ts

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        trend_ts = self._holt_winter_fit(data)
        detector = BocpdDetectorModel()
        anom_obj = detector.fit_predict(trend_ts)
        return anom_obj

    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise NotImplementedError(
            "predict is not implemented, call fit_predict() instead"
        )

    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
        **kwargs: Any,
    ) -> None:
        """
        fit can be called during priming. It's a noop for us.
        """
        raise NotImplementedError("fit is not implemented, call fit_predict() instead")
