from typing import Optional
import pandas as pd
import json

from infrastrategy.kats.detector import DetectorModel

from infrastrategy.kats.consts import (
    TimeSeriesData,
    DEFAULT_VALUE_NAME
)

from infrastrategy.kats.detectors.bocpd import (
    BOCPDetector,
    BOCPDModelType,
)

from infrastrategy.kats.detectors.detector_consts import (
    AnomalyResponse,
    ConfidenceBand,
)

class BocpdDetectorModel(DetectorModel):
    def __init__(self, serialized_model: Optional[bytes] = None, slow_drift: bool = False):
        if serialized_model is None:
            self.slow_drift = slow_drift
        else:
            model_dict = json.loads(serialized_model)
            if 'slow_drift' in model_dict:
                self.slow_drift = model_dict['slow_drift']
            else:
                self.slow_drift = slow_drift


    def serialize(self) -> bytes:
        model_dict = {'slow_drift': self.slow_drift}
        return json.dumps(model_dict).encode("utf-8")

    def _handle_missing_data_extend(
        self, data: TimeSeriesData, historical_data: TimeSeriesData
    ) -> TimeSeriesData:

        # extend() works only when there is no missing data
        # hence, we will interpolate if there is missing data
        # but we will remove the interpolated data when we
        # evaluate, to make sure that the anomaly score is
        # the same length as data
        original_time_list = (
            list(historical_data.time)
            + list(data.time)
        )

        if historical_data.is_data_missing():
            historical_data = historical_data.interpolate()
        if data.is_data_missing():
            data = data.interpolate()

        historical_data.extend(data)

        # extend has been done, now remove the interpolated data
        data = TimeSeriesData(
            pd.DataFrame({
                'time':[
                    historical_data.time.iloc[i] for i in range(len(historical_data))
                    if historical_data.time.iloc[i] in original_time_list],
                'value':[
                    historical_data.value.iloc[i] for i in range(len(historical_data))
                    if historical_data.time.iloc[i] in original_time_list]
            }),
            use_unix_time=True, unix_time_units="s", tz="US/Pacific"
        )

        return data


    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:

        self.last_N = len(data)

        #if there is historical data
        # we prepend it to data, and run
        # the detector as if we only saw data
        if historical_data is not None:
            data = self._handle_missing_data_extend(data, historical_data)

        bocpd_model = BOCPDetector(data=data)

        if not self.slow_drift:
            _ = bocpd_model.detector(
                model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=True,
                agg_cp=True
            )
        else:
            _ = bocpd_model.detector(
                model=BOCPDModelType.TREND_CHANGE_MODEL, choose_priors=False,
                agg_cp=True
            )

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob = list(change_prob_dict.values())[0]

        #construct the object
        N = len(data)
        default_ts = TimeSeriesData(time=data.time, value=pd.Series(N * [0.0]))
        score_ts = TimeSeriesData(time=data.time, value=pd.Series(change_prob))

        self.response = AnomalyResponse(
            scores=score_ts,
            confidence_band=ConfidenceBand(upper=data, lower=data),
            predicted_ts=data,
            anomaly_magnitude_ts=default_ts,
            stat_sig_ts=default_ts,
        )

        return self.response.get_last_n(self.last_N)

    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        raise ValueError("predict is not implemented, call fit_predict() instead")

    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """
        fit can be called during priming. It's a noop for us.
        """
        return
