#!/usr/bin/env python3

import json
import logging
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detector import DetectorModel
from infrastrategy.kats.detectors.detector_consts import AnomalyResponse


_MINUTES_IN_HOUR = 60
_MINUTES_IN_DAY = 24 * _MINUTES_IN_HOUR


class BaseSloDetectorModel(DetectorModel):
    """
    Base class for SLO violation prediction.
    """

    def serialize(self) -> bytes:
        """
        Serialize is not implemented since we don't store a model
        """
        return json.dumps({}).encode("utf-8")

    def fit(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> None:
        """
        fit is not implemented
        """
        return

    def predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        """
        predict is not implemented
        """
        return

    def _init_response(self, data: TimeSeriesData):
        """
        initializes a default anomaly score
        """
        n = len(data)
        default_ts = TimeSeriesData(
            time=data.time, value=pd.Series(np.zeros(len(data)))
        )

        self.response = AnomalyResponse(
            scores=default_ts,
            confidence_band=None,
            predicted_ts=None,
            anomaly_magnitude_ts=default_ts,
            stat_sig_ts=None,
        )

    def _validate_data(self, data: TimeSeriesData) -> bool:
        """
        Checks if the input data is valid
        """
        # Model expects two time series data for failure and total requests.
        return (
            len(data.value.shape) == 2
            and data.value.shape[1] == 2
            and len(data.value.iloc[:, 0]) == len(data.value.iloc[:, 1])
        )

    def _get_failure_timeseries(
        self, historical_data: TimeSeriesData, data: TimeSeriesData
    ) -> List[float]:
        # Failure time series is assumed to be the first one
        failure_historical = historical_data.value.iloc[:, 0].tolist()
        failure_data = data.value.iloc[:, 0].tolist()
        failure_historical.extend(failure_data)
        return failure_historical

    def _get_total_request_timeseries(
        self, historical_data: TimeSeriesData, data: TimeSeriesData
    ) -> List[float]:
        # Failure time series is assumed to be the first one
        total_request_historical = historical_data.value.iloc[:, 1].tolist()
        total_request = data.value.iloc[:, 1].tolist()
        total_request_historical.extend(total_request)
        return total_request_historical

    def _get_time(self, data: TimeSeriesData) -> pd.Series:
        return data.time

    @abstractmethod
    def fit_predict(
        self, data: TimeSeriesData, historical_data: Optional[TimeSeriesData] = None
    ) -> AnomalyResponse:
        # TODO
        return

    @abstractmethod
    def _predict_slo_violation(
        self,
        failure_ts: List[float],
        request_ts: List[float],
        prediction_time: List[int],
    ):
        # TODO
        return

    @abstractmethod
    def _validate_historical_data(self, data: TimeSeriesData) -> bool:
        # TODO
        return


class WindowSloDetectorModel(BaseSloDetectorModel):
    """
    WindowSloDetectorModel uses multiple windows to predict SLO violation events.
    Each window has a duration and an aceptable error budget.
    If the aggregated SLI value over a window is greater than the error budget in that
    window then SLO is violated in that window.
    We predict a SLO violation event, if SLO is violated in all the windows.
    More detail:
    https://landing.google.com/sre/workbook/chapters/alerting-on-slos/

    Example usage:
    >>> mw_detector = WindowSloDetectorModel(
        window_size=(2, 10), error_budget=(0.2, 0.1)
    )
    >>> anom = mw_detector.predict(ts_pt)
    """

    def __init__(
        self,
        serialized_model: Optional[bytes] = None,
        window_size: Optional[Union[Tuple[int, ...], int]] = (2, 10),
        error_budget: Optional[Union[Tuple[float, ...], float]] = (0.5, 0.2),
        precision_ttd_tradeoff_factor: Optional[float] = None,
        sli_error_budget: Optional[float] = None,
        windows_have_same_end: Optional[bool] = True,
    ) -> None:
        """
        Class constructor.

        Args:
            window_size: List[int]
                List of the number of points in the windows in ascending order

            error_budget: List[float]
                Acdeptable error budget in windows

            precision_ttd_tradeoff_factor: float
                A factor for choosing model parameters based on the precision to
                time-to-detection tradeoff. It should be in [0, 1] range.
                Higher values of this factor will give parameters that would
                result in higher precision (i.e. less noise).

            sli_error_budget: float
                1.0-SLO threshold, e.g. if SLO is 99% then sli_error_budget
                would be 0.01 (i.e. 1-0.99)

            windows_have_same_end: bool
                If True all windows have the same end time.
                If False all windoes have the same start time.
        """
        logging.debug(
            f"Detector with {window_size} window size and {error_budget} error budget."
        )
        self.precision_ttd_tradeoff_factor = precision_ttd_tradeoff_factor
        self.sli_error_budget = sli_error_budget
        if precision_ttd_tradeoff_factor is not None:
            # If sli_error_budget is either 0 or negative, make sure window_size
            # and error_budget are provided.
            if sli_error_budget > 0.0:
                window_size, error_budget = self._choose_model_parameters(
                    precision_ttd_tradeoff_factor, sli_error_budget
                )
            else:
                assert window_size, "window_size is not set"
                assert error_budget, "error_budget is not set"

        if isinstance(window_size, tuple):
            self.window_size = window_size
        else:
            self.window_size = (window_size,)

        if isinstance(error_budget, tuple):
            self.error_budget = error_budget
        else:
            self.error_budget = (error_budget,)
        self.windows_have_same_end = windows_have_same_end

    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData] = None,
    ) -> AnomalyResponse:
        """
        This is the main working function.

        Args:
            historical_data: TimeSeriesData
                Length of the historical data need to be larger than window_size.
                Should include failure and total request time series.

            data: TimeSeriesData
                Data where prediction is being made on.
                Should include failure and total request time series.

        Returns:
            An AnomalyResponse object of length equal to the length of the data
        """
        logging.debug(f"Received {len(data)} data points for prediction.")

        if historical_data:
            logging.debug(f"Received {len(historical_data)} historical data points.")
        else:
            logging.debug("Received None historical data.")

        assert isinstance(self.window_size, tuple), "window_size must be a tuple"
        assert len(self.window_size), "window_size should have at least one element"

        if historical_data is not None:
            assert (
                len(data) + len(historical_data) >= self.window_size[-1]
            ), "not enough data"
        else:
            assert len(data) >= self.window_size[-1], "not enough data"

        # handle cases where there is either no historical  data, or
        # not enough historical data
        data_proc, historical_data_proc = self._divide_data_if_not_enough_history(
            data=data, historical_data=historical_data
        )

        assert self._validate_data(
            data_proc
        ), "data is not valid, there should be 2 time series of equal length"
        assert self._validate_data(
            historical_data_proc
        ), "historical_data is not valid, there should be 2 time series of equal length"
        assert self._validate_historical_data(historical_data_proc), "not enough data"

        self._init_response(data)
        self.is_initialized = True

        failure_ts = self._get_failure_timeseries(historical_data_proc, data_proc)
        request_ts = self._get_total_request_timeseries(historical_data_proc, data_proc)

        predict_time = self._get_time(data_proc)
        self._predict_slo_violation(failure_ts, request_ts, predict_time)
        return self.response

    def _predict_slo_violation(
        self,
        failure_ts: List[float],
        request_ts: List[float],
        prediction_time: List[int],
    ):
        """Predicts SLO violations by computing aggregated error rate.

        Args:
            failure_ts: List[float]
                Failure time series

            request_ts: List[float]
                Request time series

            prediction_time: List[int]
                Time stamps for SLO prediction
        """
        logging.debug(f"Length of failure timeseries is {len(failure_ts)}.")
        logging.debug(f"Length of total request timeseries is {len(request_ts)}.")
        logging.debug(
            f"Length of total prediction time series is {len(prediction_time)}."
        )
        num_points_to_predict = len(prediction_time)

        # Loop over last num_points_to_predict points
        for i in range(-num_points_to_predict, 0):
            error_rate_list = []

            for j, w in enumerate(self.window_size):
                if self.windows_have_same_end:
                    aggregate_end = len(failure_ts) + i + 1
                    aggregate_start = aggregate_end - w
                else:
                    aggregate_start = i - self.window_size[-1] + 1
                    aggregate_end = aggregate_start + w

                total_failure = (
                    sum(failure_ts[aggregate_start : aggregate_end - 1])
                    + failure_ts[aggregate_end - 1]
                )
                total_request = (
                    sum(request_ts[aggregate_start : aggregate_end - 1])
                    + request_ts[aggregate_end - 1]
                )

                if total_request:
                    error_rate = float(total_failure) / total_request
                else:
                    error_rate = 0.0

                error_rate_list.append(error_rate)

                # In order to have an alert, SLO should be violated in ALL windows
                if error_rate <= self.error_budget[j]:
                    anomaly_score = 0.0
                    error_rate_list = []
                    break

                # Set anomaly score to 1 if the error rate is higher than the threshold
                # in all the windows.
                if j == len(self.window_size) - 1:
                    anomaly_score = 1.0
                    logging.debug(
                        f"Error rate exceeds the thresholds in all windows: {error_rate_list}"
                    )

            self.response.scores.value.iloc[i] = anomaly_score

    def _validate_parameters(self):
        if len(self.window_size) != len(self.error_budget):
            return False

        # window_size should be in ascending order
        for i in range(1, len(self.window_size)):
            if self.window_size[i] <= self.window_size[i - 1]:
                return False

        # error_bidget should be less than or equal to 1
        return max(self.error_budget) <= 1

    def _validate_historical_data(self, data: TimeSeriesData) -> bool:
        """
        Checks if the historical data is valid
        """
        return len(data) >= self.window_size[-1]

    def _divide_data_if_not_enough_history(
        self, data: TimeSeriesData, historical_data: TimeSeriesData
    ) -> TimeSeriesData:
        """
        Handles the case when we don't have enough historical data.
        If we don't need to update, this does not do anything
        If we need to update, this divides up the data accordingly.
        """
        # Do nothing if there are enough data points in the historical data
        # Since window_size is in the ascending order just need to compare with the last
        if historical_data and len(historical_data) >= self.window_size[-1]:
            return data, historical_data

        # when no historical data, divide the data into historical and data
        if historical_data is None:
            total_data = data
        else:
            historical_data.extend(data)
            total_data = historical_data

        # Last time for historical data
        historical_data_end_time = total_data.time.iloc[self.window_size[-1]]

        historical_data_divide = TimeSeriesData(
            time=total_data.time[total_data.time < historical_data_end_time],
            value=total_data.value[total_data.time < historical_data_end_time],
        )
        data_divide = TimeSeriesData(
            time=total_data.time[total_data.time >= historical_data_end_time],
            value=total_data.value[total_data.time >= historical_data_end_time],
        )

        return data_divide, historical_data_divide

    def _choose_model_parameters(
        self,
        precision_ttd_tradeoff_factor: float,
        sli_error_budget: float,
    ):  # -> (Tuple[int, ...], Tuple[float, ...]):
        assert (
            sli_error_budget >= 0 and sli_error_budget <= 1.0
        ), "sli_error_budget not valid"
        assert (
            precision_ttd_tradeoff_factor >= 0 and precision_ttd_tradeoff_factor <= 1
        ), "precision_ttd_tradeoff_factor not valid"
        error_rates = (1.0 * sli_error_budget, 0.1 * sli_error_budget)

        # Use a three group model set
        if precision_ttd_tradeoff_factor >= 0.66:
            window_sizes = (120, 1440)
        elif precision_ttd_tradeoff_factor >= 0.33:
            window_sizes = (60, 720)
        else:
            window_sizes = (30, 360)

        return window_sizes, error_rates
