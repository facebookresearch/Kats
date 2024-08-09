# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from datetime import timedelta
from typing import Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers
from kats.detectors.detector_consts import AnomalyResponse
from kats.detectors.prophet_detector import (
    get_holiday_dates,
    ProphetDetectorModel,
    ProphetScoreFunction,
    ProphetTrendDetectorModel,
    SeasonalityTypes,
    to_seasonality,
)
from kats.utils.simulator import Simulator

from parameterized.parameterized import parameterized

START_DATE_TEST_DATA = "2018-01-01"


class TestProphetDetector(TestCase):
    def create_random_ts(
        self, seed: int, length: int, magnitude: float, slope_factor: float
    ) -> TimeSeriesData:
        np.random.seed(seed)
        sim = Simulator(n=length, freq="1D", start=pd.to_datetime(START_DATE_TEST_DATA))

        sim.add_trend(magnitude=magnitude * np.random.rand() * slope_factor)
        sim.add_seasonality(
            magnitude * np.random.rand(),
            period=timedelta(days=7),
        )

        sim.add_noise(magnitude=0.1 * magnitude * np.random.rand())
        return sim.stl_sim()

    def create_ts(
        self,
        seed: int = 0,
        length: int = 100,
        magnitude: float = 10,
        signal_to_noise_ratio: float = 0.1,
        freq: str = "1D",
    ) -> TimeSeriesData:
        np.random.seed(seed)
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime(START_DATE_TEST_DATA))

        sim.add_seasonality(magnitude, period=timedelta(days=7))
        sim.add_noise(magnitude=signal_to_noise_ratio * magnitude)
        return sim.stl_sim()

    def create_multi_seasonality_ts(
        self,
        seed: int,
        length: int,
        freq: str,
        min_val: float,
        max_val: float,
        signal_to_noise_ratio: float,
    ) -> TimeSeriesData:
        np.random.seed(seed)

        sim = Simulator(n=length, freq=freq, start=pd.to_datetime(START_DATE_TEST_DATA))
        magnitude = (max_val - min_val) / 2

        sim.add_trend(-0.2 * magnitude)
        sim.add_seasonality(
            magnitude * (2 / 3) * np.random.rand() * 2,
            period=timedelta(days=1),
        )
        sim.add_seasonality(
            magnitude * (1 / 3) * np.random.rand(),
            period=timedelta(days=0.5),
        )
        sim.add_seasonality(
            magnitude * 0.2 * np.random.rand(),
            period=timedelta(days=7),
        )
        sim.add_noise(magnitude=signal_to_noise_ratio * magnitude)

        sim_ts = sim.stl_sim()

        self.add_trend_shift(sim_ts, length, freq, min_val + magnitude)

        return sim_ts

    def create_weekend_seasonality_ts(
        self,
        seed: int = 42,
        days: int = 56,
        freq: str = "6H",
        trend: float = 1.0,
        magnitude_weekday: float = 5,
        magnitude_weekend: float = 2,
        signal_to_noise_ratio: float = 0.1,
    ) -> TimeSeriesData:
        np.random.seed(seed)
        ts = TimeSeriesData()
        points_day = int(pd.to_timedelta("1D") / pd.to_timedelta(freq))
        weeks = int(days / 7)
        for i in range(weeks + 1):
            rest_days_weekday = 5

            if i >= weeks:
                rest_days = days % 7
                if rest_days == 0:
                    break
                rest_days_weekday = min(rest_days, 5)
            start_ts = (
                pd.to_datetime("2018-01-01")
                if len(ts) == 0
                else ts.time.iloc[-1] + pd.to_timedelta(freq)
            )
            sim = Simulator(n=rest_days_weekday * points_day, freq=freq, start=start_ts)
            sim.add_seasonality(magnitude_weekday, timedelta(days=5))
            sim.add_noise(magnitude=signal_to_noise_ratio * magnitude_weekday)
            if len(ts) == 0:
                ts = sim.stl_sim()
            else:
                ts.extend(sim.stl_sim() + trend)
            rest_days_weekend = 2
            if i >= weeks:
                # pyre-fixme[61]: `rest_days` is undefined, or not always defined.
                rest_days_weekend = rest_days - rest_days_weekday
                if rest_days_weekend == 0:
                    break
            sim = Simulator(
                n=rest_days_weekend * points_day,
                freq=freq,
                start=ts.time.iloc[-1] + pd.to_timedelta(freq),
            )
            sim.add_seasonality(magnitude_weekend, timedelta(days=2))
            sim.add_noise(magnitude=signal_to_noise_ratio * magnitude_weekend)
            ts.extend(sim.stl_sim() + trend)
        return ts

    def add_smooth_anomaly(
        self,
        ts: TimeSeriesData,
        seed: int,
        start_index: int,
        length: int,
        magnitude: float,
    ) -> None:
        # Add an anomaly that is half of a sine wave
        # start time and freq don't matter, since we only care about the values
        np.random.seed(seed)

        anomaly_sim = Simulator(
            n=length, freq="1D", start=pd.to_datetime(START_DATE_TEST_DATA)
        )
        anomaly_sim.add_seasonality(magnitude, period=timedelta(days=2 * length))
        # anomaly_sim.add_noise(magnitude=0.3 * magnitude * np.random.rand())

        anomaly_ts = anomaly_sim.stl_sim()
        for i in range(0, length):
            ts.value.iloc[start_index + i] += anomaly_ts.value[i]

    def truncate(self, ts: TimeSeriesData, start_index: int, end_index: int) -> None:
        # Set all values outside the range [start_index, end_index) to 0
        ts.value.iloc[:start_index] *= 0
        ts.value.iloc[end_index:] *= 0

    def add_trend_shift(
        self, ts: TimeSeriesData, length: int, freq: str, magnitude: float
    ) -> None:
        ts_df = ts.to_dataframe()
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime(START_DATE_TEST_DATA))
        elevation = sim.trend_shift_sim(
            cp_arr=[0, 1],
            trend_arr=[0, 0, 0],
            noise=0,
            seasonal_period=1,
            seasonal_magnitude=0,
            intercept=magnitude,
        )
        elevation_df = elevation.to_dataframe()

        ts_df_elevated = (
            ts_df.set_index("time") + elevation_df.set_index("time")
        ).reset_index()

        elevated_ts = TimeSeriesData(df=ts_df_elevated)
        ts.value = elevated_ts.value

    def horiz_translate(self, ts: TimeSeriesData, periods: int) -> None:
        ts.value = ts.value.shift(periods=periods, fill_value=0)

    def add_multiplicative_noise(self, ts: TimeSeriesData, magnitude: float) -> None:
        # Multiply all the values in ts by a number in the range [1-magnitude, 1+magnitude]
        ts.value *= np.random.rand(len(ts)) * magnitude * 2 + 1 - magnitude

    def merge_ts(self, ts1: TimeSeriesData, ts2: TimeSeriesData) -> TimeSeriesData:
        ts1_df, ts2_df = ts1.to_dataframe(), ts2.to_dataframe()
        merged_df = (ts1_df.set_index("time") + ts2_df.set_index("time")).reset_index()
        merged_ts = TimeSeriesData(df=merged_df)
        return merged_ts

    def add_multi_event(
        self,
        baseline_ts: TimeSeriesData,
        seed: int,
        length: int,
        freq: str,
        min_val: float,
        max_val: float,
        signal_to_noise_ratio: float,
        event_start_ratio: float,
        event_end_ratio: float,
        event_relative_magnitude: float,
    ) -> TimeSeriesData:
        np.random.seed(seed)
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime(START_DATE_TEST_DATA))

        event_start = int(length * event_start_ratio)
        event_end = int(length * event_end_ratio)
        duration = event_end - event_start

        magnitude = (max_val - min_val) / 2
        event_magnitude = (
            2 * magnitude * event_relative_magnitude * (signal_to_noise_ratio + 1)
        )

        event1_start = event_start + int(duration / 4)
        event1_end = event_end
        event1_magnitude = event_magnitude / 2
        event1_duration = event1_end - event1_start

        event2_start = event_start
        event2_end = event_start + int(duration / 3)
        event2_magnitude = event_magnitude / 2 / 2
        event2_duration = event2_end - event2_start

        event3_start = event_start
        event3_end = event_start + 2 * int(duration / 3)
        event3_magnitude = event_magnitude / duration / 4
        event3_duration = event3_end - event3_start
        event3_peak = event3_start + int(event3_duration / 2)

        # create event ts

        event1_ts = sim.level_shift_sim(
            seasonal_period=event1_duration // 2,
            seasonal_magnitude=event1_magnitude,
            noise=signal_to_noise_ratio * magnitude,
        )

        event2_ts = sim.level_shift_sim(
            seasonal_period=event2_duration // 2,
            seasonal_magnitude=event2_magnitude,
            noise=signal_to_noise_ratio * magnitude,
        )

        event3_ts = sim.trend_shift_sim(
            cp_arr=[event3_start, event3_peak, event3_end],
            trend_arr=[0, -event3_magnitude, +event3_magnitude, 0],
            seasonal_period=duration,
            seasonal_magnitude=0,
            intercept=0,
            noise=signal_to_noise_ratio * magnitude,
        )

        self.horiz_translate(event1_ts, event1_start - int(3 * event1_duration / 4))
        self.horiz_translate(event2_ts, event2_start - int(3 * event2_duration / 4))

        self.add_trend_shift(event1_ts, length, freq, event1_magnitude)
        self.add_trend_shift(event2_ts, length, freq, event2_magnitude)

        self.truncate(event1_ts, event1_start, event1_end)
        self.truncate(event2_ts, event2_start, event2_end)
        self.truncate(event3_ts, event3_start, event3_end)

        self.add_multiplicative_noise(event1_ts, 0.35)
        self.add_multiplicative_noise(event2_ts, 0.35)
        self.add_multiplicative_noise(event3_ts, 0.35)

        # merge the events
        events12_ts = self.merge_ts(event1_ts, event2_ts)
        event_ts = self.merge_ts(events12_ts, event3_ts)

        # merge baseline and event ts
        merged_ts = self.merge_ts(baseline_ts, event_ts)

        return merged_ts

    def calc_z_score(
        self,
        actual_val: float,
        predicted_val: float,
        upper_bound: float,
        lower_bound: float,
    ) -> float:
        # Assumes default ProphetDetectorModel param values
        actual_std = (50**0.5) * ((upper_bound - lower_bound) / 2) / 0.8
        std = max(actual_std, 1e-9)

        return (actual_val - predicted_val) / std

    def scenario_results(
        self,
        seed: int,
        include_anomaly: bool,
        use_serialized_model: bool,
    ) -> AnomalyResponse:
        """Prediction results for common data and model test scenarios"""
        ts = self.create_random_ts(seed, 100, 10, 2)

        if include_anomaly:
            self.add_smooth_anomaly(ts, seed, 90, 10, 10)

        model = ProphetDetectorModel()
        model.fit(ts[:90])

        if use_serialized_model:
            serialized_model = model.serialize()
            model = ProphetDetectorModel(serialized_model=serialized_model)

        return model.predict(ts[90:])

    # Alternate between using the current model and using serialized model
    SEED_AND_SERIALIZATIONS = [[0, True], [1, False], [2, True], [3, False], [4, True]]

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(SEED_AND_SERIALIZATIONS)
    def test_no_anomaly_prediction_length(
        self, seed: int, use_serialized_model: bool
    ) -> None:
        include_anomaly = False
        res = self.scenario_results(seed, include_anomaly, use_serialized_model)
        self.assertEqual(len(res.scores), 10)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(SEED_AND_SERIALIZATIONS)
    def test_anomaly_prediction_length(
        self, seed: int, use_serialized_model: bool
    ) -> None:
        include_anomaly = True
        res = self.scenario_results(seed, include_anomaly, use_serialized_model)
        self.assertEqual(len(res.scores), 10)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(SEED_AND_SERIALIZATIONS)
    def test_finds_no_anomaly_when_no_anomaly(
        self, seed: int, use_serialized_model: bool
    ) -> None:
        # Prophet should not find any anomalies on a well formed synthetic time series
        include_anomaly = False
        res = self.scenario_results(seed, include_anomaly, use_serialized_model)
        anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
        self.assertFalse(anomaly_found)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(SEED_AND_SERIALIZATIONS)
    def test_finds_anomaly_when_anomaly_present(
        self, seed: int, use_serialized_model: bool
    ) -> None:
        # Prophet should find anomalies
        include_anomaly = True
        res = self.scenario_results(seed, include_anomaly, use_serialized_model)
        anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
        self.assertTrue(anomaly_found)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand([[True], [False]])
    def test_fit_predict(self, vectorize: bool) -> None:
        ts = self.create_random_ts(0, 100, 10, 2)
        self.add_smooth_anomaly(ts, 0, 90, 10, 10)

        model = ProphetDetectorModel(vectorize=vectorize)
        model.fit(ts[:90])
        res0 = model.predict(ts[90:])
        # create test case for gap between training/testing time series
        res2 = model.predict(ts[95:])
        # create test case for latest testing timestamp earlier to lastest training timestamp.
        res4 = model.predict(ts[50:55])

        model = ProphetDetectorModel(vectorize=not vectorize)
        res1 = model.fit_predict(data=ts[90:], historical_data=ts[:90])
        res3 = model.fit_predict(data=ts[95:], historical_data=ts[:90])
        res5 = model.fit_predict(data=ts[50:55], historical_data=ts[:90])

        self.assertEqual(res0.scores.value.to_list(), res1.scores.value.to_list())
        self.assertEqual(res2.scores.value.to_list(), res3.scores.value.to_list())
        self.assertEqual(res4.scores.value.to_list(), res5.scores.value.to_list())

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["moderate", 0.990000],
    #  ["aggressive", 0.800000]])`.
    @parameterized.expand([["moderate", 0.99], ["aggressive", 0.8]])
    def test_outlier_removal_threshold(self, name: str, threshold: float) -> None:
        ts = self.create_random_ts(0, 365, 10, 2)
        ts_df = pd.DataFrame({"ds": ts.time, "y": ts.value})

        model = ProphetDetectorModel()

        filtered_ts_df = model._remove_outliers(ts_df, outlier_ci_threshold=threshold)

        self.assertGreaterEqual(len(ts_df), len(filtered_ts_df))

    def test_outlier_removal_uncertainty_sampling(self) -> None:
        ts = self.create_random_ts(0, 365, 10, 2)
        ts_df = pd.DataFrame({"ds": ts.time, "y": ts.value})

        model = ProphetDetectorModel()
        filtered_ts_df_moderate = model._remove_outliers(ts_df, uncertainty_samples=30)
        filtered_ts_df_high = model._remove_outliers(ts_df, uncertainty_samples=50)

        self.assertNotEqual(len(filtered_ts_df_moderate), len(filtered_ts_df_high))

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ["early event", 0.1, 0.15, 0.3, 1.5],
            ["late event", 0.1, 0.72, 0.85, -2],
            ["spiky event", 0.1, 0.5, 0.55, 5],
            ["prolonged event", 0.1, 0.35, 0.62, -1.5],
            ["noisier data", 0.5, 0.5, 0.55, 5],
        ]
    )
    def test_outlier_removal_efficacy(
        self,
        name: str,
        signal_to_noise_ratio: float,
        event_start_ratio: float,
        event_end_ratio: float,
        event_relative_magnitude: float,
    ) -> None:
        seed = 0
        length = 960
        freq = "15min"
        min_val = 0
        max_val = 1000

        baseline_ts = self.create_multi_seasonality_ts(
            seed, length, freq, min_val, max_val, signal_to_noise_ratio
        )
        test_ts = self.add_multi_event(
            baseline_ts,
            seed,
            length,
            freq,
            min_val,
            max_val,
            signal_to_noise_ratio,
            event_start_ratio,
            event_end_ratio,
            event_relative_magnitude,
        )

        model = ProphetDetectorModel()

        # Train on all data up to 0.5 days after the event
        event_end_idx = int(length * event_end_ratio)
        train_idx = (
            test_ts.time >= test_ts.time.iloc[event_end_idx] + timedelta(hours=12)
        ).idxmax()

        test_df = test_ts.to_dataframe()
        train_ts = TimeSeriesData(df=test_df.iloc[:train_idx])
        pred_ts_df_map = {}
        for remove_outliers in [False, True]:
            model.remove_outliers = remove_outliers
            # Test on all the remaining data
            pred_ts_df_map[remove_outliers] = model.fit_predict(test_ts, train_ts)

        # Model trained without outliers should have lower RMSE
        rmse_w_outliers = (
            (pred_ts_df_map[False].predicted_ts.value - test_ts.value) ** 2
        ).mean() ** 0.5
        rmse_no_outliers = (
            (pred_ts_df_map[True].predicted_ts.value - test_ts.value) ** 2
        ).mean() ** 0.5
        self.assertGreaterEqual(
            rmse_w_outliers,
            rmse_no_outliers,
            "Expected removing outliers when training model to lower prediction RMSE",
        )

    def test_default_score_func(self) -> None:
        """Test that 'deviation_from_predicted_val' is used by default

        This test verifies that the default implementation of
        ProphetDetectorModel uses the 'deviation_from_predicted_val' scoring
        function, by checking an anomaly score.
        """
        ts = self.create_ts(length=100)

        # add anomaly at index 95
        ts.value[95] += 100

        deviation_model = ProphetDetectorModel()
        deviation_response = deviation_model.fit_predict(ts[90:], ts[:90])
        self.assertEqual(
            deviation_response.scores.value[5],
            abs(
                # pyre-ignore[16]: Optional type has no attribute `value`.
                (ts.value[95] - deviation_response.predicted_ts.value[5])
                / deviation_response.predicted_ts.value[5]
            ),
        )

        # if using default score function, confidence bands should be prediction ts
        self.assertEqual(
            # pyre-ignore[16]: Optional type has no attribute `upper`.
            deviation_response.confidence_band.upper,
            deviation_response.predicted_ts,
        )
        self.assertEqual(
            # pyre-ignore[16]: Optional type has no attribute `lower`.
            deviation_response.confidence_band.lower,
            deviation_response.predicted_ts,
        )

    def test_score_func_parameter_as_z_score(self) -> None:
        """Test that score_func parameter can be set to z_score

        This test verifies that passing ProphetScoreFunction.z_score as the
        'score_func' results in ProphetDetectorModel implementing the
        'z_score' scoring function, by checking an anomaly score.
        """
        ts = self.create_ts()

        # add anomaly at index 95
        ts.value[95] += 100

        np.random.seed(0)
        z_score_model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        z_score_response = z_score_model.fit_predict(ts[90:], ts[:90])
        actual_z_score = self.calc_z_score(
            ts.value[95],
            # pyre-fixme[16]: Optional type has no attribute `value`.
            z_score_response.predicted_ts.value[5],
            # pyre-fixme[16]: Optional type has no attribute `upper`.
            z_score_response.confidence_band.upper.value[5],
            # pyre-fixme[16]: Optional type has no attribute `lower`.
            z_score_response.confidence_band.lower.value[5],
        )
        self.assertAlmostEqual(
            z_score_response.scores.value[5],
            actual_z_score,
            places=5,
        )

        # if using Z-score function, confidence bands should not prediction ts
        self.assertNotEqual(
            z_score_response.confidence_band.upper,
            z_score_response.predicted_ts,
        )
        self.assertNotEqual(
            z_score_response.confidence_band.lower,
            z_score_response.predicted_ts,
        )

        # Corrected Z-score should be the same as legacy Z-score if using default
        # scoring confidence interval
        np.random.seed(0)
        legacy_z_score_model = ProphetDetectorModel(
            score_func=ProphetScoreFunction.z_score, use_legacy_z_score=False
        )
        legacy_z_score_response = legacy_z_score_model.fit_predict(ts[90:], ts[:90])
        self.assertAlmostEqual(
            legacy_z_score_response.scores.value[5],
            actual_z_score,
            places=5,
        )

        # If using custom scoring confidence interval, corrected Z-scores will differ
        np.random.seed(0)
        legacy_z_score_model = ProphetDetectorModel(
            score_func=ProphetScoreFunction.z_score,
            use_legacy_z_score=False,
            scoring_confidence_interval=0.9,
        )
        legacy_z_score_response = legacy_z_score_model.fit_predict(ts[90:], ts[:90])
        self.assertNotAlmostEqual(
            legacy_z_score_response.scores.value[5],
            actual_z_score,
            places=5,
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["no anomaly", 0], ["with
    #  anomaly", 100]])`.
    @parameterized.expand([["no anomaly", 0], ["with anomaly", 100]])
    def test_flat_signal(self, name: str, anomaly_magnitude: float) -> None:
        """Tests the behavior of the z-score strategy on flat signals.

        This test verifies that the model's z_scores of flat signals
        with and without anomalies are identical to the actual z_scores.
        It ensures no division by zero errors occur when
        the signal has no seasonality or level shifts.
        """
        ts = self.create_ts(magnitude=0, signal_to_noise_ratio=0)

        ts.value[95] += anomaly_magnitude

        model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        response = model.fit_predict(ts[90:], ts[:90])
        actual_z_score = self.calc_z_score(
            ts.value[95],
            # pyre-ignore[16]: Optional type has no attribute `value`.
            response.predicted_ts.value[5],
            # pyre-ignore[16]: Optional type has no attribute `upper`.
            response.confidence_band.upper.value[5],
            # pyre-ignore[16]: Optional type has no attribute `lower`.
            response.confidence_band.lower.value[5],
        )
        self.assertAlmostEqual(response.scores.value[5], actual_z_score, places=5)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["no anomaly", 0], ["with
    #  anomaly", 100]])`.
    @parameterized.expand([["no anomaly", 0], ["with anomaly", 100]])
    def test_zero_noise_signal(self, name: str, anomaly_magnitude: float) -> None:
        """Tests the behavior of the z-score strategy on zero-noise signals.

        This test verifies that the model's z_scores of zero-noise signals
        with and without anomalies areidentical to the actual z_scores.
        It ensures no division by zero errors occur when the signal has
        no noise and the standard deviation of the training data is zero.
        """
        ts = self.create_ts(signal_to_noise_ratio=0)

        ts.value[95] += anomaly_magnitude

        model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        response = model.fit_predict(ts[90:], ts[:90])
        actual_z_score = self.calc_z_score(
            ts.value[95],
            # pyre-ignore[16]: Optional type has no attribute `value`.
            response.predicted_ts.value[5],
            # pyre-ignore[16]: Optional type has no attribute `upper`.
            response.confidence_band.upper.value[5],
            # pyre-ignore[16]: Optional type has no attribute `lower`.
            response.confidence_band.lower.value[5],
        )
        self.assertAlmostEqual(response.scores.value[5], actual_z_score, places=5)

    def test_heteroskedastic_noise_signal(self) -> None:
        """Tests the z-score strategy on signals with heteroskedastic noise

        This test creates synthetic data with heteroskedastic noise. Then, it adds
        anomalies of identical magnitudes to segments with different noise. Finally, it
        verifies that anomalies in low-noise segments have higher z-scores than those
        in high-noise segments. This occurs because low noise segments will have lower
        standard deviations, which result in higher z-scores.
        With call ProphetDetectorModel without weekend seasonaluty this taest fails
        """
        ts = self.create_ts(length=100 * 24, signal_to_noise_ratio=0.05, freq="1h")

        # add heteroskedastic noise to the data

        ts.value *= (
            (ts.time - pd.to_datetime(START_DATE_TEST_DATA)) % timedelta(days=7)
            > timedelta(days=3.5)
        ) * np.random.rand(100 * 24) * 2.5 + 0.5

        ts.value[93 * 24] += 100
        ts.value[96 * 24] += 100

        model = ProphetDetectorModel(
            score_func="z_score", seasonalities={SeasonalityTypes.WEEKEND: True}
        )
        response = model.fit_predict(ts[80 * 24 :], ts[: 80 * 24])

        self.assertGreater(
            response.scores.value[13 * 24], response.scores.value[16 * 24]
        )

    def test_heteroskedastic_noise_signal_with_holidays(self) -> None:
        """Tests the z-score strategy on signals with heteroskedastic noise

        This test creates synthetic data with heteroskedastic noise. Then, it adds
        anomalies of identical magnitudes to segments with different noise. Finally, it
        verifies that anomalies in low-noise segments have higher z-scores than those
        in high-noise segments. This occurs because low noise segments will have lower
        standard deviations, which result in higher z-scores.
        We are addingh holiday, to check the param works
        With call ProphetDetectorMopdel without weekend seasonaluty this taest fails
        """
        ts = self.create_ts(length=100 * 24, signal_to_noise_ratio=0.05, freq="1h")

        # add heteroskedastic noise to the data

        ts.value *= (
            (ts.time - pd.to_datetime(START_DATE_TEST_DATA)) % timedelta(days=7)
            > timedelta(days=3.5)
        ) * np.random.rand(100 * 24) * 2.5 + 0.5

        ts.value[93 * 24] += 100
        ts.value[96 * 24] += 100

        model = ProphetDetectorModel(
            score_func="z_score",
            seasonalities={SeasonalityTypes.WEEKEND: True},
            country_holidays="US",
        )
        response = model.fit_predict(ts[80 * 24 :], ts[: 80 * 24])

        self.assertGreater(
            response.scores.value[13 * 24], response.scores.value[16 * 24]
        )

    def test_heteroskedastic_noise_signal_with_specific_holidays(self) -> None:
        """Tests the z-score strategy on signals with heteroskedastic noise

        This test creates synthetic data with heteroskedastic noise. Then, it adds
        anomalies of identical magnitudes to segments with different noise. Finally, it
        verifies that anomalies in low-noise segments have higher z-scores than those
        in high-noise segments. This occurs because low noise segments will have lower
        standard deviations, which result in higher z-scores.
        We also adding value for the first day abnormakl, which shouldn;'t affects outcome as it holiday
        With call ProphetDetectorModel without weekend seasonaluty this taest fails
        """
        ts = self.create_ts(length=100 * 24, signal_to_noise_ratio=0.05, freq="1h")

        # add heteroskedastic noise to the data
        playoffs = [
            START_DATE_TEST_DATA,
            (pd.to_datetime(START_DATE_TEST_DATA) + pd.Timedelta(days=4)).strftime(
                "%Y-%m-%d"
            ),
        ]
        ts.value *= (
            (ts.time - pd.to_datetime(START_DATE_TEST_DATA)) % timedelta(days=7)
            > timedelta(days=3.5)
        ) * np.random.rand(100 * 24) * 2.5 + 0.5
        ts.value[0] += 1000
        ts.value[93 * 24] += 100
        ts.value[96 * 24] += 100

        model = ProphetDetectorModel(
            score_func="z_score",
            seasonalities={SeasonalityTypes.WEEKEND: True},
            country_holidays="US",
            holidays_list=playoffs,
        )
        response = model.fit_predict(ts[80 * 24 :], ts[: 80 * 24])

        self.assertGreater(
            response.scores.value[13 * 24], response.scores.value[16 * 24]
        )

        model = ProphetDetectorModel(
            score_func="z_score",
            seasonalities={SeasonalityTypes.WEEKEND: True},
            country_holidays="UK",
            holidays_list={"ds": playoffs, "holiday": ["playoff"] * len(playoffs)},
        )
        response = model.fit_predict(ts[80 * 24 :], ts[: 80 * 24])

        self.assertGreater(
            response.scores.value[13 * 24], response.scores.value[16 * 24]
        )

    def test_heteroskedastic_noise_signal_with_specific_holidays_mulitplier(
        self,
    ) -> None:
        """Tests the z-score strategy on signals with heteroskedastic noise

        This test creates synthetic data with heteroskedastic noise. Then, it adds
        anomalies of identical magnitudes to segments with different noise.
        We also adding value for the first day abnormakl, which shouldn;'t affects outcome as it holiday and holiday multiplier.
        And we check, that multiplier is using during holiday and using in other cases.
        With call ProphetDetectorModel without weekend seasonaluty this taest fails
        """
        ts = self.create_ts(length=100 * 24, signal_to_noise_ratio=0.05, freq="1h")

        # add heteroskedastic noise to the data
        playoffs = [
            START_DATE_TEST_DATA,
            (pd.to_datetime(START_DATE_TEST_DATA) + pd.Timedelta(days=94)).strftime(
                "%Y-%m-%d"
            ),
        ]
        holiday_in_predict: str = playoffs[1]
        ts.value *= (
            (ts.time - pd.to_datetime(START_DATE_TEST_DATA)) % timedelta(days=7)
            > timedelta(days=3.5)
        ) * np.random.rand(100 * 24) * 2.5 + 0.5
        ts.value[0] += 1000
        ts.value[93 * 24] += 100
        ts.value[96 * 24] += 100

        model = ProphetDetectorModel(
            score_func="z_score",
            seasonalities={SeasonalityTypes.WEEKEND: True},
            country_holidays="US",
            holidays_list=playoffs,
            holiday_multiplier=0,
        )
        response = model.fit_predict(ts[80 * 24 :], ts[: 80 * 24])
        value_to_check: float = response.scores.value.iloc[
            response.scores.time[
                response.scores.time == pd.to_datetime(holiday_in_predict + " 01:00:00")
            ].index[0]
        ]
        self.assertEqual(value_to_check, 0)
        value_to_check = response.scores.value.iloc[
            response.scores.time[
                response.scores.time == pd.to_datetime(holiday_in_predict + " 23:00:00")
            ].index[0]
        ]

        self.assertEqual(value_to_check, 0)
        value_to_check = response.scores.value.iloc[
            response.scores.time[
                response.scores.time
                == pd.to_datetime(holiday_in_predict + " 01:00:00")
                + pd.Timedelta(days=1)
            ].index[0]
        ]

        self.assertNotEqual(value_to_check, 0)

    def test_weekend_seasonality_noise_signal(self) -> None:
        """Tests the accuracy with heteroskedastic series and noise

        This test creates several series with different seasonalities for weekdau and weekend
        providing seasonality flag predictor provide better result, than without it.
        With call ProphetDetectorMopdel without weekend seasonaluty this taest fails

        """
        ts = TestProphetDetector().create_weekend_seasonality_ts(
            freq="6H", days=24, trend=1
        )
        ts_to_fit = ts[:40]
        ts_to_pred = ts[40:]
        model = ProphetDetectorModel(seasonalities={SeasonalityTypes.WEEKEND: True})
        model.fit(ts_to_fit)
        response = model.predict(ts_to_pred)
        predicted_ts = response.predicted_ts
        self.assertEqual(isinstance(predicted_ts, TimeSeriesData), True)
        # pyre-ignore
        res = predicted_ts.to_dataframe()
        mae = sum(abs(res.set_index("time").values[:, 0] - ts_to_pred.value)) / len(res)
        self.assertGreater(0.87, mae)
        model = ProphetDetectorModel()
        model.fit(ts_to_fit)
        response = model.predict(ts_to_pred)
        res = response.predicted_ts.to_dataframe()
        maeWeekly = sum(
            abs(res.set_index("time").values[:, 0] - ts_to_pred.value)
        ) / len(res)
        self.assertGreater(maeWeekly, mae)

        ts = TestProphetDetector().create_weekend_seasonality_ts(
            freq="6min", days=24, trend=1
        )
        ts_to_fit = ts[: 240 * 14]
        ts_to_pred = ts[240 * 14 :]
        model = ProphetDetectorModel(seasonalities=SeasonalityTypes.WEEKEND)
        model.fit(ts_to_fit)
        response = model.predict(ts_to_pred)
        res = response.predicted_ts.to_dataframe()
        mae = sum(abs(res.set_index("time").values[:, 0] - ts_to_pred.value)) / len(res)
        self.assertGreater(1.61, mae)
        model = ProphetDetectorModel()
        model.fit(ts_to_fit)
        response = model.predict(ts_to_pred)
        res = response.predicted_ts.to_dataframe()
        maeWeekly = sum(
            abs(res.set_index("time").values[:, 0] - ts_to_pred.value)
        ) / len(res)
        self.assertGreater(maeWeekly, mae)

        # ts = TestProphetDetector().create_weekend_seasonality_ts(freq="6min", days=28, trend=1)
        ts = TestProphetDetector().create_ts(freq="1D", length=24, seed=0)
        ts_to_fit = ts[:5]
        ts_to_pred = ts[5:]
        model = ProphetDetectorModel(seasonalities={SeasonalityTypes.WEEKEND: "auto"})
        model.fit(ts_to_fit)
        self.assertEqual(model.seasonalities_to_fit[SeasonalityTypes.WEEKEND], False)

        ts = TestProphetDetector().create_ts(freq="2D", length=24, seed=0)
        ts_to_fit = ts[:10]
        ts_to_pred = ts[10:]
        model = ProphetDetectorModel(seasonalities={SeasonalityTypes.WEEKEND: "auto"})
        model.fit(ts_to_fit)
        response = model.predict(ts_to_pred)
        self.assertEqual(isinstance(predicted_ts, TimeSeriesData), True)
        res = response.predicted_ts.to_dataframe()
        mae = sum(abs(res.set_index("time").values[:, 0] - ts_to_pred.value)) / len(res)
        self.assertEqual(model.seasonalities_to_fit[SeasonalityTypes.WEEKEND], "auto")
        self.assertGreater(1.5, mae)

    def test_z_score_proportional_to_anomaly_magnitude(self) -> None:
        """Tests the z-score strategy on signals with different-sized anomalies

        This test verifies that larger anomalies result in higher z-scores awhen all
        other variables are unchanged.
        """
        ts = self.create_ts(length=100 * 24, freq="1h")

        ts.value[93 * 24] += 60
        ts.value[96 * 24] += 30

        model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        response = model.fit_predict(ts[90 * 24 :], ts[: 90 * 24])

        self.assertGreater(response.scores.value[3 * 24], response.scores.value[6 * 24])

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([[3.__mul__(24)],
    #  [6.__mul__(24)]])`.
    @parameterized.expand([[3 * 24], [6 * 24]])
    def test_asymmetric_noise_signal(self, test_index: int) -> None:
        """Tests the z-score strategy on signals with asymmetric noise

        This test verifies that the asymmetric z-scores function as expected when
        exposed to asymmetric noise. The test makes predictions on test data containing
        anomalies based on training data with only positive noise and with only negative
        noise, and checks that training on data with positive noise results in lower
        z-scores for positive anomalies, and that training on data with negative noise
        results in lower z-scores for negative anomalies.
        """
        np.random.seed(0)
        test_ts = self.create_ts(length=100 * 24, freq="1h", signal_to_noise_ratio=0)
        ts1 = self.create_ts(length=100 * 24, freq="1h", signal_to_noise_ratio=0)
        ts2 = self.create_ts(length=100 * 24, freq="1h", signal_to_noise_ratio=0)

        noise = (np.random.rand(100 * 24) - 0.5) * (np.random.rand(100 * 24) > 2 / 3)
        noise *= noise > 0

        # add strictly positive noise to ts1 and strictly negative noise to ts2
        ts1.value += abs(ts1.value * noise)
        ts2.value -= abs(ts2.value * noise)

        ts1.value[93 * 24] += 20
        ts1.value[96 * 24] -= 20
        ts2.value[93 * 24] += 20
        ts2.value[96 * 24] -= 20

        model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        response1 = model.fit_predict(test_ts[90 * 24 :], ts1[: 90 * 24])
        response2 = model.fit_predict(test_ts[90 * 24 :], ts2[: 90 * 24])

        self.assertGreater(
            response2.scores.value[test_index], response1.scores.value[test_index]
        )


class TestProphetTrendDetectorModel(TestCase):
    def setUp(self) -> None:
        self.data = load_air_passengers(return_ts=False)
        self.trend_detector = ProphetTrendDetectorModel()

    def test_response_shape_for_single_series(self) -> None:
        single_ts = TimeSeriesData(self.data)
        response_single_ts = self.trend_detector.fit_predict(
            data=single_ts, historical_data=None
        )

        self.assertEqual(response_single_ts.scores.time.shape, single_ts.time.shape)

        self.assertEqual(response_single_ts.scores.value.shape, single_ts.value.shape)

        self.assertEqual(
            # pyre-ignore[16]: Optional type has no attribute `value`.
            response_single_ts.predicted_ts.value.shape,
            single_ts.value.shape,
        )

    def test_response_shape_with_historical_data(self) -> None:
        single_ts = TimeSeriesData(self.data)
        historical_ts = TimeSeriesData(self.data)
        single_ts.time = single_ts.time + pd.tseries.offsets.DateOffset(
            months=len(self.data)
        )
        response = self.trend_detector.fit_predict(single_ts, historical_ts)

        self.assertTrue(np.array_equal(response.scores.time, single_ts.time))

    def test_pmm_use_case(self) -> None:
        random.seed(100)
        time_unit = 86400
        hist_data_time = [x * time_unit for x in range(0, 28)]
        data_time = [x * time_unit for x in range(28, 35)]

        hist_data_value = [random.normalvariate(100, 10) for _ in range(0, 28)]
        data_value = [random.normalvariate(130, 10) for _ in range(28, 35)]

        hist_ts = TimeSeriesData(
            time=pd.Series(hist_data_time),
            value=pd.Series(hist_data_value),
            use_unix_time=True,
            unix_time_units="s",
        )
        data_ts = TimeSeriesData(
            time=pd.Series(data_time),
            value=pd.Series(data_value),
            use_unix_time=True,
            unix_time_units="s",
        )

        response_with_historical_data = self.trend_detector.fit_predict(
            data=data_ts, historical_data=hist_ts
        )
        self.assertEqual(
            response_with_historical_data.scores.value.shape, data_ts.value.shape
        )
        response_wo_historical_data = self.trend_detector.fit_predict(data=hist_ts)
        self.assertEqual(
            response_wo_historical_data.scores.value.shape, hist_ts.value.shape
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("day", SeasonalityTypes.DAY),
            ("week", SeasonalityTypes.WEEK),
            ("weekend", SeasonalityTypes.WEEKEND),
            ("year", SeasonalityTypes.YEAR),
            (SeasonalityTypes.DAY, SeasonalityTypes.DAY),
            (SeasonalityTypes.WEEK, SeasonalityTypes.WEEK),
            (SeasonalityTypes.WEEKEND, SeasonalityTypes.WEEKEND),
            (SeasonalityTypes.YEAR, SeasonalityTypes.YEAR),
        ]
    )
    def test_to_seasonality(
        self, actual: Union[str, SeasonalityTypes], expected: SeasonalityTypes
    ) -> None:
        self.assertEqual(to_seasonality(actual), expected)


class TestGetHolidayDates(TestCase):
    def test_no_args(self) -> None:
        result = get_holiday_dates()
        self.assertTrue(result.empty)

    def test_only_holidays(self) -> None:
        holidays = pd.DataFrame(
            {"ds": pd.date_range(start="1/1/2020", end="1/10/2020")}
        )
        result = get_holiday_dates(holidays=holidays, dates=holidays["ds"])
        pd.testing.assert_series_equal(
            result,
            pd.to_datetime(pd.Series(holidays["ds"].dt.date, name=None))
            .sort_values(ignore_index=True)
            .rename(
                None
            ),  # if name setting to None on pd.Series constructor it doesn't really change a name!
        )

    def test_only_country_holidays(self) -> None:
        dates = pd.Series(pd.date_range(start="1/1/2020", end="12/31/2020"))
        result = get_holiday_dates(country_holidays="US", dates=dates)
        self.assertFalse(result.empty)
        self.assertTrue((result.dt.year == 2020).all())

    def test_holidays_and_country_holidays(self) -> None:
        holidays = pd.DataFrame(
            {"ds": pd.date_range(start="1/1/2020", end="1/10/2020")}
        )
        dates = pd.Series(pd.date_range(start="1/1/2020", end="12/31/2020"))
        result = get_holiday_dates(
            holidays=holidays, country_holidays="US", dates=dates
        )
        self.assertFalse(result.empty)
        self.assertTrue((result.dt.year == 2020).all())
        self.assertGreater(len(result), len(holidays))
