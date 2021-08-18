# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
import unittest
from datetime import timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.prophet_detector import (
    ProphetDetectorModel,
    ProphetScoreFunction,
)
from kats.utils.simulator import Simulator

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class TestProphetDetector(TestCase):
    def create_random_ts(self, seed, length, magnitude, slope_factor):
        np.random.seed(seed)
        sim = Simulator(n=length, freq="1D", start=pd.to_datetime("2020-01-01"))

        sim.add_trend(magnitude=magnitude * np.random.rand() * slope_factor)
        sim.add_seasonality(
            magnitude * np.random.rand(),
            period=timedelta(days=7),
        )

        sim.add_noise(magnitude=0.1 * magnitude * np.random.rand())
        return sim.stl_sim()

    def create_ts(
        self, seed=0, length=100, magnitude=10, signal_to_noise_ratio=0.1, freq="1D"
    ):
        np.random.seed(seed)
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime("2020-01-01"))

        sim.add_seasonality(magnitude, period=timedelta(days=7))
        sim.add_noise(magnitude=signal_to_noise_ratio * magnitude)
        return sim.stl_sim()

    def create_multi_seasonality_ts(
        self, seed, length, freq, min_val, max_val, signal_to_noise_ratio
    ):
        np.random.seed(seed)

        sim = Simulator(n=length, freq=freq, start=pd.to_datetime("2020-01-01"))
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

    def add_smooth_anomaly(self, ts, seed, start_index, length, magnitude):
        # Add an anomaly that is half of a sine wave
        # start time and freq don't matter, since we only care about the values
        np.random.seed(seed)

        anomaly_sim = Simulator(n=length, freq="1D", start=pd.to_datetime("2020-01-01"))
        anomaly_sim.add_seasonality(magnitude, period=timedelta(days=2 * length))
        # anomaly_sim.add_noise(magnitude=0.3 * magnitude * np.random.rand())

        anomaly_ts = anomaly_sim.stl_sim()
        for i in range(0, length):
            ts.value.iloc[start_index + i] += anomaly_ts.value[i]

    def truncate(self, ts, start_index, end_index):
        # Set all values outside the range [start_index, end_index) to 0
        ts.value.iloc[:start_index] *= 0
        ts.value.iloc[end_index:] *= 0

    def add_trend_shift(self, ts, length, freq, magnitude):
        ts_df = ts.to_dataframe()
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime("2020-01-01"))
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

    def horiz_translate(self, ts, periods):
        ts.value = ts.value.shift(periods=periods, fill_value=0)

    def add_multiplicative_noise(self, ts, magnitude):
        # Multiply all the values in ts by a number in the range [1-magnitude, 1+magnitude]
        ts.value *= np.random.rand(len(ts)) * magnitude * 2 + 1 - magnitude

    def merge_ts(self, ts1, ts2):
        ts1_df, ts2_df = ts1.to_dataframe(), ts2.to_dataframe()
        merged_df = (ts1_df.set_index("time") + ts2_df.set_index("time")).reset_index()
        merged_ts = TimeSeriesData(df=merged_df)
        return merged_ts

    def add_multi_event(
        self,
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
    ):

        np.random.seed(seed)
        sim = Simulator(n=length, freq=freq, start=pd.to_datetime("2020-01-01"))

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
            seasonal_period=event1_duration / 2,
            seasonal_magnitude=event1_magnitude,
            noise=signal_to_noise_ratio * magnitude,
        )

        event2_ts = sim.level_shift_sim(
            seasonal_period=event2_duration / 2,
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

    def calc_stds(self, predicted_val, upper_bound, lower_bound):
        actual_upper_std = (50 ** 0.5) * (upper_bound - predicted_val) / 0.8
        actual_lower_std = (50 ** 0.5) * (predicted_val - lower_bound) / 0.8

        upper_std = max(actual_upper_std, 1e-9)
        lower_std = max(actual_lower_std, 1e-9)

        return upper_std, lower_std

    def calc_z_score(self, actual_val, predicted_val, upper_bound, lower_bound):
        upper_std, lower_std = self.calc_stds(predicted_val, upper_bound, lower_bound)

        if actual_val > predicted_val:
            return (actual_val - predicted_val) / upper_std
        else:
            return (actual_val - predicted_val) / lower_std

    def test_no_anomaly(self) -> None:
        # Prophet should not find any anomalies on a well formed synthetic time series
        for i in range(0, 5):
            ts = self.create_random_ts(i, 100, 10, 2)

            model = ProphetDetectorModel()
            model.fit(ts[:90])

            # alternate between using the current model and using serialized model
            if i % 2 == 0:
                serialized_model = model.serialize()
                model = ProphetDetectorModel(serialized_model=serialized_model)

            res = model.predict(ts[90:])
            self.assertEqual(len(res.scores), 10)
            anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
            self.assertFalse(anomaly_found)

    def test_anomaly(self) -> None:
        # Prophet should find anomalies
        for i in range(0, 5):
            ts = self.create_random_ts(i, 100, 10, 2)
            self.add_smooth_anomaly(ts, i, 90, 10, 10)

            model = ProphetDetectorModel()
            model.fit(ts[:90])

            # alternate between using the current model and using serialized model
            if i % 2 == 0:
                serialized_model = model.serialize()
                model = ProphetDetectorModel(serialized_model=serialized_model)

            res = model.predict(ts[90:])
            self.assertEqual(len(res.scores), 10)
            anomaly_found = res.scores.min < -0.3 or res.scores.max > 0.3
            self.assertTrue(anomaly_found)

    def test_fit_predict(self) -> None:
        ts = self.create_random_ts(0, 100, 10, 2)
        self.add_smooth_anomaly(ts, 0, 90, 10, 10)

        model = ProphetDetectorModel()
        model.fit(ts[:90])
        res0 = model.predict(ts[90:])

        model = ProphetDetectorModel()
        res1 = model.fit_predict(data=ts[90:], historical_data=ts[:90])

        self.assertEqual(res0.scores.value.to_list(), res1.scores.value.to_list())

    def test_outlier_removal_threshold(self):
        ts = self.create_random_ts(0, 365, 10, 2)
        ts_df = pd.DataFrame({"ds": ts.time, "y": ts.value})

        model = ProphetDetectorModel()

        filtered_ts_df = model._remove_outliers(ts_df, outlier_ci_threshold=0.99)

        aggressively_filtered_ts_df = model._remove_outliers(
            ts_df, outlier_ci_threshold=0.8
        )

        self.assertGreaterEqual(len(ts_df), len(filtered_ts_df))
        self.assertGreaterEqual(len(filtered_ts_df), len(aggressively_filtered_ts_df))

    def test_outlier_removal_efficacy(self):
        def _subtest(
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
        ):
            model = ProphetDetectorModel()

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
                "Expected removing outliers when training model to lower prediciton RMSE",
            )

        baseline_ts = self.create_multi_seasonality_ts(0, 960, "15min", 0, 1000, 0.1)

        with self.subTest("Testing with early event"):
            _subtest(baseline_ts, 0, 960, "15min", 0, 1000, 0.1, 0.15, 0.3, 1.5)

        with self.subTest("Testing with late event"):
            _subtest(baseline_ts, 0, 960, "15min", 0, 1000, 0.1, 0.72, 0.85, -2)

        with self.subTest("Testing with spiky event"):
            _subtest(baseline_ts, 0, 960, "15min", 0, 1000, 0.1, 0.5, 0.55, 5)

        with self.subTest("Testing with prolonged event"):
            _subtest(baseline_ts, 0, 960, "15min", 0, 1000, 0.1, 0.35, 0.62, -1.5)

        noisy_ts = self.create_multi_seasonality_ts(0, 960, "15min", 0, 1000, 0.5)

        with self.subTest("Testing with noisy underlying data"):
            _subtest(noisy_ts, 0, 960, "15min", 0, 1000, 0.5, 0.5, 0.55, 5)

    def test_score_parameter(self):
        """Tests the behavior of the score_func parameter.

        This test verifies:
        (1) the default implementation of ProphetDetectorModel
        uses the 'deviation_from_predicted_val' scoring function;
        (2) passing ProphetScoreFunction.z_score as the 'score_func' results in
        ProphetDetectorModel implementing the 'z_score' scoring function;
        (3) the anomaly scores returned by each of these functions
        are identical to the actual deviation and actual z_score.
        """
        ts = self.create_ts()

        # add anomaly at index 95
        ts.value[95] += 100

        deviation_model = ProphetDetectorModel()
        deviation_response = deviation_model.fit_predict(ts[90:], ts[:90])
        self.assertEqual(
            deviation_response.scores.value[5],
            abs(
                (ts.value[95] - deviation_response.predicted_ts.value[5])
                / deviation_response.predicted_ts.value[5]
            ),
        )

        z_score_model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
        z_score_response = z_score_model.fit_predict(ts[90:], ts[:90])
        actual_z_score = self.calc_z_score(
            ts.value[95],
            z_score_response.predicted_ts.value[5],
            z_score_response.confidence_band.upper.value[5],
            z_score_response.confidence_band.lower.value[5],
        )
        self.assertAlmostEqual(
            z_score_response.scores.value[5], actual_z_score, places=15
        )

    def test_flat_signal(self):
        """Tests the behavior of the z-score strategy on flat signals.

        This test verifies that the model's z_scores of flat signals
        with and without anomalies are identical to the actual z_scores.
        It ensures no division by zero errors occur when
        the signal has no seasonality or level shifts.
        """
        ts = self.create_ts(magnitude=0, signal_to_noise_ratio=0)

        for anomaly_magnitude in (0, 100):
            ts.value[95] += anomaly_magnitude

            model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
            response = model.fit_predict(ts[90:], ts[:90])
            actual_z_score = self.calc_z_score(
                ts.value[95],
                response.predicted_ts.value[5],
                response.confidence_band.upper.value[5],
                response.confidence_band.lower.value[5],
            )
            self.assertAlmostEqual(response.scores.value[5], actual_z_score, places=15)

    def test_zero_noise_signal(self):
        """Tests the behavior of the z-score strategy on zero-noise signals.

        This test verifies that the model's z_scores of zero-noise signals
        with and without anomalies areidentical to the actual z_scores.
        It ensures no division by zero errors occur when the signal has
        no noise and the standard deviation of the training data is zero.
        """
        ts = self.create_ts(signal_to_noise_ratio=0)

        for anomaly_magnitude in (0, 100):
            ts.value[95] += anomaly_magnitude

            model = ProphetDetectorModel(score_func=ProphetScoreFunction.z_score)
            response = model.fit_predict(ts[90:], ts[:90])
            actual_z_score = self.calc_z_score(
                ts.value[95],
                response.predicted_ts.value[5],
                response.confidence_band.upper.value[5],
                response.confidence_band.lower.value[5],
            )
            self.assertAlmostEqual(response.scores.value[5], actual_z_score, places=15)

    @unittest.skip(
        "Prophet doesn't learn heteroskedastic seasonality with params used by ProphetDetectorModel"
    )
    def test_heteroskedastic_noise_signal(self):
        """Tests the z-score strategy on signals with heteroskedastic noise

        This test creates synthetic data with heteroskedastic noise. Then, it adds
        anomalies of identical magnitudes to segments with different noise. Finally, it
        verifies that anomalies in low-noise segments have higher z-scores than those
        in high-noise segments. This occurs because low noise segments will have lower
        standard deviations, which result in higher z-scores.
        """
        ts = self.create_ts(length=100 * 24, signal_to_noise_ratio=0.05, freq="1h")

        # add heteroskedastic noise to the data

        ts.value *= (
            (ts.time - pd.to_datetime("2020-01-01")) % timedelta(days=7)
            > timedelta(days=3.5)
        ) * np.random.rand(100 * 24) + 0.5

        ts.value[93 * 24] += 100
        ts.value[96 * 24] += 100

        model = ProphetDetectorModel(score_func="z_score")
        response = model.fit_predict(ts[90 * 24 :], ts[: 90 * 24])

        self.assertGreater(response.scores.value[3 * 24], response.scores.value[6 * 24])

    def test_z_score_proportional_to_anomaly_magnitude(self):
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

    def test_asymmetric_noise_signal(self):
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
            response2.scores.value[3 * 24], response1.scores.value[3 * 24]
        )
        self.assertGreater(
            response2.scores.value[6 * 24], response1.scores.value[6 * 24]
        )
