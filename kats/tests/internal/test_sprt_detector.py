# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.sprt_detector import (
    SPRTDetectorModel,
)

TOL = 1e-10  # numeric tolerance


class TestStreamingSPRTDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(928)
        self.n = 72
        time = pd.date_range("2021-01-01", periods=self.n, freq="H")
        value_a = np.random.normal(1, 0.1, self.n)
        value_b = np.concatenate(
            [np.random.normal(1, 0.2, 60), np.random.normal(2, 0.2, 12)]
        )
        variance_a = np.random.normal(0.1, 0.001, self.n)
        variance_b = np.random.normal(0.2, 0.001, self.n)
        sample_count_a = [1000] * 72
        sample_count_b = [100] * 72
        df_b_greater = pd.DataFrame(
            {
                "time": time,
                "value_a": value_a,
                "value_b": value_b,
                "variance_a": variance_a,
                "variance_b": variance_b,
                "sample_count_a": sample_count_a,
                "sample_count_b": sample_count_b,
            }
        )
        tsd = TimeSeriesData(
            df=df_b_greater,
            time=df_b_greater["time"],
            tz="utc",
        )
        model = SPRTDetectorModel(
            alpha=0.05,
            beta=0.08,
            effect_size=1,
            distribution="normal",
            test_direction="b_greater",
        )
        _ = model.fit_predict(data=tsd[:48])
        pre_serialized_model = model.serialize()
        self.anomaly_score = TimeSeriesData(
            time=pd.Series(), value=pd.Series([], name="ts_value")
        )

        for i in range(48, self.n):
            model = SPRTDetectorModel(serialized_model=pre_serialized_model)
            self.anomaly_score.extend(
                model.fit_predict(
                    data=tsd[i : i + 1], historical_data=tsd[i - 48 : i]
                ).scores,
                validate=False,
            )
            pre_serialized_model = model.serialize()

        self.anomaly_score.validate_data(
            validate_frequency=True, validate_dimension=False
        )

    def test_streaming_length_match(self) -> None:
        self.assertEqual(len(self.anomaly_score), self.n - 48)

    def test_streaming_value_match(self) -> None:
        self.assertTrue((self.anomaly_score.value.value > 0).sum() == 12)


class TestAGreaterNormalSPRTDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(929)
        self.n = 72
        time = pd.date_range("2021-01-01", periods=self.n, freq="H")
        value_a = np.concatenate(
            [np.random.normal(1, 0.2, 60), np.random.normal(2, 0.2, 12)]
        )
        value_b = np.random.normal(1, 0.1, self.n)

        variance_a = np.random.normal(0.2, 0.001, self.n)
        variance_b = np.random.normal(0.1, 0.001, self.n)
        sample_count_a = [1000] * 72
        sample_count_b = [100] * 72
        df_a_greater = pd.DataFrame(
            {
                "time": time,
                "value_a": value_a,
                "value_b": value_b,
                "variance_a": variance_a,
                "variance_b": variance_b,
                "sample_count_a": sample_count_a,
                "sample_count_b": sample_count_b,
            }
        )
        self.df = df_a_greater
        self.tsd = TimeSeriesData(
            df=df_a_greater,
            time=df_a_greater["time"],
            tz="utc",
        )
        self.model = SPRTDetectorModel(
            alpha=0.05,
            beta=0.08,
            effect_size=1.0,
            distribution="normal",
            test_direction="a_greater",
        )
        self.serialized_model = self.model.serialize()
        self.anomaly_response = self.model.fit_predict(self.tsd)

    def test_serialized_model(self) -> None:
        self.assertIsInstance(self.serialized_model, bytes)

    def test_new_model(self) -> None:
        model_new = SPRTDetectorModel(
            self.serialized_model, effect_size=None, alpha=0.1
        )
        self.assertEqual(model_new.alpha, 0.05)

    def test_model(self) -> None:
        self.assertNotEqual(
            self.model,
            SPRTDetectorModel(
                alpha=0.05,
                beta=0.08,
                effect_size=1.0,
                distribution="normal",
                test_direction="a_greater",
            ),
        )

    def test_anomaly_score_consistency(self) -> None:
        tsd1 = TimeSeriesData(
            df=self.df.loc[:63],
            time=self.df.loc[:63]["time"],
            tz="utc",
        )
        tsd2 = TimeSeriesData(
            df=self.df.loc[64:],
            time=self.df.loc[64:]["time"],
            tz="utc",
        )
        anomaly_response1 = self.model.fit_predict(tsd1)
        anomaly_response2 = self.model.fit_predict(tsd2, historical_data=tsd1)
        ar_scores1 = anomaly_response1.scores
        ar_scores1.extend(anomaly_response2.scores)
        self.assertFalse(
            any(np.abs(ar_scores1.value - self.anomaly_response.scores.value)) > TOL
        )


class TestBGreaterBinomialSPRTDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(930)
        self.n = 72
        time = pd.date_range("2021-01-01", periods=self.n, freq="H")
        value_a = np.random.binomial(n=100, p=0.7, size=self.n)
        value_b = np.concatenate(
            [
                np.random.binomial(n=100, p=0.7, size=60),
                np.random.binomial(n=100, p=0.95, size=12),
            ]
        )

        variance_a = np.random.normal(0.2, 0.001, self.n)
        variance_b = np.random.normal(0.1, 0.001, self.n)
        sample_count_a = [1000] * 72
        sample_count_b = [100] * 72
        df_b_greater = pd.DataFrame(
            {
                "time": time,
                "value_a": value_a,
                "value_b": value_b,
                "variance_a": variance_a,
                "variance_b": variance_b,
                "sample_count_a": sample_count_a,
                "sample_count_b": sample_count_b,
            }
        )
        self.df = df_b_greater
        self.tsd = TimeSeriesData(
            df=df_b_greater,
            time=df_b_greater["time"],
            tz="utc",
        )
        self.model = SPRTDetectorModel(
            alpha=0.05,
            beta=0.08,
            effect_size=10,
            distribution="binomial",
            test_direction="b_greater",
        )
        self.serialized_model = self.model.serialize()
        self.anomaly_response = self.model.fit_predict(self.tsd)

    def test_serialized_model(self) -> None:
        self.assertIsInstance(self.serialized_model, bytes)

    def test_new_model(self) -> None:
        model_new = SPRTDetectorModel(
            self.serialized_model, effect_size=None, alpha=0.1
        )
        self.assertEqual(model_new.alpha, 0.05)

    def test_model(self) -> None:
        self.assertNotEqual(
            self.model,
            SPRTDetectorModel(
                alpha=0.05,
                beta=0.08,
                effect_size=1.0,
                distribution="binomial",
                test_direction="b_greater",
            ),
        )

    def test_anomaly_score_consistency(self) -> None:
        tsd1 = TimeSeriesData(
            df=self.df.loc[:63],
            time=self.df.loc[:63]["time"],
            tz="utc",
        )
        tsd2 = TimeSeriesData(
            df=self.df.loc[64:],
            time=self.df.loc[64:]["time"],
            tz="utc",
        )
        anomaly_response1 = self.model.fit_predict(tsd1)
        anomaly_response2 = self.model.fit_predict(tsd2, historical_data=tsd1)
        ar_scores1 = anomaly_response1.scores
        ar_scores1.extend(anomaly_response2.scores)
        self.assertFalse(
            any(np.abs(ar_scores1.value - self.anomaly_response.scores.value)) > TOL
        )
