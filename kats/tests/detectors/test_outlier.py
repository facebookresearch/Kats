# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.data.utils import load_data, load_air_passengers
from kats.detectors.outlier import (
    MultivariateAnomalyDetector,
    MultivariateAnomalyDetectorType,
    OutlierDetector,
)
from kats.models.bayesian_var import BayesianVARParams
from kats.models.var import VARParams

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


# Anomaly detection tests
class OutlierDetectionTest(TestCase):
    def setUp(self):
        data = load_air_passengers(return_ts=False)
        self.ts_data = TimeSeriesData(data)
        data_2 = data.copy()
        data_2["y_2"] = data_2["y"]
        self.ts_data_2 = TimeSeriesData(data_2)

        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        self.ts_data_daily = TimeSeriesData(daily_data)

    def test_additive_overrides(self) -> None:
        m = OutlierDetector(self.ts_data, "additive")

        m.detector()
        outliers = m.remover(interpolate=True)

        m2 = OutlierDetector(self.ts_data, "logarithmic")

        m2.detector()
        outliers2 = m2.remover(interpolate=True)

        self.assertEqual(outliers.value.all(), outliers2.value.all())

    def test_outlier_detection_additive(self) -> None:
        m = OutlierDetector(self.ts_data, "additive")

        m.detector()
        m.remover(interpolate=True)

        m2 = OutlierDetector(self.ts_data_daily, "additive")
        m2.detector()
        m2.remover(interpolate=True)
        # test for multiple time series
        m3 = OutlierDetector(self.ts_data_2, "additive")
        m3.detector()
        m3.remover(interpolate=True)

    def test_outlier_detection_multiplicative(self) -> None:
        m = OutlierDetector(self.ts_data, "multiplicative")
        m.detector()
        m.remover(interpolate=True)

        m2 = OutlierDetector(self.ts_data_daily, "multiplicative")
        m2.detector()
        m2.remover(interpolate=True)
        # test for multiple time series
        m3 = OutlierDetector(self.ts_data_2, "additive")
        m3.detector()
        m3.remover(interpolate=True)

    def test_outlier_detector_exception(self) -> None:
        data = self.ts_data.to_dataframe()
        data_new = pd.concat([data, data])
        ts_data_new = TimeSeriesData(data_new)

        with self.assertLogs(level="ERROR"):
            m = OutlierDetector(ts_data_new)
            m.detector()

    def test_output_scores_shape(self) -> None:
        # test for a single time series
        m = OutlierDetector(self.ts_data, "additive")
        m.detector()
        output_scores = m.output_scores

        self.assertIsNotNone(output_scores)
        if output_scores is not None:
            self.assertEqual(output_scores.shape[0], m.data.value.shape[0])

        # test that there are not any nan values in output score
        self.assertFalse(output_scores["y"].isnull().any())
        # test for more than 1 time series
        m2 = OutlierDetector(self.ts_data_2, "additive")
        m2.detector()
        output_scores2 = m2.output_scores

        self.assertIsNotNone(output_scores2)
        if output_scores2 is not None:
            self.assertEqual(output_scores2.shape, m2.data.value.shape)


class MultivariateVARDetectorTest(TestCase):
    def setUp(self):
        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_var_detector(self) -> None:
        np.random.seed(10)

        params = VARParams(maxlags=2)
        d = MultivariateAnomalyDetector(self.TSData_multi, params, training_days=60)
        anomaly_score_df = d.detector()
        self.assertCountEqual(
            list(anomaly_score_df.columns),
            list(self.TSData_multi.value.columns)
            + ["overall_anomaly_score", "p_value"],
        )
        d.plot()
        alpha = 0.05
        anomalies = d.get_anomaly_timepoints(alpha)
        d.get_anomalous_metrics(anomalies[0], top_k=3)

    def test_bayesian_detector(self) -> None:
        np.random.seed(10)

        params = BayesianVARParams(p=2)
        d = MultivariateAnomalyDetector(
            self.TSData_multi,
            params,
            training_days=60,
            model_type=MultivariateAnomalyDetectorType.BAYESIAN_VAR,
        )
        anomaly_score_df = d.detector()
        self.assertCountEqual(
            list(anomaly_score_df.columns),
            list(self.TSData_multi.value.columns)
            + ["overall_anomaly_score", "p_value"],
        )
        d.plot()
        alpha = 0.05
        anomalies = d.get_anomaly_timepoints(alpha)
        d.get_anomalous_metrics(anomalies[0], top_k=3)

    def test_runtime_errors(self) -> None:
        DATA_multi = self.TSData_multi.to_dataframe()
        DATA_multi2 = pd.concat([DATA_multi, DATA_multi])
        TSData_multi2 = TimeSeriesData(DATA_multi2)
        params = VARParams(maxlags=2)

        with self.assertRaises(RuntimeError):
            d = MultivariateAnomalyDetector(TSData_multi2, params, training_days=60)
            d.detector()

        DATA_multi3 = pd.merge(
            DATA_multi, DATA_multi, how="inner", on="time", suffixes=("_1", "_2")
        )
        TSData_multi3 = TimeSeriesData(DATA_multi3)
        with self.assertRaises(RuntimeError):
            d2 = MultivariateAnomalyDetector(TSData_multi3, params, training_days=60)
            d2.detector()
