# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.detectors.multivariate_detector import MultivariateAnomalyDetectorModel


class TestMultivariateAnomalyDetectorModel(TestCase):
    def setUp(self) -> None:
        self.data = load_data("multivariate_anomaly_simulated_data.csv")
        self.outlier_detector = MultivariateAnomalyDetectorModel()

    def test_response_shape_for_multi_series(self) -> None:
        multi_ts = TimeSeriesData(self.data)
        response_multi_ts = self.outlier_detector.fit_predict(
            data=multi_ts, historical_data=None
        )

        self.assertEqual(response_multi_ts.scores.time.shape, multi_ts.time.shape)

        self.assertEqual(response_multi_ts.scores.value.shape, multi_ts.value.shape)

        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `value`.
            response_multi_ts.predicted_ts.value.shape,
            multi_ts.value.shape,
        )

    def test_response_shape_with_historical_data(self) -> None:
        single_ts = TimeSeriesData(self.data)
        historical_ts = TimeSeriesData(self.data)
        single_ts.time = single_ts.time + pd.tseries.offsets.DateOffset(
            days=len(self.data)
        )
        response = self.outlier_detector.fit_predict(single_ts, historical_ts)

        self.assertTrue(np.array_equal(response.scores.time, single_ts.time))
