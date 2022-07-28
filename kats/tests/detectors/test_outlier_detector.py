# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers
from kats.detectors.outlier_detector import OutlierDetectorModel


class TestOutlierDetectorModel(TestCase):
    def setUp(self) -> None:
        self.data = load_air_passengers(return_ts=False)
        self.outlier_detector = OutlierDetectorModel()

    def test_response_shape_for_single_series(self) -> None:
        single_ts = TimeSeriesData(self.data)
        response_single_ts = self.outlier_detector.fit_predict(
            data=single_ts, historical_data=None
        )

        self.assertEqual(response_single_ts.scores.time.shape, single_ts.time.shape)

        self.assertEqual(response_single_ts.scores.value.shape, single_ts.value.shape)

        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `value`.
            response_single_ts.predicted_ts.value.shape,
            single_ts.value.shape,
        )

    def test_response_shape_for_multi_series(self) -> None:
        data = self.data.copy()
        data["y_2"] = data["y"]
        multi_ts = TimeSeriesData(data)
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
            months=len(self.data)
        )
        response = self.outlier_detector.fit_predict(single_ts, historical_ts)

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

        response_with_historical_data = self.outlier_detector.fit_predict(
            data=data_ts, historical_data=hist_ts
        )
        self.assertEqual(
            response_with_historical_data.scores.value.shape, data_ts.value.shape
        )
        response_wo_historical_data = self.outlier_detector.fit_predict(data=hist_ts)
        self.assertEqual(
            response_wo_historical_data.scores.value.shape, hist_ts.value.shape
        )

    def test_remover_usecase(self) -> None:
        # manually add outlier on the date of '1950-12-01'
        self.data.loc[self.data.time == "1950-12-01", "y"] *= 5
        # manually add outlier on the date of '1959-12-01'
        self.data.loc[self.data.time == "1959-12-01", "y"] *= 4

        single_ts = TimeSeriesData(self.data)

        response_with_interpolate = self.outlier_detector.fit_predict(
            single_ts, historical_ts=None, interpolate=True
        )

        response_with_no_interpolate = self.outlier_detector.fit_predict(
            data=single_ts, historical_data=None, interpolate=False
        )

        # When no iterpolate argument is given by default it is taking False
        response_with_default_interpolate = self.outlier_detector.fit_predict(
            data=single_ts, historical_data=None
        )

        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `time`.
            response_with_interpolate.predicted_ts.time.shape,
            single_ts.time.shape,
        )
        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `value`.
            response_with_interpolate.predicted_ts.value.shape,
            single_ts.value.shape,
        )

        self.assertEqual(
            response_with_no_interpolate.predicted_ts.time.shape, single_ts.time.shape
        )
        self.assertEqual(
            response_with_no_interpolate.predicted_ts.value.shape, single_ts.value.shape
        )

        self.assertEqual(
            response_with_default_interpolate.predicted_ts.time.shape,
            single_ts.time.shape,
        )
        self.assertEqual(
            response_with_default_interpolate.predicted_ts.value.shape,
            single_ts.value.shape,
        )
