# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from operator import attrgetter
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers
from kats.detectors.bocpd_model import BocpdDetectorModel, BocpdTrendDetectorModel
from kats.utils.simulator import Simulator
from parameterized import parameterized


class BocpdDetectorModelTest(TestCase):
    first_cp_begin: int = 100
    first_cp_end: int = 200
    second_cp_begin: int = 350

    def setUp(self) -> None:
        sim = Simulator(n=450, start="2018-01-01")

        cp_array_input = [
            BocpdDetectorModelTest.first_cp_begin,
            BocpdDetectorModelTest.first_cp_end,
            BocpdDetectorModelTest.second_cp_begin,
        ]

        self.ts_length = 450

        self.level_ts = sim.level_shift_sim(
            random_seed=100,
            cp_arr=cp_array_input,
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        # Define detector model with no historical data
        bocpd_detector = BocpdDetectorModel()
        self.no_history_model = bocpd_detector.fit_predict(data=self.level_ts)

        # setup history model
        history_model_ts_length = 450
        history_model_ts_history_length = 100
        self.history_model_scores_length = (
            history_model_ts_length - history_model_ts_history_length
        )

        level_ts_history = TimeSeriesData(
            time=self.level_ts.time.iloc[:history_model_ts_history_length],
            value=pd.Series(
                self.level_ts.value.iloc[:history_model_ts_history_length],
                name="value",
            ),
        )

        level_ts_data = TimeSeriesData(
            time=self.level_ts.time.iloc[history_model_ts_history_length:],
            value=pd.Series(
                self.level_ts.value.iloc[history_model_ts_history_length:],
                name="value",
            ),
        )

        history_bocpd_detector = BocpdDetectorModel()
        self.history_model = history_bocpd_detector.fit_predict(
            historical_data=level_ts_history, data=level_ts_data
        )

        # setup of slow drift model
        self.slow_drift_model_ts_length = 200

        drift_model_sim = Simulator(
            n=self.slow_drift_model_ts_length, start="2018-01-01"
        )
        drift_model_trend_ts = drift_model_sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )

        drift_model_bocpd_detector = BocpdDetectorModel(slow_drift=True)
        self.slow_drift_model = drift_model_bocpd_detector.fit_predict(
            data=drift_model_trend_ts
        )

        # setup of no drift model
        no_drift_model_bocpd_detector = BocpdDetectorModel(slow_drift=False)
        self.serialized_no_drift_model = no_drift_model_bocpd_detector.serialize()

        self.ignore_drift_parameter_model = BocpdDetectorModel(
            serialized_model=self.serialized_no_drift_model, slow_drift=True
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            [95, 105],  # Interval 1
            [195, 205],  # Interval 2
            [345, 355],  # Interval 3
        ]
    )
    def test_no_history_threshold(self, from_interval: int, to_interval: int) -> None:
        anom = self.no_history_model
        threshold = 0.4

        # we have set changepoints at 100, 200, 350
        # we want to make sure those are detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(
            np.max(anom.scores.value.values[from_interval:to_interval]) > threshold
        )

    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "of_no_history_model",
                "no_history_model.scores",
                "ts_length",
            ],
            [
                "of_history_model",
                "history_model.scores",
                "history_model_scores_length",
            ],
            [
                "of_slow_drift_model",
                "slow_drift_model.scores",
                "slow_drift_model_ts_length",
            ],
        ]
    )
    def test_scores_length_validation(
        self, name: str, attribute_predicted: str, attribute_actual: str
    ) -> None:
        self.assertEqual(
            len(attrgetter(attribute_predicted)(self)),
            attrgetter(attribute_actual)(self),
        )

    # pyre-ignore Undefined attribute [16]: Module parameterized.parameterized has no attribute expand.
    @parameterized.expand(
        [
            [95, 105],  # Interval 1
            [245, 255],  # Interval 2
        ]
    )
    def test_history_threshold(self, from_interval: int, to_interval: int) -> None:
        threshold = 0.4
        # we test for the two changepoints in 200, 350, but shifted by 100
        # since that is the length of the history
        self.assertTrue(
            np.max(self.history_model.scores.value.values[from_interval:to_interval])
            > threshold
        )

    def test_slow_drift_threshold(self) -> None:
        threshold = 0.4
        # we have set changepoints at 100
        # we want to make sure that is detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(
            np.max(self.slow_drift_model.scores.value.values[95:105]) > threshold
        )

    def test_serialize_ignore_slow_drift_parm(self) -> None:
        # In setup section:
        # In serialized_no_drift_model we used slow_drift=False.
        # But while reading the serialized model, we used slow_drift=True.
        # The parameter slow_drift=True must be ignored, and the model must have
        # the original parameter slow_drift=False (in the serialized model)
        self.assertEqual(self.ignore_drift_parameter_model.slow_drift, False)

    def test_serialize_fit_predict_with_ignore_slow_drift_parm(self) -> None:
        # See setup section to find how the ignore_drift_parameter_model is setup
        anom = self.ignore_drift_parameter_model.fit_predict(data=self.level_ts)
        self.assertEqual(len(anom.scores), self.ts_length)

    def test_missing_data(self) -> None:
        # this data is in the same format as OneDetection
        # it also crosses the daylight savings time
        history_time_list = (
            (
                pd.date_range(
                    "2020-03-01", "2020-03-10", tz="US/Pacific", freq="1d"
                ).astype(int)
                / 1e9
            )
            .astype(int)
            .to_list()
        )

        data_time_list = (
            (
                pd.date_range(
                    "2020-03-11", "2020-03-20", tz="US/Pacific", freq="1d"
                ).astype(int)
                / 1e9
            )
            .astype(int)
            .to_list()
        )

        history = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": (history_time_list[:5] + history_time_list[6:]),
                    "value": np.random.randn(len(history_time_list) - 1),
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )

        data = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": (data_time_list[:5] + data_time_list[6:]),
                    "value": np.random.randn(len(data_time_list) - 1),
                }
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )
        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(historical_data=history, data=data)

        self.assertEqual(len(anom.scores), len(data))


class TestBocpdTrendDetectorModel(TestCase):
    def setUp(self) -> None:
        self.data = load_air_passengers(return_ts=False)
        self.trend_detector = BocpdTrendDetectorModel()

    def test_response_shape_for_single_series(self) -> None:
        single_ts = TimeSeriesData(self.data)
        response_single_ts = self.trend_detector.fit_predict(
            data=single_ts, historical_data=None
        )

        self.assertEqual(response_single_ts.scores.time.shape, single_ts.time.shape)

        self.assertEqual(response_single_ts.scores.value.shape, single_ts.value.shape)

        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `value`.
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
