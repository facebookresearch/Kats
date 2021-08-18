# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.bocpd_model import BocpdDetectorModel
from kats.utils.simulator import Simulator

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class BocpdDetectorModelTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    def setUp(self):
        self.sim = Simulator(n=450, start="2018-01-01")

        self.cp_array_input = [
            BocpdDetectorModelTest.first_cp_begin,
            BocpdDetectorModelTest.first_cp_end,
            BocpdDetectorModelTest.second_cp_begin,
        ]

        self.ts_length = 450
        self.sigma = 0.05

        self.level_arr = [1.35, 1.05, 1.35, 1.2]

    def test_no_history(self) -> None:

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(data=level_ts)
        self.assertEqual(len(anom.scores), self.ts_length)
        threshold = 0.4

        # we have set changepoints at 100, 200, 350
        # we want to make sure those are detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[195:205]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[345:355]) > threshold)

    def test_history(self) -> None:
        ts_length = 450
        ts_history_length = 100

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        level_ts_history = TimeSeriesData(
            time=level_ts.time.iloc[:ts_history_length],
            value=pd.Series(level_ts.value.iloc[:ts_history_length], name="value"),
        )

        level_ts_data = TimeSeriesData(
            time=level_ts.time.iloc[ts_history_length:],
            value=pd.Series(level_ts.value.iloc[ts_history_length:], name="value"),
        )

        bocpd_detector = BocpdDetectorModel()
        anom = bocpd_detector.fit_predict(
            historical_data=level_ts_history, data=level_ts_data
        )
        self.assertEqual(len(anom.scores), ts_length - ts_history_length)

        threshold = 0.4
        # same as above.
        # we test for the two changepoints in 200, 350, but shifted by 100
        # since that is the length of the history
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)
        self.assertTrue(np.max(anom.scores.value.values[245:255]) > threshold)

    def test_slow_drift(self) -> None:
        ts_length = 200

        sim = Simulator(n=ts_length, start="2018-01-01")
        trend_ts = sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )
        bocpd_detector = BocpdDetectorModel(slow_drift=True)
        anom = bocpd_detector.fit_predict(data=trend_ts)
        self.assertEqual(len(anom.scores), ts_length)
        threshold = 0.4

        # we have set changepoints at 100
        # we want to make sure that is detected
        # we set some slack for them be detected
        # 5 time points before/after
        self.assertTrue(np.max(anom.scores.value.values[95:105]) > threshold)

    def test_serialize(self) -> None:

        level_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=self.sigma,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_detector = BocpdDetectorModel(slow_drift=False)
        ser_model = bocpd_detector.serialize()

        # check that it ignores the slow_drift parameter
        # and considers the serialized one instead
        bocpd_detector2 = BocpdDetectorModel(
            serialized_model=ser_model, slow_drift=True
        )
        self.assertEqual(bocpd_detector2.slow_drift, False)

        anom = bocpd_detector2.fit_predict(data=level_ts)
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
