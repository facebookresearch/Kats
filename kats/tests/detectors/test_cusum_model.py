# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from operator import attrgetter
from typing import Callable
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import IRREGULAR_GRANULARITY_ERROR, TimeSeriesData
from kats.detectors.cusum_model import CUSUMDetectorModel, CusumScoreFunction
from parameterized.parameterized import parameterized


class TestIncreaseCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        self.scan_window = 24 * 60 * 60  # in seconds
        self.historical_window = 3 * 24 * 60 * 60  # in seconds
        self.test_data_window = 16  # in hours
        self.regression_sum_score = 12  #
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        self.tsd = TimeSeriesData(df_increase)
        self.tsd_value_name = self.tsd.value.name
        data = self.tsd[-self.test_data_window :]

        self.model = CUSUMDetectorModel(
            scan_window=self.scan_window, historical_window=self.historical_window
        )

        self.score_tsd = self.model.fit_predict(
            data=data, historical_data=self.tsd[: -self.test_data_window]
        ).scores

        self.score_tsd_percentage_change = self.model._predict(
            data=data,
            score_func=CusumScoreFunction.percentage_change.value,
        ).score

        self.score_tsd_z_score = self.model._predict(
            data=data, score_func=CusumScoreFunction.z_score.value
        ).score

        self.serialized_model = self.model.serialize()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("increase_length_match", len, "score_tsd", "test_data_window"),
            (
                # make sure the time series name are the same
                "increase_time_series_name_match",
                lambda x: x.value.name,
                "score_tsd",
                "tsd_value_name",
            ),
            (
                # the regression is detected
                "increase_regression_detected",
                # pyre-ignore[16]: bool has no attribute sum
                lambda x: (x.value > 0).sum(),
                "score_tsd",
                "regression_sum_score",
            ),
            (
                "increase_percentage_change_length_match",
                len,
                "score_tsd_percentage_change",
                "test_data_window",
            ),
            (
                # the regression is detected
                "increase_percentage_change_regression_detected",
                lambda x: (x.value > 0).sum(),
                "score_tsd_percentage_change",
                "regression_sum_score",
            ),
            (
                "increase_z_score_length_match",
                len,
                "score_tsd_z_score",
                "test_data_window",
            ),
            (
                # the regression is detected
                "increase_z_score_regression_detected",
                lambda x: (x.value > 0).sum(),
                "score_tsd_z_score",
                "regression_sum_score",
            ),
        ]
    )
    def test_score_tsd(
        self,
        name: str,
        func_: Callable[[TimeSeriesData], float],
        attr1: str,
        attr2: str,
    ) -> None:
        self.assertEqual(func_(attrgetter(attr1)(self)), attrgetter(attr2)(self))

    def test_serialized_model(self) -> None:
        self.assertIsInstance(self.serialized_model, bytes)

    def test_new_model(self) -> None:
        model_new = CUSUMDetectorModel(self.serialized_model)
        self.assertEqual(model_new, self.model)

    def test_model(self) -> None:
        self.assertNotEqual(
            self.model,
            CUSUMDetectorModel(
                scan_window=self.scan_window, historical_window=self.historical_window
            ),
        )


class TestDecreaseCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        scan_window = 24 * 60 * 60  # in seconds
        historical_window = 3 * 24 * 60 * 60  # in seconds
        self.test_data_window = 6  # in hours
        df_decrease = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(2, 0.2, 156), np.random.normal(1, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_decrease)
        data = tsd[-self.test_data_window :]

        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )

        _ = model.fit_predict(
            data=data, historical_data=tsd[: -self.test_data_window]
        ).scores

        self.score_tsd = model._predict(
            data=data, score_func=CusumScoreFunction.change.value
        ).score

        self.score_tsd_percentage_change = model._predict(
            data=data,
            score_func=CusumScoreFunction.percentage_change.value,
        ).score

        self.score_tsd_z_score = model._predict(
            data=data, score_func=CusumScoreFunction.z_score.value
        ).score

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("decrease_length_match", len, "score_tsd"),
            (
                # the regression is detected
                "decrease_regression_detected",
                # pyre-ignore[16]: bool has no attribute sum
                lambda x: (x.value < 0).sum(),
                "score_tsd",
            ),
            (
                "decrease_percentage_change_length_match",
                len,
                "score_tsd_percentage_change",
            ),
            (
                # the regression is detected
                "decrease_percentage_change_regression_detected",
                lambda x: (x.value < 0).sum(),
                "score_tsd_percentage_change",
            ),
            (
                "decrease_z_score_length_match",
                len,
                "score_tsd_z_score",
            ),
            (
                # the regression is detected
                "decrease_z_score_regression_detected",
                lambda x: (x.value < 0).sum(),
                "score_tsd_z_score",
            ),
        ]
    )
    def test_score_tsd(
        self, name: str, func_: Callable[[TimeSeriesData], float], attr: str
    ) -> None:
        self.assertEqual(func_(attrgetter(attr)(self)), self.test_data_window)


class TestAdhocCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        self.historical_window = 48 * 60 * 60  # in seconds
        self.scan_window = 11 * 60 * 60 + 50  # in seconds
        n = 168
        self.const_0 = 0
        self.const_12 = 12
        self.const_24 = 24
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [
                        np.random.normal(1, 0.2, 48),
                        np.random.normal(0.2, 0.1, 12),
                        np.random.normal(1, 0.2, 60),
                        np.random.normal(2, 0.2, 24),
                        np.random.normal(0.9, 0.2, 24),
                    ]
                ),
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
            }
        )
        self.tsd = TimeSeriesData(df_increase)

        model = CUSUMDetectorModel(
            scan_window=self.scan_window, historical_window=self.historical_window
        )
        self.score_tsd = model.fit_predict(data=self.tsd).scores

        # test not enough data
        model = CUSUMDetectorModel(
            scan_window=self.scan_window, historical_window=self.historical_window
        )
        self.score_tsd_not_enough_data = model.fit_predict(
            data=self.tsd[-4:], historical_data=self.tsd[-8:-4]
        ).scores

        model = CUSUMDetectorModel(
            scan_window=self.scan_window, historical_window=2 * 3600
        )
        self.score_tsd_fixed_historical_window = model.fit_predict(
            data=self.tsd[-8:]
        ).scores

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("adhoc", len, "score_tsd", len, "tsd"),
            (
                # the regression is went away
                "adhoc_regression_went_away",
                lambda x: x.value[-6:].sum(),
                "score_tsd",
                lambda x: x,
                "const_0",
            ),
            (
                # the increase regression is detected
                "adhoc_regression_increased",
                # pyre-ignore[16]: bool has no attribute sum
                lambda x: (x.value > 0.5).sum(),
                "score_tsd",
                lambda x: x,
                "const_24",
            ),
            (
                # the decrease regression is detected
                "adhoc_regression_decreased",
                lambda x: (x.value < -0.45).sum(),
                "score_tsd",
                lambda x: x,
                "const_12",
            ),
            (
                # test not enough data
                "adhoc_not_enough_data_length_match",
                len,
                "score_tsd_not_enough_data",
                lambda x: len(x[-4:]),
                "tsd",
            ),
            (
                # test not enough data
                "adhoc_not_enough_data_zero_sum_score",
                lambda x: x.value.sum(),
                "score_tsd_not_enough_data",
                lambda x: x,
                "const_0",
            ),
            (
                "adhoc_length_match",
                len,
                "score_tsd_fixed_historical_window",
                lambda x: len(x[-8:]),
                "tsd",
            ),
            (
                "adhoc_zero_sum_score",
                lambda x: x.value.sum(),
                "score_tsd_fixed_historical_window",
                lambda x: x,
                "const_0",
            ),
        ]
    )
    def test_score_tsd(
        self,
        name: str,
        func_1: Callable[[TimeSeriesData], float],
        attr1: str,
        func_2: Callable[[TimeSeriesData], float],
        attr2: str,
    ) -> None:
        self.assertEqual(
            func_1(attrgetter(attr1)(self)), func_2(attrgetter(attr2)(self))
        )


class TestMissingDataCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        df = pd.DataFrame(
            {
                "ts_value": [0] * 8,
                "time": [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-08",
                    "2020-01-09",
                ],
            }
        )
        self.tsd = TimeSeriesData(df)
        # We also assume a bad input here
        model = CUSUMDetectorModel(
            scan_window=2 * 24 * 3600,
            historical_window=2 * 24 * 3600,
        )
        self.score_tsd = model.fit_predict(
            data=self.tsd,
        ).scores

    def test_missing_data_length_match(self) -> None:
        self.assertEqual(len(self.score_tsd), len(self.tsd))

    def test_missing_data_value_match(self) -> None:
        self.assertTrue((self.score_tsd.time.values == self.tsd.time.values).all())


class TestStreamingCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        self.n = 72
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 60), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=self.n, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)
        # Priming the model
        model = CUSUMDetectorModel(
            historical_window=historical_window, scan_window=scan_window
        )
        model.fit(data=tsd[:48])
        pre_serialized_model = model.serialize()

        self.anomaly_score = TimeSeriesData(
            time=pd.Series(), value=pd.Series([], name="ts_value")
        )
        # feeding 1 new data point a time
        for i in range(48, self.n):
            model = CUSUMDetectorModel(
                serialized_model=pre_serialized_model,
                historical_window=historical_window,
                scan_window=scan_window,
            )
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
        self.assertTrue(8 <= (self.anomaly_score.value > 0).sum() <= 12)


class TestDecomposingSeasonalityCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        historical_window = 10 * 24 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        n = 480
        periodicity = 24
        self.const_0 = 0
        self.const_10 = 10

        df_sin = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
                "ts_value": np.concatenate([20 * np.ones(n // 2), 21 * np.ones(n // 2)])
                + 4 * np.sin(2 * np.pi / periodicity * np.arange(0, n)),
            }
        )

        # removing a few data points to test the missing value handling as well
        self.tsd = TimeSeriesData(pd.concat([df_sin[:100], df_sin[103:]]))

        model = CUSUMDetectorModel(
            scan_window=scan_window,
            historical_window=historical_window,
            remove_seasonality=True,
            score_func=CusumScoreFunction.percentage_change,
        )
        self.score_tsd = model.fit_predict(
            data=self.tsd,
        ).scores

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            (
                "decomposing_seasonality_length_match",
                len,
                "score_tsd",
                len,
                "tsd",
                lambda x, y: x == y,
            ),
            (
                # the scores set to zero after about 7 days
                "decomposing_seasonality_score_after_seven_days",
                lambda x: x.value[-72:].sum(),
                "score_tsd",
                lambda x: x,
                "const_0",
                lambda x, y: x == y,
            ),
            (
                # the increase regression is detected and is on for about 7 days
                # statsmodels version difference will result in different STL results
                "decomposing_seasonality_regression_detected",
                # pyre-ignore[16]: bool has no attribute sumiiuivjtlilhgvhfijkngviirdvbggdrh
                lambda x: np.abs((x.value > 0.01).sum() - 168),
                "score_tsd",
                lambda x: x,
                "const_10",
                lambda x, y: x < y,
            ),
            (
                # make sure the time series time are the same
                "decomposing_seasonality_time_series_same",
                lambda x: x.time.values,
                "score_tsd",
                lambda x: x.time.values,
                "tsd",
                lambda x, y: (x == y).all(),
            ),
            (
                # make sure the time series name are the same
                "decomposing_seasonality_time_series_name_check",
                lambda x: x.value.name,
                "score_tsd",
                lambda x: x.value.name,
                "tsd",
                lambda x, y: x == y,
            ),
        ]
    )
    def test_score_tsd(
        self,
        name: str,
        func_1: Callable[[TimeSeriesData], float],
        attr1: str,
        func_2: Callable[[TimeSeriesData], float],
        attr2: str,
        func_sup: Callable[[float, float], bool],
    ) -> None:
        self.assertTrue(
            func_sup(func_1(attrgetter(attr1)(self)), func_2(attrgetter(attr2)(self)))
        )


class TestMissingDataRemoveSeasonalityCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        x = np.random.normal(0.5, 3, 998)
        time_val0 = list(
            pd.date_range(start="2018-02-03 14:59:59", freq="1800s", periods=1000)
        )
        time_val = time_val0[:300] + time_val0[301:605] + time_val0[606:]
        self.tsd = TimeSeriesData(
            pd.DataFrame({"time": time_val, "value": pd.Series(x)})
        )

        time_val01 = list(
            pd.date_range(start="2018-02-03 14:00:04", freq="1800s", periods=1000)
        )
        time_val1 = time_val01[:300] + time_val01[301:605] + time_val01[606:]
        self.tsd1 = TimeSeriesData(
            pd.DataFrame({"time": time_val1, "value": pd.Series(x)})
        )

    def test_interpolation(self) -> None:
        # base = 59 * 60 + 59 or = -1
        model = CUSUMDetectorModel(
            scan_window=360000,
            historical_window=360000,
            remove_seasonality=True,
        )
        score_tsd = model.fit_predict(data=self.tsd).scores
        self.assertEqual(len(score_tsd), len(self.tsd))
        self.assertTrue((score_tsd.time.values == self.tsd.time.values).all())

        # base = 4
        model1 = CUSUMDetectorModel(
            scan_window=360000,
            historical_window=360000,
            remove_seasonality=True,
        )
        score_tsd1 = model1.fit_predict(data=self.tsd1).scores
        self.assertEqual(len(score_tsd1), len(self.tsd1))
        self.assertTrue((score_tsd1.time.values == self.tsd1.time.values).all())


class TestRaiseCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        self.historical_window = 48 * 60 * 60  # in seconds
        self.scan_window = 24 * 60 * 60  # in seconds
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )

        self.tsd = TimeSeriesData(df_increase)

    def test_raise_direction(self) -> None:
        with self.assertRaisesRegex(ValueError, "direction can only be right or left"):
            model = CUSUMDetectorModel(
                scan_window=self.scan_window, historical_window=self.historical_window
            )
            model._time2idx(self.tsd, self.tsd.time.iloc[0], "")

    def test_raise_model_instantiation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "You must provide either serialized model or values for scan_window and historical_window.",
        ):
            _ = CUSUMDetectorModel()

    def test_raise_time_series_freq(self) -> None:
        with self.assertRaisesRegex(ValueError, IRREGULAR_GRANULARITY_ERROR):
            model = CUSUMDetectorModel(
                scan_window=self.scan_window, historical_window=self.historical_window
            )
            model.fit_predict(
                data=TimeSeriesData(
                    df=pd.DataFrame(
                        {
                            "value": [0] * 8,
                            "time": [
                                "2020-01-01",
                                "2020-01-02",
                                "2020-01-04",
                                "2020-01-05",
                                "2020-01-07",
                                "2020-01-08",
                                "2020-01-10",
                                "2020-01-11",
                            ],
                        }
                    )
                )
            )

    def test_raise_predict_not_implemented(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"predict is not implemented, call fit_predict\(\) instead"
        ):
            model = CUSUMDetectorModel(
                scan_window=self.scan_window, historical_window=self.historical_window
            )
            model.predict(data=self.tsd)


class TestDeltaReturnCUSUMDetectorModel(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        self.scan_window = 24 * 60 * 60  # in seconds
        self.historical_window = 3 * 24 * 60 * 60  # in seconds
        self.test_data_window = 16  # in hours
        self.regression_sum_score = 12  #
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        self.tsd = TimeSeriesData(df_increase)
        self.tsd_value_name = self.tsd.value.name
        data = self.tsd[-self.test_data_window :]

        self.model = CUSUMDetectorModel(
            scan_window=self.scan_window, historical_window=self.historical_window
        )

        self.delta = self.model.fit_predict(
            data=data, historical_data=self.tsd[: -self.test_data_window]
        ).anomaly_magnitude_ts

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("change_point_delta", len, "delta"),
        ]
    )
    def test_change_tsd(
        self,
        name: str,
        func_: Callable[[TimeSeriesData], float],
        attr1: str,
        # attr2: str,
    ) -> None:
        self.assertTrue(func_(attrgetter(attr1)(self)) > 0)


class TestCUSUMDetectorModelWindowsErrors(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        control_time = pd.date_range(start="2018-01-06", freq="D", periods=(100))
        control_val = np.random.normal(0, 5, 100)
        self.data_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))

    def test_errors(self) -> None:
        # case 1: scan_window < 2 * frequency_sec
        model = CUSUMDetectorModel(
            scan_window=1 * 24 * 60 * 60, historical_window=2 * 24 * 60 * 60
        )
        with self.assertRaises(ValueError):
            _ = model.fit_predict(data=self.data_ts)

        # case 2: historical_window < 2 * frequency_sec
        model = CUSUMDetectorModel(
            scan_window=2 * 24 * 60 * 60, historical_window=1 * 24 * 60 * 60
        )
        with self.assertRaises(ValueError):
            _ = model.fit_predict(data=self.data_ts)

        # case 3: step_window < frequency_sec
        model = CUSUMDetectorModel(
            scan_window=5 * 24 * 60 * 60,
            historical_window=5 * 24 * 60 * 60,
            step_window=24 * 60 * 60 // 2,
        )
        with self.assertRaises(ValueError):
            _ = model.fit_predict(data=self.data_ts)

        # case 4: step_window >= scan_window
        model = CUSUMDetectorModel(
            scan_window=5 * 24 * 60 * 60,
            historical_window=5 * 24 * 60 * 60,
            step_window=20 * 24 * 60 * 60,
        )
        with self.assertRaises(ValueError):
            _ = model.fit_predict(data=self.data_ts)


class TestCUSUMDetectorModelChangeDirection(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        control_time = pd.date_range(start="2018-01-06", freq="D", periods=(100))
        control_val = np.random.normal(0, 5, 100)
        self.data_ts = TimeSeriesData(time=control_time, value=pd.Series(control_val))

    def test_direction_list(self) -> None:
        model = CUSUMDetectorModel(
            scan_window=10 * 24 * 60 * 60,
            historical_window=10 * 24 * 60 * 60,
            threshold=0.01,
            delta_std_ratio=1.0,
            serialized_model=None,
            change_directions=["increase"],
            score_func=CusumScoreFunction.percentage_change,
        )
        anom = model.fit_predict(data=self.data_ts)
        self.assertEqual(len(anom.scores), len(self.data_ts))

    def test_direction_str(self) -> None:
        # case 1: scan_window < 2 * frequency_sec
        model = CUSUMDetectorModel(
            scan_window=10 * 24 * 60 * 60,
            historical_window=10 * 24 * 60 * 60,
            threshold=0.01,
            delta_std_ratio=1.0,
            serialized_model=None,
            change_directions="increase",
            score_func=CusumScoreFunction.percentage_change,
        )
        self.assertEqual(model.change_directions, ["increase"])
        anom = model.fit_predict(data=self.data_ts)
        self.assertEqual(len(anom.scores), len(self.data_ts))


class TestCUSUMDetectorModelIrregularGranularityError(TestCase):
    def setUp(self) -> None:
        np.random.seed(100)
        ts_time = list(
            pd.date_range(start="2018-01-06 00:00:00", freq="60s", periods=(100))
        )
        ts_time = ts_time[:60] + ts_time[61:80] + ts_time[81:]
        ts_time[82] = pd.to_datetime("2018-01-06 01:24:02")
        ts_time[85] = pd.to_datetime("2018-01-06 01:27:22")
        ts_val = np.random.normal(0, 5, 98)
        self.data_ts = TimeSeriesData(time=pd.Series(ts_time), value=pd.Series(ts_val))

    def test_irregular_granularity_error(self) -> None:
        model = CUSUMDetectorModel(
            scan_window=10 * 60,
            historical_window=10 * 60,
            threshold=0.01,
            delta_std_ratio=1.0,
            serialized_model=None,
            change_directions=["increase"],
            remove_seasonality=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            IRREGULAR_GRANULARITY_ERROR,
        ):
            _ = model.fit_predict(data=self.data_ts)
