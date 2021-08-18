# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.cusum_model import (
    CUSUMDetectorModel,
    CusumScoreFunction,
)

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


class TestCUSUMDetectorModel(TestCase):
    def test_increase(self) -> None:
        np.random.seed(100)
        scan_window = 24 * 60 * 60  # in seconds
        historical_window = 3 * 24 * 60 * 60  # in seconds
        test_data_window = 16  # in hours
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)

        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(
            data=tsd[-test_data_window:], historical_data=tsd[:-test_data_window]
        ).scores

        self.assertEqual(len(score_tsd), test_data_window)
        # make sure the time series name are the same
        self.assertTrue(score_tsd.value.name == tsd.value.name)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)
        score_tsd = model._predict(
            data=tsd[-test_data_window:],
            score_func=CusumScoreFunction.percentage_change.value,
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.z_score.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value > 0).sum(), 12)

        serialized_model = model.serialize()
        self.assertIsInstance(serialized_model, bytes)
        model_new = CUSUMDetectorModel(serialized_model)
        self.assertEqual(model_new, model)
        self.assertNotEqual(
            model,
            CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            ),
        )

    def test_decrease(self) -> None:
        np.random.seed(100)
        scan_window = 24 * 60 * 60  # in seconds
        historical_window = 3 * 24 * 60 * 60  # in seconds
        test_data_window = 6  # in hours
        df_decrease = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(2, 0.2, 156), np.random.normal(1, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_decrease)

        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(
            data=tsd[-test_data_window:], historical_data=tsd[:-test_data_window]
        ).scores
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.change.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)
        score_tsd = model._predict(
            data=tsd[-test_data_window:],
            score_func=CusumScoreFunction.percentage_change.value,
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)
        score_tsd = model._predict(
            data=tsd[-test_data_window:], score_func=CusumScoreFunction.z_score.value
        )
        self.assertEqual(len(score_tsd), test_data_window)
        # the regression is detected
        self.assertEqual((score_tsd.value < 0).sum(), test_data_window)

    def test_adhoc(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 11 * 60 * 60 + 50  # in seconds
        n = 168
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
        tsd = TimeSeriesData(df_increase)
        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(data=tsd).scores
        self.assertEqual(len(score_tsd), len(tsd))
        # the regression is went away
        self.assertEqual(score_tsd.value[-6:].sum(), 0)
        # the increase regression is detected
        self.assertEqual((score_tsd.value > 0.5).sum(), 24)
        # the decrease regression is detected
        self.assertEqual((score_tsd.value < -0.45).sum(), 12)

        # test not enough data
        model = CUSUMDetectorModel(
            scan_window=scan_window, historical_window=historical_window
        )
        score_tsd = model.fit_predict(data=tsd[-4:], historical_data=tsd[-8:-4]).scores

        self.assertEqual(len(score_tsd), len(tsd[-4:]))
        self.assertEqual(score_tsd.value.sum(), 0)

        model = CUSUMDetectorModel(scan_window=scan_window, historical_window=3600)
        score_tsd = model.fit_predict(data=tsd[-8:]).scores

        self.assertEqual(len(score_tsd), len(tsd[-8:]))
        self.assertEqual(score_tsd.value.sum(), 0)

    def test_missing_data(self) -> None:
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
        tsd = TimeSeriesData(df)
        # We also assume a bad input here
        model = CUSUMDetectorModel(
            scan_window=24 * 3600,
            historical_window=2 * 24 * 3600,
        )
        score_tsd = model.fit_predict(
            data=tsd,
        ).scores

        self.assertEqual(len(score_tsd), len(tsd))
        self.assertTrue((score_tsd.time.values == tsd.time.values).all())

    def test_streaming(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        n = 72
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 60), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
            }
        )
        tsd = TimeSeriesData(df_increase)
        # Priming the model
        model = CUSUMDetectorModel(
            historical_window=historical_window, scan_window=scan_window
        )
        model.fit(data=tsd[:48])
        pre_serialized_model = model.serialize()

        anomaly_score = TimeSeriesData(
            time=pd.Series(), value=pd.Series([], name="ts_value")
        )
        # feeding 1 new data point a time
        for i in range(48, n):
            model = CUSUMDetectorModel(
                serialized_model=pre_serialized_model,
                historical_window=historical_window,
                scan_window=scan_window,
            )
            anomaly_score.extend(
                model.fit_predict(
                    data=tsd[i : i + 1], historical_data=tsd[i - 48 : i]
                ).scores,
                validate=False,
            )
            pre_serialized_model = model.serialize()
        anomaly_score.validate_data(validate_frequency=True, validate_dimension=False)
        self.assertEqual(len(anomaly_score), n - 48)
        self.assertTrue(8 <= (anomaly_score.value > 0).sum() <= 12)

    def test_decomposing_seasonality(self) -> None:
        np.random.seed(100)
        historical_window = 10 * 24 * 60 * 60  # in seconds
        scan_window = 12 * 60 * 60  # in seconds
        n = 480
        periodicity = 24

        df_sin = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=n, freq="H"),
                "ts_value": np.concatenate([20 * np.ones(n // 2), 21 * np.ones(n // 2)])
                + 4 * np.sin(2 * np.pi / periodicity * np.arange(0, n)),
            }
        )

        # removing a few data points to test the missing value handling as well
        tsd = TimeSeriesData(pd.concat([df_sin[:100], df_sin[103:]]))

        model = CUSUMDetectorModel(
            scan_window=scan_window,
            historical_window=historical_window,
            remove_seasonality=True,
            score_func=CusumScoreFunction.percentage_change,
        )
        score_tsd = model.fit_predict(
            data=tsd,
        ).scores

        self.assertEqual(len(score_tsd), len(tsd))
        # the scores set to zero after about 7 days
        self.assertEqual(score_tsd.value[-72:].sum(), 0)
        # the increase regression is detected and is on for about 7 days
        # statsmodels version difference will result in different STL results
        self.assertLess(np.abs((score_tsd.value > 0.01).sum() - 168), 10)
        # make sure the time series time are the same
        self.assertTrue((score_tsd.time.values == tsd.time.values).all())
        # make sure the time series name are the same
        self.assertTrue(score_tsd.value.name == tsd.value.name)

    def test_raise(self) -> None:
        np.random.seed(100)
        historical_window = 48 * 60 * 60  # in seconds
        scan_window = 24 * 60 * 60  # in seconds
        df_increase = pd.DataFrame(
            {
                "ts_value": np.concatenate(
                    [np.random.normal(1, 0.2, 156), np.random.normal(1.5, 0.2, 12)]
                ),
                "time": pd.date_range("2020-01-01", periods=168, freq="H"),
            }
        )

        tsd = TimeSeriesData(df_increase)
        with self.assertRaisesRegex(
            ValueError,
            "Step window should smaller than scan window to ensure we have overlap for scan windows.",
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window,
                step_window=scan_window * 2,
                historical_window=historical_window,
            )

        with self.assertRaisesRegex(ValueError, "direction can only be right or left"):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            )
            model._time2idx(tsd, tsd.time.iloc[0], "")

        with self.assertRaisesRegex(
            ValueError,
            "You must either provide serialized model or values for scan_window and historical_window.",
        ):
            model = CUSUMDetectorModel()

        with self.assertRaisesRegex(
            ValueError, "Not able to infer freqency of the time series"
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
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

        with self.assertRaisesRegex(
            ValueError, r"predict is not implemented, call fit_predict\(\) instead"
        ):
            model = CUSUMDetectorModel(
                scan_window=scan_window, historical_window=historical_window
            )
            model.predict(data=tsd)
