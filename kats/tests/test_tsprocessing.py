#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from data_ai.event_correlation.synthetic_correlation_generator import TimeSeriesGen
from kats.consts import TimeSeriesData
from kats.detectors.residual_translation import KDEResidualTranslator
from kats.utils.decomposition import TimeSeriesDecomposition
from kats.utils.simulator import Simulator
from scipy.stats import ks_2samp  # @manual

data = pd.read_csv("kats/kats/data/air_passengers.csv")
data.columns = ["time", "y"]
ts_data = TimeSeriesData(data)
# generate multiple series
data_2 = data.copy()
data_2["y_2"] = data_2["y"]
ts_data_2 = TimeSeriesData(data_2)
# generate TimeSeriesData without "time" as time column name
data_nonstandard_name = data.copy()
data_nonstandard_name.columns = ["ds", "y"]
ts_data_nonstandard_name = TimeSeriesData(df=data_nonstandard_name, time_col_name="ds")

daily_data = pd.read_csv("kats/kats/data/peyton_manning.csv")
daily_data.columns = ["time", "y"]
ts_data_daily = TimeSeriesData(daily_data)

DATA_multi = pd.read_csv("kats/kats/data/cdn_working_set.csv")
TSData_multi = TimeSeriesData(DATA_multi)


class DecompositionTest(TestCase):
    def test_asserts(self):
        with self.assertRaises(ValueError):
            timeseries = TimeSeriesData(DATA_multi)
            TimeSeriesDecomposition(timeseries, "additive")

    def test_defaults(self):
        m1 = TimeSeriesDecomposition(ts_data, "additive")
        output1 = m1.decomposer()

        m2 = TimeSeriesDecomposition(ts_data, "logarithmic")
        output2 = m2.decomposer()

        self.assertEqual(output1["trend"].value.all(), output2["trend"].value.all())
        self.assertEqual(
            output1["seasonal"].value.all(), output2["seasonal"].value.all()
        )
        self.assertEqual(output1["rem"].value.all(), output2["rem"].value.all())

        m3 = TimeSeriesDecomposition(ts_data, "additive", "STL2")
        output3 = m3.decomposer()

        self.assertEqual(output1["trend"].value.all(), output3["trend"].value.all())
        self.assertEqual(
            output1["seasonal"].value.all(), output3["seasonal"].value.all()
        )
        self.assertEqual(output1["rem"].value.all(), output3["rem"].value.all())

    def test_nonstandard_time_col_name(self):
        m = TimeSeriesDecomposition(ts_data_nonstandard_name, "multiplicative")
        m.decomposer()
        self.assertEqual(
            m.results["trend"].time_col_name, ts_data_nonstandard_name.time_col_name
        )
        self.assertEqual(
            m.results["seasonal"].time_col_name, ts_data_nonstandard_name.time_col_name
        )
        self.assertEqual(
            m.results["rem"].time_col_name, ts_data_nonstandard_name.time_col_name
        )

    def test_decomposition_additive(self):
        m = TimeSeriesDecomposition(ts_data, "additive")
        output = m.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data.time), "y": ts_data.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    + output["seasonal"].value
                    + output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )

        self.assertAlmostEqual(
            np.mean((out["y_actuals"] - out["y_decomposed"]) ** 2), 0, 5
        )

        m_seasonal = TimeSeriesDecomposition(ts_data, "additive", "seasonal_decompose")
        output = m_seasonal.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data.time), "y": ts_data.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    + output["seasonal"].value
                    + output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )

        self.assertAlmostEqual(
            np.mean((out["y_actuals"] - out["y_decomposed"]) ** 2), 0, 5
        )

        m2 = TimeSeriesDecomposition(ts_data_daily, "additive")
        output = m2.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data_daily.time), "y": ts_data_daily.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    + output["seasonal"].value
                    + output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )
        self.assertAlmostEqual(
            np.mean((out2["y_actuals"] - out2["y_decomposed"]) ** 2), 0, 5
        )

        m2_seasonal = TimeSeriesDecomposition(
            ts_data_daily, "additive", "seasonal_decompose"
        )
        output = m2_seasonal.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data_daily.time), "y": ts_data_daily.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    + output["seasonal"].value
                    + output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )
        self.assertAlmostEqual(
            np.mean((out2["y_actuals"] - out2["y_decomposed"]) ** 2), 0, 5
        )

    def test_decomposition_multiplicative(self):
        m = TimeSeriesDecomposition(ts_data, "multiplicative")
        output = m.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data.time), "y": ts_data.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    * output["seasonal"].value
                    * output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )

        self.assertAlmostEqual(
            np.mean((out["y_actuals"] - out["y_decomposed"]) ** 2), 0, 5
        )

        m_seas = TimeSeriesDecomposition(
            ts_data, "multiplicative", "seasonal_decompose"
        )
        output = m_seas.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data.time), "y": ts_data.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    * output["seasonal"].value
                    * output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )

        self.assertAlmostEqual(
            np.mean((out["y_actuals"] - out["y_decomposed"]) ** 2), 0, 5
        )
        m2 = TimeSeriesDecomposition(ts_data_daily, "multiplicative")
        output = m2.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data_daily.time), "y": ts_data_daily.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    * output["seasonal"].value
                    * output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )
        self.assertAlmostEqual(
            np.mean((out2["y_actuals"] - out2["y_decomposed"]) ** 2), 0, 5
        )

        m2_seas = TimeSeriesDecomposition(
            ts_data_daily, "multiplicative", "seasonal_decompose"
        )
        output = m2_seas.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(ts_data_daily.time), "y": ts_data_daily.value}
            ),
            pd.DataFrame.from_dict(
                {
                    "time": output["trend"].time,
                    "y": output["trend"].value
                    * output["seasonal"].value
                    * output["rem"].value,
                }
            ),
            how="inner",
            on="time",
            suffixes=("_actuals", "_decomposed"),
        )
        self.assertAlmostEqual(
            np.mean((out2["y_actuals"] - out2["y_decomposed"]) ** 2), 0, 5
        )

    def test_plot(self):
        m = TimeSeriesDecomposition(ts_data, "multiplicative")
        m.decomposer()

        m.plot()

    def test_multiplicative_assert(self):
        data_new = data.copy()
        data_new["y"] = -1.0 * data_new["y"]
        ts_data_new = TimeSeriesData(data_new)
        print(ts_data_new)
        with self.assertLogs(level="ERROR"):
            m = TimeSeriesDecomposition(ts_data_new, "multiplicative")
            m.decomposer()

    def test_new_freq(self):
        def process_time(z):
            x0, x1 = z.split(" ")
            time = (
                "-".join(y.rjust(2, "0") for y in x0.split("/"))
                + "20 "
                + ":".join(y.rjust(2, "0") for y in x1.split(":"))
                + ":00"
            )

            return datetime.strptime(time, "%m-%d-%Y %H:%M:%S")

        df_15_min = DATA_multi

        df_15_min["ts"] = df_15_min["time"].apply(process_time)

        df_15_min_dict = {}

        for i in range(0, 4):

            df_15_min_temp = df_15_min.copy()
            df_15_min_temp["ts"] = [
                x + timedelta(minutes=15 * i) for x in df_15_min_temp["ts"]
            ]
            df_15_min_dict[i] = df_15_min_temp

        df_15_min_ts = pd.concat(df_15_min_dict.values()).sort_values(by="ts")[
            ["ts", "V1"]
        ]

        df_15_min_ts.columns = ["time", "y"]

        df_ts = TimeSeriesData(df_15_min_ts)

        m = TimeSeriesDecomposition(df_ts, "additive", method="STL")
        m.decomposer()
        m2 = TimeSeriesDecomposition(df_ts, "additive", method="seasonal_decompose")
        m2.decomposer()


# class KDEResidualTranslatorTest(TestCase):
#     def setUp(self) -> None:
#         self._y = ts_data
#         yhat = pd.DataFrame(
#             {"value": self._y.value.rolling(7).mean().shift(1), "time": self._y.time}
#         )
#         self._yhat = TimeSeriesData(yhat)
#         self._residual = self._y - self._yhat

#     def test_setup(self) -> None:
#         self.assertEquals(self._yhat.value.isnull().sum(), 7)

#     def test_illegal_truncated_fracs(self) -> None:
#         with self.assertRaises(ValueError):
#             KDEResidualTranslator(-0.1, 0.9)
#         with self.assertRaises(ValueError):
#             KDEResidualTranslator(1.1, 2.0)
#         with self.assertRaises(ValueError):
#             KDEResidualTranslator(0.1, -0.9)
#         with self.assertRaises(ValueError):
#             KDEResidualTranslator(0.1, 1.9)
#         with self.assertRaises(ValueError):
#             KDEResidualTranslator(0.9, 0.8)

#     def test_y_yhat(self) -> None:
#         trn = KDEResidualTranslator()
#         trn = trn.fit(y=self._y, yhat=self._yhat)
#         self._test_residual_trn(trn)

#     def _test_residual(self) -> None:
#         trn = KDEResidualTranslator()
#         for name in self._series_names:
#             dataset = self._get_dataset_for_name(name)[["y", "yhat"]]
#             dataset["residual"] = dataset.yhat - dataset.y
#             dataset.drop(["y", "yhat"], axis=1, inplace=True)
#             trn = trn.fit(dataset)
#             self._test_residual_trn(trn)

#     def _test_residual_trn(self, trn: KDEResidualTranslator) -> None:
#         np.testing.assert_allclose(
#             np.exp(trn.predict_log_proba(residual=self._residual).value),
#             trn.predict_proba(residual=self._residual).value,
#         )
#         proba = trn.predict_proba(residual=self._residual)
#         self.assertTrue(np.all((proba.value >= 0) & (proba.value <= 1)))
#         ks = ks_2samp(
#             trn.kde_.sample(len(self._residual)).flatten(), self._residual.value
#         )
#         self.assertTrue(ks.statistic < 0.1 or ks.pvalue >= 0.2)


class SimulatorTest(TestCase):
    def test_arima_sim(self):
        sim = Simulator(n=10, freq="MS", start=pd.to_datetime("2011-01-01 00:00:00"))

        np.random.seed(100)
        ts = sim.arima_sim(ar=[0.1, 0.05], ma=[0.04, 0.1], d=1)

        expected_value = pd.Series(
            [
                0.797342,
                1.494317,
                1.608064,
                1.186103,
                2.147635,
                1.772615,
                0.750320,
                2.159774,
                3.744138,
                3.944730,
            ]
        )
        self.assertEqual(True, (ts.value - expected_value).all())
        self.assertEqual(len(ts.time), 10)

    def test_stl_sim_additive(self):
        # Create a STL-based simulated object
        sim = Simulator(n=100, freq="1D", start=pd.to_datetime("2011-01-01"))
        np.random.seed(614)
        sim.add_trend(magnitude=10)
        sim.add_seasonality(5, period=timedelta(days=7))
        sim.add_noise(magnitude=2)
        sim_ts = sim.stl_sim()
        # Compare the obtained simulated time series to
        # the original TimeSeriesGen simulation from event correlation
        generator1 = TimeSeriesGen(start="2011-01-01", n_data_points=100, interval="1D")
        generator1.add_trend(magnitude=10)
        np.random.seed(614)
        generator1.add_seasonality(5, period=timedelta(days=7))
        generator1.add_noise(magnitude=2)
        gen_ts = generator1.gen_series()
        gen_ts_series = TimeSeriesData(
            gen_ts.reset_index().rename(columns={"index": "time", "0": "value"})
        )
        self.assertEqual(True, (gen_ts_series.value == sim_ts.value).all())
        self.assertEqual(True, (gen_ts_series.time == sim_ts.time).all())

    def test_stl_sim_multiplicative(self):
        # Create a STL-based simulated object
        sim = Simulator(n=100, freq="1D", start=pd.to_datetime("2011-01-01"))
        np.random.seed(614)
        sim.add_trend(magnitude=5, multiply=True)
        sim.add_seasonality(10, period=timedelta(days=14))
        sim.add_noise(magnitude=1, multiply=True)
        sim_ts = sim.stl_sim()
        # Compare the obtained simulated time series to
        # the original TimeSeriesGen simulation from event correlation
        generator2 = TimeSeriesGen(start="2011-01-01", n_data_points=100, interval="1D")
        generator2.add_trend(magnitude=5, multiply=True)
        np.random.seed(614)
        generator2.add_seasonality(10, period=timedelta(days=14))
        generator2.add_noise(magnitude=1, multiply=True)
        gen_ts = generator2.gen_series()
        gen_ts_series = TimeSeriesData(
            gen_ts.reset_index().rename(columns={"index": "time", "0": "value"})
        )
        self.assertEqual(True, (gen_ts_series.value == sim_ts.value).all())
        self.assertEqual(True, (gen_ts_series.time == sim_ts.time).all())

    def test_level_shift(self):
        sim = Simulator(n=450, start="2018-01-01")
        ts = sim.level_shift_sim()

        self.assertEqual(len(ts), 450)

        sim2 = Simulator(n=450, start="2018-01-01")
        ts2 = sim2.level_shift_sim(
            cp_arr=[100, 200],
            level_arr=[3, 20, 2],
            noise=3,
            seasonal_period=7,
            seasonal_magnitude=3,
            anomaly_arr = [50, 150, 250],
            z_score_arr = [10, -10, 20],
        )

        self.assertEqual(len(ts2), 450)

    def test_level_shift_mvn_indep(self):

        sim = Simulator(n=450, start="2018-01-01")
        ts = sim.level_shift_multivariate_indep_sim()
        self.assertEqual(len(ts), 450)

        ts_df = ts.to_dataframe()
        self.assertEqual(ts_df.shape[1], 4)  # time + n_dim

        sim2 = Simulator(n=450, start="2018-01-01")
        ts2 = sim2.level_shift_multivariate_indep_sim(
            cp_arr=[100, 200],
            level_arr=[3, 20, 2],
            noise=3,
            seasonal_period=7,
            seasonal_magnitude=3,
        )

        self.assertEqual(len(ts2), 450)

        ts2_df = ts2.to_dataframe()
        self.assertEqual(ts2_df.shape[1], 4)  # time + n_dim

    def test_trend_shift(self):
        sim = Simulator(n=450, start="2018-01-01")
        ts = sim.trend_shift_sim()
        self.assertEqual(len(ts), 450)

        sim2 = Simulator(n=450, start="2018-01-01")
        ts2 = sim2.trend_shift_sim(
            cp_arr=[100, 200],
            trend_arr=[3, 20, 2],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=3,
            anomaly_arr = [50, 150, 250],
            z_score_arr = [10, -10, 20],
        )

        self.assertEqual(len(ts2), 450)


if __name__ == "__main__":
    unittest.main()
