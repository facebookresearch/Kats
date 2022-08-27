# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.detectors.residual_translation import KDEResidualTranslator
from kats.utils.decomposition import TimeSeriesDecomposition
from kats.utils.simulator import Simulator
from scipy.stats import ks_2samp
from statsmodels.tsa.seasonal import seasonal_decompose, STL


class DecompositionTest(TestCase):
    def setUp(self) -> None:
        data = load_air_passengers(return_ts=False)
        self.ts_data = TimeSeriesData(data)

        data_nonstandard_name = data.copy()
        data_nonstandard_name.columns = ["ds", "y"]
        self.ts_data_nonstandard_name = TimeSeriesData(
            df=data_nonstandard_name, time_col_name="ds"
        )

        daily_data = load_data("peyton_manning.csv")
        daily_data.columns = ["time", "y"]
        self.ts_data_daily = TimeSeriesData(daily_data)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_asserts(self) -> None:
        with self.assertRaises(ValueError):
            TimeSeriesDecomposition(self.TSData_multi, "additive")

    def test_defaults(self) -> None:
        m1 = TimeSeriesDecomposition(self.ts_data, "additive")
        output1 = m1.decomposer()

        m2 = TimeSeriesDecomposition(self.ts_data, "logarithmic")
        output2 = m2.decomposer()

        self.assertEqual(output1["trend"].value.all(), output2["trend"].value.all())
        self.assertEqual(
            output1["seasonal"].value.all(), output2["seasonal"].value.all()
        )
        self.assertEqual(output1["rem"].value.all(), output2["rem"].value.all())

        m3 = TimeSeriesDecomposition(self.ts_data, "additive", "STL2")
        output3 = m3.decomposer()

        self.assertEqual(output1["trend"].value.all(), output3["trend"].value.all())
        self.assertEqual(
            output1["seasonal"].value.all(), output3["seasonal"].value.all()
        )
        self.assertEqual(output1["rem"].value.all(), output3["rem"].value.all())

    def test_nonstandard_time_col_name(self) -> None:
        m = TimeSeriesDecomposition(self.ts_data_nonstandard_name, "multiplicative")
        m.decomposer()
        self.assertEqual(
            # pyre-fixme[16]: `TimeSeriesDecomposition` has no attribute `results`.
            m.results["trend"].time_col_name,
            self.ts_data_nonstandard_name.time_col_name,
        )
        self.assertEqual(
            m.results["seasonal"].time_col_name,
            self.ts_data_nonstandard_name.time_col_name,
        )
        self.assertEqual(
            m.results["rem"].time_col_name, self.ts_data_nonstandard_name.time_col_name
        )

    def test_decomposition_additive(self) -> None:
        m = TimeSeriesDecomposition(self.ts_data, "additive")
        output = m.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(self.ts_data.time), "y": self.ts_data.value}
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

        m_seasonal = TimeSeriesDecomposition(
            self.ts_data, "additive", "seasonal_decompose"
        )
        output = m_seasonal.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(self.ts_data.time), "y": self.ts_data.value}
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

        m2 = TimeSeriesDecomposition(self.ts_data_daily, "additive")
        output = m2.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {
                    "time": pd.DatetimeIndex(self.ts_data_daily.time),
                    "y": self.ts_data_daily.value,
                }
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
            self.ts_data_daily, "additive", "seasonal_decompose"
        )
        output = m2_seasonal.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {
                    "time": pd.DatetimeIndex(self.ts_data_daily.time),
                    "y": self.ts_data_daily.value,
                }
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

    def test_decomposition_multiplicative(self) -> None:
        m = TimeSeriesDecomposition(self.ts_data, "multiplicative")
        output = m.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(self.ts_data.time), "y": self.ts_data.value}
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
            self.ts_data, "multiplicative", "seasonal_decompose"
        )
        output = m_seas.decomposer()

        out = pd.merge(
            pd.DataFrame.from_dict(
                {"time": pd.DatetimeIndex(self.ts_data.time), "y": self.ts_data.value}
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
        m2 = TimeSeriesDecomposition(self.ts_data_daily, "multiplicative")
        output = m2.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {
                    "time": pd.DatetimeIndex(self.ts_data_daily.time),
                    "y": self.ts_data_daily.value,
                }
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
            self.ts_data_daily, "multiplicative", "seasonal_decompose"
        )
        output = m2_seas.decomposer()

        out2 = pd.merge(
            pd.DataFrame.from_dict(
                {
                    "time": pd.DatetimeIndex(self.ts_data_daily.time),
                    "y": self.ts_data_daily.value,
                }
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

    def test_plot(self) -> None:
        m = TimeSeriesDecomposition(self.ts_data, "multiplicative")
        m.decomposer()

        m.plot()

    def test_multiplicative_assert(self) -> None:
        data_new = self.ts_data.to_dataframe().copy()
        data_new["y"] = -1.0 * data_new["y"]
        ts_data_new = TimeSeriesData(data_new)
        print(ts_data_new)
        with self.assertLogs(level="ERROR"):
            m = TimeSeriesDecomposition(ts_data_new, "multiplicative")
            m.decomposer()

    def test_new_freq(self) -> None:
        DATA_multi = self.TSData_multi.to_dataframe()
        df_15_min = DATA_multi[["time", "1"]]
        df_15_min["time"] = list(
            pd.date_range(end="2020-02-01", periods=df_15_min.shape[0], freq="25T")
        )
        df_15_min["time"] = df_15_min["time"].astype("str")
        df_15_min.columns = ["time", "y"]

        df_ts = TimeSeriesData(df_15_min)

        m = TimeSeriesDecomposition(df_ts, "additive", method="STL")
        m.decomposer()
        m2 = TimeSeriesDecomposition(df_ts, "additive", method="seasonal_decompose")
        m2.decomposer()

    def test_10_minutes_level_dense_data(self) -> None:
        sim = Simulator(
            n=2 * 144, freq="10T", start=pd.to_datetime("2021-01-01")
        )  # 2 days of data
        sim.add_trend(magnitude=1)
        sim.add_seasonality(magnitude=50, period=timedelta(days=1))
        sim.add_noise(magnitude=10)
        # dates are dense, there is no gaps between dates
        dense_dates_ts = sim.stl_sim()
        dense_dates_df = dense_dates_ts.to_dataframe()

        m = TimeSeriesDecomposition(
            dense_dates_ts,
            "additive",
            method="STL",
        )
        decomp = m.decomposer()
        m2 = TimeSeriesDecomposition(
            dense_dates_ts,
            "additive",
            method="STL",
            period=144,
            robust=True,
        )
        decomp2 = m2.decomposer()
        m3 = TimeSeriesDecomposition(
            dense_dates_ts,
            "additive",
            method="seasonal_decompose",
        )
        decomp3 = m3.decomposer()

        # check that interpolate doesn't add new data points
        self.assertEqual(
            len(decomp["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )
        self.assertEqual(
            len(decomp2["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )
        self.assertEqual(
            len(decomp3["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )

        # check if decomposition does what it intends to do
        stl = STL(dense_dates_df.reset_index().value, period=2)
        true_results = stl.fit()
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp["seasonal"].to_dataframe()["season"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp["trend"].to_dataframe()["trend"].values
            ).all()
        )
        stl = STL(dense_dates_df.reset_index().value, period=144, robust=True)
        true_results = stl.fit()
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp2["seasonal"].to_dataframe()["season"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp2["trend"].to_dataframe()["trend"].values
            ).all()
        )
        true_results = seasonal_decompose(
            dense_dates_df.reset_index().value, period=2, model="additive"
        )
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp3["seasonal"].to_dataframe()["seasonal"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp3["trend"].to_dataframe()["trend"].values
            )[
                1:-1
            ].all()  # at the beginning and end NaNs
        )

    def test_10_minutes_level_sparse_data(self) -> None:
        # create data
        sim = Simulator(
            n=2 * 144, freq="10T", start=pd.to_datetime("2021-01-01")
        )  # 2 days of data
        sim.add_trend(magnitude=1)
        sim.add_seasonality(magnitude=50, period=timedelta(days=1))
        sim.add_noise(magnitude=10)
        dense_dates_ts = sim.stl_sim()
        # dates are sparse, there are some gaps between dates
        sparse_dates_df = dense_dates_ts.to_dataframe().copy()
        sparse_dates_df["time"] = sparse_dates_df["time"].map(
            lambda x: x + pd.Timedelta(365, "D")
            if (x >= pd.Timestamp(2021, 1, 2)) & (x < pd.Timestamp(2021, 1, 3))
            else x
        )
        sparse_dates_ts = TimeSeriesData(sparse_dates_df)

        # do decomposition
        m = TimeSeriesDecomposition(
            sparse_dates_ts,
            "additive",
            method="STL",
        )
        decomp = m.decomposer()
        m2 = TimeSeriesDecomposition(
            sparse_dates_ts,
            "additive",
            method="STL",
            period=144,
            robust=True,
        )
        decomp2 = m2.decomposer()
        m3 = TimeSeriesDecomposition(
            sparse_dates_ts,
            "additive",
            method="seasonal_decompose",
        )
        decomp3 = m3.decomposer()

        # check that interpolate doesn't add new data points
        self.assertEqual(
            len(decomp["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )
        self.assertEqual(
            len(decomp2["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )
        self.assertEqual(
            len(decomp3["trend"].to_dataframe()), len(dense_dates_ts.to_dataframe())
        )

        # check if decomposition does what it intends to do
        stl = STL(sparse_dates_df.reset_index().value, period=2)
        true_results = stl.fit()
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp["seasonal"].to_dataframe()["season"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp["trend"].to_dataframe()["trend"].values
            ).all()
        )
        stl = STL(sparse_dates_df.reset_index().value, period=144, robust=True)
        true_results = stl.fit()
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp2["seasonal"].to_dataframe()["season"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp2["trend"].to_dataframe()["trend"].values
            ).all()
        )
        true_results = seasonal_decompose(
            sparse_dates_df.reset_index().value, period=2, model="additive"
        )
        self.assertTrue(
            (
                true_results.seasonal.values
                == decomp3["seasonal"].to_dataframe()["seasonal"].values
            ).all()
        )
        self.assertTrue(
            (
                true_results.trend.values
                == decomp3["trend"].to_dataframe()["trend"].values
            )[
                1:-1
            ].all()  # at the beginning and end NaNs
        )


class KDEResidualTranslatorTest(TestCase):
    def setUp(self) -> None:
        data = load_air_passengers(return_ts=False)
        data.columns = ["time", "value"]
        self._y = TimeSeriesData(data)
        yhat = pd.DataFrame(
            {"value": self._y.value.rolling(7).mean().shift(1), "time": self._y.time}
        )
        self._yhat = TimeSeriesData(yhat)
        self._residual = self._y - self._yhat

    def test_setup(self) -> None:
        self.assertEquals(self._yhat.value.isnull().sum(), 7)

    def test_illegal_truncated_fracs(self) -> None:
        with self.assertRaises(ValueError):
            KDEResidualTranslator(-0.1, 0.9)
        with self.assertRaises(ValueError):
            KDEResidualTranslator(1.1, 2.0)
        with self.assertRaises(ValueError):
            KDEResidualTranslator(0.1, -0.9)
        with self.assertRaises(ValueError):
            KDEResidualTranslator(0.1, 1.9)
        with self.assertRaises(ValueError):
            KDEResidualTranslator(0.9, 0.8)

    def test_y_yhat(self) -> None:
        trn = KDEResidualTranslator()
        trn = trn.fit(y=self._y, yhat=self._yhat)
        self._test_residual_trn(trn)

    # def _test_residual(self) -> None:
    #     trn = KDEResidualTranslator()
    #     for name in self._series_names:
    #         dataset = self._get_dataset_for_name(name)[["y", "yhat"]]
    #         dataset["residual"] = dataset.yhat - dataset.y
    #         dataset.drop(["y", "yhat"], axis=1, inplace=True)
    #         trn = trn.fit(dataset)
    #         self._test_residual_trn(trn)

    def _test_residual_trn(self, trn: KDEResidualTranslator) -> None:
        np.testing.assert_allclose(
            np.exp(trn.predict_log_proba(residual=self._residual).value),
            trn.predict_proba(residual=self._residual).value,
        )
        proba = trn.predict_proba(residual=self._residual)
        self.assertTrue(np.all(proba.value >= 0) and np.all(proba.value <= 1))
        ks = ks_2samp(
            # pyre-fixme [16]: Optional type has no attribute `sample`
            trn.kde_.sample(len(self._residual)).flatten(),
            self._residual.value,
        )
        self.assertTrue(ks.statistic < 0.1 or ks.pvalue >= 0.2)


class SimulatorTest(TestCase):
    def test_arima_sim(self) -> None:
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

    def test_stl_sim_additive(self) -> None:
        # Create a STL-based simulated object
        sim = Simulator(n=100, freq="1D", start=pd.to_datetime("2011-01-01"))
        np.random.seed(614)
        sim.add_trend(magnitude=10)
        sim.add_seasonality(5, period=timedelta(days=7))
        sim.add_noise(magnitude=2)
        sim_ts = sim.stl_sim()
        # Compare the obtained simulated time series to
        # the original simulated data
        generator1 = Simulator(n=100, freq="D", start="2011-01-01")
        generator1.add_trend(magnitude=10)
        np.random.seed(614)
        generator1.add_seasonality(magnitude=5, period=timedelta(days=7))
        generator1.add_noise(magnitude=2)
        gen_ts_series = generator1.stl_sim()
        # pyre-fixme[16]: `bool` has no attribute `all`.
        self.assertEqual(True, (gen_ts_series.value == sim_ts.value).all())
        self.assertEqual(True, (gen_ts_series.time == sim_ts.time).all())

    def test_stl_sim_multiplicative(self) -> None:
        # Create a STL-based simulated object
        sim = Simulator(n=100, freq="1D", start=pd.to_datetime("2011-01-01"))
        np.random.seed(614)
        sim.add_trend(magnitude=5, multiply=True)
        sim.add_seasonality(10, period=timedelta(days=14))
        sim.add_noise(magnitude=1, multiply=True)
        sim_ts = sim.stl_sim()
        # Compare the obtained simulated time series to
        # the original simulated data
        generator2 = Simulator(n=100, freq="D", start="2011-01-01")
        generator2.add_trend(magnitude=5, multiply=True)
        np.random.seed(614)
        generator2.add_seasonality(magnitude=10, period=timedelta(days=14))
        generator2.add_noise(magnitude=1, multiply=True)
        gen_ts_series = generator2.stl_sim()
        # pyre-fixme[16]: `bool` has no attribute `all`.
        self.assertEqual(True, (gen_ts_series.value == sim_ts.value).all())
        self.assertEqual(True, (gen_ts_series.time == sim_ts.time).all())

    def test_level_shift(self) -> None:
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
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        self.assertEqual(len(ts2), 450)

        sim3 = Simulator(n=450, start="2018-01-01")
        ts3 = sim3.level_shift_sim(
            cp_arr=[],
            level_arr=[3],
            noise=3,
            seasonal_period=7,
            seasonal_magnitude=3,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        self.assertEqual(len(ts3), 450)

    def test_level_shift_mvn_indep(self) -> None:

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

    def test_trend_shift(self) -> None:
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
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        self.assertEqual(len(ts2), 450)

        sim3 = Simulator(n=450, start="2018-01-01")
        ts3 = sim3.trend_shift_sim(
            cp_arr=[],
            trend_arr=[3],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=3,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        self.assertEqual(len(ts3), 450)

    def test_injected_anomalies(self) -> None:
        date_start_str = "2020-03-01"
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
        previous_seq = [date_start + timedelta(days=x) for x in range(60)]
        values = np.random.randn(len(previous_seq))
        ts_init = TimeSeriesData(
            pd.DataFrame({"time": previous_seq[0:60], "y_str": values[0:60]})
        )

        sim2 = Simulator(n=450, start="2018-01-01")

        ts_level = sim2.inject_level_shift(
            ts_input=ts_init, cp_arr=[10, 30, 40], level_arr=[10, -10]
        )
        self.assertEqual(len(ts_init), len(ts_level))

        ts_trend = sim2.inject_trend_shift(
            ts_input=ts_init, cp_arr=[10, 20, 30], trend_arr=[1, -3]
        )
        self.assertEqual(len(ts_init), len(ts_trend))

        ts_spike = sim2.inject_spikes(
            ts_input=ts_init, anomaly_arr=[10, 20, 30], z_score_arr=[10, -10, 2]
        )
        self.assertEqual(len(ts_init), len(ts_spike))


if __name__ == "__main__":
    unittest.main()
