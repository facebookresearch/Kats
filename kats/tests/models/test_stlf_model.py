# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import timedelta
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models import (
    linear_model,
    prophet,
    quadratic_model,
    simple_heuristic_model,
    theta,
)
from kats.models.stlf import STLFModel, STLFParams
from kats.utils.simulator import Simulator
from parameterized.parameterized import parameterized


def load_data_std_cols(path: str) -> pd.DataFrame:
    df = load_data(path)
    df.columns = ["time", "y"]
    return df


METHODS = ["theta", "prophet", "linear", "quadratic", "simple"]


class testSTLFModel(TestCase):
    def setUp(self) -> None:

        sim = Simulator(
            n=3 * 144, freq="10T", start=pd.to_datetime("2021-01-01")
        )  # 3 days of data
        sim.add_trend(magnitude=1)
        sim.add_seasonality(magnitude=50, period=timedelta(days=1))
        sim.add_noise(magnitude=0)
        dense_dates_10min_df = sim.stl_sim().to_dataframe()
        dense_dates_10min_df.rename(columns={"value": "y"}, inplace=True)

        # Load data for each test
        self.TEST_DATA = {
            "daily": {
                "ts": TimeSeriesData(load_data_std_cols("peyton_manning.csv")),
            },
            "multi": {
                "ts": TimeSeriesData(
                    load_data("multivariate_anomaly_simulated_data.csv")
                ),
            },
            "dens_ds_10min": {
                "ts": TimeSeriesData(dense_dates_10min_df),
            },
        }

    # pyre-fixme[56]
    @parameterized.expand([("daily", m) for m in METHODS])
    def test_fit_forecast(self, dataset: str, method: str, steps: int = 5) -> None:
        ts = self.TEST_DATA[dataset]["ts"]
        params = STLFParams(m=12, method=method)
        train, truth = ts[:-steps], ts[-steps:]
        m = STLFModel(train, params)
        m.fit()
        pred = m.predict(steps=steps).iloc[:, 1:].to_numpy()

        # check whether the values are close and shapes are correct
        truth = truth.to_dataframe().y.to_numpy()
        self.assertTrue((truth - pred[:, 1]).max() < 2)  # check actual vs true
        self.assertTrue(all(pred[:, 2] >= pred[:, 0]))  # check upper > lower bounds

    # pyre-fixme[56]
    @parameterized.expand([("multi", m) for m in METHODS])
    def test_invalid_predict_length(self, dataset: str, method: str) -> None:
        ts = self.TEST_DATA[dataset]["ts"]
        params = STLFParams(m=10000, method=method)
        self.assertRaises(
            ValueError,
            STLFModel,
            ts,
            params,
        )

    def test_invalid_params(self) -> None:
        self.assertRaises(
            ValueError,
            STLFParams,
            method="random_model",
            m=12,
        )

    # pyre-fixme[56]
    @parameterized.expand([("multi", m) for m in METHODS])
    def test_model_param(self, dataset: str, method: str) -> None:
        ts = self.TEST_DATA[dataset]["ts"]
        params = STLFParams(m=12, method=method)
        params.validate_params()
        self.assertRaises(
            ValueError,
            STLFModel,
            ts,
            params,
        )

    # pyre-fixme[56]
    @parameterized.expand([("daily", m) for m in METHODS])
    def test_str(self, dataset: str, method: str) -> None:
        ts = self.TEST_DATA[dataset]["ts"]
        params = STLFParams(m=12, method=method)
        params.validate_params()
        m = STLFModel(ts, params)
        self.assertEqual(m.__str__(), "STLF")

    # pyre-fixme[56]
    @parameterized.expand([("daily", m) for m in METHODS])
    def test_fit_forecast_no_default_params(
        self, dataset: str, method: str, steps: int = 5
    ) -> None:
        ts = self.TEST_DATA[dataset]["ts"]
        if method == "prophet":
            method_params = prophet.ProphetParams(seasonality_mode="multiplicative")
        elif method == "theta":
            method_params = theta.ThetaParams(m=2)
        elif method == "linear":
            method_params = linear_model.LinearModelParams(alpha=0.01)
        elif method == "simple":
            method_params = simple_heuristic_model.SimpleHeuristicModelParams()
        else:
            method_params = quadratic_model.QuadraticModelParams(alpha=0.05)
        params = STLFParams(m=12, method=method, method_params=method_params)
        train, truth = ts[:-steps], ts[-steps:]
        m = STLFModel(train, params)
        m.fit()
        pred = m.predict(steps=steps).iloc[:, 1:].to_numpy()

        # check whether the values are close and shapes are correct
        truth = truth.to_dataframe().y.to_numpy()
        self.assertTrue((truth - pred[:, 1]).max() < 2)  # check actual vs true
        self.assertTrue(all(pred[:, 2] >= pred[:, 0]))  # check upper > lower bounds

    # pyre-fixme[56]
    @parameterized.expand(
        [("daily", m, "additive") for m in METHODS]
        + [("daily", m, "multiplicative") for m in METHODS]
    )
    def test_fit_forecast_decomposition_parameter(
        self, dataset: str, method: str, decomposition_method: str, steps: int = 5
    ) -> None:

        ts = self.TEST_DATA[dataset]["ts"]
        train, truth = ts[:-steps], ts[-steps:]
        params = STLFParams(
            m=12,
            method=method,
            decomposition=decomposition_method,
        )
        m = STLFModel(train, params)
        m.fit()
        pred = m.predict(steps=steps).iloc[:, 1:].to_numpy()
        truth = truth.to_dataframe().y.to_numpy()

        self.assertTrue((truth - pred[:, 1]).max() < 2)  # check actual vs true
        self.assertTrue(all(pred[:, 2] >= pred[:, 0]))  # check upper > lower bounds

    def test_fit_forecast_simple_model(self) -> None:

        steps = 7

        ts = self.TEST_DATA["daily"]["ts"]
        train, truth = ts[:-steps], ts[-steps:]
        params = STLFParams(
            m=12,
            method="simple",
            decomposition="additive",
        )
        m = STLFModel(train, params)
        m.fit()
        pred = m.predict(steps=steps).iloc[:, 1:].to_numpy()
        truth = truth.to_dataframe().y.to_numpy()

        self.assertTrue((truth - pred[:, 1]).max() < 2)  # check actual vs true
        self.assertTrue(all(pred[:, 2] >= pred[:, 0]))  # check upper > lower bounds

    # pyre-fixme[56]
    @parameterized.expand([("dens_ds_10min", m) for m in METHODS])
    def test_fit_forecast_minute(
        self, dataset: str, method: str, steps: int = 144
    ) -> None:

        ts = self.TEST_DATA[dataset]["ts"]
        params = STLFParams(m=144, method=method, decomposition="additive")
        train, truth = ts[:-steps], ts[-steps:]
        m = STLFModel(train, params)
        m.fit()
        pred = m.predict(steps=steps)

        # check whether the values are close and shapes are correct
        truth = truth.value.values
        self.assertTrue((truth - pred["fcst"]).max() < 4)  # check actual vs true
        self.assertTrue(
            all(pred["fcst_upper"] >= pred["fcst_lower"])
        )  # check upper > lower bounds


if __name__ == "__main__":
    unittest.main()
