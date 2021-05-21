#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import pandas as pd
import numpy as np
from unittest import TestCase
from kats.consts import TimeSeriesData
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    theta,
    sarima,
)
from kats.models.ensemble.ensemble import (
    BaseEnsemble,
    BaseModelParams,
    EnsembleParams,
)
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models.ensemble.median_ensemble import MedianEnsembleModel
from kats.models.ensemble.weighted_avg_ensemble import WeightedAvgEnsemble

if "kats/tests" in os.getcwd():
    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/air_passengers.csv"
            )
        )

    daily_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/peyton_manning.csv"
            )
        )

    multi_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/multivariate_anomaly_simulated_data.csv"
            )
        )
elif "/home/runner/work/" in os.getcwd(): # for Githun Action
    data_path = "kats/data/air_passengers.csv"
    daily_data_path = "kats/data/peyton_manning.csv"
    multi_data_path = "kats/data/multivariate_anomaly_simulated_data.csv"
else:
    data_path = "kats/kats/data/air_passengers.csv"
    daily_data_path = "kats/kats/data/peyton_manning.csv"
    multi_data_path = "kats/kats/data/multivariate_anomaly_simulated_data.csv"

DATA = pd.read_csv(data_path)
DATA.columns = ["time", "y"]
TSData = TimeSeriesData(DATA)

DATA_daily = pd.read_csv(daily_data_path)
DATA_daily.columns = ["time", "y"]
TSData_daily = TimeSeriesData(DATA_daily)

DATA_multi = pd.read_csv(multi_data_path)
TSData_multi = TimeSeriesData(DATA_multi)

np.random.seed(123321)
DATA_dummy = pd.DataFrame({
    "time": pd.date_range(start="2019-01-01", end="2019-12-31", freq="D"),
    "y": [x + np.random.randint(20) for x in range(365)],
})
TSData_dummy = TimeSeriesData(DATA_dummy)


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


class testBaseEnsemble(TestCase):
    def test_fit_forecast(self) -> None:
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ProphetParams`.
                BaseModelParams("prophet", prophet.ProphetParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )

        m = BaseEnsemble(TSData, params)
        m.fit()
        m._predict_all(steps=30, freq="MS")
        m.plot()

        m_daily = BaseEnsemble(TSData_daily, params)
        m_daily.fit()
        m_daily._predict_all(steps=30, freq="D")
        m.plot()

        m_dummy = BaseEnsemble(TSData_dummy, params)
        m_dummy.fit()
        m_dummy._predict_all(steps=30, freq="D")
        m_dummy.plot()

        # test __str__ method
        self.assertEqual(m.__str__(), "Ensemble")

    def test_others(self) -> None:
        # test validate_param in base params
        # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
        base_param = BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1))
        base_param.validate_params()

        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ProphetParams`.
                BaseModelParams("prophet", prophet.ProphetParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            BaseEnsemble,
            TSData_multi,
            params,
        )

        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("random_model_name", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            BaseEnsemble,
            TSData,
            params,
        )


class testMedianEnsemble(TestCase):
    def test_fit_forecast(self) -> None:
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ProphetParams`.
                BaseModelParams("prophet", prophet.ProphetParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )
        m = MedianEnsembleModel(data=TSData, params=params)
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

        m_daily = MedianEnsembleModel(data=TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m_daily.plot()

        m_dummy = MedianEnsembleModel(data=TSData_dummy, params=params)
        m_dummy.fit()
        m_dummy.predict(steps=30, freq="D")
        m_dummy.plot()

        # test __str__ method
        self.assertEqual(m_daily.__str__(), "Median Ensemble")

    def test_others(self) -> None:
        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            MedianEnsembleModel,
            TSData_multi,
            params,
        )


class testWeightedAvgEnsemble(TestCase):
    def test_fit_forecast(self) -> None:
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                BaseModelParams(
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `ProphetParams`.
                    "prophet", prophet.ProphetParams(seasonality_mode="multiplicative")
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )
        m = WeightedAvgEnsemble(data=TSData, params=params)
        m.fit()
        m.predict(steps=30, freq="MS", err_method="mape")
        m.plot()

        m_daily = WeightedAvgEnsemble(data=TSData_daily, params=params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()

        m_dummy = WeightedAvgEnsemble(data=TSData_dummy, params=params)
        m_dummy.fit()
        m_dummy.predict(steps=30, freq="D")
        m_dummy.plot()

        # test __str__ method
        self.assertEqual(m.__str__(), "Weighted Average Ensemble")

    def test_others(self) -> None:
        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `HoltWintersParams`.
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            WeightedAvgEnsemble,
            TSData_multi,
            params,
        )


class testKatsEnsemble(TestCase):
    def test_fit_forecast(self) -> None:
        model_params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ProphetParams`.
                BaseModelParams("prophet", prophet.ProphetParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ThetaParams`.
                BaseModelParams("theta", theta.ThetaParams(m=12)),
            ]
        )
        aggregations = ["median", "weightedavg"]
        decomps = ["additive", "multiplicative"]

        for agg in aggregations:
            for decomp in decomps:
                KatsEnsembleParam = {
                    "models": model_params,
                    "aggregation": agg,
                    "seasonality_length": 12,
                    "decomposition_method": decomp,
                }

                m = KatsEnsemble(data=TSData, params=KatsEnsembleParam)
                m.fit()
                m.predict(steps=30)
                m.aggregate()
                m.plot()

                m = KatsEnsemble(data=TSData, params=KatsEnsembleParam)
                m.fit()
                m.forecast(steps=30)
                m.aggregate()
                m.plot()

                m = KatsEnsemble(data=TSData_dummy, params=KatsEnsembleParam)
                m.fit()
                m.forecast(steps=30)
                m.aggregate()
                m.plot()

    def test_others(self) -> None:
        model_params = EnsembleParams(
            [
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ARIMAParams`.
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
                    # pyre-fixme[6]: Expected `Model` for 2nd param but got
                    #  `SARIMAParams`.
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ProphetParams`.
                BaseModelParams("prophet", prophet.ProphetParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `LinearModelParams`.
                BaseModelParams("linear", linear_model.LinearModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got
                #  `QuadraticModelParams`.
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                # pyre-fixme[6]: Expected `Model` for 2nd param but got `ThetaParams`.
                BaseModelParams("theta", theta.ThetaParams(m=12)),
            ]
        )

        KatsEnsembleParam = {
            "models": model_params,
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "random_decomp",
        }
        # test invalid decomposition method
        m = KatsEnsemble(data=TSData, params=KatsEnsembleParam)
        m.validate_params()

        # test invalid seasonality length
        KatsEnsembleParam = {
            "models": model_params,
            "aggregation": "median",
            "seasonality_length": 1000000,
            "decomposition_method": "additive",
        }

        self.assertRaises(
            ValueError,
            KatsEnsemble,
            TSData,
            KatsEnsembleParam,
        )

        # test logging with default executors
        KatsEnsembleParam = {
            "models": model_params,
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "random_decomp",
            "fitExecutor": None,
            "forecastExecutor": None,
        }
        with self.assertLogs(level="INFO"):
            m = KatsEnsemble(data=TSData, params=KatsEnsembleParam)

        # test non-seasonal data
        KatsEnsembleParam = {
            "models": model_params,
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "additive",
        }

        dummy_ts = TimeSeriesData(
            time=pd.date_range(start='2020-01-01', end='2020-05-31', freq='D'),
            value=pd.Series(list(range(152))),
        )
        m = KatsEnsemble(data=dummy_ts, params=KatsEnsembleParam)


if __name__ == "__main__":
    unittest.main()
