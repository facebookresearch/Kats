# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
import unittest.mock as mock
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)
from kats.models.ensemble.ensemble import BaseEnsemble, BaseModelParams, EnsembleParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models.ensemble.median_ensemble import MedianEnsembleModel
from kats.models.ensemble.weighted_avg_ensemble import WeightedAvgEnsemble
from parameterized.parameterized import parameterized

np.random.seed(123321)
DATA_dummy = pd.DataFrame(
    {
        "time": pd.date_range(start="2019-01-01", end="2019-12-31", freq="D"),
        "y": [x + np.random.randint(20) for x in range(365)],
    }
)
TSData_dummy = TimeSeriesData(DATA_dummy)


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


def get_fake_preds(
    ts_data: TimeSeriesData, fcst_periods: int, fcst_freq: str
) -> pd.DataFrame:
    time = pd.date_range(
        start=ts_data.time.iloc[-1], periods=fcst_periods + 1, freq=fcst_freq
    )[1:]
    fcst = np.random.uniform(0, 100, len(time))
    return pd.DataFrame(
        {
            "time": {i: t for i, t in enumerate(time)},
            "fcst": {i: t for i, t in enumerate(fcst)},
            "fcst_lower": {i: t for i, t in enumerate(fcst + 10)},
            "fcst_upper": {i: t for i, t in enumerate(fcst - 10)},
        }
    )


class testBaseEnsemble(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

        self.TSData_dummy = TSData_dummy

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["TSData", 30, "MS"],
    #  ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]])`.
    @parameterized.expand(
        [["TSData", 30, "MS"], ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]]
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_fit_forecast(self, ts_data_name, steps: int, freq: str) -> None:
        ts_data = getattr(self, ts_data_name)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)
        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams(
                    "quadratic",
                    quadratic_model.QuadraticModelParams(),
                ),
            ]
        )
        m = BaseEnsemble(ts_data, params)

        with mock.patch("kats.models.ensemble.ensemble.Pool") as mock_pooled:
            mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
            mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)

            # fit the ensemble model
            m.fit()
            mock_pooled.assert_called()
            mock_fit_model.assert_called()

            # no predictions should be made yet
            mock_fit_model.return_value.predict.assert_not_called()

            # now run predict for each of the component models
            m._predict_all(steps=steps, freq=freq)

            # now predict should have been called
            mock_fit_model.return_value.predict.assert_called_with(
                steps, freq=f"{freq}"
            )

            self.assertEqual(m.__str__(), "Ensemble")

    def test_others(self) -> None:
        # test validate_param in base params
        base_param = BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1))
        base_param.validate_params()

        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            BaseEnsemble,
            self.TSData_multi,
            params,
        )

        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                BaseModelParams("random_model_name", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            BaseEnsemble,
            self.TSData,
            params,
        )


class testMedianEnsemble(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

        self.TSData_dummy = TSData_dummy

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["TSData", 30, "MS"],
    #  ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]])`.
    @parameterized.expand(
        [["TSData", 30, "MS"], ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]]
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_fit_forecast(self, ts_data_name, steps: int, freq: str) -> None:
        ts_data = getattr(self, ts_data_name)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)[
            ["time", "fcst"]
        ]
        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams(
                    "quadratic",
                    quadratic_model.QuadraticModelParams(),
                ),
            ]
        )
        m = MedianEnsembleModel(data=ts_data, params=params)

        with mock.patch("kats.models.ensemble.ensemble.Pool") as mock_pooled:
            mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
            mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)

            # fit the ensemble model
            m.fit()

            mock_pooled.assert_called()
            mock_fit_model.assert_called()

            # no predictions should be made yet
            mock_fit_model.return_value.predict.assert_not_called()

            # now run predict on the ensemble model
            m.predict(steps=steps, freq=freq)
            mock_fit_model.return_value.predict.assert_called_with(
                steps, freq=f"{freq}"
            )
            m.plot()

            # test __str__ method
            self.assertEqual(m.__str__(), "Median Ensemble")

    def test_others(self) -> None:
        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            MedianEnsembleModel,
            self.TSData_multi,
            params,
        )


class testWeightedAvgEnsemble(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

        self.TSData_dummy = TSData_dummy

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["TSData", 30, "MS"],
    #  ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]])`.
    @parameterized.expand(
        [["TSData", 30, "MS"], ["TSData_daily", 30, "D"], ["TSData_dummy", 30, "D"]]
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_fit_forecast(self, ts_data_name, steps: int, freq: str) -> None:
        ts_data = getattr(self, ts_data_name)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)[
            ["time", "fcst"]
        ]
        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
                BaseModelParams(
                    "sarima",
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
                    "prophet",
                    prophet.ProphetParams(seasonality_mode="multiplicative"),
                ),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
            ]
        )

        m = WeightedAvgEnsemble(ts_data, params=params)

        with mock.patch("kats.models.ensemble.ensemble.Pool") as mock_pooled:
            mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
            mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)

            # fit the ensemble model
            m.fit()
            mock_pooled.assert_called()

            with mock.patch(
                "kats.models.ensemble.weighted_avg_ensemble.Pool"
            ) as mock_weighted_pooled:
                mock_backtest = (
                    mock_weighted_pooled.return_value.apply_async.return_value.get
                )
                # the backtester should just return a random number here
                mock_backtest.return_value = np.random.rand()
                m.predict(steps=steps, freq=freq)
                mock_backtest.assert_called()
                m.plot()

            # test __str__ method
            self.assertEqual(m.__str__(), "Weighted Average Ensemble")

    def test_others(self) -> None:
        # validate params in EnsembleParams
        params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )

        self.assertRaises(
            ValueError,
            WeightedAvgEnsemble,
            self.TSData_multi,
            params,
        )


class testKatsEnsemble(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()
        self.TSData_dummy = TSData_dummy

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["TSData", 30, "MS"],
    #  ["TSData", 30, "D"], ["TSData_dummy", 30, "D"]])`.
    @parameterized.expand(
        [["TSData", 30, "MS"], ["TSData", 30, "D"], ["TSData_dummy", 30, "D"]]
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_fit_median_forecast(self, ts_data_name, steps: int, freq: str) -> None:
        ts_data = getattr(self, ts_data_name)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)
        model_params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                BaseModelParams("theta", theta.ThetaParams(m=12)),
            ]
        )
        decomps = ["additive", "multiplicative"]

        for decomp in decomps:
            KatsEnsembleParam = {
                "models": model_params,
                "aggregation": "median",
                "seasonality_length": 12,
                "decomposition_method": decomp,
            }

            m = KatsEnsemble(data=ts_data, params=KatsEnsembleParam)

            with mock.patch("multiprocessing.managers.SyncManager.Pool") as mock_pooled:
                mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
                mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)
                # fit the model
                m.fit()
                mock_pooled.assert_called()
                # no predictions should be made yet
                mock_fit_model.return_value.predict.assert_not_called()
                # now run predict on the ensemble model
                m.predict(steps=steps)
                mock_fit_model.return_value.predict.assert_called_with(steps)
                m.aggregate()
                m.plot()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([["TSData", 30, "MS"],
    #  ["TSData", 30, "D"], ["TSData_dummy", 30, "D"]])`.
    @parameterized.expand(
        [["TSData", 30, "MS"], ["TSData", 30, "D"], ["TSData_dummy", 30, "D"]]
    )
    def test_fit_weightedavg_forecast(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        ts_data_name,
        steps: int,
        freq: str,
    ) -> None:
        ts_data = getattr(self, ts_data_name)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)
        model_params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                BaseModelParams("theta", theta.ThetaParams(m=12)),
            ]
        )
        decomps = ["additive", "multiplicative"]

        for decomp in decomps:
            KatsEnsembleParam = {
                "models": model_params,
                "aggregation": "weightedavg",
                "seasonality_length": 12,
                "decomposition_method": decomp,
            }
            m = KatsEnsemble(data=ts_data, params=KatsEnsembleParam)

            with mock.patch("multiprocessing.managers.SyncManager.Pool") as mock_pooled:
                mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
                mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)
                mock_fit_model.return_value.__add__ = mock.MagicMock(
                    return_value=np.random.rand()
                )
                # fit the model
                m.fit()
                mock_pooled.assert_called()

                # no predictions should be made yet
                mock_fit_model.return_value.predict.assert_not_called()
                # backtesting should be done after calling fit
                mock_fit_model.return_value.__add__.assert_called_with(
                    sys.float_info.epsilon
                )

                # now run predict on the ensemble model
                m.predict(steps=steps)
                mock_fit_model.return_value.predict.assert_called_with(steps)
                m.aggregate()
                m.plot()

                # reset all the mocks and make sure they're not called
                mock_pooled.reset_mock()
                mock_pooled.assert_not_called()
                mock_fit_model.return_value.predict.assert_not_called()
                mock_fit_model.return_value.__add__.assert_not_called()

                # now retry the above with forecast rather than fit/predict
                m.forecast(steps=30)
                mock_pooled.assert_called()

                # backtesting should be done after calling fit
                mock_fit_model.return_value.__add__.assert_called_with(
                    sys.float_info.epsilon
                )

                # now run predict on the ensemble model
                # m.predict(steps=steps)
                mock_fit_model.return_value.predict.assert_called_with(steps)
                m.aggregate()
                m.plot()

    def test_others(self) -> None:
        model_params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
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
                BaseModelParams("prophet", prophet.ProphetParams()),
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
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
        m = KatsEnsemble(data=self.TSData, params=KatsEnsembleParam)
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
            self.TSData,
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
            m = KatsEnsemble(data=self.TSData, params=KatsEnsembleParam)

        # test non-seasonal data
        KatsEnsembleParam = {
            "models": model_params,
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "additive",
        }

        dummy_ts = TimeSeriesData(
            time=pd.date_range(start="2020-01-01", end="2020-05-31", freq="D"),
            value=pd.Series(list(range(152))),
        )
        m = KatsEnsemble(data=dummy_ts, params=KatsEnsembleParam)


if __name__ == "__main__":
    unittest.main()
