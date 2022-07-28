# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
import unittest.mock as mock
from typing import Any, Dict
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
from kats.models.model import Model
from parameterized.parameterized import parameterized

np.random.seed(123321)
DATA_dummy = pd.DataFrame(
    {
        "time": pd.date_range(start="2019-01-01", end="2019-12-31", freq="D"),
        "y": [x + np.random.randint(20) for x in range(365)],
    }
)

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


# pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
def get_predict_model(m: Model, model_name: str, steps: int, freq: str) -> np.ndarray:
    """Get model prediction based on model_name."""
    if model_name == "BaseEnsemble":
        # pyre-fixme[16]: `Model` has no attribute `_predict_all`.
        return m._predict_all(steps=steps, freq=freq)
    else:
        # pyre-fixme[7]: Expected `ndarray` but got `None`.
        return m.predict(steps=steps, freq=freq)


def get_ensemble_param(ts_param: Dict[str, bool]) -> EnsembleParams:
    """Returns EnsembleParams based on which base_models are included."""
    base_model_list = [
        BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1))
        if ts_param["arima"]
        else "",
        BaseModelParams("holtwinters", holtwinters.HoltWintersParams())
        if ts_param["holtwinters"]
        else "",
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
        )
        if ts_param["sarima"]
        else "",
        BaseModelParams("prophet", prophet.ProphetParams())
        if ts_param["prophet"]
        else "",
        BaseModelParams("linear", linear_model.LinearModelParams())
        if ts_param["linear"]
        else "",
        BaseModelParams("quadratic", quadratic_model.QuadraticModelParams())
        if ts_param["quadratic"]
        else "",
        BaseModelParams("theta", theta.ThetaParams(m=12)) if ts_param["theta"] else "",
    ]
    return EnsembleParams(
        # pyre-fixme[6]: For 1st param expected `List[BaseModelParams]` but got
        #  `List[Union[BaseModelParams, str]]`.
        [base_model for base_model in base_model_list if base_model != ""]
    )


# test params, True or False means that base_emodel is included in this ensemble_model_param or not.
# params_base is being used for the BaseEnsemble model, and similarly the same for the KatsEnsemble
TEST_PARAM = {
    "params_base": {
        "arima": True,
        "holtwinters": True,
        "sarima": True,
        "prophet": True,
        "linear": True,
        "quadratic": True,
        "theta": False,
    },
    "params_multivariate_data": {
        "arima": True,
        "holtwinters": True,
        "sarima": False,
        "prophet": False,
        "linear": False,
        "quadratic": False,
        "theta": False,
    },
    "params_kats": {
        "arima": True,
        "holtwinters": False,
        "sarima": True,
        "prophet": True,
        "linear": True,
        "quadratic": True,
        "theta": True,
    },
}


TEST_DATA: Dict[str, Dict[str, Any]] = {
    "monthly": {
        "ts": load_air_passengers(),
        "params": {
            "base": get_ensemble_param(TEST_PARAM["params_base"]),
            "kats": get_ensemble_param(TEST_PARAM["params_kats"]),
        },
    },
    "daily": {
        "ts": TimeSeriesData(
            load_data("peyton_manning.csv").set_axis(["time", "y"], axis=1)
        ),
        "params": {
            "base": get_ensemble_param(TEST_PARAM["params_base"]),
            "kats": get_ensemble_param(TEST_PARAM["params_kats"]),
        },
    },
    "multivariate": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv")),
        "params": get_ensemble_param(TEST_PARAM["params_multivariate_data"]),
    },
    "dummy": {
        "ts": TimeSeriesData(DATA_dummy),
        "params": {
            "base": get_ensemble_param(TEST_PARAM["params_base"]),
            "kats": get_ensemble_param(TEST_PARAM["params_kats"]),
        },
    },
}


class testEnsembleModels(TestCase):
    """Test Three models BaseEnsemble, MedianEnsembleModel and WeightedAvgEnsemble."""

    def test_valid_params_base_params(self) -> None:
        """Check using a valid model params for BaseModelParams."""
        # test validate_param in base params
        base_param = BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1))
        base_param.validate_params()

    def test_invalid_params_in_ensemble_params(self) -> None:
        """Check using a non-valid model name in EnsembleParams results in error."""
        # validate params in EnsembleParams
        TSData = TEST_DATA["monthly"]["ts"]
        params = EnsembleParams(
            [
                BaseModelParams("random_model_name", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams("holtwinters", holtwinters.HoltWintersParams()),
            ]
        )
        self.assertRaises(
            ValueError,
            BaseEnsemble,
            TSData,
            params,
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "BaseEnsemble_monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                BaseEnsemble,
                False,
            ],
            [
                "BaseEnsemble_daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"]["base"],
                30,
                "D",
                BaseEnsemble,
                False,
            ],
            [
                "BaseEnsemble_dummy",
                TEST_DATA["dummy"]["ts"],
                TEST_DATA["dummy"]["params"]["base"],
                30,
                "D",
                BaseEnsemble,
                False,
            ],
            [
                "MedianEnsembleModel_monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                MedianEnsembleModel,
                False,
            ],
            [
                "MedianEnsembleModel_daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"]["base"],
                30,
                "D",
                MedianEnsembleModel,
                False,
            ],
            [
                "MedianEnsembleModel_dummy",
                TEST_DATA["dummy"]["ts"],
                TEST_DATA["dummy"]["params"]["base"],
                30,
                "D",
                MedianEnsembleModel,
                False,
            ],
            [
                "WeightedAvgEnsemble_monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                WeightedAvgEnsemble,
                False,
            ],
            [
                "WeightedAvgEnsemble_daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"]["base"],
                30,
                "D",
                WeightedAvgEnsemble,
                False,
            ],
            [
                "WeightedAvgEnsemble_dummy",
                TEST_DATA["dummy"]["ts"],
                TEST_DATA["dummy"]["params"]["base"],
                30,
                "D",
                WeightedAvgEnsemble,
                True,
            ],
        ]
    )
    def test_forecast(
        self,
        ts_data_name: str,
        ts_data: TimeSeriesData,
        # pyre-fixme[2]: Parameter must be annotated.
        params,
        steps: int,
        # pyre-fixme[2]: Parameter must be annotated.
        freq,
        # pyre-fixme[2]: Parameter must be annotated.
        model,
        backtester: bool,
    ) -> None:
        """Test forecast."""
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)[
            ["time", "fcst"]
        ]

        m = model(data=ts_data, params=params)

        with mock.patch("kats.models.ensemble.ensemble.Pool") as mock_pooled:
            mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
            mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)

            # fit the ensemble model
            m.fit()
            mock_pooled.assert_called()

            if backtester:
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
            else:
                mock_fit_model.assert_called()
                # no predictions should be made yet
                mock_fit_model.return_value.predict.assert_not_called()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "BaseEnsemble",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                BaseEnsemble,
                False,
                False,
            ],
            [
                "MedianEnsembleModel",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                MedianEnsembleModel,
                False,
                True,
            ],
            [
                "WeightedAvgEnsemble",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["base"],
                30,
                "MS",
                WeightedAvgEnsemble,
                True,
                True,
            ],
        ]
    )
    def test_predict_plot(
        self,
        ts_model_name: str,
        ts_data: TimeSeriesData,
        # pyre-fixme[2]: Parameter must be annotated.
        params,
        steps: int,
        # pyre-fixme[2]: Parameter must be annotated.
        freq,
        # pyre-fixme[2]: Parameter must be annotated.
        model,
        backtester: bool,
        plot: bool,
    ) -> None:
        m = model(data=ts_data, params=params)
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)[
            ["time", "fcst"]
        ]
        with mock.patch("kats.models.ensemble.ensemble.Pool") as mock_pooled:
            mock_fit_model = mock_pooled.return_value.apply_async.return_value.get
            mock_fit_model.return_value.predict = mock.MagicMock(return_value=preds)

            # fit the ensemble model
            m.fit()
            mock_pooled.assert_called()
            if backtester:
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
                    if plot:
                        m.plot()
            else:
                mock_fit_model.assert_called()
                # no predictions should be made yet
                mock_fit_model.return_value.predict.assert_not_called()
                # now run predict on the ensemble model
                get_predict_model(m, ts_model_name, steps=steps, freq=freq)
                mock_fit_model.return_value.predict.assert_called_with(
                    steps, freq=f"{freq}"
                )
                if plot:
                    m.plot()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "BaseEnsemble",
                BaseEnsemble,
                "Ensemble",
            ],
            [
                "MedianEnsembleModel",
                MedianEnsembleModel,
                "Median Ensemble",
            ],
            [
                "WeightedAvgEnsemble",
                WeightedAvgEnsemble,
                "Weighted Average Ensemble",
            ],
        ]
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_name(self, ts_model_name: str, model, model_name: str) -> None:
        """Test name of the model according to the model used."""
        # test __str__ method
        m = model(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"]["base"])
        self.assertEqual(m.__str__(), model_name)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "BaseEnsemble",
                BaseEnsemble,
            ],
            [
                "MedianEnsembleModel",
                MedianEnsembleModel,
            ],
            [
                "WeightedAvgEnsemble",
                WeightedAvgEnsemble,
            ],
        ]
    )
    def test_invalid_params_ensemble_params(
        self,
        ts_model_name: str,
        # pyre-fixme[24]: Generic type `Model` expects 1 type parameter.
        model: Model,
    ) -> None:
        # validate params in EnsembleParams
        TSData_multi = TEST_DATA["multivariate"]["ts"]
        params = TEST_DATA["multivariate"]["params"]

        self.assertRaises(
            ValueError,
            # pyre-fixme[6]: For 2nd param expected `(...) -> Any` but got
            #  `Model[typing.Any]`.
            model,
            TSData_multi,
            params,
        )


class testKatsEnsemble(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "monthly_kats",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["kats"],
                30,
                "MS",
            ],
            [
                "daily_kats",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"]["kats"],
                30,
                "D",
            ],
            [
                "dummy_kats",
                TEST_DATA["dummy"]["ts"],
                TEST_DATA["dummy"]["params"]["kats"],
                30,
                "D",
            ],
        ]
    )
    def test_fit_median_forecast(
        self,
        ts_data_name: str,
        ts_data: TimeSeriesData,
        params: Dict[str, Any],
        steps: int,
        freq: str,
    ) -> None:
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)

        decomps = ["additive", "multiplicative"]

        for decomp in decomps:
            KatsEnsembleParam = {
                "models": params,
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

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"]["kats"],
                30,
                "MS",
            ],
            [
                "daily",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"]["kats"],
                30,
                "D",
            ],
            [
                "dummy",
                TEST_DATA["dummy"]["ts"],
                TEST_DATA["dummy"]["params"]["kats"],
                30,
                "D",
            ],
        ]
    )
    def test_fit_weightedavg_forecast(
        self,
        ts_data_name: str,
        ts_data: TimeSeriesData,
        params: Dict[str, Any],
        steps: int,
        freq: str,
    ) -> None:
        preds = get_fake_preds(ts_data, fcst_periods=steps, fcst_freq=freq)

        decomps = ["additive", "multiplicative"]

        for decomp in decomps:
            KatsEnsembleParam = {
                "models": params,
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
                m.predict(steps=steps)
                mock_fit_model.return_value.predict.assert_called_with(steps)
                m.aggregate()
                m.plot()

    def test_invalid_decomposition_method(self) -> None:
        KatsEnsembleParam = {
            "models": get_ensemble_param(TEST_PARAM["params_kats"]),
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "random_decomp",
        }
        m = KatsEnsemble(data=TEST_DATA["monthly"]["ts"], params=KatsEnsembleParam)
        m.validate_params()

    def test_invalid_seasonality_length(self) -> None:
        KatsEnsembleParam = {
            "models": get_ensemble_param(TEST_PARAM["params_kats"]),
            "aggregation": "median",
            "seasonality_length": 1000000,
            "decomposition_method": "additive",
        }

        self.assertRaises(
            ValueError,
            KatsEnsemble,
            TEST_DATA["monthly"]["ts"],
            KatsEnsembleParam,
        )

    def test_logging_default_executors(self) -> None:
        KatsEnsembleParam = {
            "models": get_ensemble_param(TEST_PARAM["params_kats"]),
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "random_decomp",
            "fitExecutor": None,
            "forecastExecutor": None,
        }
        with self.assertLogs(level="INFO"):
            KatsEnsemble(data=TEST_DATA["monthly"]["ts"], params=KatsEnsembleParam)

    def test_non_seasonal_data(self) -> None:
        KatsEnsembleParam = {
            "models": get_ensemble_param(TEST_PARAM["params_kats"]),
            "aggregation": "median",
            "seasonality_length": 12,
            "decomposition_method": "additive",
        }

        dummy_ts = TimeSeriesData(
            time=pd.date_range(start="2020-01-01", end="2020-05-31", freq="D"),
            value=pd.Series(list(range(152))),
        )
        KatsEnsemble(data=dummy_ts, params=KatsEnsembleParam)


if __name__ == "__main__":
    unittest.main()
