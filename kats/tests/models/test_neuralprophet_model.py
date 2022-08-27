# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import sys
import unittest
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from kats.compat import pandas
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers, load_data
from kats.models.neuralprophet import NeuralProphetModel, NeuralProphetParams
from kats.tests.models.test_models_dummy_data import (
    NONSEASONAL_FUTURE_DF,
    NONSEASONAL_INPUT,
)
from parameterized.parameterized import parameterized


TEST_DATA: Dict[str, Any] = {
    "nonseasonal": {
        "ts": TimeSeriesData(NONSEASONAL_INPUT),
        "future_df": NONSEASONAL_FUTURE_DF,
        "params": NeuralProphetParams(epochs=5),
    },
    "daily": {
        "ts": TimeSeriesData(
            load_data("peyton_manning.csv").set_axis(["time", "y"], axis=1)
        ),
        "params": NeuralProphetParams(epochs=5),
        "params_custom_seasonality": NeuralProphetParams(
            epochs=5,
            custom_seasonalities=[
                {
                    "name": "semi_annually",
                    "period": 365.25 / 2,
                    "fourier_order": 5,
                },
            ],
        ),
    },
    "monthly": {
        "ts": load_air_passengers(),
        "params": NeuralProphetParams(epochs=5),
        "params_custom_seasonality": NeuralProphetParams(
            epochs=5,
            custom_seasonalities=[
                {
                    "name": "monthly",
                    "period": 30.5,
                    "fourier_order": 5,
                },
            ],
        ),
    },
    "multivariate": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv"))
    },
}


class NeuralProphetModelTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # pyre-fixme[33]: Given annotation cannot contain `Any`.
        original_import_fn: Callable[
            [
                str,
                Optional[Mapping[str, Any]],
                Optional[Mapping[str, Any]],
                Sequence[str],
                int,
            ],
            Any,
        ] = builtins.__import__

        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        def mock_neuralprophet_import(module: Any, *args: Any, **kwargs: Any) -> None:
            if module == "neuralprophet":
                raise ImportError
            else:
                return original_import_fn(module, *args, **kwargs)

        cls.mock_imports = patch(
            "builtins.__import__", side_effect=mock_neuralprophet_import
        )

    def test_neuralprophet_not_installed(self) -> None:
        # Unload prophet module so its imports can be mocked as necessary
        del sys.modules["kats.models.neuralprophet"]

        with self.mock_imports:
            from kats.models.neuralprophet import (
                NeuralProphetModel,
                NeuralProphetParams,
            )

            self.assertRaises(RuntimeError, NeuralProphetParams)
            self.assertRaises(
                RuntimeError,
                NeuralProphetModel,
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
            )

        # Restore the prophet module
        del sys.modules["kats.models.neuralprophet"]
        from kats.models.neuralprophet import NeuralProphetModel, NeuralProphetParams

        # Confirm that the module has been properly reloaded -- should not
        # raise an exception anymore
        NeuralProphetModel(
            TEST_DATA["daily"]["ts"],
            NeuralProphetParams(
                epochs=5,
            ),
        )

    def test_default_parameters(self) -> None:
        """
        Check that the default parameters are as expected. The expected values
        are hard coded.
        """
        expected_defaults = NeuralProphetParams(
            growth="linear",
            changepoints=None,
            n_changepoints=10,
            changepoints_range=0.9,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto",
            seasonality_mode="additive",
            custom_seasonalities=None,
        )

        actual_defaults = vars(NeuralProphetParams())

        # Expected params should be valid
        expected_defaults.validate_params()

        for param, exp_val in vars(expected_defaults).items():
            msg = "param: {param}, expected default: {exp_val}, actual default: {val}".format(
                param=param, exp_val=exp_val, val=actual_defaults[param]
            )
            self.assertEqual(exp_val, actual_defaults[param], msg)

    def test_multivar(self) -> None:
        # Prophet model does not support multivariate time series data
        self.assertRaises(
            ValueError,
            NeuralProphetModel,
            TEST_DATA["multivariate"]["ts"],
            NeuralProphetParams(epochs=5),
        )

    def test_exec_plot(self) -> None:
        m = NeuralProphetModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        m.fit(freq="MS")
        fcst = m.predict(steps=30)
        m.plot(fcst)

    def test_name(self) -> None:
        m = NeuralProphetModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        self.assertEqual("NeuralProphet", m.__str__())

    def test_search_space(self) -> None:
        self.assertEqual(
            [
                {
                    "name": "yearly_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "weekly_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "daily_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "seasonality_mode",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
                {
                    "name": "changepoints_range",
                    "type": "choice",
                    "value_type": "float",
                    "values": list(np.arange(0.85, 0.96, 0.01)),  # last value is 0.95
                    "is_ordered": True,
                },
            ],
            NeuralProphetModel.get_parameter_search_space(),
        )

    # Testing extra regressors
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [
                "lagged regressors",
                5,
                3,
                [
                    {
                        "names": "reg1",
                        "regularization": 0.1,
                        "normalize": True,
                    },
                    {
                        "names": "reg2",
                    },
                ],
                None,
            ],
            [
                "future regressors",
                5,
                0,
                None,
                [
                    {
                        "name": "reg1",
                        "regularization": 0.1,
                        "normalize": True,
                    },
                    {
                        "name": "reg2",
                    },
                ],
            ],
        ]
    )
    def test_regressors(
        self,
        testcase_name: str,
        epochs: Optional[int],
        n_lags: int,
        extra_lagged_regressors: Optional[List[Dict[str, Any]]],
        extra_future_regressors: Optional[List[Dict[str, Any]]],
    ) -> None:
        tmp_df = TEST_DATA["daily"]["ts"].to_dataframe()
        tmp_df["reg1"] = np.arange(len(tmp_df))
        tmp_df["reg2"] = np.arange(len(tmp_df), 0, -1)
        ts = TimeSeriesData(tmp_df)

        future = pd.DataFrame(
            {
                "ds": pd.date_range("2013-05-01", periods=30),
                "reg1": np.arange(30),
                "reg2": np.arange(30, 0, -1),
            }
        )

        m_daily = NeuralProphetModel(
            ts,
            NeuralProphetParams(
                epochs=epochs,
                n_lags=n_lags,
                extra_lagged_regressors=extra_lagged_regressors,
                extra_future_regressors=extra_future_regressors,
            ),
        )
        m_daily.fit(freq="D")
        fcst = m_daily.predict(steps=30, future=future)
        m_daily.plot(fcst)


if __name__ == "__main__":
    unittest.main()
