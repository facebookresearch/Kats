# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import sys
import unittest
from typing import Optional
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data, load_air_passengers
from kats.models.prophet import ProphetModel, ProphetParams
from kats.tests.models.test_models_dummy_data import (
    NONSEASONAL_INPUT,
    NONSEASONAL_FUTURE_DF,
    AIR_FCST_30_PROPHET,
    AIR_FCST_30_PROPHET_CAP_AND_FLOOR,
    PEYTON_FCST_30_PROPHET_CAP_AND_FLOOR,
    AIR_FCST_30_PROPHET_INCL_HIST,
    PEYTON_FCST_15_PROPHET_INCL_HIST,
    AIR_FCST_15_PROPHET_LOGISTIC_CAP,
    PEYTON_FCST_30_PROPHET_DAILY_CAP,
    AIR_FCST_30_PROPHET_CUSTOM_SEASONALITY,
    PEYTON_FCST_30_PROPHET_CUSTOM_SEASONALITY,
    NONSEASONAL_FCST_15_PROPHET_ARG_FUTURE,
)
from pandas.util.testing import assert_frame_equal
from parameterized import parameterized

TEST_DATA = {
    "nonseasonal": {
        "ts": TimeSeriesData(NONSEASONAL_INPUT),
        "future_df": NONSEASONAL_FUTURE_DF,
        "params": ProphetParams(),
    },
    "daily": {
        "ts": TimeSeriesData(
            load_data("peyton_manning.csv").set_axis(["time", "y"], axis=1)
        ),
        "params": ProphetParams(),
        "params_cap_and_floor": ProphetParams(cap=1000, floor=10, growth="logistic"),
        "params_logistic_cap": ProphetParams(growth="logistic", cap=20),
        "params_custom_seasonality": ProphetParams(
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
        "params": ProphetParams(),
        "params_cap_and_floor": ProphetParams(cap=1000, floor=10, growth="logistic"),
        "params_logistic_cap": ProphetParams(growth="logistic", cap=1000),
        "params_custom_seasonality": ProphetParams(
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


class ProphetModelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

        original_import_fn = builtins.__import__

        def mock_prophet_import(module, *args, **kwargs):
            if module == "fbprophet":
                raise ImportError
            else:
                return original_import_fn(module, *args, **kwargs)

        cls.mock_imports = patch("builtins.__import__", side_effect=mock_prophet_import)

    def test_fbprophet_not_installed(self) -> None:
        # Unload prophet module so its imports can be mocked as necessary
        del sys.modules["kats.models.prophet"]

        with self.mock_imports:
            from kats.models.prophet import ProphetModel, ProphetParams

            self.assertRaises(RuntimeError, ProphetParams)
            self.assertRaises(
                RuntimeError,
                ProphetModel,
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
            )

        # Restore the prophet module
        del sys.modules["kats.models.prophet"]
        from kats.models.prophet import ProphetModel, ProphetParams

        # Confirm that the module has been properly reloaded -- should not
        # raise an exception anymore
        ProphetModel(TEST_DATA["daily"]["ts"], ProphetParams())

    def test_default_parameters(self) -> None:
        """
        Check that the default parameters are as expected. The expected values
        are hard coded.
        """
        expected_defaults = ProphetParams(
            growth="linear",
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto",
            holidays=None,
            seasonality_mode="additive",
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            cap=None,
            floor=None,
            custom_seasonalities=None,
        )

        actual_defaults = vars(ProphetParams())

        # Expected params should be valid
        expected_defaults.validate_params()

        for param, exp_val in vars(expected_defaults).items():
            msg = "param: {param}, expected default: {exp_val}, actual default: {val}".format(
                param=param, exp_val=exp_val, val=actual_defaults[param]
            )
            self.assertEqual(actual_defaults[param], exp_val, msg)

    def test_invalid_params(self) -> None:
        params = ProphetParams(growth="logistic")
        self.assertRaises(ValueError, params.validate_params)

        params = ProphetParams(
            custom_seasonalities=[
                {
                    "name": "monthly",
                },
            ],
        )
        self.assertRaises(ValueError, params.validate_params)

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "monthly",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                30,
                False,
                "MS",
                None,
                None,
                AIR_FCST_30_PROPHET,
            ],
            [
                "monthly, cap and floor",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params_cap_and_floor"],
                30,
                False,
                "MS",
                None,
                None,
                AIR_FCST_30_PROPHET_CAP_AND_FLOOR,
            ],
            [
                "daily, cap and floor",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_cap_and_floor"],
                30,
                False,
                "D",
                None,
                None,
                PEYTON_FCST_30_PROPHET_CAP_AND_FLOOR,
            ],
            [
                "monthly, historical",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params"],
                30,
                True,
                "MS",
                None,
                None,
                AIR_FCST_30_PROPHET_INCL_HIST,
            ],
            [
                "daily, historical",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
                15,
                True,
                "D",
                None,
                None,
                PEYTON_FCST_15_PROPHET_INCL_HIST,
            ],
            [
                "monthly, logistic with cap",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params_logistic_cap"],
                15,
                False,
                "MS",
                None,
                None,
                AIR_FCST_15_PROPHET_LOGISTIC_CAP,
            ],
            [
                "daily, logistic with cap",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_logistic_cap"],
                30,
                False,
                "D",
                None,
                None,
                PEYTON_FCST_30_PROPHET_DAILY_CAP,
            ],
            [
                "monthly, custom seasonality",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params_custom_seasonality"],
                30,
                False,
                "MS",
                None,
                None,
                AIR_FCST_30_PROPHET_CUSTOM_SEASONALITY,
            ],
            [
                "daily, custom seasonality",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params_custom_seasonality"],
                30,
                False,
                "D",
                None,
                None,
                PEYTON_FCST_30_PROPHET_CUSTOM_SEASONALITY,
            ],
            [
                "optional predict params",
                TEST_DATA["nonseasonal"]["ts"],
                TEST_DATA["nonseasonal"]["params"],
                30,
                False,
                None,
                TEST_DATA["nonseasonal"]["future_df"],
                True,
                NONSEASONAL_FCST_15_PROPHET_ARG_FUTURE,
            ],
        ]
    )
    def test_forecast(
        self,
        testcase_name: str,
        ts: TimeSeriesData,
        params: ProphetParams,
        steps: int,
        include_history: bool,
        freq: Optional[str],
        future: Optional[pd.DataFrame],
        raw: Optional[bool],
        truth: pd.DataFrame,
    ) -> None:
        kwargs = {}
        if freq is not None:
            kwargs["freq"] = freq
        if future is not None:
            kwargs["future"] = future
        if raw is not None:
            kwargs["raw"] = raw

        params.validate_params()
        m = ProphetModel(data=ts, params=params)
        m.fit()
        forecast_df = m.predict(steps=steps, include_history=include_history, **kwargs)

        assert_frame_equal(forecast_df, truth, check_exact=False)

    def test_multivar(self) -> None:
        # Prophet model does not support multivariate time series data
        self.assertRaises(
            ValueError,
            ProphetModel,
            TEST_DATA["multivariate"]["ts"],
            ProphetParams(),
        )

    def test_exec_plot(self):
        m = ProphetModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        m.fit()
        m.predict(steps=30, freq="MS")
        m.plot()

    def test_name(self):
        m = ProphetModel(TEST_DATA["daily"]["ts"], TEST_DATA["daily"]["params"])
        self.assertEqual(m.__str__(), "Prophet")

    def test_search_space(self):
        self.assertEqual(
            ProphetModel.get_parameter_search_space(),
            [
                {
                    "name": "seasonality_prior_scale",
                    "type": "choice",
                    "value_type": "float",
                    "values": list(np.logspace(-2, 1, 10, endpoint=True)),
                    "is_ordered": True,
                },
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
                    "name": "changepoint_prior_scale",
                    "type": "choice",
                    "value_type": "float",
                    "values": list(np.logspace(-3, 0, 10, endpoint=True)),
                    "is_ordered": True,
                },
                {
                    "name": "changepoint_range",
                    "type": "choice",
                    "value_type": "float",
                    "values": list(np.arange(0.8, 0.96, 0.01)),  # last value is 0.95
                    "is_ordered": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
