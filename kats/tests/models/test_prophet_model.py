# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data, load_air_passengers
from kats.models.prophet import ProphetModel, ProphetParams
from parameterized import parameterized


TEST_DATA = {
    "nonseasonal": {
        "ts": TimeSeriesData(
            pd.DataFrame(
                {
                    "time": pd.date_range("1960-12-01", "1963-01-01", freq="m"),
                    "y": np.random.randn(25),
                }
            )
        ),
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
    def test_params(self) -> None:
        # Test default value
        params = ProphetParams()
        params.validate_params()

        # Test invalid params
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
                None,
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
                None,
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
                None,
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
                None,
            ],
            [
                "daily, historical",
                TEST_DATA["daily"]["ts"],
                TEST_DATA["daily"]["params"],
                30,
                True,
                "D",
                None,
                None,
                None,
            ],
            [
                "monthly, logistic with cap",
                TEST_DATA["monthly"]["ts"],
                TEST_DATA["monthly"]["params_logistic_cap"],
                30,
                False,
                "MS",
                None,
                None,
                None,
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
                None,
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
                None,
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
                None,
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
        truth: TimeSeriesData,
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
        m.predict(steps=steps, include_history=include_history, **kwargs)
        # TODO: validate results

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
