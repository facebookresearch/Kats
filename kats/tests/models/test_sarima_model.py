# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import unittest
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.sarima import SARIMAModel, SARIMAParams


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class SARIMAModelTest(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

        DATA_daily = load_data("peyton_manning.csv")
        DATA_daily.columns = ["time", "y"]
        self.TSData_daily = TimeSeriesData(DATA_daily)

        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_fit_forecast(self) -> None:
        def setUp(self):
            DATA = load_data("air_passengers.csv")
            DATA.columns = ["time", "y"]
            self.TSData = TimeSeriesData(DATA)

        params = SARIMAParams(
            p=2,
            d=1,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
        )
        params.validate_params()
        m = SARIMAModel(self.TSData, params)
        m.fit(
            start_params=None,
            # pyre-fixme[6]: Expected `bool` for 2nd param but got `None`.
            transformed=None,
            # pyre-fixme[6]: Expected `bool` for 3rd param but got `None`.
            includes_fixed=None,
            cov_type=None,
            cov_kwds=None,
            method="lbfgs",
            maxiter=50,
            # pyre-fixme[6]: Expected `bool` for 8th param but got `int`.
            full_output=1,
            disp=False,
            callback=None,
            return_params=False,
            optim_score=None,
            # pyre-fixme[6]: Expected `bool` for 13th param but got `None`.
            optim_complex_step=None,
            optim_hessian=None,
            low_memory=False,
        )
        m.predict(steps=30, freq="MS")
        m.plot()

        m_daily = SARIMAModel(self.TSData_daily, params)
        m_daily.fit()
        m_daily.predict(steps=30, freq="D")
        m.plot()

    def test_exog_forecast(self) -> None:
        # Prepping data
        steps = 10

        DATA_multi = self.TSData_multi.to_dataframe()
        endog = DATA_multi["0"][:-steps]
        time = self.TSData_multi.time_to_index()[:-steps]

        exog = DATA_multi["1"][:-steps].values
        fcst_exog = DATA_multi["1"][-steps:].values  # exog to be used for predictions

        ts_data = TimeSeriesData(value=endog, time=time)

        params = SARIMAParams(
            p=2,
            d=1,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
            exog=exog,
        )

        params.validate_params()
        m = SARIMAModel(ts_data, params)
        m.fit(
            start_params=None,
            # pyre-fixme[6]: Expected `bool` for 2nd param but got `None`.
            transformed=None,
            # pyre-fixme[6]: Expected `bool` for 3rd param but got `None`.
            includes_fixed=None,
            cov_type=None,
            cov_kwds=None,
            method="lbfgs",
            maxiter=50,
            # pyre-fixme[6]: Expected `bool` for 8th param but got `int`.
            full_output=1,
            disp=False,
            callback=None,
            return_params=False,
            optim_score=None,
            # pyre-fixme[6]: Expected `bool` for 13th param but got `None`.
            optim_complex_step=None,
            optim_hessian=None,
            low_memory=False,
        )
        m.predict(steps=steps, exog=fcst_exog, freq="D")
        m.plot()

        # should raise a value error if exogenous variables aren't used to predict
        with self.assertRaises(ValueError):
            m.predict(steps=steps, freq="D")

    def test_others(self) -> None:
        params = SARIMAParams(
            p=2,
            d=1,
            q=1,
            trend="ct",
            seasonal_order=(1, 0, 1, 12),
            enforce_invertibility=False,
            enforce_stationarity=False,
        )
        params.validate_params()
        m = SARIMAModel(self.TSData, params)

        # test search space
        self.assertEqual(
            m.get_parameter_search_space(),
            [
                {
                    "name": "p",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "d",
                    "type": "choice",
                    "values": list(range(1, 3)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "q",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "seasonal_order",
                    "type": "choice",
                    "values": [
                        (1, 0, 1, 7),
                        (1, 0, 2, 7),
                        (2, 0, 1, 7),
                        (2, 0, 2, 7),
                        (1, 1, 1, 7),
                        (0, 1, 1, 7),
                    ],
                    # Note: JSON representation must be 'int', 'float', 'bool' or 'str'.
                    # so we use 'str' here instead of 'Tuple'
                    # when doing HPT, we need to convert it back to tuple
                    "value_type": "str",
                },
                {
                    "name": "trend",
                    "type": "choice",
                    "values": ["n", "c", "t", "ct"],
                    "value_type": "str",
                },
            ],
        )

        # test __str__ method
        self.assertEqual(m.__str__(), "SARIMA")

        # test input error
        self.assertRaises(
            ValueError,
            SARIMAModel,
            self.TSData_multi,
            params,
        )


if __name__ == "__main__":
    unittest.main()
