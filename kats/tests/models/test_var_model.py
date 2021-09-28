# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models.var import VARModel, VARParams


class testVARModel(TestCase):
    def setUp(self):
        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)

    def test_fit_forecast(self) -> None:
        params = VARParams()
        m = VARModel(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True)

    def test_model_wrong_param(self) -> None:
        params = VARParams()
        input_data = TimeSeriesData(pd.DataFrame())
        with self.assertRaises(ValueError):
            m = VARModel(input_data, params)
            m.fit()
            m.predict(steps=30, include_history=True)

    @patch("pandas.concat")
    def test_predict_exception(self, mock_obj) -> None:
        mock_obj.side_effect = Exception
        with self.assertRaisesRegex(
            Exception, "^Fail to generate in-sample forecasts for historical data"
        ):
            params = VARParams()
            m = VARModel(self.TSData_multi, params)
            m.fit()
            m.predict(steps=30, include_history=True)

    def test_trivial_path(self) -> None:
        params = VARParams()
        params.validate_params()
        m = VARModel(self.TSData_multi, params)
        print(m)
        with self.assertRaises(NotImplementedError):
            VARModel.get_parameter_search_space()


if __name__ == "__main__":
    unittest.main()
