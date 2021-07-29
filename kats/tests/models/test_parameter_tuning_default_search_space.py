# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

from kats.models.arima import ARIMAModel
from kats.models.holtwinters import HoltWintersModel
from kats.models.linear_model import LinearModel
from kats.models.prophet import ProphetModel
from kats.models.quadratic_model import QuadraticModel
from kats.models.sarima import SARIMAModel
from kats.models.theta import ThetaModel
from kats.models.var import VARModel
from kats.utils.time_series_parameter_tuning import (
    TimeSeriesParameterTuning,
)


class TestParameterTuningDefaultSearchSpace(TestCase):
    def test_parameter_tuning_default_search_space_arima(self) -> None:
        search_space = ARIMAModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_prophet(self) -> None:
        search_space = ProphetModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_linear_model(self) -> None:
        search_space = LinearModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_quadratic_model(self) -> None:
        search_space = QuadraticModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_sarima_model(self) -> None:
        search_space = SARIMAModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_holtwinters_model(self) -> None:
        search_space = HoltWintersModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)

    def test_parameter_tuning_default_search_space_var_model(self) -> None:
        self.assertRaises(NotImplementedError, VARModel.get_parameter_search_space)

    def test_parameter_tuning_default_search_space_theta_model(self) -> None:
        search_space = ThetaModel.get_parameter_search_space()
        TimeSeriesParameterTuning.validate_parameters_format(search_space)


if __name__ == "__main__":
    unittest.main()
