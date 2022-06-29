# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_air_passengers


class DataValidationTest(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()

    def test_data_validation(self) -> None:
        # add the extra data point to break the frequency.
        extra_point = pd.DataFrame(
            [["1900-01-01", 2], ["2020-01-01", 2]], columns=["time", "y"]
        )
        DATA = self.TSData.to_dataframe()
        data_with_extra_point = DATA.copy().append(extra_point)

        tsData_with_missing_point = TimeSeriesData(data_with_extra_point)

        tsData_with_missing_point.validate_data(
            validate_frequency=False, validate_dimension=False
        )
        tsData_with_missing_point.validate_data(
            validate_frequency=False, validate_dimension=True
        )
        with self.assertRaises(ValueError, msg="Frequency validation should fail."):
            tsData_with_missing_point.validate_data(
                validate_frequency=True, validate_dimension=False
            )
        with self.assertRaises(ValueError, msg="Frequency validation should fail."):
            tsData_with_missing_point.validate_data(
                validate_frequency=True, validate_dimension=True
            )


if __name__ == "__main__":
    unittest.main()
