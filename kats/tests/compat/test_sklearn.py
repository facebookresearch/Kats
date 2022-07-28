# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from kats.compat import compat, sklearn


class TestSklearn(unittest.TestCase):
    def setUp(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 1.0])
        sample_weight = np.array([1.0, 1.0])
        self.orig_args = [y_true, y_pred, sample_weight]
        self.args = [y_true, y_pred]
        self.kwargs = {
            "sample_weight": sample_weight,
            "multioutput": "uniform_average",
            "squared": True,
        }

    def test_version(self) -> None:
        self.assertTrue(sklearn.version == compat.Version("sklearn"))

    @patch("kats.compat.sklearn.mse")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_mean_squared_error(self, sklearn_mse: Any) -> None:
        sklearn_mse.return_value = 42.0
        result = sklearn.mean_squared_error(*self.orig_args)
        self.assertEqual(42.0, result)
        sklearn_mse.assert_called_once_with(*self.args, **self.kwargs)

    @patch("kats.compat.sklearn.mse")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_root_mean_squared_error(self, sklearn_mse: Any) -> None:
        sklearn_mse.return_value = 42.0
        result = sklearn.mean_squared_error(*self.orig_args, squared=False)
        # sqrt part of the mocked call
        self.assertEqual(42.0, result)
        kwargs = dict(self.kwargs)
        kwargs["squared"] = False
        sklearn_mse.assert_called_once_with(*self.args, **kwargs)

    @patch("kats.compat.sklearn.version", compat.Version("0.24"))
    @patch("kats.compat.sklearn.msle")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_mean_squared_log_error(self, sklearn_msle: Any) -> None:
        sklearn_msle.return_value = 42.0
        result = sklearn.mean_squared_log_error(*self.orig_args)
        self.assertEqual(42.0, result)
        kwargs = dict(self.kwargs)
        del kwargs["squared"]
        sklearn_msle.assert_called_once_with(*self.args, **kwargs)

    @patch("kats.compat.sklearn.version", compat.Version("0.24"))
    @patch("kats.compat.sklearn.msle")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_root_mean_squared_log_error_24(self, sklearn_msle: Any) -> None:
        sklearn_msle.return_value = 36.0
        result = sklearn.mean_squared_log_error(*self.orig_args, squared=False)
        # sqrt applied after-the-fact
        self.assertEqual(6.0, result)
        kwargs = dict(self.kwargs)
        del kwargs["squared"]
        sklearn_msle.assert_called_once_with(*self.args, **kwargs)

    @patch("kats.compat.sklearn.version", compat.Version("1.0"))
    @patch("kats.compat.sklearn.msle")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_root_mean_squared_log_error(self, sklearn_msle: Any) -> None:
        sklearn_msle.return_value = 36.0
        result = sklearn.mean_squared_log_error(*self.orig_args, squared=False)
        # sqrt part of the mocked call
        self.assertEqual(36.0, result)
        kwargs = dict(self.kwargs)
        kwargs["squared"] = False
        sklearn_msle.assert_called_once_with(*self.args, **kwargs)


if __name__ == "__main__":
    unittest.main()
