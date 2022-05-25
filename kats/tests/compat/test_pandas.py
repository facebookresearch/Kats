# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from typing import Any, Dict
from unittest.mock import ANY, patch

import pandas as pd
from kats.compat import compat, pandas


class TestPandas(unittest.TestCase):
    assert_frame_equal_args: Dict[str, Any] = {}
    assert_series_equal_args: Dict[str, Any] = {}
    assert_index_equal_args: Dict[str, Any] = {}

    def setUp(self) -> None:
        for method in (
            "assert_frame_equal",
            "assert_series_equal",
            "assert_index_equal",
        ):
            args = {}
            setattr(self, f"{method}_args", args)
            for k in inspect.signature(getattr(pandas, method)).parameters:
                if k not in {"left", "right"}:
                    args[k] = ANY

    def test_version(self) -> None:
        self.assertTrue(pandas.version == compat.Version("pandas"))

    @patch("kats.compat.pandas.version", compat.Version("1.0"))
    @patch("kats.compat.pandas.pdt.assert_frame_equal")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_assert_frame_equal_10(self, assert_frame_equal: Any) -> None:
        assert_frame_equal.return_value = False
        df = pd.DataFrame()
        result = pandas.assert_frame_equal(
            df, df, check_less_precise=2, check_flags=False, check_freq=False, rtol=0.01
        )
        self.assertFalse(result)
        args = dict(self.assert_frame_equal_args)
        args["check_less_precise"] = 2
        # drop args from other versions
        del args["rtol"]
        del args["atol"]
        del args["check_freq"]
        del args["check_flags"]
        assert_frame_equal.assert_called_once_with(df, df, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.1"))
    @patch("kats.compat.pandas.pdt.assert_frame_equal")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_assert_frame_equal_11(self, assert_frame_equal: Any) -> None:
        assert_frame_equal.return_value = False
        df = pd.DataFrame()
        result = pandas.assert_frame_equal(
            df, df, check_less_precise=2, check_flags=False, rtol=0.01
        )
        self.assertFalse(result)
        args = dict(self.assert_frame_equal_args)
        args["rtol"] = 0.01
        # drop args from other versions
        del args["check_less_precise"]
        del args["check_flags"]
        assert_frame_equal.assert_called_once_with(df, df, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.2"))
    @patch("kats.compat.pandas.pdt.assert_frame_equal")
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def test_assert_frame_equal_12(self, assert_frame_equal: Any) -> None:
        assert_frame_equal.return_value = False
        df = pd.DataFrame()
        result = pandas.assert_frame_equal(df, df, check_less_precise=2, rtol=0.01)
        self.assertFalse(result)
        args = dict(self.assert_frame_equal_args)
        # drop args from other versions
        del args["check_less_precise"]
        args["rtol"] = 0.01
        assert_frame_equal.assert_called_once_with(df, df, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.0"))
    @patch("kats.compat.pandas.pdt.assert_index_equal")
    @patch("kats.compat.pandas.pdt.assert_series_equal")
    def test_assert_series_equal_10(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_series_equal: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_index_equal: Any,
    ) -> None:
        assert_series_equal.return_value = False
        s = pd.Series(dtype=int)
        result = pandas.assert_series_equal(
            s, s, check_less_precise=2, check_flags=False, check_freq=False, rtol=0.01
        )
        self.assertFalse(result)
        args = dict(self.assert_series_equal_args)
        args["check_less_precise"] = 2
        # drop args from other versions
        del args["check_category_order"]
        del args["check_freq"]
        del args["check_flags"]
        del args["rtol"]
        del args["atol"]
        del args["check_index"]
        assert_series_equal.assert_called_once_with(s, s, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.0.2"))
    @patch("kats.compat.pandas.pdt.assert_index_equal")
    @patch("kats.compat.pandas.pdt.assert_series_equal")
    def test_assert_series_equal_102(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_series_equal: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_index_equal: Any,
    ) -> None:
        # Check that on pandas 1.0.2, `check_less_precise` is passed and not later args
        assert_series_equal.return_value = False
        s = pd.Series(dtype=int)
        result = pandas.assert_series_equal(
            s,
            s,
            check_less_precise=2,
            check_category_order=True,
            check_flags=False,
            check_freq=False,
            rtol=0.01,
        )
        self.assertFalse(result)
        args = dict(self.assert_series_equal_args)
        args["check_less_precise"] = 2
        args["check_category_order"] = True
        # drop args from other versions
        del args["check_freq"]
        del args["check_flags"]
        del args["rtol"]
        del args["atol"]
        del args["check_index"]
        assert_series_equal.assert_called_once_with(s, s, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.1"))
    @patch("kats.compat.pandas.pdt.assert_index_equal")
    @patch("kats.compat.pandas.pdt.assert_series_equal")
    def test_assert_series_equal_11(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_series_equal: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_index_equal: Any,
    ) -> None:
        # Check that on pandas 1.1, `check_less_precise` is passed and not later args
        assert_series_equal.return_value = False
        s = pd.Series(dtype=int)
        result = pandas.assert_series_equal(
            s,
            s,
            check_less_precise=2,
            check_category_order=True,
            check_flags=False,
            check_freq=False,
            rtol=0.01,
        )
        self.assertFalse(result)
        args = dict(self.assert_series_equal_args)
        args["check_category_order"] = True
        args["check_freq"] = False
        args["rtol"] = 0.01
        # drop args from other versions
        del args["check_less_precise"]
        del args["check_flags"]
        del args["check_index"]
        assert_series_equal.assert_called_once_with(s, s, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.2"))
    @patch("kats.compat.pandas.pdt.assert_index_equal")
    @patch("kats.compat.pandas.pdt.assert_series_equal")
    def test_assert_series_equal_12(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_series_equal: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_index_equal: Any,
    ) -> None:
        # Check that on pandas 1.2, `check_less_precise` is not passed and not later args
        assert_series_equal.return_value = False
        s = pd.Series(dtype=int)
        result = pandas.assert_series_equal(
            s,
            s,
            check_less_precise=2,
            check_category_order=True,
            check_flags=False,
            check_freq=False,
            rtol=0.01,
        )
        self.assertFalse(result)
        args = dict(self.assert_series_equal_args)
        args["check_category_order"] = True
        args["check_freq"] = False
        args["check_flags"] = False
        args["rtol"] = 0.01
        # drop args from other versions
        del args["check_less_precise"]
        del args["check_index"]
        assert_series_equal.assert_called_once_with(s, s, **args)

    @patch("kats.compat.pandas.version", compat.Version("1.3"))
    @patch("kats.compat.pandas.pdt.assert_index_equal")
    @patch("kats.compat.pandas.pdt.assert_series_equal")
    def test_assert_series_equal_13(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_series_equal: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        assert_index_equal: Any,
    ) -> None:
        assert_series_equal.return_value = False
        s = pd.Series(dtype=int)
        result = pandas.assert_series_equal(
            s,
            s,
            check_less_precise=2,
            check_category_order=True,
            check_flags=False,
            check_freq=False,
            rtol=0.01,
            check_index=False,
        )
        self.assertFalse(result)
        args = dict(self.assert_series_equal_args)
        args["check_category_order"] = True
        args["check_freq"] = False
        args["check_flags"] = False
        args["rtol"] = 0.01
        args["check_index"] = False
        # drop args from other versions
        del args["check_less_precise"]
        assert_series_equal.assert_called_once_with(s, s, **args)


if __name__ == "__main__":
    unittest.main()
