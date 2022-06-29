# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, Union

import pandas as pd
import pandas.testing as pdt
from kats.compat import compat
from pandas.core.algorithms import safe_sort


version: compat.Version = compat.Version("pandas")


def convert_precision(check_less_precise: Union[bool, int], rtol: float) -> int:
    """Convert rtol and check_less_precise argument."""
    if check_less_precise is True:
        precise = 3
    elif check_less_precise is False:
        precise = 5
    else:
        precise = check_less_precise
    precise = min(precise, int(abs(math.log10(rtol))))
    return precise


def assert_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    check_dtype: bool = True,
    check_index_type: Union[bool, str] = "equiv",
    check_column_type: Union[bool, str] = "equiv",
    check_frame_type: bool = True,
    check_less_precise: Union[bool, int] = False,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = "DataFrame",
) -> None:
    kwargs: Dict[str, Any] = {
        "check_dtype": check_dtype,
        "check_index_type": check_index_type,
        "check_column_type": check_column_type,
        "check_frame_type": check_frame_type,
        "check_names": check_names,
        "by_blocks": by_blocks,
        "check_exact": check_exact,
        "check_datetimelike_compat": check_datetimelike_compat,
        "check_categorical": check_categorical,
        "check_like": check_like,
        "obj": obj,
    }
    if version < "1.1":
        kwargs["check_less_precise"] = convert_precision(check_less_precise, rtol)
    else:
        kwargs["check_freq"] = check_freq
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
        if version >= "1.2":
            kwargs["check_flags"] = check_flags
    pdt.assert_frame_equal(left, right, **kwargs)
    if (
        check_freq
        and version < "1.1"
        and isinstance(left.index, (pd.DatetimeIndex, pd.TimedeltaIndex))
        and hasattr(left.index, "freq")
        and hasattr(right.index, "freq")
    ):
        assert left.index.freq == right.index.freq, (left.index.freq, right.index.freq)
    if (
        check_flags
        and version < "1.2"
        and hasattr(left, "flags")
        and hasattr(right, "flags")
    ):
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"


def assert_series_equal(
    left: pd.Series,
    right: pd.Series,
    check_dtype: bool = True,
    check_index_type: Union[bool, str] = "equiv",
    check_series_type: Union[bool, str] = "equiv",
    check_less_precise: Union[bool, int] = False,
    check_names: bool = True,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = "Series",
    check_index: bool = True,
) -> None:
    kwargs: Dict[str, Any] = {
        "check_dtype": check_dtype,
        "check_index_type": check_index_type,
        "check_series_type": check_series_type,
        "check_names": check_names,
        "check_exact": check_exact,
        "check_datetimelike_compat": check_datetimelike_compat,
        "check_categorical": check_categorical,
        "obj": obj,
    }
    if version < "1.0.2":
        kwargs["check_less_precise"] = convert_precision(check_less_precise, rtol)
    elif version < "1.1":
        kwargs["check_less_precise"] = convert_precision(check_less_precise, rtol)
        kwargs["check_category_order"] = check_category_order
    elif version < "1.2":
        kwargs["check_category_order"] = check_category_order
        kwargs["check_freq"] = check_freq
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    elif version < "1.3":
        kwargs["check_category_order"] = check_category_order
        kwargs["check_freq"] = check_freq
        kwargs["check_flags"] = check_flags
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    else:
        kwargs["check_category_order"] = check_category_order
        kwargs["check_freq"] = check_freq
        kwargs["check_flags"] = check_flags
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
        kwargs["check_index"] = check_index

    pdt.assert_series_equal(left, right, **kwargs)
    if (
        check_freq
        and version < "1.1"
        and isinstance(left.index, (pd.DatetimeIndex, pd.TimedeltaIndex))
        and hasattr(left.index, "freq")
        and hasattr(right.index, "freq")
    ):
        # pyre-fixme[16]: `Index` has no attribute `freq`.
        assert left.index.freq == right.index.freq, (left.index.freq, right.index.freq)
    if (
        check_flags
        and version < "1.2"
        and hasattr(left, "flags")
        and hasattr(right, "flags")
    ):
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"
    if check_index and version < "1.3":
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_less_precise=check_less_precise,
            check_exact=check_exact,
            check_categorical=check_categorical,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )


def assert_index_equal(
    left: pd.Index,
    right: pd.Index,
    exact: Union[bool, str] = "equiv",
    check_names: bool = True,
    check_less_precise: Union[bool, int] = False,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = "Index",
) -> None:
    if not check_order and version < "1.2":
        left = pd.Index(safe_sort(left))
        right = pd.Index(safe_sort(right))

    kwargs: Dict[str, Any] = {
        "exact": exact,
        "check_names": check_names,
        "check_exact": check_exact,
        "check_categorical": check_categorical,
        "obj": obj,
    }
    if version < "1.1":
        kwargs["check_less_precise"] = convert_precision(check_less_precise, rtol)
    elif version < "1.2":
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    else:
        kwargs["check_order"] = check_order
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    pdt.assert_index_equal(left, right, **kwargs)
