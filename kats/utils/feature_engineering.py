# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Dict, Optional, Union

import numpy as np
import pandas as pd


# Map months to seasons.
# Equivalent to ((index.month + 1) % 12) // 3 but 15x faster.
# 0: winter, 1: spring, 2: summer, 3: fall
_SEASON_MAP = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
    12: 0,
}

# Map hours to daytime periods.
# 0: sleep, 1: breakfast, 2: morning work/lunch, 3: afternoon, 4: dinnner, 5: evening
_DAYTIME_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 4,
    19: 4,
    20: 5,
    21: 5,
    22: 5,
    23: 0,
}


def _map(index: Union[np.ndarray, pd.Int64Index], map: Dict[int, int]) -> np.ndarray:
    """Map values to other values efficiently.

    Args:
        index: the integer index.
    Returns:
        The mapped values.
    """
    # About 4x faster than .to_series().apply(lambda).
    # About 20x faster than to_series().replace().
    values = index.to_numpy()
    result = values.copy()
    for k, v in map.items():
        result[values == k] = v
    return result


def date_features(s: pd.Series, result: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute date features for each row of a time series.

    Args:
        s: univariate time series values indexed by date.
        result: the result DataFrame to put the features into. If None, create
            one using the same index and values as s.

    Returns:
        The result with date features added.
    """
    if result is None:
        result = pd.DataFrame(s, copy=False)
    index = cast(pd.DatetimeIndex, s.index)

    result["year"] = index.year
    result["month"] = index.month
    result["day"] = index.day
    result["dayofweek"] = index.dayofweek
    result["dayofyear"] = index.dayofyear
    result["quarter"] = index.quarter
    result["season"] = _map(index.month, _SEASON_MAP)
    result["weekofyear"] = index.weekofyear
    try:
        # Work around numpy Deprecation Warning about parsing timezones
        # by converting to UTC and removing the tz info.
        dates = index.tz_convert(None).to_numpy()
    except TypeError:
        # No timezone.
        dates = index.to_numpy()
    first_of_month = pd.to_datetime(dates.astype("datetime64[M]"))
    week_of_month = np.ceil((first_of_month.dayofweek + index.day) / 7.0)
    result["weekofmonth"] = week_of_month.astype(int)
    # result["is_holiday"] = ?
    # result["holiday_types"] = ?
    result["is_weekend"] = index.dayofweek >= 5
    result["is_leap_year"] = index.is_leap_year
    result["is_leap_day"] = (index.month == 2) & (index.day == 29)
    result["is_month_end"] = index.is_month_end
    result["is_quarter_end"] = index.is_month_end & (index.month % 4 == 3)

    return result


def time_features(s: pd.Series, result: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute time features for each row of a time series.

    Args:
        s: univariate time series values indexed by time.
        result: the result DataFrame to put the features into. If None, create
            one using the same index and values as s.

    Returns:
        The result with time features added.
    """
    if result is None:
        result = pd.DataFrame(s, copy=False)
    index = cast(pd.DatetimeIndex, s.index)

    result["hour"] = index.hour
    result["minute"] = index.minute
    result["second"] = index.second
    result["milliseconds"] = index.microsecond / 1000
    result["quarterhour"] = index.minute // 15 + 1
    result["hourofweek"] = index.dayofweek * 24 + index.hour
    result["daytime"] = _map(index.hour, _DAYTIME_MAP)

    # No vectorized methods for these:
    dst = np.zeros(len(result))
    utcoffset = np.zeros(len(result))
    for i, dt in enumerate(index):
        try:
            dst[i] = dt.dst().total_seconds()
            utcoffset[i] = dt.utcoffset().total_seconds()
        except AttributeError:
            # No timezone
            dst[i] = np.nan
            utcoffset[i] = np.nan
    result["dst"] = dst
    result["utcoffset"] = utcoffset
    return result


def datetime_features(
    s: pd.Series, result: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute date and time features for each row of a time series.

    Args:
        s: univariate time series values indexed by datetime.
        result: the result DataFrame to put the features into. If None, create
            one using the same index and values as s.

    Returns:
        The result with date and time features added.
    """
    result = date_features(s, result)
    return time_features(s, result)


def timestamp_datetime_features(t: pd.Timestamp) -> Dict[str, Any]:
    """Compute date/time features for a single timestamp.

    Convenience method; equivalent to datetime_features() on a singleton series.

    Args:
        t: the timestamp

    Returns:
        A dictionary of features.
    """
    df = datetime_features(pd.Series([0], index=[t], name="_"))
    result = df.to_dict(orient="records")[0]
    del result["_"]
    return result


def circle_encode(
    ts: pd.DataFrame, features: Dict[str, int], modulo: bool = False
) -> pd.DataFrame:
    """Circularly encode features.

    Args:
        ts: the multivariate time series data.
        features: dictionary of features to encode and the periodicity of those
                  features.
        modulo: if True, first reduce the features modulo the periodicity.

    Returns:
        The circularly encoded features. Each feature f generates two columns,
        f_cos and f_sin.
    """
    result = pd.DataFrame(index=ts.index)
    pi2 = np.pi * 2.0
    if modulo:
        for feat, period in features.items():
            f = (ts[feat] % period) * (pi2 / period)
            result[f"{feat}_cos"] = np.cos(f)
            result[f"{feat}_sin"] = np.sin(f)
    else:
        for feat, period in features.items():
            f = ts[feat] * (pi2 / period)
            result[f"{feat}_cos"] = np.cos(f)
            result[f"{feat}_sin"] = np.sin(f)
    return result
