# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains some of the key data structures in the Kats library,
including :class:`TimeSeriesData`, :class:`TimeSeriesChangePoint`, and
:class:`TimeSeriesIterator`.

:class:`TimeSeriesChangePoint` is the return type of many of the Kats detection
algorithms.

:class:`TimeSeriesData` is the fundamental data structure in the Kats library,
that gives uses access to a host of forecasting, detection, and utility
algorithms right at the user's fingertips.
"""

from __future__ import annotations

import copy
import datetime
import logging
from collections.abc import Iterable
from enum import Enum, auto, unique
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.tseries.frequencies import to_offset

FigSize = Tuple[int, int]


# Constants
DEFAULT_TIME_NAME = "time"  # Default name for the time column in TimeSeriesData
DEFAULT_VALUE_NAME = "value"  # Default name for the value column in TimeSeriesData
PREFIX_OP_1 = "_kats.1"  # Internal prefix used when merging two TimeSeriesData objects
PREFIX_OP_2 = (
    "_kats.2"  # Second internal prefix used when merging two TimeSeriesData objects
)
INTERPOLATION_METHODS = {
    "linear",
    "bfill",
    "ffill",
}  # List of possible interpolation methods


def _log_error(msg: str) -> ValueError:
    logging.error(msg)
    return ValueError(msg)


class TimeSeriesChangePoint:
    """Object returned by detector classes.

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
    """

    def __init__(self, start_time, end_time, confidence: float) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._confidence = confidence

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def confidence(self) -> float:
        return self._confidence

    def __repr__(self):
        return (
            f"TimeSeriesChangePoint(start_time: {self.start_time}, end_time: "
            f"{self.end_time}, confidence: {self.confidence})"
        )

    def __str__(self):
        return (
            f"TimeSeriesChangePoint(start_time: {self.start_time}, end_time: "
            f"{self.end_time}, confidence: {self.confidence})"
        )


class TimeSeriesData:
    """The fundamental Kats data structure to store a time series.

    In order to access much of the functionality in the Kats library, users
    must initialize the :class:`TimeSeriesData` class with their data first.

    Initialization. :class:`TimeSeriesData` can be initialized from the
    following data sources:

        - `pandas.DataFrame`
        - `pandas.Series`
        - `pandas.DatetimeIndex`

    Typical usage example for initialization:

    >>> import pandas as pd
    >>> df = pd.read_csv("/kats/data/air_passengers.csv")
    >>> ts = TimeSeriesData(df=df, time_col_name="ds")

    Initialization arguments (all optional, but must choose one way to
    initialize e.g. `pandas.DataFrame`):

    - df: A `pandas.DataFrame` storing the time series (default None).
    - sort_by_time: A boolean indicating whether the :class:`TimeSeriesData`
        should be sorted by time (default True).
    - time: a `pandas.Series` or `pandas.DatetimeIndex` storing the time
        values (default None).
    - value: A pandas.Series or pandas.DataFrame storing the series value(s)
        (default None).
    - time_col_name: A string representing the value of the time column (
        default "time")
    - date_format: A string specifying the format of the date/time in the
        time column. Useful for faster parsing, and required
        `pandas.to_datetime()` cannot parse the column otherwise (default None).
    - use_unix_time: A boolean indicating if the time is represented as
        unix time (default False).
    - unix_time_units: A string indicating the units of the unix time -- only
        used if `use_unix_time=True` (default "ns").
    - tz: A string representing the timezone of the time values (default None).
    - tz_ambiguous: A string representing how to handle ambiguous timezones
        (default "raise").
    - tz_nonexistant: A string representing how to handle nonexistant timezone
        values (default "raise").

    Raises:
      ValueError: Invalid params passed when trying to create the
        :class:`TimeSeriesData`.

    Operations. Many operations that you can do with `pandas.DataFrame` objects
    are also applicable to :class:`TimeSeriesData`. For example:

      >>> ts[0:2] # Slicing
      >>> ts_1 == ts_2 # Equality
      >>> ts_1.extend(ts_2) # Extend
      >>> ts.plot(cols=["y"]) # Visualize

    Utility Functions. Many utility functions for converting
    :class:`TimeSeriesData` objects to other common data structures exist.
    For example:

      >>> ts.to_dataframe() # Convert to pandas.DataFrame
      >>> ts.to_array() # Convert to numpy.ndarray

    Attributes:
      time: A `pandas.Series` object storing the time values of the time
        series.
      value: A `pandas.Series` (if univariate) or `pandas.DataFrame` (if
        multivariate) object storing the values of each field in the time
        series.
      min: A float or `pandas.Series` representing the min value(s) of the
        time series.
      max: A float or `pandas.Series` representing the max value(s) of the
        time series.
    """

    _min: float = np.nan
    _max: float = np.nan

    def __init__(  # noqa C901
        self,
        df: Optional[pd.DataFrame] = None,
        sort_by_time: bool = True,
        time: Union[pd.Series, pd.DatetimeIndex, None] = None,
        value: Union[pd.Series, pd.DataFrame, None] = None,
        time_col_name: str = DEFAULT_TIME_NAME,
        date_format: Optional[str] = None,
        use_unix_time: bool = False,
        unix_time_units: str = "ns",
        tz: Optional[str] = None,
        tz_ambiguous: Union[str, np.ndarray] = "raise",
        tz_nonexistent: str = "raise",
    ) -> None:
        """Initializes :class:`TimeSeriesData` class with arguments provided."""
        self.time_col_name = time_col_name

        # If DataFrame is passed
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                msg = (
                    "Argument df needs to be a pandas.DataFrame but is of type "
                    f"{type(df)}."
                )
                raise _log_error(msg)
            # If empty DataFrame is passed then create an empty object
            if df.empty:
                self._time = pd.Series([], name=time_col_name, dtype=float)
                self._value = pd.Series([], name=DEFAULT_VALUE_NAME, dtype=float)
                logging.warning("Initializing empty TimeSeriesData object")
            # Otherwise initialize TimeSeriesData from DataFrame
            else:
                # Ensuring time column is present in DataFrame
                if self.time_col_name not in df.columns:
                    msg = f"Time column {self.time_col_name} not in DataFrame"
                    raise _log_error(msg)
                # Parsing time column into correct format
                df = df.copy()
                df.reset_index(inplace=True, drop=True)
                df[self.time_col_name] = self._set_time_format(
                    series=df[self.time_col_name],
                    date_format=date_format,
                    use_unix_time=use_unix_time,
                    unix_time_units=unix_time_units,
                    tz=tz,
                    tz_ambiguous=tz_ambiguous,
                    tz_nonexistent=tz_nonexistent,
                )
                # Sorting by time if necessary
                if sort_by_time:
                    df.sort_values(self.time_col_name, inplace=True)
                    df.reset_index(inplace=True, drop=True)
                else:
                    logging.warning(
                        (
                            "Please make sure the time series is sorted by time or "
                            "set 'sort_by_time' as True."
                        )
                    )
                self._time = df[self.time_col_name]
                self._value = df[[x for x in df.columns if x != self.time_col_name]]
                self._set_univariate_values_to_series()

        # If separate objects are passed
        elif time is not None and value is not None:
            if not (
                (
                    isinstance(time, pd.core.series.Series)
                    or isinstance(time, pd.DatetimeIndex)
                )
                and (
                    isinstance(value, pd.core.series.Series)
                    or isinstance(value, pd.DataFrame)
                )
            ):
                msg = (
                    f"Invalid types: time is {type(time)} when it must be a "
                    + "pandas.Series or pandas.DatetimeIndex and value is "
                    + f"{type(value)} when it must be a pandas.DataFrame or "
                    + "pandas.Series"
                )
                raise _log_error(msg)
            if isinstance(time, pd.DatetimeIndex):
                self._time = pd.Series(time)
            else:
                self._time = cast(pd.Series, time.reset_index(drop=True))
            self._value = value.reset_index(drop=True)
            self._set_univariate_values_to_series()
            # Set time col name
            if time.name:
                self.time_col_name = time.name
            else:
                self._time.rename(DEFAULT_TIME_NAME, inplace=True)
            # Make sure the value series has a name
            if (
                isinstance(self._value, pd.core.series.Series)
                and self._value.name is None
            ):
                # pyre-fixme[6]: Expected `Union[typing.Callable[[Optional[typing.Has...
                self._value.rename(DEFAULT_VALUE_NAME, inplace=True)
            # Checking for emptiness
            if self.time.empty and self.value.empty:
                logging.warning("Initializing empty TimeSeriesData object")
                self.time = pd.Series([], name=time_col_name)
                if isinstance(value, pd.DataFrame):
                    self.value = pd.Series([], name=DEFAULT_VALUE_NAME)
                else:
                    self.value = pd.Series(
                        [], name=value.name if value.name else DEFAULT_VALUE_NAME
                    )
            # Raise exception if only one of time and value is empty
            elif self.time.empty or self.value.empty:
                msg = "One of time or value is empty while the other is not"
                raise _log_error(msg)
            # If time values are passed then standardizing format
            else:
                self.time = cast(
                    pd.Series,
                    self._set_time_format(
                        self.time,
                        date_format=date_format,
                        use_unix_time=use_unix_time,
                        unix_time_units=unix_time_units,
                        tz=tz,
                        tz_ambiguous=tz_ambiguous,
                        tz_nonexistent=tz_nonexistent,
                    ).reset_index(drop=True),
                )

        # If None is passed
        elif not time and not value:
            self._time = pd.Series([], name=time_col_name)
            self._value = pd.Series([], name=DEFAULT_VALUE_NAME)
            logging.warning("Initializing empty TimeSeriesData object")

        # Error if only one of time or value is None
        else:
            msg = "One of time or value is empty while the other is not"
            raise _log_error(msg)

        # Validate values
        if not self.value.empty and not (
            (
                isinstance(self.value, pd.core.series.Series)
                and is_numeric_dtype(self.value)
            )
            or (
                isinstance(self.value, pd.DataFrame)
                and all(is_numeric_dtype(self.value[col]) for col in self.value)
            )
        ):
            msg = f"Time series data is type {self.value.dtype} but must be numeric"
            raise _log_error(msg)

        self._calc_min_max_values()

    @property
    def time(self) -> pd.Series:
        """Returns the time values of the series.

        Returns:
          A `pandas.Series` representing the time values of the time series.
        """
        return self._time

    @time.setter
    def time(self, time_values: pd.Series) -> None:
        """Sets the time values of the :class:`TimeSeriesData`.

        Args:
          time_values. A `pandas.Series` with the updated time values.
        """
        self._time = time_values

    @property
    def value(self) -> Union[pd.Series, pd.DataFrame]:
        """Returns the value(s) of the series.

        Returns:
          A `pandas.Series` or `pandas.DataFrame` representing the value(s) of the
          time series.
        """
        return self._value

    @value.setter
    def value(self, values: Union[pd.Series, pd.DataFrame]) -> None:
        """Sets the value(s) of the :class:`TimeSeriesData.`

        Args:
          values: A `pandas.Series` or `pandas.DataFrame` with the updated
          values(s).
        """

        self._value = values
        # updates for min/max values are necessary once values are updated
        self._calc_min_max_values()

    @property
    def min(self) -> Union[pd.Series, float]:
        """Returns the min value(s) of the series.

        Returns:
          A `pandas.Series` or float representing the min value(s) of the
          time series.
        """
        return self._min

    @property
    def max(self) -> Union[pd.Series, float]:
        """Returns the max value(s) of the series.

        Returns:
          A `pandas.Series` or float representing the max value(s) of the
          time series.
        """
        return self._max

    def __eq__(self, other: object) -> bool:
        # Currently "__eq__" only works with other TimeSeriesData objects.
        if not isinstance(other, TimeSeriesData):
            return NotImplemented

        # Check if time values are equal
        try:
            assert_series_equal(self.time, other.time, check_dtype=False)
        except AssertionError:
            return False

        # If both objects are univariate
        if isinstance(self.value, pd.Series) and isinstance(other.value, pd.Series):
            # Check if value Series are equal
            try:
                assert_series_equal(self.value, other.value, check_dtype=False)
            except AssertionError:
                return False
        # If both objects are multivariate
        elif isinstance(self.value, pd.DataFrame) and isinstance(
            other.value, pd.DataFrame
        ):
            # Check if value DataFrames are equal (ignore column order)
            try:
                assert_frame_equal(
                    self.value.sort_index(axis=1),
                    other.value.sort_index(axis=1),
                    check_names=True,
                    check_dtype=False,
                )
            except AssertionError:
                return False
        # Otherwise one TimeSeriesData is univariate and the other is multivariate
        else:
            return False

        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __sub__(self, other: object) -> TimeSeriesData:
        return self._perform_op(other, OperationsEnum.SUB)

    def __truediv__(self, other: object) -> TimeSeriesData:
        return self._perform_op(other, OperationsEnum.DIV)

    def __add__(self, other: object) -> TimeSeriesData:
        return self._perform_op(other, OperationsEnum.ADD)

    def __mul__(self, other: object) -> TimeSeriesData:
        return self._perform_op(other, OperationsEnum.MUL)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, sliced) -> TimeSeriesData:
        if isinstance(sliced, str) or (
            isinstance(sliced, Iterable) and all(isinstance(s, str) for s in sliced)
        ):
            return TimeSeriesData(
                time=self.time,
                value=self.value[sliced],
                time_col_name=self.time_col_name,
            )
        return TimeSeriesData(
            time=self.time[sliced],
            value=self.value[sliced],
            time_col_name=self.time_col_name,
        )

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    def _repr_html_(self) -> str:
        return self.to_dataframe()._repr_html_()

    def _set_univariate_values_to_series(self):
        # This hack is required since downstream models are expecting value of
        # type Series in case of univariate time series
        if isinstance(self.value, pd.DataFrame) and self.value.shape[1] == 1:
            self.value = self.value.iloc[:, 0]

    def is_empty(self) -> bool:
        """Checks if the :class:`TimeSeriesData` is empty.

        Returns:
          False if :class:`TimeSeriesData` does not have any datapoints.
          Otherwise return True.
        """

        return self.value.empty and self.time.empty

    def _set_time_format(
        self,
        series: pd.Series,
        date_format: Optional[str],
        use_unix_time: Optional[bool],
        unix_time_units: Optional[str],
        tz: Optional[str] = None,
        tz_ambiguous: Union[str, np.ndarray] = "raise",
        tz_nonexistent: str = "raise",
    ) -> pd.core.series.Series:
        """Parses time format when initializing :class:`TimeSeriesData`."""

        # Checking if time column is of type pandas datetime
        if not is_datetime(series):
            # If we should use unix time
            if use_unix_time:
                try:
                    if tz:
                        return (
                            pd.to_datetime(
                                series.values, unit=unix_time_units, utc=True
                            )
                            .tz_convert(tz)
                            .to_series()
                            .reset_index(drop=True)
                        )
                    else:
                        return pd.to_datetime(series, unit=unix_time_units)
                except ValueError:
                    logging.error("Failed to parse unix time")
                    logging.debug(
                        "Could not parse time column "
                        + f"{list(series)} using unix units "
                        + f"{unix_time_units}"
                    )
                    raise ValueError("Unable to parse unix time")
            # Otherwise try to parse string
            else:
                try:
                    if tz:
                        return (
                            pd.to_datetime(series.values, format=date_format)
                            .tz_localize(
                                tz, ambiguous=tz_ambiguous, nonexistent=tz_nonexistent
                            )
                            .to_series()
                            .reset_index(drop=True)
                        )
                    else:
                        return pd.to_datetime(series, format=date_format)
                except ValueError:
                    logging.error("Failed to parse time")
                    logging.debug(
                        "Could not parse time column "
                        + f"{list(series)} automatically "
                        + "or by using specified format "
                        + f"{date_format}"
                    )
                    raise ValueError("Unable to parse time with format specified")
        else:
            return series

    def extend(self, other: object, validate: bool = True) -> None:
        """
        Extends :class:`TimeSeriesData` with another :class:`TimeSeriesData`
        object.

        Args:
          other: The other :class:`TimeSeriesData` object (currently
            only other :class:`TimeSeriesData` objects are supported).
          validate (optional): A boolean representing if the new
            :class:`TimeSeriesData` should be validated (default True).

        Raises:
          ValueError: The object passed was not an instance of
            :class:`TimeSeriesData`.
        """

        if not isinstance(other, TimeSeriesData):
            raise TypeError("extend must take another TimeSeriesData object")
        # Concatenate times
        self.time = pd.concat([self.time, other.time], ignore_index=True).reset_index(
            drop=True
        )
        # Convert values to DataFrame if needed
        cur_value = self.value
        other_value = other.value
        if isinstance(self.value, pd.Series):
            cur_value = pd.DataFrame(cur_value)
        if isinstance(other.value, pd.Series):
            other_value = pd.DataFrame(other_value)
        # Concatenate values
        self.value = pd.concat([cur_value, other_value], ignore_index=True).reset_index(
            drop=True
        )
        # Merge value back to Series if required
        self._set_univariate_values_to_series()
        # Validate that frequency is constant if required
        if validate:
            self.validate_data(validate_frequency=True, validate_dimension=False)

    def time_to_index(self) -> pd.DatetimeIndex:
        """
        Utility function converting the time in the :class:`TimeSeriesData`
        object to a `pandas.DatetimeIndex`.

        Returns:
          A `pandas.DatetimeIndex` representation of the time values of the series.
        """

        return pd.DatetimeIndex(self.time)

    def validate_data(self, validate_frequency: bool, validate_dimension: bool) -> None:
        """
        Validates the time series for correctness (on both frequency and
        dimension).

        Args:
          validate_frequency: A boolean indicating whether the
            :class:`TimeSeriesData` should be validated for constant frequency.
          validate_dimension: A boolean indicating whether the
            :class:`TimeSeriesData` should be validated for having both the
            same number of timesteps and values.

        Raises:
          ValueError: The frequency and/or dimensions were invalid.
        """

        # check the time frequency is constant
        if validate_frequency and pd.infer_freq(self.time_to_index()) is None:
            raise ValueError("Only constant frequency is supported for time!")

        if validate_dimension and len(self.time) != self.value.shape[0]:
            raise ValueError("time and value has different length (dimension)!")

    def _calc_min_max_values(self):
        # Get maximum and minimum values
        if not self.value.empty:
            if isinstance(self.value, pd.core.series.Series):
                self._min = np.nanmin(self.value.values)
                self._max = np.nanmax(self.value.values)
            else:
                self._min = self.value.min(skipna=True)
                self._max = self.value.max(skipna=True)
        else:
            self._min = np.nan
            self._max = np.nan

    def is_data_missing(self) -> bool:
        """
        Checks if data is missing from the time series.

        This is very similar to :meth:`validate_data()` but will not raise an
        error.

        Returns:
          True when data is missing from the time series. Otherwise False.
        """

        # pd.infer_freq needs at least 3 time points.
        # here we tackle the case less than 3 time points
        if len(self.time) < 3:
            return False

        if pd.infer_freq(self.time_to_index()) is None:
            return True
        else:
            return False

    def freq_to_timedelta(self):
        """
        Returns a `pandas.Timedelta` representation of the
        :class:`TimeSeriesdata` frequency.

        Returns:
          A `pandas.Timedelta` object representing the frequency of the
          :class:`TimeSeriesData`.
        """

        return pd.Timedelta(to_offset(pd.infer_freq(self.time_to_index())))

    def tz(
        self,
    ) -> Union[datetime.tzinfo, dateutil.tz.tz.tzfile, None]:
        """
        Returns the timezone of the :class:`TimeSeriesData`.

        Returns:
          A timezone aware object representing the timezone of the
          :class:`TimeSeriesData`. Returns None when there is no timezone
          present.

        For more info, see:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.tz.html.
        """

        return self.time_to_index().tz

    def is_univariate(self):
        """Returns whether the :class:`TimeSeriesData` is univariate.

        Returns:
          True if the :class:`TimeSeriesData` is univariate. False otherwise.
        """

        return len(self.value.shape) == 1

    def to_dataframe(self, standard_time_col_name: bool = False) -> pd.DataFrame:
        """
        Converts the :class:`TimeSeriesData` object into a `pandas.DataFrame`.

        Args:
          standard_time_col (optional): True if the DataFrame's time column name
            should be "time". To keep the same time column name as the current
            :class:`TimeSeriesData` object, leave as False (default False).
        """

        time_col_name = (
            DEFAULT_TIME_NAME if standard_time_col_name else self.time_col_name
        )
        output_df = pd.DataFrame(dict(zip((time_col_name,), (self.time,))))
        if isinstance(self.value, pd.Series):
            if self.value.name is not None:
                output_df[self.value.name] = self.value
            else:
                output_df[DEFAULT_VALUE_NAME] = self.value
        elif isinstance(self.value, pd.DataFrame):
            output_df = pd.concat([output_df, self.value], axis=1).reset_index(
                drop=True
            )
        else:
            raise ValueError(f"Wrong value type: {type(self.value)}")
        return output_df

    def to_array(self) -> np.ndarray:
        """Converts the :class:`TimeSeriesData` object to a `numpy.ndarray`.

        Returns:
          A `numpy.ndarray` representation of the time series.
        """

        return self.to_dataframe().to_numpy()

    def _get_binary_op_other_arg(self, other: object) -> TimeSeriesData:
        if isinstance(other, float) or isinstance(other, int):
            if isinstance(self.value, pd.Series):
                return TimeSeriesData(
                    pd.DataFrame(
                        dict(
                            zip(
                                (DEFAULT_TIME_NAME, self.value.name),
                                (self.time, pd.Series(other, index=self.time.index)),
                            )
                        )
                    )
                )
            else:
                # TODO: implement multivariate time series operation with constant
                raise NotImplementedError("Operation on multivariate")

        if not isinstance(other, TimeSeriesData):
            raise TypeError("Binary op must take another TimeSeriesData object")
        if not self.time.equals(other.time):
            raise ValueError("BBinary op must take a TimeSeriesData with same time")
        return other

    def _perform_op(self, other: object, op_type: "OperationsEnum") -> TimeSeriesData:
        # Extract DataFrames with same time column name for joining
        self_df = self.to_dataframe(standard_time_col_name=True)
        other_df = self._get_binary_op_other_arg(other).to_dataframe(
            standard_time_col_name=True
        )
        # Join DataFrames on time column
        combo_df = pd.merge(
            self_df,
            other_df,
            on=DEFAULT_TIME_NAME,
            how="outer",
            suffixes=(PREFIX_OP_1, PREFIX_OP_2),
        )
        # Map the final column name to the sub column names
        col_map = {}
        for col_name in list(combo_df.columns):
            if PREFIX_OP_1 in col_name:
                prefix = col_name.split(PREFIX_OP_1)[0]
                col_map[prefix] = col_map.get(prefix, []) + [col_name]
            elif PREFIX_OP_2 in col_name:
                prefix = col_name.split(PREFIX_OP_2)[0]
                col_map[prefix] = col_map.get(prefix, []) + [col_name]

        for col_name in list(col_map.keys()):
            # Perform operation on two columns and merge back to one column
            col_1, col_2 = col_map[col_name]
            if op_type == OperationsEnum.ADD:
                combo_df[col_name] = combo_df[col_1] + combo_df[col_2]
            elif op_type == OperationsEnum.SUB:
                combo_df[col_name] = combo_df[col_1] - combo_df[col_2]
            elif op_type == OperationsEnum.DIV:
                combo_df[col_name] = combo_df[col_1] / combo_df[col_2]
            elif op_type == OperationsEnum.MUL:
                combo_df[col_name] = combo_df[col_1] * combo_df[col_2]
            else:
                raise ValueError("Unsupported Operations Type")
            combo_df.drop([col_1, col_2], axis=1, inplace=True)
        # Set columns only present in one of the objects to None
        final_col_list = set([DEFAULT_TIME_NAME] + list(col_map.keys()))
        for col_name in list(combo_df.columns):
            if col_name not in final_col_list:
                combo_df[col_name] = np.nan
        # Change time col name back if needed
        if self.time_col_name != DEFAULT_TIME_NAME:
            combo_df[self.time_col_name] = combo_df[DEFAULT_TIME_NAME]
            combo_df.drop(DEFAULT_TIME_NAME, axis=1, inplace=True)
        return TimeSeriesData(df=combo_df, time_col_name=self.time_col_name)

    def infer_freq_robust(self) -> pd.Timedelta:
        """
        This method is a more robust way to infer the frequency of the time
        series in the presence of missing data. It looks at the diff of the
        time series, and decides the frequency by majority voting.

        Returns:
          A `pandas.Timedelta` object representing the frequency of the series.

        Raises:
          ValueError: The :class:`TimeSeriesData` has less than 2 data points.
        """

        df = self.to_dataframe()

        if df.shape[0] <= 1:
            raise ValueError("Cannot find frequency for less than two data points")

        freq_counts = (
            df[self.time_col_name].diff().value_counts().sort_values(ascending=False)
        )

        frequency = freq_counts.index[0]

        return frequency

    def interpolate(
        self,
        freq: Optional[Union[str, pd.Timedelta]] = None,
        method: str = "linear",
        remove_duplicate_time=False,
    ) -> TimeSeriesData:
        """
        Interpolate missing date if `time` doesn't have constant frequency.

        The following options are available:
          - linear
          - backward fill
          - forward fill

        See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
        for more detail on these options.

        Args:
          freq: A string representing the pre-defined freq of the time series.
          method: A string representing the method to impute the missing time
            and data. See the above options (default "linear").
          remove_duplicate_index: A boolean to auto-remove any duplicate time
            values, as interpolation in this case due to the need to index
            on time (default False).

        Returns:
            A new :class:`TimeSeriesData` object with interpolated data.
        """

        if not freq:
            freq = self.infer_freq_robust()

        # convert to pandas.DataFrame so that we can leverage the built-in methods
        df = self.to_dataframe()

        # Linear interpolation fails if a column has an int type - convert to float
        if method == "linear":
            for col in list(df):
                if col != self.time_col_name:
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        raise ValueError(
                            f"Column {col} is invalid type: {df[col].dtype}"
                        )

        df.set_index(self.time_col_name, inplace=True)

        # Removing duplicate time index values if needed
        if remove_duplicate_time:
            df = df[~df.index.duplicated()]

        if method == "linear":
            df = df.resample(freq).interpolate(method="linear")

        elif method == "ffill":
            df = df.resample(freq).ffill()

        elif method == "bfill":
            df = df.resample(freq).bfill()

        else:
            # method is not supported
            raise ValueError(f"the given method is not supported: {method}")

        df = df.reset_index().rename(columns={"index": self.time_col_name})
        return TimeSeriesData(df, time_col_name=self.time_col_name)

    def plot(
        self,
        cols: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        grid: bool = True,
        figsize: Optional[FigSize] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        grid_kwargs: Optional[Dict[str, Any]] = None,
    ) -> plt.Axes:
        """Plots the time series.

        Args:
            cols: List of variable names to plot against time. If None,
                plot all variables in the time series data.
            ax: optional Axes to use. If None, create one.
            grid: if True, draw gridlines.
            figsize: if ax is None, the figsize to create. If None, defaults to
                (10, 6).
            plot_kwargs: optional additional arguments to pass to pandas.plot().
            grid_kwargs: optional additional arguments to pass to Axes.grid().
        Returns:
            The matplotlib Axes.
        """
        if self.is_empty():
            raise ValueError("No data to plot")
        # Make sure columns are valid
        df = self.to_dataframe()
        all_cols = list(df.columns)
        all_cols.remove(self.time_col_name)
        if cols is None:
            cols = all_cols
        elif not set(cols).issubset(all_cols):
            logging.error(f"Columns to plot: {cols} are not all in the timeseries")
            raise ValueError("Invalid columns passed")
        if figsize is None:
            figsize = (10, 6)
        if plot_kwargs is None:
            plot_kwargs = {}
        grid_kwargs_ = {"which": "major", "c": "gray", "ls": "-", "lw": 1, "alpha": 0.2}
        if grid_kwargs is not None:
            grid_kwargs_.update(**grid_kwargs)

        # Plot
        logging.info("Plotting time series")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if grid:
            ax.grid(True, **grid_kwargs_)
        fig.tight_layout()
        # pyre-ignore[29]: `pd.core.accessor.CachedAccessor` is not a function.
        df.plot(x=self.time_col_name, y=cols, ax=ax, **plot_kwargs)
        return ax


class TimeSeriesIterator:
    def __init__(self, ts: TimeSeriesData) -> None:
        self.ts = copy.deepcopy(ts)
        self.ts.value = pd.DataFrame(ts.value)
        self.start = 0

    def __iter__(self):
        self.a = pd.DataFrame(
            list(self.ts.value.iloc[:, 0]), index=list(self.ts.time), columns=["y"]
        )
        return self

    def __next__(self):
        if self.start < self.ts.value.shape[1]:
            x = pd.DataFrame(
                list(self.ts.value.iloc[:, self.start]),
                index=list(self.ts.time),
                columns=["y"],
            )
            self.start += 1
            return x
        else:
            raise StopIteration


class TSIterator:
    """Iterates through the values of a single timeseries.

    Produces a timeseries with a single point, in case of an
    univariate time series, or a timeseries with an array indicating
    the values at the given location, for a multivariate time series.

    Attributes:
        ts: The input timeseries.
    """

    def __init__(self, ts: TimeSeriesData) -> None:
        self.ts = ts
        self.curr = 0

    def __iter__(self):
        return self

    def __next__(self) -> TimeSeriesData:
        if self.curr < len(self.ts.time):
            if self.ts.is_univariate():
                ret = TimeSeriesData(
                    time=pd.Series(self.ts.time[self.curr]),
                    value=pd.Series(self.ts.value.iloc[self.curr], name=self.curr),
                )
            else:
                ret = TimeSeriesData(
                    time=pd.Series(self.ts.time[self.curr]),
                    value=pd.DataFrame(
                        self.ts.value.iloc[self.curr], columns=[self.curr]
                    ),
                )
            self.curr += 1
            return ret
        else:
            raise StopIteration


class Params:
    def __init__(self):
        pass

    def validate_params(self):
        pass


@unique
class ModelEnum(Enum):
    """
    This enum lists the options of models to be set for default search space in
    hyper-parameter tuning.
    """

    ARIMA = auto()
    SARIMA = auto()
    PROPHET = auto()
    HOLTWINTERS = auto()
    LINEAR = auto()
    QUADRATIC = auto()


@unique
class SearchMethodEnum(Enum):
    """
    This enum lists the options of search algorithms to be used in
    hyper-parameter tuning.
    """

    GRID_SEARCH = auto()
    RANDOM_SEARCH_UNIFORM = auto()
    RANDOM_SEARCH_SOBOL = auto()
    BAYES_OPT = auto()


@unique
class OperationsEnum(Enum):
    """
    This enum lists all the mathematical operations that can be performed on
    :class:`TimeSeriesData` objects.
    """

    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()


__all__ = [
    "ModelEnum",
    "OperationsEnum",
    "Params",
    "SearchMethodEnum",
    "TimeSeriesChangePoint",
    "TimeSeriesData",
    "TimeSeriesIterator",
    "TSIterator",
]
