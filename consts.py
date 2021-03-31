#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from enum import Enum, auto, unique
from typing import List, Optional, Union, cast, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.tseries.frequencies import to_offset


# Constants
DEFAULT_TIME_NAME = "time"
DEFAULT_VALUE_NAME = "value"
PREFIX_OP_1 = "_kats.1"
PREFIX_OP_2 = "_kats.2"
INTERPOLATION_METHODS = {"linear", "bfill", "ffill"}


class TimeSeriesChangePoint:
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
        return f"TimeSeriesChangePoint(start_time: {self.start_time}, end_time: {self.end_time}, confidence: {self.confidence})"

    def __str__(self):
        return f"TimeSeriesChangePoint(start_time: {self.start_time}, end_time: {self.end_time}, confidence: {self.confidence})"


class TimeSeriesData:
    def __init__(  # noqa C901
        self,
        df: Optional[pd.DataFrame] = None,
        sort_by_time: bool = True,
        time: Union[pd.Series, pd.DatetimeIndex, None] = None,
        value: Union[pd.Series, pd.DataFrame, None] = None,
        time_col_name: str = DEFAULT_TIME_NAME,
        date_format: str = None,
        use_unix_time: bool = False,
        unix_time_units: str = "ns",
        tz: Optional[str] = None,
        tz_ambiguous: Union[str, np.ndarray] = "raise",
        tz_nonexistent: str = "raise",
    ) -> None:
        self.time_col_name = time_col_name

        # If DataFrame is passed
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                logging.error(
                    "Argument df needs to be a pandas.DataFrame but is of "
                    + f"type {type(df)}"
                )
                raise ValueError(
                    "Incorrect object types passed to initialize TimeSeriesData"
                )
            # If empty DataFrame is passed then create an empty object
            if df.empty:
                self.time = pd.Series([], name=time_col_name)
                self.value = pd.Series([], name=DEFAULT_VALUE_NAME)
                logging.warning("Initializing empty TimeSeriesData object")
            # Otherwise initialize TimeSeriesData from DataFrame
            else:
                # Ensuring time column is present in DataFrame
                if self.time_col_name not in df.columns:
                    logging.error(
                        f"Time column: {self.time_col_name}, not in DataFrame"
                    )
                    raise ValueError("No time column found in dataframe")
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
                        "Please make sure the time series is sorted by time or set "
                        + "'sort_by_time' as True."
                    )
                self.time = df[self.time_col_name]
                self.value = df[[x for x in df.columns if x != self.time_col_name]]
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
                logging.error(
                    f"Invalid types: time is {type(time)} when it must be a "
                    + "pandas.Series or pandas.DatetimeIndex and value is "
                    + f"{type(value)} when it must be a pandas.DataFrame or "
                    + "pandas.Series"
                )
                raise ValueError(
                    "Incorrect object types passed to initialize TimeSeriesData"
                )
            if isinstance(time, pd.DatetimeIndex):
                self.time = pd.Series(time)
            else:
                self.time = time
            self.value = value
            self._set_univariate_values_to_series()
            # Set time col name
            if time.name:
                self.time_col_name = time.name
            # Resetting indices
            self.time = self.time.reset_index(drop=True)
            self.value = self.value.reset_index(drop=True)
            # Checking for emptiness
            if self.time.empty and self.value.empty:
                logging.warning("Initializing empty TimeSeriesData object")
                self.time = pd.Series([], name=time_col_name)
                if isinstance(value, pd.DataFrame):
                    self.value = pd.Series([], name=DEFAULT_VALUE_NAME)
                else:
                    self.value = pd.Series([], name=value.name if value.name else DEFAULT_VALUE_NAME)
            # Raise exception if only one of time and value is empty
            elif self.time.empty or self.value.empty:
                logging.error(
                    "Series objects for time and value must both be empty or both have values"
                )
                raise ValueError("One of time or value is empty while the other is not")
            # If time values are passed then standardizing format
            else:
                self.time = self._set_time_format(
                    self.time,
                    date_format=date_format,
                    use_unix_time=use_unix_time,
                    unix_time_units=unix_time_units,
                    tz=tz,
                    tz_ambiguous=tz_ambiguous,
                    tz_nonexistent=tz_nonexistent,
                ).reset_index(drop=True)

        # If None is passed
        elif not time and not value:
            self.time = pd.Series([], name=time_col_name)
            self.value = pd.Series([], name=DEFAULT_VALUE_NAME)
            logging.warning("Initializing empty TimeSeriesData object")

        # Error if only one of time or value is None
        else:
            logging.error(
                "Objects for time and value must both be None or both have values"
            )
            raise ValueError("One of time or value is empty while the other is not")

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
            raise ValueError("Data for time series is not a numeric type")
            logging.error("Values passed are not of a numeric type")
            logging.debug(
                f"dtype of values was {self.value.dtype} which is not numeric"
            )

        self._calc_min_max_values()

    @property
    def time(self) -> pd.Series:
        return self._time

    @time.setter
    def time(self, time_values: pd.Series) -> None:
        self._time = time_values

    @property
    def value(self) -> Union[pd.Series, pd.DataFrame]:
        return self._value

    @value.setter
    def value(self, values: Union[pd.Series, pd.DataFrame]) -> None:
        self._value = values
        # updates for min/max values are necessary once values are updated
        self._calc_min_max_values()

    @property
    def min(self) -> Union[pd.Series, float]:
        return self._min

    @property
    def max(self) -> Union[pd.Series, float]:
        return self._max

    @min.setter  # we do not allow setting min directly
    def min(self, min: Any):
        msg = "Assigning mininum value to TimeSeriesData is not allowed."
        logging.error(msg)
        raise ValueError(msg)

    @max.setter  # we do not allow setting max directly
    def max(self, max: Any):
        msg = "Assigning maximum value to TimeSeriesData is not allowed."
        logging.error(msg)
        raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        # It is recommended for "__eq__" to work with arbitrary objects
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

    def __sub__(self, other: object) -> "TimeSeriesData":
        return self._perform_op(other, OperationsEnum.SUB)

    def __truediv__(self, other: object) -> "TimeSeriesData":
        return self._perform_op(other, OperationsEnum.DIV)

    def __add__(self, other: object) -> "TimeSeriesData":
        return self._perform_op(other, OperationsEnum.ADD)

    def __mul__(self, other: object) -> "TimeSeriesData":
        return self._perform_op(other, OperationsEnum.MUL)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, sliced) -> "TimeSeriesData":
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
        """
        Returns False if TimeSeriesData does not have any datapoints

        Returns True otherwise.
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
        return pd.DatetimeIndex(self.time)

    def validate_data(self, validate_frequency: bool, validate_dimension: bool) -> None:
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
        returns True when data is missing from the time series
        This is very similar to validate_data but will not raise an error
        """
        # pd.infer_freq needs atleast 3 time points.
        # here we tackle the case less than 3 time points
        if len(self.time) < 3:
            return False

        if pd.infer_freq(self.time_to_index()) is None:
            return True
        else:
            return False

    def freq_to_timedelta(self):
        return pd.Timedelta(to_offset(pd.infer_freq(self.time_to_index())))

    def tz(self):
        return self.time_to_index().tz

    def is_univariate(self):
        return len(self.value.shape) == 1

    def to_dataframe(self, standard_time_col_name: bool = False) -> pd.DataFrame:
        """
        Transform back to pd.DataFrame
        standard_time_col: bool If True time column name is set to 'time'
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
        return self.to_dataframe().to_numpy()

    def _get_binary_op_other_arg(self, other: object) -> "TimeSeriesData":
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
                raise NotImplementedError("Operation on multivariate ")

        if not isinstance(other, TimeSeriesData):
            raise TypeError("Binary op must take another TimeSeriesData object")
        other = cast(TimeSeriesData, other)
        if not self.time.equals(other.time):
            raise ValueError("BBinary op must take a TimeSeriesData with same time")
        return other

    def _perform_op(self, other: object, op_type: "OperationsEnum") -> "TimeSeriesData":
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
        This is a more robust way to infer the frequency of the time series
        in the presence of missing data.
        It looks at the diff of the time series, and decides the frequency
        by majority voting
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
        self, freq: str = None, method: str = "linear", remove_duplicate_time=False
    ) -> "TimeSeriesData":
        """
        Interpolate missing date if `time` doesn't has constant frequency

        Parameters
        -----------
        - freq: pre-defined freq of the time series
        - method: method to impute the missing time and data
            - linear
            - backward fill
            - forward fill
        - remove_duplicate_index: boolean to auto-remove any duplicate time
            values, as interpolation in this case due to the need to index
            on time. Default False.

        Returns: new TimeSeriesData object with interpolated data
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

    def plot(self, cols: List[str]) -> None:
        """
        Plot TimeSeriesData
        cols: List of variables to plot (against time)
        """
        if self.is_empty():
            return
        # Make sure columns are valid
        df = self.to_dataframe()
        all_cols = list(df.columns)
        all_cols.remove(self.time_col_name)
        if not set(cols).issubset(all_cols):
            logging.error(f"Columns to plot: {cols} are not all in the timeseries")
            raise ValueError("Invalid columns passed")
        # Plot
        logging.info("Plotting time series")
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)
        for col in cols:
            ax.plot(
                df[self.time_col_name].to_numpy(),
                df[col].to_numpy(),
                "k",
            )
        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        fig.tight_layout()
        self.to_dataframe().plot(x=self.time_col_name, y=cols, ax=ax)


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
                    value=pd.Series(self.ts.value.iloc[self.curr]),
                )
            else:
                ret = TimeSeriesData(
                    time=pd.Series(self.ts.time[self.curr]),
                    value=pd.DataFrame(self.ts.value.iloc[self.curr]),
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
    TimeSeriesData objects
    """

    ADD = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()
