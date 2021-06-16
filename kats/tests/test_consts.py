#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
import pytz
from dateutil import parser
from dateutil.relativedelta import relativedelta
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

from kats.consts import DEFAULT_TIME_NAME, DEFAULT_VALUE_NAME, TimeSeriesData, TSIterator

# tentative, for test purpose
print(os.getcwd())

# Constant values to reuse across test cases
if "kats/tests" in os.getcwd():
    DATA_FILE = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/air_passengers.csv"
            )
            )
elif "/home/runner/work/" in os.getcwd(): # for Githun Action
    DATA_FILE = "kats/data/air_passengers.csv"
else:
    DATA_FILE = "kats/kats/data/air_passengers.csv"
TIME_COL_NAME = "ds"
VALUE_COL_NAME = "y"
AIR_DF = pd.read_csv(DATA_FILE)
AIR_DF_DATETIME = AIR_DF.copy(deep=True)
AIR_DF_DATETIME.ds = AIR_DF_DATETIME.ds.apply(lambda x: parser.parse(x))
AIR_DF_UNIXTIME = AIR_DF.copy(deep=True)
AIR_DF_UNIXTIME.ds = AIR_DF_DATETIME.ds.apply(
    lambda x: (x - datetime(1970, 1, 1)).total_seconds()
)
AIR_DF_WITH_DEFAULT_NAMES = AIR_DF.copy(deep=True)
AIR_DF_WITH_DEFAULT_NAMES.columns = [DEFAULT_TIME_NAME, DEFAULT_VALUE_NAME]
MULTIVAR_AIR_DF = AIR_DF.copy(deep=True)
MULTIVAR_AIR_DF[VALUE_COL_NAME + "_1"] = MULTIVAR_AIR_DF.y * 2
MULTIVAR_AIR_DF_DATETIME = MULTIVAR_AIR_DF.copy(deep=True)
MULTIVAR_AIR_DF_DATETIME.ds = MULTIVAR_AIR_DF_DATETIME.ds.apply(
    lambda x: parser.parse(x)
)
MULTIVAR_VALUE_DF = MULTIVAR_AIR_DF[[VALUE_COL_NAME, VALUE_COL_NAME + "_1"]]
AIR_TIME_SERIES = AIR_DF.ds
AIR_TIME_SERIES_PD_DATETIME = pd.to_datetime(AIR_TIME_SERIES)
AIR_TIME_SERIES_UNIXTIME = AIR_TIME_SERIES_PD_DATETIME.apply(
    lambda x: (x - datetime(1970, 1, 1)).total_seconds()
)
AIR_VALUE_SERIES = AIR_DF.y
AIR_TIME_DATETIME_INDEX = pd.DatetimeIndex(AIR_TIME_SERIES)
EMPTY_DF = pd.DataFrame()
EMPTY_TIME_SERIES = pd.Series([], name=DEFAULT_TIME_NAME)
EMPTY_VALUE_SERIES = pd.Series([], name=DEFAULT_VALUE_NAME)
EMPTY_VALUE_SERIES_NO_NAME = pd.Series([])
EMPTY_TIME_DATETIME_INDEX = pd.DatetimeIndex(pd.Series([]))
EMPTY_DF_WITH_COLS = pd.concat([EMPTY_TIME_SERIES, EMPTY_VALUE_SERIES], axis=1)
NUM_YEARS_OFFSET = 12


class TimeSeriesDataInitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Univariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_from_df = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        # Univariate TimeSeriesData initialized from a pd.DataFrame with time
        # as a datetime.datetime object
        cls.ts_from_df_datetime = TimeSeriesData(
            df=AIR_DF_DATETIME, time_col_name=TIME_COL_NAME
        )
        # Univariate TimeSeriesData initialized from a pd.DataFrame with time
        # as unix time
        cls.ts_from_df_with_unix = TimeSeriesData(
            df=AIR_DF_UNIXTIME,
            use_unix_time=True,
            unix_time_units="s",
            time_col_name=TIME_COL_NAME,
        )
        # Multivariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_from_df_multi = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        # Multivariate TimeSeriesData initialized from a pd.DataFrame with time
        # as a datetime.datetime object
        cls.ts_from_df_multi_datetime = TimeSeriesData(
            df=MULTIVAR_AIR_DF_DATETIME, time_col_name=TIME_COL_NAME
        )
        # Univariate TimeSeriesData initialized from two pd.Series with time
        # as a string
        cls.ts_from_series_univar_no_datetime = TimeSeriesData(
            time=AIR_TIME_SERIES, value=AIR_VALUE_SERIES
        )
        # Univariate TimeSeriesData initialized from two pd.Series with time
        # as a pd.Timestamp
        cls.ts_from_series_univar_with_datetime = TimeSeriesData(
            time=AIR_TIME_SERIES_PD_DATETIME, value=AIR_VALUE_SERIES
        )
        # Univariate TimeSeriesData initialized from two pd.Series with time
        # as unix time
        cls.ts_from_series_with_unix = TimeSeriesData(
            time=AIR_TIME_SERIES_UNIXTIME,
            value=AIR_VALUE_SERIES,
            use_unix_time=True,
            unix_time_units="s",
            time_col_name=TIME_COL_NAME,
        )
        # Univariate TimeSeriesData initialized with time as a pd.Series and
        # value as a pd.DataFrame
        cls.ts_from_series_and_df_univar = TimeSeriesData(
            time=AIR_TIME_SERIES, value=AIR_VALUE_SERIES.to_frame()
        )
        # Multivariate TimeSeriesData initialized from a pd.Series for time
        # and DataFrame for value
        cls.ts_from_series_and_df_multivar = TimeSeriesData(
            time=AIR_TIME_SERIES, value=MULTIVAR_VALUE_DF
        )
        # Univariate TimeSeriesData initialized with time as a pd.DateTimeIndex
        # and value as a pd.Series
        cls.ts_from_index_and_series_univar = TimeSeriesData(
            time=AIR_TIME_DATETIME_INDEX,
            value=AIR_VALUE_SERIES,
            time_col_name=TIME_COL_NAME,
        )
        # Multivariate TimeSeriesData initialized with time as a
        # pd.DateTimeIndex and value as a pd.DataFrame
        cls.ts_from_index_and_series_multivar = TimeSeriesData(
            time=AIR_TIME_DATETIME_INDEX,
            value=MULTIVAR_VALUE_DF,
            time_col_name=TIME_COL_NAME,
        )
        # TimeSeriesData initialized from None Objects
        cls.ts_df_none = TimeSeriesData(df=None)
        cls.ts_time_none_and_value_none = TimeSeriesData(time=None, value=None)
        # TimeSeriesData initialized from Empty Objects
        cls.ts_df_empty = TimeSeriesData(df=EMPTY_DF)
        cls.ts_time_empty_value_empty = TimeSeriesData(
            time=EMPTY_TIME_SERIES, value=EMPTY_VALUE_SERIES
        )
        cls.ts_time_empty_value_empty_no_name = TimeSeriesData(
            time=EMPTY_TIME_SERIES, value=EMPTY_VALUE_SERIES_NO_NAME
        )
        cls.ts_time_empty_value_empty_df = TimeSeriesData(
            time=EMPTY_TIME_SERIES, value=EMPTY_DF
        )
        cls.ts_time_empty_value_empty_df_with_cols = TimeSeriesData(
            time=EMPTY_TIME_SERIES, value=EMPTY_DF_WITH_COLS
        )

        # univariate data with missing time
        cls.ts_univariate_missing = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-05"],
                    "value": [1, 2, 3, 4],
                }
            )
        )

        # multivariate data with missing time
        cls.ts_multi_missing = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-05"],
                    "value1": [1, 2, 3, 4],
                    "value2": [4, 3, 2, 1],
                }
            )
        )

        # univariate data with unixtime in US/Pacific with time zone
        cls.unix_list = (
            (
                pd.date_range(
                    "2020-03-01", "2020-03-10", tz="US/Pacific", freq="1d"
                ).astype(int)
                / 1e9
            )
            .astype(int)
            .to_list()
        )
        cls.ts_univar_PST_tz = TimeSeriesData(
            df=pd.DataFrame({"time": cls.unix_list, "value": [0] * 10}),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )
        # univariate data with unixtime in US/Pacific without time zone
        cls.ts_univar_PST = TimeSeriesData(
            df=pd.DataFrame({"time": cls.unix_list, "value": [0] * 10}),
            use_unix_time=True,
            unix_time_units="s",
        )
        # univariate data with date str with tz
        date = ["2020-10-31", "2020-11-01", "2020-11-02"]
        cls.ts_univar_str_date_tz = TimeSeriesData(
            df=pd.DataFrame({"time": date, "value": [0] * 3}),
            date_format="%Y-%m-%d",
            tz="US/Pacific",
        )
        # univariate data with date str without tz
        cls.ts_univar_str_date = TimeSeriesData(
            df=pd.DataFrame({"time": date, "value": [0] * 3}),
            date_format="%Y-%m-%d",
        )

        # univariate data in US/Pacific Time Zone with missing data
        cls.ts_univar_PST_missing_tz = TimeSeriesData(
            df=pd.DataFrame(
                {"time": (cls.unix_list[0:4] + cls.unix_list[7:10]), "value": [0] * 7}
            ),
            use_unix_time=True,
            unix_time_units="s",
            tz="US/Pacific",
        )

    # Testing univariate time series intialized from a DataFrame
    def test_init_from_df_univar(self) -> None:
        # DataFrame with string time
        assert_series_equal(self.ts_from_df.time, AIR_TIME_SERIES_PD_DATETIME)
        assert_series_equal(self.ts_from_df.value, AIR_VALUE_SERIES)
        # DataFrame with datetime time
        assert_series_equal(self.ts_from_df_datetime.time, AIR_TIME_SERIES_PD_DATETIME)
        assert_series_equal(self.ts_from_df_datetime.value, AIR_VALUE_SERIES)
        # DataFrame with unix time
        assert_series_equal(self.ts_from_df_with_unix.time, AIR_TIME_SERIES_PD_DATETIME)
        assert_series_equal(self.ts_from_df_with_unix.value, AIR_VALUE_SERIES)

    # Testing multivariate time series initialized from a DataFrame
    def test_init_from_df_multi(self) -> None:
        assert_series_equal(self.ts_from_df_multi.time, AIR_TIME_SERIES_PD_DATETIME)
        assert_frame_equal(self.ts_from_df_multi.value, MULTIVAR_VALUE_DF)

    # Testing univiarite time series initialized from a Series and Series/DataFrame
    def test_init_from_series_univar(self) -> None:
        # time and value from Series, with time as string
        assert_series_equal(
            self.ts_from_series_univar_no_datetime.time, AIR_TIME_SERIES_PD_DATETIME
        )
        # time and value from Series, with time as pd.Timestamp
        assert_series_equal(
            self.ts_from_series_univar_with_datetime.time, AIR_TIME_SERIES_PD_DATETIME
        )
        assert_series_equal(
            self.ts_from_series_univar_no_datetime.value, AIR_VALUE_SERIES
        )
        # time and value from Series, with time as unix time
        assert_series_equal(
            self.ts_from_series_with_unix.time, AIR_TIME_SERIES_PD_DATETIME
        )
        assert_series_equal(self.ts_from_series_with_unix.value, AIR_VALUE_SERIES)
        # time from Series and value from DataFrame
        assert_series_equal(
            self.ts_from_series_and_df_univar.time, AIR_TIME_SERIES_PD_DATETIME
        )
        print(type(self.ts_from_series_and_df_univar.value))
        assert_series_equal(self.ts_from_series_and_df_univar.value, AIR_VALUE_SERIES)

    # Testing multivariate time series initialized from a Series/DataFrame
    def test_init_from_series_multivar(self) -> None:
        # Testing multivariate time series initialized from a
        assert_series_equal(
            self.ts_from_series_and_df_multivar.time, AIR_TIME_SERIES_PD_DATETIME
        )
        assert_frame_equal(self.ts_from_series_and_df_multivar.value, MULTIVAR_VALUE_DF)

    # Testing univariate time series with time initialized as a
    # pd.DateTimeIndex
    def test_init_from_index_univar(self) -> None:
        assert_series_equal(
            self.ts_from_index_and_series_univar.time, AIR_TIME_SERIES_PD_DATETIME
        )
        assert_series_equal(
            self.ts_from_index_and_series_univar.value, AIR_VALUE_SERIES
        )

    # Testing multivariate time series with time initialized as a
    # pd.DateTimeIndex
    def test_init_from_index_multivar(self) -> None:
        assert_series_equal(
            self.ts_from_index_and_series_multivar.time, AIR_TIME_SERIES_PD_DATETIME
        )
        assert_frame_equal(
            self.ts_from_index_and_series_multivar.value, MULTIVAR_VALUE_DF
        )

    # Testing initialization from None Objects
    def test_none(self) -> None:
        # Testing initialization from None DataFrame
        assert_series_equal(self.ts_df_none.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_df_none.value, EMPTY_VALUE_SERIES)
        # Testing initialization from two None Series
        assert_series_equal(self.ts_time_none_and_value_none.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_time_none_and_value_none.value, EMPTY_VALUE_SERIES)

    # Testing initialization from Empty Objects
    def test_empty(self) -> None:
        # Testing intialization from empty DataFrame
        assert_series_equal(self.ts_df_empty.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_df_empty.value, EMPTY_VALUE_SERIES)
        # Testing intialization from two empty Series
        assert_series_equal(self.ts_time_empty_value_empty.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_time_empty_value_empty.value, EMPTY_VALUE_SERIES)
        # Testing intialization from two empty no name Series
        assert_series_equal(self.ts_time_empty_value_empty_no_name.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_time_empty_value_empty_no_name.value, EMPTY_VALUE_SERIES)
        # Testing initialization from time as empty Series and value as empty
        # DataFrame
        assert_series_equal(self.ts_time_empty_value_empty_df.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_time_empty_value_empty_df.value, EMPTY_VALUE_SERIES)
        # Testing initialization from time as empty Series and value as empty
        # DataFrame
        assert_series_equal(self.ts_time_empty_value_empty_df_with_cols.time, EMPTY_TIME_SERIES)
        assert_series_equal(self.ts_time_empty_value_empty_df_with_cols.value, EMPTY_VALUE_SERIES)

    # Testing incorrect initializations
    def test_incorrect_init_types(self) -> None:
        with self.assertRaises(ValueError):
            # Incorret initialization with DF
            # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]` for 1st
            #  param but got `List[Variable[_T]]`.
            TimeSeriesData(df=[])
            # Incorrect initialization with value
            TimeSeriesData(time=AIR_TIME_SERIES, value=None)
            # pyre-fixme[6]: Expected `Union[None, pd.core.frame.DataFrame,
            #  pd.core.series.Series]` for 2nd param but got `List[Variable[_T]]`.
            TimeSeriesData(time=AIR_TIME_SERIES, value=[])
            # Incorrect initialization with time
            TimeSeriesData(time=None, value=AIR_VALUE_SERIES)
            # pyre-fixme[6]: Expected `Union[None,
            #  pd.core.indexes.datetimes.DatetimeIndex, pd.core.series.Series]` for 1st
            #  param but got `List[Variable[_T]]`.
            TimeSeriesData(time=[], value=AIR_VALUE_SERIES)
            # Incorrect initialization with time and value
            # pyre-fixme[6]: Expected `Union[None,
            #  pd.core.indexes.datetimes.DatetimeIndex, pd.core.series.Series]` for 1st
            #  param but got `List[Variable[_T]]`.
            TimeSeriesData(time=[], value=[])

    # Testing DataFrame conversion
    def test_to_dataframe(self) -> None:
        # Univariate case
        assert_frame_equal(self.ts_from_df.to_dataframe(), AIR_DF_DATETIME)
        # Multivariate case
        assert_frame_equal(
            self.ts_from_df_multi_datetime.to_dataframe(), MULTIVAR_AIR_DF_DATETIME
        )
        # Series Cases
        assert_frame_equal(
            self.ts_from_series_univar_no_datetime.to_dataframe(), AIR_DF_DATETIME
        )
        assert_frame_equal(
            self.ts_from_series_univar_with_datetime.to_dataframe(), AIR_DF_DATETIME
        )
        # Series/DataFrame Cases
        assert_frame_equal(
            self.ts_from_series_and_df_univar.to_dataframe(), AIR_DF_DATETIME
        )
        assert_frame_equal(
            self.ts_from_series_and_df_multivar.to_dataframe(), MULTIVAR_AIR_DF_DATETIME
        )
        # Empty/None Cases
        assert_frame_equal(self.ts_df_none.to_dataframe(), EMPTY_DF_WITH_COLS)
        assert_frame_equal(
            self.ts_time_none_and_value_none.to_dataframe(), EMPTY_DF_WITH_COLS
        )
        assert_frame_equal(self.ts_df_empty.to_dataframe(), EMPTY_DF_WITH_COLS)
        assert_frame_equal(
            self.ts_time_empty_value_empty.to_dataframe(), EMPTY_DF_WITH_COLS
        )
        assert_frame_equal(
            self.ts_time_empty_value_empty_df.to_dataframe(), EMPTY_DF_WITH_COLS
        )

    # Testing Data Interpolate
    def test_interpolate(self) -> None:
        # univariate
        self.assertEqual(
            self.ts_univariate_missing.interpolate(freq="D", method="linear"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value": [1, 2, 3, 3.5, 4],
                    }
                )
            ),
        )

        self.assertEqual(
            self.ts_univariate_missing.interpolate(freq="D", method="ffill"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value": [1, 2, 3, 3, 4],
                    }
                )
            ),
        )

        self.assertEqual(
            self.ts_univariate_missing.interpolate(freq="D", method="bfill"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value": [1, 2, 3, 4, 4],
                    }
                )
            ),
        )

        # multivariate
        self.assertEqual(
            self.ts_multi_missing.interpolate(freq="D", method="linear"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value1": [1, 2, 3, 3.5, 4],
                        "value2": [4, 3, 2, 1.5, 1],
                    }
                )
            ),
        )

        self.assertEqual(
            self.ts_multi_missing.interpolate(freq="D", method="ffill"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value1": [1, 2, 3, 3, 4],
                        "value2": [4, 3, 2, 2, 1],
                    }
                )
            ),
        )

        self.assertEqual(
            self.ts_multi_missing.interpolate(freq="D", method="bfill"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value1": [1, 2, 3, 4, 4],
                        "value2": [4, 3, 2, 1, 1],
                    }
                )
            ),
        )

        # test with no frequency given univariate
        self.assertEqual(
            self.ts_univariate_missing.interpolate(method="linear"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value": [1, 2, 3, 3.5, 4],
                    }
                )
            ),
        )

        # no frequency given, for multivariate
        self.assertEqual(
            self.ts_multi_missing.interpolate(method="linear"),
            TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": [
                            "2010-01-01",
                            "2010-01-02",
                            "2010-01-03",
                            "2010-01-04",
                            "2010-01-05",
                        ],
                        "value1": [1, 2, 3, 3.5, 4],
                        "value2": [4, 3, 2, 1.5, 1],
                    }
                )
            ),
        )

    def test_to_array(self) -> None:
        # Univariate case
        np.testing.assert_array_equal(
            self.ts_from_df.to_array(), AIR_DF_DATETIME.to_numpy()
        )
        # Multivariate case
        np.testing.assert_array_equal(
            self.ts_from_df_multi_datetime.to_array(),
            MULTIVAR_AIR_DF_DATETIME.to_numpy(),
        )
        # Series Cases
        np.testing.assert_array_equal(
            self.ts_from_series_univar_no_datetime.to_array(),
            AIR_DF_DATETIME.to_numpy(),
        )
        np.testing.assert_array_equal(
            self.ts_from_series_univar_with_datetime.to_array(),
            AIR_DF_DATETIME.to_numpy(),
        )
        # Series/DataFrame Cases
        np.testing.assert_array_equal(
            self.ts_from_series_and_df_univar.to_array(), AIR_DF_DATETIME.to_numpy()
        )
        np.testing.assert_array_equal(
            self.ts_from_series_and_df_multivar.to_array(),
            MULTIVAR_AIR_DF_DATETIME.to_numpy(),
        )
        # Empty/None Cases
        np.testing.assert_array_equal(self.ts_df_none.to_array(), np.empty)
        np.testing.assert_array_equal(
            self.ts_time_none_and_value_none.to_array(), np.empty
        )
        np.testing.assert_array_equal(self.ts_df_empty.to_array(), np.empty)
        np.testing.assert_array_equal(
            self.ts_time_empty_value_empty.to_array(), np.empty
        )
        np.testing.assert_array_equal(
            self.ts_time_empty_value_empty_df.to_array(), np.empty
        )

    def test_tz(self) -> None:
        self.ts_univar_PST_tz.validate_data(
            validate_frequency=True, validate_dimension=True
        )
        self.assertEqual(self.ts_univar_PST_tz.freq_to_timedelta(), pd.Timedelta("1d"))
        self.assertEqual(self.ts_univar_PST_tz.tz(), pytz.timezone("US/Pacific"))
        self.assertTrue(
            (
                np.array(self.unix_list)
                == (self.ts_univar_PST_tz.time.values.astype(int) / 1e9).astype(int)
            ).all()
        )

        with self.assertRaisesRegex(
            ValueError, "Only constant frequency is supported for time!"
        ):
            self.ts_univar_PST.validate_data(
                validate_frequency=True, validate_dimension=True
            )

        self.ts_univar_str_date.validate_data(
            validate_frequency=True, validate_dimension=True
        )
        self.assertEqual(
            self.ts_univar_str_date.freq_to_timedelta(), pd.Timedelta("1d")
        )

        self.ts_univar_str_date_tz.validate_data(
            validate_frequency=True, validate_dimension=True
        )
        self.assertEqual(
            self.ts_univar_str_date_tz.freq_to_timedelta(), pd.Timedelta("1d")
        )
        self.assertEqual(self.ts_univar_PST_tz.tz(), pytz.timezone("US/Pacific"))

        # test ambiguous
        tsd = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": [
                        "2018-10-28 01:30:00",
                        "2018-10-28 02:00:00",
                        "2018-10-28 02:30:00",
                        "2018-10-28 02:00:00",
                        "2018-10-28 02:30:00",
                        "2018-10-28 03:00:00",
                        "2018-10-28 03:30:00",
                    ],
                    "value": [0] * 7,
                }
            ),
            tz="CET",
            tz_ambiguous="infer",
        )
        tsd.validate_data(validate_frequency=True, validate_dimension=True)

        # test nonexistent
        tsd = TimeSeriesData(
            df=pd.DataFrame(
                {
                    "time": [
                        "2020-03-08 02:00:00",
                        "2020-03-08 02:30:00",
                        "2020-03-08 03:00:00",
                    ],
                    "value": [0] * 3,
                }
            ),
            tz="US/Pacific",
            tz_nonexistent="shift_forward",
        )

    def test_infer_freq_robust(self) -> None:
        self.assertEqual(
            self.ts_univariate_missing.infer_freq_robust(),
            pd.Timedelta(value=1, unit="D"),
        )

        self.assertEqual(
            self.ts_univar_PST_missing_tz.infer_freq_robust(),
            pd.Timedelta(value=1, unit="D"),
        )

    def test_is_data_missing(self) -> None:
        self.assertEqual(self.ts_univariate_missing.is_data_missing(), True)

        self.assertEqual(self.ts_univar_PST_missing_tz.is_data_missing(), True)

        self.assertEqual(self.ts_from_series_and_df_univar.is_data_missing(), False)

        self.assertEqual(self.ts_from_series_and_df_multivar.is_data_missing(), False)

    def test_min_max_values(self) -> None:
        # test min/max value for univariate
        self.assertEqual(self.ts_from_df.min, np.nanmin(self.ts_from_df.value.values))
        self.assertEqual(self.ts_from_df.max, np.nanmax(self.ts_from_df.value.values))

        # test min/max value for multivariate
        self.assertEqual(
            # pyre-fixme[16]: `float` has no attribute `equals`.
            self.ts_from_df_multi.min.equals(
                self.ts_from_df_multi.value.min(skipna=True)
            ),
            True,
        )
        self.assertEqual(
            self.ts_from_df_multi.max.equals(
                self.ts_from_df_multi.value.max(skipna=True)
            ),
            True,
        )

        # test min/max value for empty TS
        empty_ts = TimeSeriesData(pd.DataFrame())
        self.assertEqual(np.isnan(empty_ts.min), True)
        self.assertEqual(np.isnan(empty_ts.max), True)

        # test if min/max changes if values are re-assigned for univariate
        ts_from_df_new = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        new_val = np.random.randn(len(AIR_DF))
        ts_from_df_new.value = pd.Series(new_val)
        self.assertEqual(ts_from_df_new.min, np.min(new_val))
        self.assertEqual(ts_from_df_new.max, np.max(new_val))

        # test if min/max changes if values are re-assigned with NaNs for univariate
        new_val[-1] = np.nan
        ts_from_df_new.value = pd.Series(new_val)
        self.assertEqual(ts_from_df_new.min, np.nanmin(new_val))
        self.assertEqual(ts_from_df_new.max, np.nanmax(new_val))

        # test min/max changes if values are re-assigned for multivariate
        ts_from_df_multi_new = TimeSeriesData(
            MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        new_val_multi = np.random.randn(
            MULTIVAR_VALUE_DF.shape[0], MULTIVAR_VALUE_DF.shape[1] - 1
        )
        ts_from_df_multi_new.value = pd.DataFrame(new_val_multi)
        self.assertEqual(
            ts_from_df_multi_new.min.equals(pd.DataFrame(new_val_multi).min()), True
        )
        self.assertEqual(
            ts_from_df_multi_new.max.equals(pd.DataFrame(new_val_multi).max()), True
        )

        # test min/max changes if values are re-assigned with NaNs for multivariate
        new_val_multi[0] = np.nan
        ts_from_df_multi_new.value = pd.DataFrame(new_val_multi)
        self.assertEqual(
            ts_from_df_multi_new.min.equals(
                pd.DataFrame(new_val_multi).min(skipna=True)
            ),
            True,
        )
        self.assertEqual(
            ts_from_df_multi_new.max.equals(
                pd.DataFrame(new_val_multi).max(skipna=True)
            ),
            True,
        )


class TimeSeriesDataOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Creating DataFrames
        # DataFrame with date offset
        transformed_df_date = AIR_DF_DATETIME.copy(deep=True)
        transformed_df_date.ds = transformed_df_date.ds.apply(
            lambda x: x + relativedelta(years=NUM_YEARS_OFFSET)
        )
        transformed_df_date_concat = AIR_DF.append(
            transformed_df_date, ignore_index=True
        )
        transformed_df_date_double = AIR_DF_DATETIME.copy(deep=True)
        transformed_df_date_double.ds = transformed_df_date.ds.apply(
            lambda x: x + relativedelta(years=NUM_YEARS_OFFSET * 2)
        )
        transformed_df_date_concat_double = AIR_DF.append(
            transformed_df_date_double, ignore_index=True
        )
        # DataFrames with value offset
        transformed_df_value = AIR_DF.copy(deep=True)
        transformed_df_value.y = transformed_df_value.y.apply(lambda x: x * 2)
        transformed_df_value_inv = AIR_DF.copy(deep=True)
        transformed_df_value_inv.y = transformed_df_value_inv.y.apply(lambda x: x * -1)
        # DataFrame with date and value offset
        transformed_df_date_and_value = transformed_df_date.copy(deep=True)
        transformed_df_date_and_value.y = transformed_df_date_and_value.y.apply(
            lambda x: x * 2
        )
        # DataFrame with date offset (multivariate)
        transformed_df_date_multi = transformed_df_date.copy(deep=True)
        transformed_df_date_multi[VALUE_COL_NAME + "_1"] = (
            transformed_df_date_multi.y * 2
        )
        transformed_df_date_concat_multi = MULTIVAR_AIR_DF.append(
            transformed_df_date_multi, ignore_index=True
        )
        transformed_df_date_concat_mixed = MULTIVAR_AIR_DF_DATETIME.append(
            transformed_df_date
        )
        transformed_df_date_double_multi = transformed_df_date_double.copy(deep=True)
        transformed_df_date_double_multi[VALUE_COL_NAME + "_1"] = (
            transformed_df_date_double_multi.y * 2
        )
        transformed_df_date_concat_double_multi = MULTIVAR_AIR_DF.append(
            transformed_df_date_double_multi, ignore_index=True
        )
        transformed_df_date_concat_double_mixed = MULTIVAR_AIR_DF_DATETIME.append(
            transformed_df_date_double
        )
        # DataFrame with value offset (multivariate)
        transformed_df_value_none_multi = MULTIVAR_AIR_DF.copy(deep=True)
        transformed_df_value_none_multi.y = transformed_df_value_none_multi.y_1
        transformed_df_value_none_multi.y_1 = np.nan
        # DataFrame with date and value offset (multivariate)
        transformed_df_date_and_value_multi = transformed_df_date_and_value.copy(
            deep=True
        )
        transformed_df_date_and_value_multi[VALUE_COL_NAME + "_1"] = (
            transformed_df_date_and_value_multi.y * 2
        )
        # DataFrame with all constant values
        df_zeros = AIR_DF.copy(deep=True)
        df_zeros.y.values[:] = 0
        df_ones = AIR_DF.copy(deep=True)
        df_ones.y.values[:] = 1
        df_twos = df_ones.copy(deep=True)
        df_twos.y.values[:] = 2
        df_neg_ones = AIR_DF.copy(deep=True)
        df_neg_ones.y.values[:] = -1
        df_ones_multi = df_ones.copy(deep=True)
        df_ones_multi[VALUE_COL_NAME + "_1"] = df_ones_multi.y * 2

        # Creating TimeSeriesData objects
        # Univariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_univ_1 = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_univ_2 = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_univ_default_names = TimeSeriesData(df=AIR_DF_WITH_DEFAULT_NAMES)
        cls.ts_univ_default_names_2 = TimeSeriesData(df=AIR_DF_WITH_DEFAULT_NAMES)

        # Multivariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_multi_1 = TimeSeriesData(df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_multi_2 = TimeSeriesData(df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME)

        # TimeSeriesData with date offset
        cls.ts_date_transform_univ = TimeSeriesData(
            df=transformed_df_date, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_univ = TimeSeriesData(
            df=transformed_df_date_concat, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_double_univ = TimeSeriesData(
            df=transformed_df_date_double, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_double_univ = TimeSeriesData(
            df=transformed_df_date_concat_double, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData with date offset (multivariate)
        cls.ts_date_transform_multi = TimeSeriesData(
            df=transformed_df_date_multi, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_multi = TimeSeriesData(
            df=transformed_df_date_concat_multi, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_mixed = TimeSeriesData(
            df=transformed_df_date_concat_mixed, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_double_multi = TimeSeriesData(
            df=transformed_df_date_double_multi, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_double_multi = TimeSeriesData(
            df=transformed_df_date_concat_double_multi, time_col_name=TIME_COL_NAME
        )
        cls.ts_date_transform_concat_double_mixed = TimeSeriesData(
            df=transformed_df_date_concat_double_mixed, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData with value offset
        cls.ts_value_transform_univ = TimeSeriesData(
            df=transformed_df_value, time_col_name=TIME_COL_NAME
        )
        cls.ts_value_transform_inv_univ = TimeSeriesData(
            df=transformed_df_value_inv, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData with value offset (multivariate)
        cls.ts_value_transform_none_multi = TimeSeriesData(
            df=transformed_df_value_none_multi, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData with date and value offset
        cls.ts_date_and_value_transform_univ = TimeSeriesData(
            df=transformed_df_date_and_value, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData with date and value offset (multivariate)
        cls.ts_date_and_value_transform_multi = TimeSeriesData(
            df=transformed_df_date_and_value_multi, time_col_name=TIME_COL_NAME
        )
        # TimeSeriesData object with all constant values
        cls.ts_zero = TimeSeriesData(df=df_zeros, time_col_name=TIME_COL_NAME)
        cls.ts_ones = TimeSeriesData(df=df_ones, time_col_name=TIME_COL_NAME)
        cls.ts_twos = TimeSeriesData(df=df_twos, time_col_name=TIME_COL_NAME)
        cls.ts_neg_ones = TimeSeriesData(df=df_neg_ones, time_col_name=TIME_COL_NAME)
        cls.ts_ones_multi = TimeSeriesData(
            df=df_ones_multi, time_col_name=TIME_COL_NAME
        )
        # Empty TimeSeriesData Object
        cls.ts_empty = TimeSeriesData(df=EMPTY_DF)
        cls.ts_empty_with_cols = TimeSeriesData(
            df=EMPTY_DF_WITH_COLS, time_col_name=TIME_COL_NAME
        )
        # Copies for Extended objects
        cls.ts_univ_extend = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_univ_extend_2 = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_univ_extend_err = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        cls.ts_multi_extend = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_multi_extend_2 = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_multi_extend_3 = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_multi_extend_4 = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_multi_extend_err = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_multi_extend_err_2 = TimeSeriesData(
            df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME
        )
        cls.ts_empty_extend = TimeSeriesData(df=EMPTY_DF)
        cls.ts_empty_extend_err = TimeSeriesData(df=EMPTY_DF)

        # Other values
        cls.length = len(AIR_DF)

    def test_eq(self) -> None:
        # Univariate equality
        self.assertTrue(self.ts_univ_1 == self.ts_univ_2)
        # Multivariate equality
        self.assertTrue(self.ts_multi_1 == self.ts_multi_2)
        # Univariate inequality
        self.assertFalse(self.ts_univ_1 == self.ts_date_transform_univ)
        self.assertFalse(self.ts_univ_1 == self.ts_value_transform_univ)
        self.assertFalse(self.ts_univ_1 == self.ts_date_and_value_transform_univ)
        # Multivariate inequality
        self.assertFalse(self.ts_multi_1 == self.ts_date_transform_multi)
        self.assertFalse(self.ts_multi_1 == self.ts_value_transform_none_multi)
        self.assertFalse(self.ts_multi_1 == self.ts_date_and_value_transform_multi)
        # Univariate vs. Multivariate inequality
        self.assertFalse(self.ts_univ_1 == self.ts_multi_1)
        self.assertFalse(self.ts_multi_1 == self.ts_univ_1)

    def test_ne(self) -> None:
        # Univariate equality
        self.assertFalse(self.ts_univ_1 != self.ts_univ_2)
        # Multivariate equality
        self.assertFalse(self.ts_multi_1 != self.ts_multi_2)
        # Univariate inequality
        self.assertTrue(self.ts_univ_1 != self.ts_date_transform_univ)
        self.assertTrue(self.ts_univ_1 != self.ts_value_transform_univ)
        self.assertTrue(self.ts_univ_1 != self.ts_date_and_value_transform_univ)
        # Multivariate inequality
        self.assertTrue(self.ts_multi_1 != self.ts_date_transform_multi)
        self.assertTrue(self.ts_multi_1 != self.ts_value_transform_none_multi)
        self.assertTrue(self.ts_multi_1 != self.ts_date_and_value_transform_multi)
        # Univariate vs. Multivariate inequality
        self.assertTrue(self.ts_univ_1 != self.ts_multi_1)
        self.assertTrue(self.ts_multi_1 != self.ts_univ_1)

    def test_add(self) -> None:
        # Add same DataFrames
        self.assertEqual(self.ts_univ_1 + self.ts_univ_2, self.ts_value_transform_univ)
        # Add different DataFrames
        self.assertEqual(
            self.ts_univ_1 + self.ts_value_transform_inv_univ, self.ts_zero
        )
        # Add Univariate and Multivariate DataFrames
        self.assertEqual(
            self.ts_univ_1 + self.ts_multi_1, self.ts_value_transform_none_multi
        )
        # Empty Case
        self.assertEqual(self.ts_empty + self.ts_empty, self.ts_empty)
        # Add DataFrames with different dates
        with self.assertRaises(ValueError):
            self.ts_univ_1 + self.ts_date_transform_univ

    def test_sub(self) -> None:
        # Subtract same DataFrames
        self.assertEqual(self.ts_univ_1 - self.ts_univ_2, self.ts_zero)
        # Subtract different DataFrames
        self.assertEqual(
            self.ts_univ_1 - self.ts_value_transform_inv_univ,
            self.ts_value_transform_univ,
        )
        # Subtract Univariate and Multivariate DataFrames
        self.assertEqual(
            self.ts_multi_1 - self.ts_value_transform_inv_univ,
            self.ts_value_transform_none_multi,
        )
        # Empty Case
        self.assertEqual(self.ts_empty - self.ts_empty, self.ts_empty)
        # Subtract DataFrames with different dates
        with self.assertRaises(ValueError):
            self.ts_univ_1 - self.ts_date_transform_univ

    def test_div(self) -> None:
        # Divide same DataFrames
        self.assertEqual(self.ts_univ_1 / self.ts_univ_2, self.ts_ones)
        # Divide different DataFrames
        self.assertEqual(
            self.ts_univ_1 / self.ts_value_transform_inv_univ, self.ts_neg_ones
        )
        # Divide Univariate and Multivariate DataFrames
        self.assertEqual(
            self.ts_value_transform_univ / self.ts_ones_multi,
            self.ts_value_transform_none_multi,
        )
        # Empty Case
        self.assertEqual(self.ts_empty / self.ts_empty, self.ts_empty)
        # Divide DataFrames with different dates
        with self.assertRaises(ValueError):
            self.ts_univ_1 / self.ts_date_transform_univ

    def test_mul(self) -> None:
        # Multiply same DataFrames
        self.assertEqual(self.ts_ones * self.ts_ones, self.ts_ones)
        # Multiply different DataFrames
        self.assertEqual(self.ts_univ_1 * self.ts_twos, self.ts_value_transform_univ)
        # Multiply Univariate and Multivariate DataFrames
        self.assertEqual(
            self.ts_multi_1 * self.ts_twos, self.ts_value_transform_none_multi
        )
        # Empty Case
        self.assertEqual(self.ts_empty * self.ts_empty, self.ts_empty)
        # Multiply DataFrames with different dates
        with self.assertRaises(ValueError):
            self.ts_univ_1 * self.ts_date_transform_univ

    def test_len(self) -> None:
        # Normal case
        self.assertEqual(len(self.ts_univ_1), self.length)
        # Empty case
        self.assertEqual(len(self.ts_empty), 0)

    def test_empty(self) -> None:
        # Empty case
        self.assertTrue(self.ts_empty.is_empty())
        # Not empty case
        self.assertFalse(self.ts_univ_1.is_empty())

    def test_extend(self) -> None:
        # Testing cases with validate=True
        # Univariate case
        self.ts_univ_extend.extend(self.ts_date_transform_univ)
        self.assertEqual(self.ts_univ_extend, self.ts_date_transform_concat_univ)
        # Multivariate case
        self.ts_multi_extend.extend(self.ts_date_transform_multi)
        self.assertEqual(self.ts_multi_extend, self.ts_date_transform_concat_multi)
        # Univariate and multivariate case
        self.ts_multi_extend_2.extend(self.ts_date_transform_univ)
        self.assertEqual(self.ts_multi_extend_2, self.ts_date_transform_concat_mixed)
        # Empty case
        self.ts_univ_default_names.extend(self.ts_empty)
        self.assertEqual(self.ts_univ_default_names, self.ts_univ_default_names_2)
        # Catching errors
        with self.assertRaises(ValueError):
            self.ts_univ_extend_err.extend(self.ts_date_transform_double_univ)
            # Multivariate case
            self.ts_multi_extend_err.extend(self.ts_date_transform_double_multi)
            # Univariate and multivariate case
            self.ts_multi_extend_err_2.extend(self.ts_date_transform_double_univ)
            # Empty case
            self.ts_empty_extend_err.extend(self.ts_empty)
        # Testing cases with validate=False
        # Univariate case
        self.ts_univ_extend_2.extend(self.ts_date_transform_double_univ, validate=False)
        self.assertEqual(
            self.ts_univ_extend_2, self.ts_date_transform_concat_double_univ
        )
        # Multivariate case
        self.ts_multi_extend_3.extend(
            self.ts_date_transform_double_multi, validate=False
        )
        self.assertEqual(
            self.ts_multi_extend_3, self.ts_date_transform_concat_double_multi
        )
        # Univariate and multivariate case
        self.ts_multi_extend_4.extend(
            self.ts_date_transform_double_univ, validate=False
        )
        self.assertEqual(
            self.ts_multi_extend_4, self.ts_date_transform_concat_double_mixed
        )
        # Empty case
        self.ts_empty_extend.extend(self.ts_empty, validate=False)
        self.assertEqual(self.ts_empty_extend, self.ts_empty)

    def test_get_item(self) -> None:
        # Univariate test case
        self.assertEqual(
            self.ts_date_transform_concat_univ[: len(self.ts_univ_1)], self.ts_univ_1
        )
        # Multivariate test case
        self.assertEqual(
            self.ts_date_transform_concat_multi[: len(self.ts_multi_1)], self.ts_multi_1
        )
        # Full/Empty cases
        self.assertEqual(self.ts_univ_1[:], self.ts_univ_1)
        self.assertEqual(
            self.ts_univ_1[0:0],
            TimeSeriesData(
                time=pd.Series(name=TIME_COL_NAME),
                value=pd.Series(name=VALUE_COL_NAME),
                time_col_name=TIME_COL_NAME,
            )
        )

    def test_plot(self) -> None:
        # Univariate test case
        print(self.ts_univ_1.to_dataframe().head())
        print(self.ts_univ_1.to_dataframe().columns)
        self.ts_univ_1.plot(cols = ["y"])

        # Multivariate test case
        self.ts_multi_1.plot(cols = ["y", "y_1"])


class TimeSeriesDataMiscTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Creating TimeSeriesData objects
        # Univariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_univ = TimeSeriesData(df=AIR_DF, time_col_name=TIME_COL_NAME)
        # Multivariate TimeSeriesData initialized from a pd.DataFrame
        cls.ts_multi = TimeSeriesData(df=MULTIVAR_AIR_DF, time_col_name=TIME_COL_NAME)

    def test_is_univariate(self) -> None:
        # Univariate case
        self.assertTrue(self.ts_univ.is_univariate())
        # Multivariate case
        self.assertFalse(self.ts_multi.is_univariate())

    def test_time_to_index(self) -> None:
        # Univariate case
        assert_index_equal(self.ts_univ.time_to_index(), AIR_TIME_DATETIME_INDEX)
        # Multivariate case
        assert_index_equal(self.ts_multi.time_to_index(), AIR_TIME_DATETIME_INDEX)

    def test_repr(self) -> None:
        # Univariate case
        self.assertEqual(self.ts_univ.__repr__(), AIR_DF_DATETIME.__repr__())
        # Multivariate case
        self.assertEqual(self.ts_multi.__repr__(), MULTIVAR_AIR_DF_DATETIME.__repr__())

    def test_repr_html(self) -> None:
        # Univariate case
        self.assertEqual(self.ts_univ._repr_html_(), AIR_DF_DATETIME._repr_html_())
        # Multivariate case
        self.assertEqual(
            self.ts_multi._repr_html_(), MULTIVAR_AIR_DF_DATETIME._repr_html_()
        )


class TSIteratorTest(unittest.TestCase):
    def test_ts_iterator_univariate_next(self) -> None:
        df = pd.DataFrame(
            [["2020-03-01", 100], ["2020-03-02", 120], ["2020-03-03", 130]],
            columns=["time", "y"],
        )
        kats_data = TimeSeriesData(df=df)
        kats_iterator = TSIterator(kats_data)
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-01")]))
        assert_series_equal(val.value, pd.Series([100]))
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-02")]))
        assert_series_equal(val.value, pd.Series([120]))
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-03")]))
        assert_series_equal(val.value, pd.Series([130]))

    def test_ts_iterator_multivariate_next(self) -> None:
        df = pd.DataFrame(
            [
                ["2020-03-01", 100, 200],
                ["2020-03-02", 120, 220],
                ["2020-03-03", 130, 230],
            ],
            columns=["time", "y1", "y2"],
        )
        kats_data = TimeSeriesData(df=df)
        kats_iterator = TSIterator(kats_data)
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-01")]))
        assert_series_equal(val.value, pd.Series([100, 200], name=0))
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-02")]))
        assert_series_equal(val.value, pd.Series([120, 220], name=1))
        val = next(kats_iterator)
        assert_series_equal(val.time, pd.Series([pd.Timestamp("2020-03-03")]))
        assert_series_equal(val.value, pd.Series([130, 230], name=2))
    def test_ts_iterator_comprehension(self) -> None:
        kats_data = TimeSeriesData(
            time=pd.to_datetime(
                np.array([1596225347, 1596225348, 1596225349]), unit="s", utc=True
            ),
            value=pd.Series(np.array([1, 2, 4])),
        )
        kats_iterator = TSIterator(kats_data)
        kats_list = list(kats_iterator)
        val = kats_list[0]
        assert_series_equal(
            val.time, pd.Series([pd.Timestamp("2020-07-31 19:55:47+0000", tz="UTC")])
        )
        assert_series_equal(val.value, pd.Series([1]))
        val = kats_list[1]
        assert_series_equal(
            val.time, pd.Series([pd.Timestamp("2020-07-31 19:55:48+0000", tz="UTC")])
        )
        assert_series_equal(val.value, pd.Series([2]))
        val = kats_list[2]
        assert_series_equal(
            val.time, pd.Series([pd.Timestamp("2020-07-31 19:55:49+0000", tz="UTC")])
        )
        assert_series_equal(val.value, pd.Series([4]))
