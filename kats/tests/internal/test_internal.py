# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import re
import unittest
from datetime import datetime, time, timedelta
from statistics import median
from unittest import TestCase

import numpy as np
import pandas as pd
from analytics.bamboo import Bamboo as bb
from kats.consts import Params, TimeSeriesData
from kats.detectors import changepoint_evaluator, robust_stat_detection
from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.outlier import OutlierDetector
from kats.internal.internal_benchmark_data.detector_annotation import (
    plotly_timeseries,
    AnnotatedTimeSeries,
)
from kats.internal.internal_benchmark_data.detector_evaluation import (
    FBDetectBenchmarkEvaluator,
)
from kats.internal.special_events.pinnacle_forecast import (
    PinnacleSpecialEventModel,
    PinnacleSpecialEventParams,
)
from kats.internal.special_events.special_events_base import (
    BaseSpecialEventsModel,
)
from kats.internal.special_events.special_events_basic import (
    BasicSpecialEventsSingleTSModel,
    BasicSpecialEventsSingleTSParams,
)
from kats.utils.simulator import Simulator
from libfb.py.testutil import is_devserver


DEVSERVER_TEST = unittest.skipUnless(is_devserver(), "Tests only run on devservers.")

DATA = pd.read_csv("kats/kats/data/air_passengers.csv")
DATA.columns = ["time", "y"]

DATA_daily = pd.read_csv("kats/kats/data/peyton_manning.csv")
DATA_daily.columns = ["time", "y"]

if "kats/tests" in os.getcwd():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname("__file__"), "../", "data/air_passengers.csv")
    )

    daily_data_path = os.path.abspath(
        os.path.join(os.path.dirname("__file__"), "../", "data/peyton_manning.csv")
    )
else:
    data_path = "kats/kats/data/air_passengers.csv"
    daily_data_path = "kats/kats/data/peyton_manning.csv"

DATA = pd.read_csv(data_path)
DATA.columns = ["time", "y"]

DATA_daily = pd.read_csv(daily_data_path)
DATA_daily.columns = ["time", "y"]


def read_fb_alm_minute(country, list_ds, n_bases):

    all_list_ds = []
    for ds in list_ds:
        ds = datetime.strptime(ds, "%Y-%m-%d")
        for i in range(n_bases):
            base_ds = ds - timedelta(days=7 * (i + 1))
            all_list_ds.append("'" + base_ds.strftime("%Y-%m-%d") + "'")
        all_list_ds.append("'" + ds.strftime("%Y-%m-%d") + "'")

    all_list_ds = ",".join(all_list_ds)

    sql = f"""
                SELECT event_time, alm_count_ex_bg
                FROM cea_alm_country
                WHERE country = '{country}'
                      AND ds in ({all_list_ds})
            """

    df = bb.presto(sql, "cea")

    return df


class DataValidationTest(TestCase):
    def test_data_validation(self) -> None:
        # add the extra data point to break the frequency.
        extra_point = pd.DataFrame(
            [["1900-01-01", 2], ["2020-01-01", 2]], columns=["time", "y"]
        )
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


class BaseSpecialEventsModelTest(TestCase):
    def test_params_validation(self) -> None:

        # params for now empty
        params = Params()

        # test special events table
        special_events_df = pd.DataFrame(
            data=[
                ("NYE", datetime(2020, 12, 31).date()),
                ("Election 2020", datetime(2020, 11, 8)),
            ],
            columns=["special_event", "ds"],
        )

        bad_special_events_df = special_events_df.copy().rename(
            columns={"special_event": "wrong_column_name"}
        )

        m = BaseSpecialEventsModel(
            data=DATA, params=params, special_events=special_events_df
        )
        m.validate_inputs()

        with self.assertRaises(
            ValueError, msg="Column name for the special event table should fail"
        ):
            m = BaseSpecialEventsModel(
                data=DATA, params=params, special_events=bad_special_events_df
            )


class BasicSpecialEventsSingleTSModelTest(TestCase):
    def test_params_validation(self) -> None:

        historical_special_events_incorrect_format1 = pd.DataFrame(
            {
                "holidays": "fake",
                "ds": pd.to_datetime(
                    [
                        "2012-05-30",
                        "2012-08-11",
                        "2012-10-05",
                        "2013-01-24",
                        "2013-04-04",
                    ]
                ),
            }
        )

        historical_special_events_incorrect_format2 = pd.DataFrame(
            {
                "special_event": "fake",
                "date": pd.to_datetime(
                    [
                        "2012-05-30",
                        "2012-08-11",
                        "2012-10-05",
                        "2013-01-24",
                        "2013-04-04",
                    ]
                ),
            }
        )

        future_special_events_correct_format = pd.DataFrame(
            {"special_event": "fake", "ds": pd.to_datetime(["2013-06-08"])}
        )

        future_special_events_incorrect_format1 = pd.DataFrame(
            {"special_event": "fake", "date": pd.to_datetime(["2013-06-08"])}
        )

        future_special_events_incorrect_format2 = pd.DataFrame(
            {"holidays": "fake", "ds": pd.to_datetime(["2013-06-08"])}
        )

        with self.assertRaises(
            ValueError, msg="Please provide time column in your special_events data"
        ):
            BasicSpecialEventsSingleTSParams(
                future_special_events=future_special_events_incorrect_format1,
                granularity_data="days",
                base_seasonality="weekly",
                base_window=4,
                keep_intermediate_results=True,
            )

        with self.assertRaises(
            ValueError,
            msg="Please provide special_event column in your special_events data",
        ):
            BasicSpecialEventsSingleTSParams(
                future_special_events=future_special_events_incorrect_format2,
                granularity_data="days",
                base_seasonality="weekly",
                base_window=4,
                keep_intermediate_results=True,
            )

        params = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events_correct_format,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        with self.assertRaises(
            ValueError,
            msg="Please provide special_event column in your special_events data",
        ):
            BasicSpecialEventsSingleTSModel(
                DATA_daily,
                params=params,
                special_events=historical_special_events_incorrect_format1,
            )

        with self.assertRaises(
            ValueError, msg="Please provide time column in your special_events data"
        ):
            BasicSpecialEventsSingleTSModel(
                DATA_daily,
                params=params,
                special_events=historical_special_events_incorrect_format2,
            )

    def test_fit_forecast_daily_granularity(self) -> None:

        historical_special_events = pd.DataFrame(
            {
                "special_event": "fake",
                "ds": pd.to_datetime(
                    [
                        "2012-05-30",
                        "2012-08-11",
                        "2012-10-05",
                        "2013-01-24",
                        "2013-04-04",
                    ]
                ),
                "lower_window": 0,
                "upper_window": 0,
            }
        )

        future_special_events = pd.DataFrame(
            {"special_event": "fake", "ds": pd.to_datetime(["2013-06-08"])}
        )

        params = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        m = BasicSpecialEventsSingleTSModel(
            DATA_daily, params=params, special_events=historical_special_events
        )

        fcst = m.predict()

        # checks intermediate result
        all_special_events = {}
        for col in m.processed_data.columns:
            special_event_name = re.sub("__.*", "", col)
            if re.search("__", col) and special_event_name not in all_special_events:
                all_special_events[special_event_name] = []
            if re.search("_base$", col):
                event_name = re.sub("_base", "", col)
                self.assertEqual(
                    m.processed_data.loc[
                        ~(m.processed_data.is_event), event_name
                    ].median(),
                    m.processed_data.loc[m.processed_data.is_event, col].values[0],
                )
            elif re.search("_multiplier$", col):
                event_name = re.sub("_multiplier", "", col)
                multi = (
                    1
                    + (
                        m.processed_data.loc[
                            m.processed_data.is_event, event_name
                        ].values[0]
                        - m.processed_data.loc[
                            ~(m.processed_data.is_event), event_name
                        ].median()
                    )
                    / m.processed_data.loc[
                        ~(m.processed_data.is_event), event_name
                    ].median()
                )
                self.assertEqual(
                    multi,
                    m.processed_data.loc[m.processed_data.is_event, col].values[0],
                )
                all_special_events[special_event_name].append(multi)

        for special_event in all_special_events:
            self.assertEqual(
                np.mean(all_special_events[special_event]),
                m.processed_data.loc[
                    m.processed_data.is_event, f"mean_{special_event}"
                ].values[0],
            )
            print(all_special_events[special_event])
            self.assertEqual(
                np.std(
                    all_special_events[special_event], ddof=1
                ),  # pandas default ddof is 1
                m.processed_data.loc[
                    m.processed_data.is_event, f"std_{special_event}"
                ].values[0],
            )
            for p in params.percentiles:
                p_num = float(p[1:]) / 100
                self.assertEqual(
                    np.quantile(all_special_events[special_event], p_num),
                    m.processed_data.loc[
                        m.processed_data.is_event, f"{p}_{special_event}"
                    ].values[0],
                )

        # checks the predicted forest results
        for prefix in params.multiplier_prefixes:
            self.assertTrue(
                abs(
                    fcst.loc[
                        fcst.time == np.datetime64("2013-06-08T00:00:00.000000000"),
                        "fcst",
                    ].values[0]
                    * m.processed_data.loc[
                        (m.processed_data.days_to_event == 0), f"{prefix}_fake"
                    ].values[0]
                    - fcst.loc[
                        fcst.time == np.datetime64("2013-06-08T00:00:00.000000000"),
                        f"{prefix}_fcst_adjusted",
                    ].values[0]
                )
                < 0.0001
            )

    @DEVSERVER_TEST
    def test_fit_forecast_minute_granularity(self) -> None:

        # historical actuals
        minute_df = read_fb_alm_minute(
            country="US",
            list_ds=[
                "2019-12-31",
                "2018-12-31",
                "2017-12-31",
                "2016-12-31",
                "2015-12-31",
            ],
            n_bases=4,
        )

        # forecast that needs to be adjusted
        # for now let's fake it
        fcst_minute_2020 = (
            minute_df[minute_df.event_time >= "2019-12-31"]
            .rename(columns={"event_time": "time", "alm_count_ex_bg": "fcst"})
            .copy()
        )
        fcst_minute_2020.time = fcst_minute_2020.time.map(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            .replace(year=2020)
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        # future special event
        predict_nye = pd.DataFrame(
            {"special_event": "NYE", "ds": pd.to_datetime(["2020-12-31"])}
        )

        # set up the parameters
        params = BasicSpecialEventsSingleTSParams(
            future_special_events=predict_nye,
            granularity_data="minutes",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        # historical special events
        all_nye = pd.DataFrame(
            {
                "special_event": "NYE",
                "ds": pd.to_datetime(
                    [
                        "2020-12-31",
                        "2019-12-31",
                        "2018-12-31",
                        "2017-12-31",
                        "2016-12-31",
                        "2015-12-31",
                    ]
                ),
                "lower_window": 0,
                "upper_window": 0,
            }
        )

        # set up the model
        m = BasicSpecialEventsSingleTSModel(
            minute_df.rename(columns={"event_time": "time", "alm_count_ex_bg": "y"}),
            params=params,
            special_events=all_nye,
            original_fcst=fcst_minute_2020,
        )

        # predict
        m.fit()
        fcst = m.predict()

        # set up the model with no forecast
        m_no_fcst = BasicSpecialEventsSingleTSModel(
            minute_df.rename(columns={"event_time": "time", "alm_count_ex_bg": "y"}),
            params=params,
            special_events=all_nye,
            generate_forecast=False,
        )

        # predict
        m_no_fcst.fit()
        no_fcst = m_no_fcst.predict()

        # testing that no fcst is output when generate_forecast is set to False
        for col in no_fcst.columns:
            self.assertTrue("fcst" not in col)

        # testing if output of no_fcst is the same as the the pocessed data of the model with forecast
        self.assertTrue(no_fcst.equals(m.processed_data))

        _time = time(0, 19, 0)

        self.assertEqual(
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "NYE__2015-12-31_base",
            ].values[0],
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event % 7 == 0)
                & (m.processed_data.days_to_event != 0),
                "NYE__2015-12-31",
            ].median(),
        )

        # check the mutliplier for the 2015-12-31 NYE is calculated correctly
        self.assertEqual(
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "NYE__2015-12-31_multiplier",
            ].values[0],
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "NYE__2015-12-31",
            ].values[0]
            / m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "NYE__2015-12-31_base",
            ].values[0],
        )

        # check the mutliplier for NYE is calculated correctly
        self.assertEqual(
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "mean_NYE",
            ].values[0],
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                [
                    "NYE__2015-12-31_multiplier",
                    "NYE__2016-12-31_multiplier",
                    "NYE__2017-12-31_multiplier",
                    "NYE__2018-12-31_multiplier",
                    "NYE__2019-12-31_multiplier",
                ],
            ]
            .mean(axis=1)
            .values[0],
        )
        self.assertEqual(
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                "std_NYE",
            ].values[0],
            m.processed_data.loc[
                (m.processed_data.minutes == _time)
                & (m.processed_data.days_to_event == 0),
                [
                    "NYE__2015-12-31_multiplier",
                    "NYE__2016-12-31_multiplier",
                    "NYE__2017-12-31_multiplier",
                    "NYE__2018-12-31_multiplier",
                    "NYE__2019-12-31_multiplier",
                ],
            ]
            .std(axis=1)
            .values[0],
        )
        for p in params.percentiles:
            p_num = float(p[1:]) / 100
            self.assertEqual(
                m.processed_data.loc[
                    (m.processed_data.minutes == _time)
                    & (m.processed_data.days_to_event == 0),
                    f"{p}_NYE",
                ].values[0],
                m.processed_data.loc[
                    (m.processed_data.minutes == _time)
                    & (m.processed_data.days_to_event == 0),
                    [
                        "NYE__2015-12-31_multiplier",
                        "NYE__2016-12-31_multiplier",
                        "NYE__2017-12-31_multiplier",
                        "NYE__2018-12-31_multiplier",
                        "NYE__2019-12-31_multiplier",
                    ],
                ]
                .quantile(p_num, axis=1)
                .values[0],
            )

        # testing if forecast prediction is correct
        for prefix in params.multiplier_prefixes:
            self.assertTrue(
                abs(
                    fcst.loc[
                        fcst.time == np.datetime64("2020-12-31T00:19:00.000000000"),
                        "fcst",
                    ].values[0]
                    * m.processed_data.loc[
                        (m.processed_data.minutes == _time)
                        & (m.processed_data.days_to_event == 0),
                        f"{prefix}_NYE",
                    ].values[0]
                    - fcst.loc[
                        fcst.time == np.datetime64("2020-12-31T00:19:00.000000000"),
                        f"{prefix}_fcst_adjusted",
                    ].values[0]
                )
                < 0.0001
            )

    @DEVSERVER_TEST
    def test_median_base_replace_minutes_granularity(self) -> None:
        # historical actuals
        minute_df = read_fb_alm_minute(
            country="US",
            list_ds=[
                "2019-12-31",
                "2018-12-31",
                "2017-12-31",
                "2016-12-31",
                "2015-12-31",
            ],
            n_bases=4,
        )

        # forecast that needs to be adjusted
        # for now let's fake it
        fcst_minute_2020 = (
            minute_df[minute_df.event_time >= "2019-12-01"]
            .rename(columns={"event_time": "time", "alm_count_ex_bg": "fcst"})
            .copy()
        )
        fcst_minute_2020.time = fcst_minute_2020.time.map(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            .replace(year=2020)
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        # future special event
        predict_nye = pd.DataFrame(
            {"special_event": "NYE", "ds": pd.to_datetime(["2020-12-31"])}
        )

        # set up the parameters
        params = BasicSpecialEventsSingleTSParams(
            future_special_events=predict_nye,
            granularity_data="minutes",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        # historical special events
        all_nye = pd.DataFrame(
            {
                "special_event": "NYE",
                "ds": pd.to_datetime(
                    [
                        "2020-12-31",
                        "2019-12-31",
                        "2018-12-31",
                        "2017-12-31",
                        "2016-12-31",
                        "2015-12-31",
                    ]
                ),
                "lower_window": 0,
                "upper_window": 0,
            }
        )

        # set up the model
        m = BasicSpecialEventsSingleTSModel(
            minute_df.rename(columns={"event_time": "time", "alm_count_ex_bg": "y"}),
            params=params,
            special_events=all_nye,
            original_fcst=fcst_minute_2020,
            median_base_replace=True,
        )

        # fit
        m.fit()

        # testing if base forecasts of future special event are replaced by the median of window base forecasts
        _time = time(8, 24, 0)
        _time_fcst = list(m.original_fcst.loc[m.original_fcst.minutes == _time]["fcst"])
        self.assertEqual(_time_fcst[-1], np.median(_time_fcst[:-1]))

    def test_median_base_replace_daily_granularity(self) -> None:
        historical_special_events = pd.DataFrame(
            {
                "special_event": "fake",
                "ds": pd.to_datetime(
                    [
                        "2012-05-30",
                        "2012-08-11",
                        "2012-10-05",
                        "2013-01-24",
                        "2013-04-04",
                    ]
                ),
                "lower_window": 0,
                "upper_window": 0,
            }
        )

        future_special_events = pd.DataFrame(
            {"special_event": "fake", "ds": pd.to_datetime(["2013-06-08"])}
        )

        params = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        m = BasicSpecialEventsSingleTSModel(
            DATA_daily,
            params=params,
            special_events=historical_special_events,
            median_base_replace=True,
        )

        # fit
        m.fit()

        # testing if base forecasts of future special event are replaced by the median of window base forecasts
        self.assertEqual(
            np.median(list(m.original_fcst.iloc[[10, 17, 24, 31, 38]]["fcst"])[:-1]),
            list(m.original_fcst.iloc[[10, 17, 24, 31, 38]]["fcst"])[-1],
        )

        # test the case that original_fcst has no enough data
        m2 = BasicSpecialEventsSingleTSModel(
            DATA_daily,
            params=params,
            special_events=historical_special_events,
            original_fcst=pd.DataFrame(
                {"time": [pd.to_datetime("2013-06-08")], "fcst": [6.88888]}
            ),
            median_base_replace=True,
        )

        # fit
        m2.fit()

        # testing if base forecasts of future special event are replaced by the median of window base forecasts
        self.assertEqual(
            np.median(list(m2.original_fcst.iloc[[10, 17, 24, 31, 38]]["fcst"])[:-1]),
            list(m2.original_fcst.iloc[[10, 17, 24, 31, 38]]["fcst"])[-1],
        )

    def test_base_calculation_with_exclusion_of_special_dates(self) -> None:

        # create fake events for holidays in 2014-2016 that falls in the base window
        orig_data_fake = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2014-11-26",
                        "2014-12-03",
                        "2014-12-10",
                        "2014-12-17",
                        "2014-12-24",
                        "2014-12-31",
                        "2015-11-26",
                        "2015-12-03",
                        "2015-12-10",
                        "2015-12-17",
                        "2015-12-24",
                        "2015-12-31",
                        "2016-11-26",
                        "2016-12-03",
                        "2016-12-10",
                        "2016-12-17",
                        "2016-12-24",
                    ]
                ),
                "y": pd.to_numeric(list(range(1, 18))),
            }
        )

        historical_special_events = pd.DataFrame(
            {
                "special_event": ("NYE", "NYE", "Xmas", "Xmas"),
                "ds": pd.to_datetime(
                    [
                        "2014-12-31",
                        "2015-12-31",
                        "2014-12-24",
                        "2015-12-24",
                    ]
                ),
                "lower_window": 0,
                "upper_window": 0,
            }
        )

        future_special_events = pd.DataFrame(
            {
                "special_event": ("NYE", "Xmas"),
                "ds": pd.to_datetime(["2016-12-31", "2016-12-24"]),
            }
        )

        params_without_special_events_exclusion = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
        )

        m_without_special_events_exclusion = BasicSpecialEventsSingleTSModel(
            orig_data_fake,
            params=params_without_special_events_exclusion,
            special_events=historical_special_events,
        )
        # fcst_without_special_events_exclusion
        _ = m_without_special_events_exclusion.predict()

        # included special events (every NYE, and one Thanksgiving (2015-11-26) that falls in the base window)
        params_with_special_events_exclusion = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
            special_events_exclusion_dates=[
                "2014-12-24",
                "2015-12-24",
                "2016-12-24",
                "2015-11-26",
            ],
        )

        m_with_special_events_exclusion = BasicSpecialEventsSingleTSModel(
            orig_data_fake,
            params=params_with_special_events_exclusion,
            special_events=historical_special_events,
        )

        fcst_with_special_events_exclusion = m_with_special_events_exclusion.predict()
        self.assertIsNotNone(fcst_with_special_events_exclusion)

        # checks intermediate result
        # 1. NYE: expected median value should equal calculated median value
        # 1.1 test if it holds without exluding any special events dates
        expected_median_without_special_events_exclusion_NYE_2014 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2014-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-17"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-24"].y.values[0],
            ]
        )
        median_without_special_events_exclusion_NYE_2014 = (
            m_without_special_events_exclusion.processed_data.loc[
                (m_without_special_events_exclusion.processed_data.is_event)
            ]["NYE__2014-12-31_base"].values[0]
        )

        self.assertEqual(
            expected_median_without_special_events_exclusion_NYE_2014,
            median_without_special_events_exclusion_NYE_2014,
        )

        expected_median_without_special_events_exclusion_NYE_2015 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2015-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-17"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-24"].y.values[0],
            ]
        )
        median_without_special_events_exclusion_NYE_2015 = (
            m_without_special_events_exclusion.processed_data.loc[
                (m_without_special_events_exclusion.processed_data.is_event)
            ]["NYE__2015-12-31_base"].values[0]
        )

        self.assertEqual(
            expected_median_without_special_events_exclusion_NYE_2015,
            median_without_special_events_exclusion_NYE_2015,
        )

        # 1.2 test if it holds after exluding all NYE dates
        # affecting the base calculation of both 2014 and 2015
        # without excluding Xmas dates, four lagged dates (base window = 4) are used
        # while after excluding Xmas dates, only three lagged dates are used

        expected_median_with_special_events_exclusion_NYE_2014 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2014-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-17"].y.values[0],
            ]
        )
        median_with_special_events_exclusion_NYE_2014 = (
            m_with_special_events_exclusion.processed_data.loc[
                (m_with_special_events_exclusion.processed_data.is_event)
            ]["NYE__2014-12-31_base"].values[0]
        )

        self.assertEqual(
            expected_median_with_special_events_exclusion_NYE_2014,
            median_with_special_events_exclusion_NYE_2014,
        )

        expected_median_with_special_events_exclusion_NYE_2015 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2015-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-17"].y.values[0],
            ]
        )
        median_with_special_events_exclusion_NYE_2015 = (
            m_with_special_events_exclusion.processed_data.loc[
                (m_with_special_events_exclusion.processed_data.is_event)
            ]["NYE__2015-12-31_base"].values[0]
        )

        self.assertEqual(
            expected_median_with_special_events_exclusion_NYE_2015,
            median_with_special_events_exclusion_NYE_2015,
        )

        # 2. Xmas: expected median value should equal calculated median value
        # 2.1 test if it holds without exluding any special events dates (almost identical to test 1.1)
        expected_median_without_special_events_exclusion_Xmas_2014 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2014-11-26"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-17"].y.values[0],
            ]
        )
        median_without_special_events_exclusion_Xmas_2014 = (
            m_without_special_events_exclusion.processed_data.loc[
                (m_without_special_events_exclusion.processed_data.is_event)
            ]["Xmas__2014-12-24_base"].values[0]
        )

        self.assertEqual(
            expected_median_without_special_events_exclusion_Xmas_2014,
            median_without_special_events_exclusion_Xmas_2014,
        )

        expected_median_without_special_events_exclusion_Xmas_2015 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2015-11-26"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-17"].y.values[0],
            ]
        )
        median_without_special_events_exclusion_Xmas_2015 = (
            m_without_special_events_exclusion.processed_data.loc[
                (m_without_special_events_exclusion.processed_data.is_event)
            ]["Xmas__2015-12-24_base"].values[0]
        )

        self.assertEqual(
            expected_median_without_special_events_exclusion_Xmas_2015,
            median_without_special_events_exclusion_Xmas_2015,
        )

        # 1.2 test if it holds after exluding only one Thanksgiving date
        # affecting the base calculation of 2015, but not 2014
        # for Xmas 2014, four lagged dates (base window = 4) are used
        # while for Xmas 2015, only three lagged dates are used

        expected_median_with_special_events_exclusion_Xmas_2014 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2014-11-26"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2014-12-17"].y.values[0],
            ]
        )
        median_with_special_events_exclusion_Xmas_2014 = (
            m_with_special_events_exclusion.processed_data.loc[
                (m_with_special_events_exclusion.processed_data.is_event)
            ]["Xmas__2014-12-24_base"].values[0]
        )

        self.assertEqual(
            expected_median_with_special_events_exclusion_Xmas_2014,
            median_with_special_events_exclusion_Xmas_2014,
        )

        expected_median_with_special_events_exclusion_Xmas_2015 = median(
            [
                orig_data_fake[orig_data_fake["time"] == "2015-12-03"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-10"].y.values[0],
                orig_data_fake[orig_data_fake["time"] == "2015-12-17"].y.values[0],
            ]
        )
        median_with_special_events_exclusion_Xmas_2015 = (
            m_with_special_events_exclusion.processed_data.loc[
                (m_with_special_events_exclusion.processed_data.is_event)
            ]["Xmas__2015-12-24_base"].values[0]
        )

        self.assertEqual(
            expected_median_with_special_events_exclusion_Xmas_2015,
            median_with_special_events_exclusion_Xmas_2015,
        )

        # 3. test that one single date works (rather than inputting a list of dates)
        params_with_special_events_exclusion_single = BasicSpecialEventsSingleTSParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_seasonality="weekly",
            base_window=4,
            keep_intermediate_results=True,
            # pyre-fixme[6]: Expected `Optional[typing.List[str]]` for 6th
            #  param but got `str`.
            special_events_exclusion_dates="2015-11-26",
        )

        m_with_special_events_exclusion_single = BasicSpecialEventsSingleTSModel(
            orig_data_fake,
            params=params_with_special_events_exclusion_single,
            special_events=historical_special_events,
        )

        fcst_with_special_events_exclusion_single = (
            m_with_special_events_exclusion_single.predict()
        )
        self.assertIsNotNone(fcst_with_special_events_exclusion_single)

        # makes sure using one date still works
        median_with_special_events_exclusion_Xmas_2015 = (
            m_with_special_events_exclusion_single.processed_data.loc[
                (m_with_special_events_exclusion_single.processed_data.is_event)
            ]["Xmas__2015-12-24_base"].values[0]
        )

        self.assertEqual(
            expected_median_with_special_events_exclusion_Xmas_2015,
            median_with_special_events_exclusion_Xmas_2015,
        )


class PinnacleSpecialEventModelTest(TestCase):
    def test_validate_inputs(self) -> None:
        # invalid future date
        future_special_events = pd.DataFrame(
            {"special_event": "past_date", "ds": ["2020-09-01"]}
        )
        data = pd.DataFrame({"time": ["2020-09-01", "2020-09-02"], "y": [1, 5]})
        driver_train = pd.DataFrame(
            {"time": ["2020-09-01", "2020-09-02"], "y1": [2, 11]}
        )

        driver_future = pd.DataFrame({"time": ["2020-09-01"], "y1": [13]})

        params = PinnacleSpecialEventParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_window=1,
        )
        with self.assertRaises(
            ValueError,
            msg="The dates to forecast already exists in training data. Please specify future dates.",
        ):
            PinnacleSpecialEventModel(data, driver_train, driver_future, params)

        # base window is wider than data range
        future_special_events = pd.DataFrame(
            {"special_event": "past_date", "ds": ["2020-09-10"]}
        )
        driver_future = pd.DataFrame({"time": ["2020-09-10"], "y1": [13]})
        params = PinnacleSpecialEventParams(
            future_special_events=future_special_events,
            granularity_data="days",
            base_window=7,
        )

        with self.assertRaises(
            ValueError,
            msg="Please specify base window that is smaller than number of rows of training data.",
        ):
            PinnacleSpecialEventModel(data, driver_train, driver_future, params)

    # def test_fit_and_predict_daily_forecast(self) -> None:
    #     future_special_events = pd.DataFrame(
    #         {"special_event": "fake", "ds": pd.to_datetime(["2020-09-08"])}
    #     )
    #     data = pd.DataFrame(
    #         {
    #             "time": [
    #                 "2020-09-01",
    #                 "2020-09-02",
    #                 "2020-09-03",
    #                 "2020-09-04",
    #                 "2020-09-05",
    #                 "2020-09-06",
    #             ],
    #             "y": [1, 5, 1, 6, 1, 5],
    #         }
    #     )

    #     driver_train = pd.DataFrame(
    #         {
    #             "time": [
    #                 "2020-09-01",
    #                 "2020-09-02",
    #                 "2020-09-03",
    #                 "2020-09-04",
    #                 "2020-09-05",
    #                 "2020-09-06",
    #             ],
    #             "y1": [2, 11, 2, 10, 2, 12],
    #         }
    #     )

    #     driver_future = pd.DataFrame({"time": ["2020-09-08"], "y1": [13]})

    #     params = PinnacleSpecialEventParams(
    #         future_special_events=future_special_events,
    #         granularity_data="days",
    #         base_window=1,
    #         peak_percentile=50,
    #     )

    #     m = PinnacleSpecialEventModel(data, driver_train, driver_future, params)
    #     m.fit()
    #     self.assertEqual(m.threshold, 4.0)
    #     self.assertEqual(m.correlated_columns, ["y1"])

    #     result = m.predict()
    #     self.assertEqual(
    #         result.columns.tolist(),
    #         ["time", "fcst", "date", "minutes", "fcst_adjusted"],
    #     )
    #     self.assertEqual(result.fcst_adjusted[1], 13.435503736713882)


class RegressionDetectionEvaluationTest(TestCase):
    def testFBDetectBenchmarkEvaluator(self) -> None:
        eva = FBDetectBenchmarkEvaluator(CUSUMDetector)
        res = eva.evaluate()
        self.assertEqual(res["true_positives"], 221)
        self.assertEqual(res["false_positives"], 72)
        self.assertEqual(res["true_negatives"], 33)
        self.assertEqual(res["false_negatives"], 8)
        self.assertEqual(res["uncertains"], 14)

        with self.assertRaises(ValueError):
            FBDetectBenchmarkEvaluator(OutlierDetector)


class AnnotatedTimeSeriesTest(TestCase):
    def test_plotly_timeseries(self):
        sim2 = Simulator(n=450, start="2018-01-01")
        ts2 = sim2.level_shift_sim(
            cp_arr=[100, 200, 350],
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.075,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        human_label = [50, 98, 150, 198, 250, 350]

        # just test that these functions do not fail
        plotly_timeseries(ts2)
        plotly_timeseries(ts2, human_label)

    def test_annotation_upload(self):
        sim2 = Simulator(n=450, start="2018-01-01")
        ts2 = sim2.level_shift_sim(
            cp_arr=[100, 200, 350],
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.075,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        ts3 = sim2.level_shift_sim(
            random_seed=200,
            cp_arr=[100, 200, 350],
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.075,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        ts4 = sim2.level_shift_sim(
            random_seed=300,
            cp_arr=[100, 200, 350],
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.075,
            anomaly_arr=[50, 150, 250],
            z_score_arr=[10, -10, 20],
        )

        human_label = [50, 98, 150, 198, 250, 350]

        ts_with_labels = AnnotatedTimeSeries(
            hive_tablename="test_kats_internal_benchmark_upload_dummy",
            hive_table_ds="2021-09-01",
            partition_name="test_partition",
        )

        ts_with_labels.add_timeseries(
            dataset_name="synthetic", ts=ts2, anno={"0": human_label}
        )

        ts_with_labels.add_timeseries(
            dataset_name="synthetic1", ts=ts3, anno={"0": human_label}
        )

        ts_with_labels.add_timeseries(
            dataset_name="synthetic2", ts=ts4, anno={"0": human_label}
        )

        ts_df = ts_with_labels.get_dataframe()

        # check that a dataframe with 3 rows is formed
        self.assertEqual(ts_df.shape[0], 3)

        # check that this dataframe works with evaluator
        turing_2 = changepoint_evaluator.TuringEvaluator(
            detector=robust_stat_detection.RobustStatDetector
        )
        model_params = {"p_value_cutoff": 5e-3, "comparison_window": 2}
        eval_agg_df_2 = turing_2.evaluate(data=ts_df, model_params=model_params)

        self.assertEqual(eval_agg_df_2.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
