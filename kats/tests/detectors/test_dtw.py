# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest.mock as mock
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.dtwcpd import (
    DTWCPDChangePoint,
    DTWCPDDetector,
    DTWTimeSeriesTooSmallException,
)


class DTWCPDTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    sigma = 0.05  # std. dev
    num_points = 450

    def test_clear_spike_in_zero_data_yields_cp(self) -> None:
        N = 200
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-01-01", periods=N, freq="D"),
                "ts1": np.zeros(N),
                "ts2": np.zeros(N),
                "ts3": np.zeros(N),
            }
        )
        df.loc[100, "ts3"] = 500
        ts = TimeSeriesData(df)

        dtw_model = DTWCPDDetector(data=ts, sliding_window_size=50, skip_size=10)

        cps = dtw_model.detector()

        expected_result = [
            DTWCPDChangePoint(
                start_time=pd.Timestamp("2021-03-02 00:00:00", freq="D"),
                end_time=pd.Timestamp("2021-04-20 00:00:00", freq="D"),
                confidence=1e9,
                ts_name="ts3",
            )
        ]

        # Single change point at the desired location.
        self.assertEqual(cps, expected_result)

    def test_no_spike_in_zero_data_yields_no_cp(self) -> None:

        N = 200
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-01-01", periods=N, freq="D"),
                "ts1": np.zeros(N),
                "ts2": np.zeros(N),
                "ts3": np.zeros(N),
            }
        )
        ts = TimeSeriesData(df)

        dtw_model = DTWCPDDetector(data=ts, sliding_window_size=50, skip_size=10)

        cps = dtw_model.detector()

        self.assertEqual(cps, [])

    def test_two_similar_spikes_in_zero_data_yields_cp(self) -> None:
        N = 200
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-01-01", periods=N, freq="D"),
                "ts1": np.zeros(N),
                "ts2": np.zeros(N),
                "ts3": np.zeros(N),
                "ts4": np.zeros(N),
                "ts5": np.zeros(N),
            }
        )
        df.loc[100, "ts3"] = 500

        df.loc[150, "ts1"] = 10
        df.loc[50, "ts2"] = 20

        ts = TimeSeriesData(df)

        dtw_model = DTWCPDDetector(data=ts, sliding_window_size=50, skip_size=10)

        cps = dtw_model.detector()

        expected_result = [
            DTWCPDChangePoint(
                start_time=pd.Timestamp("2021-03-02 00:00:00", freq="D"),
                end_time=pd.Timestamp("2021-04-20 00:00:00", freq="D"),
                confidence=mock.ANY,
                ts_name="ts3",
            )
        ]

        self.assertEqual(cps, expected_result)

    def test_wrong_sliding_window_wrong_size(self) -> None:
        N = 200
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-01-01", periods=N, freq="D"),
                "ts1": np.zeros(N),
                "ts2": np.zeros(N),
                "ts3": np.zeros(N),
                "ts4": np.zeros(N),
                "ts5": np.zeros(N),
            }
        )
        df.loc[100, "ts3"] = 500

        df.loc[150, "ts1"] = 10
        df.loc[50, "ts2"] = 20

        ts = TimeSeriesData(df)

        with self.assertRaises(DTWTimeSeriesTooSmallException):
            dtw_model = DTWCPDDetector(data=ts, sliding_window_size=500, skip_size=100)
            dtw_model.detector()

    def test_one_time_series(self) -> None:
        N = 200
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-01-01", periods=N, freq="D"),
                "ts1": np.zeros(N),
            }
        )
        df.loc[100, "ts1"] = 500
        ts = TimeSeriesData(df)

        with self.assertRaises(DTWTimeSeriesTooSmallException):
            dtw_model = DTWCPDDetector(data=ts, sliding_window_size=50, skip_size=10)
            dtw_model.detector()

    def test_DTWDistance(self) -> None:
        s1 = [1.0, 2, 3, 4, 5, 6]
        s2 = [3.0, 0, 0, 2, 0, 0]
        w = 2
        distance = DTWCPDDetector.DTWDistance(s1, s2, w)
        self.assertAlmostEqual(distance, 8.660254037844387)

    def test_LB_Keogh(self) -> None:
        s1 = [1.0, 2, 3, 4, 5, 6]
        s2 = [3.0, 0, 0, 2, 0, 0]
        w = 2
        lowerBound = DTWCPDDetector.LB_Keogh(s1, s2, w)
        self.assertAlmostEqual(lowerBound, 5.385164807134504)

    # TODO: test it handles time indices

    # TODO: Test Complex traffic pattern repeats (e.g., with a few increasing spikes etc)

    # TODO: Test non-zero data (with random low level noise in traffic)

    # TODO: what about checking in the same time series?
