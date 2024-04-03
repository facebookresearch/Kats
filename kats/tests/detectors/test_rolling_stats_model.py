# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import (
    DataError,
    DataInsufficientError,
    InternalError,
    ParameterError,
    TimeSeriesData,
)
from kats.detectors.rolling_stats_model import RollingStatsModel

from parameterized.parameterized import parameterized


def generate_ts_data(
    granularities_sec: int = 60,
    length: int = 100,
    date_start_str: str = "2020-01-01 00:00:00",
    include_start_point: bool = True,
) -> TimeSeriesData:
    np.random.seed(0)
    date_start = datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
    ts_val = np.random.normal(10, 3, length)
    if include_start_point:
        ts_time = pd.date_range(
            start=date_start, freq=str(granularities_sec) + "s", periods=length
        )
    else:
        ts_time = pd.date_range(
            start=date_start, freq=str(granularities_sec) + "s", periods=length + 1
        )[1:]
    ts = TimeSeriesData(pd.DataFrame({"time": ts_time, "value": pd.Series(ts_val)}))
    return ts


def generate_data_with_sudden_granularity_changes(length: int = 200) -> TimeSeriesData:
    """
    Generate data with sudden changes in data granularity
    (e.g., before 2 weeks ago, hourly data; after 2 weeks ago, 15 minute granularity)
    Common with ODS data because of ODS rollups
    """
    np.random.seed(0)
    len_list = [length // 4, length // 4, length // 4, length - length // 4 * 3]
    gran_list = [3600, 900, 600, 300]
    ts = None
    for i in range(3):
        if ts is None:
            ts = generate_ts_data(
                granularities_sec=gran_list[i],
                length=len_list[i],
                date_start_str="2020-01-01 00:00:00",
            )
        else:
            ts0 = generate_ts_data(
                granularities_sec=gran_list[i],
                length=len_list[i],
                date_start_str=str(ts.time.iloc[-1]),
                include_start_point=False,
            )
            ts.extend(ts0, validate=False)
    assert ts is not None
    return ts


def generate_data_with_individual_missing_datapoints(
    granularities_sec: int = 60,
    length: int = 100,
) -> TimeSeriesData:
    """
    Generate data with individual missing datapoints
    Most often found when counting in Scuba without fills
    """
    np.random.seed(0)

    ts = generate_ts_data(
        granularities_sec=granularities_sec,
        length=length,
    )

    missing_points = sorted(set({length // 3, length // 3 * 2, length - 3}))
    ts_1, ts_2, ts_3, ts_4 = (
        ts[: missing_points[0]],
        ts[missing_points[0] + 1 : missing_points[1]],
        ts[missing_points[1] + 1 : missing_points[2]],
        ts[missing_points[2] + 1 :],
    )

    ts_1.extend(ts_2, validate=False)
    ts_1.extend(ts_3, validate=False)
    ts_1.extend(ts_4, validate=False)

    return ts_1


def generate_irregular_granularity_data(
    granularities_sec: int = 60,
    length: int = 200,
    percentage: float = 0.7,
    seed: int = 0,
) -> TimeSeriesData:
    """
    Generate irregular granularity data
    Most often found in ODS
    """
    np.random.seed(seed)
    n = int(length * percentage)

    ts_val = np.random.normal(10, 3, n)
    ts_time = pd.date_range(
        start="2020-01-01", freq=str(granularities_sec) + "s", periods=length
    )[np.random.choice(list(range(length)), n, replace=False)]

    ts = TimeSeriesData(pd.DataFrame({"time": ts_time, "value": pd.Series(ts_val)}))
    return ts


class TestRollingStatsModels(TestCase):
    # pyre-ignore[56]
    @parameterized.expand(
        [
            # score func, window, point_based, seasonality
            ("z_score", 10, False, False),
            ("z_score", 10, True, False),
            ("z_score", 20, True, False),
            ("z_score", 20, False, False),
            ("mad", 10, True, False),
            ("mad", 10, False, False),
            ("iqr", 20, True, False),
            ("iqr", 20, False, False),
            ("modified_z_score_mad", 10, True, False),
            ("modified_z_score_mad", 10, False, False),
            ("modified_z_score_iqr", 20, True, False),
            ("modified_z_score_iqr", 20, False, False),
            ("iqr_median_deviation", 20, True, False),
            ("iqr_median_deviation", 20, False, False),
        ]
    )
    def test_irregular_granularity_data(
        self,
        score_func: str,
        window_size: int,
        point_based: bool,
        remove_seasonality: bool,
    ) -> None:
        ig_ts = generate_irregular_granularity_data(percentage=0.85)

        window_size_points = window_size
        if not point_based:
            window_size *= 60

        model = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:window_size_points],
            data=ig_ts[window_size_points:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[window_size_points:]))

        model1 = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
            allow_expanding_window=True,
        )

        anom1 = model1.fit_predict(
            data=ig_ts,
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))

    # pyre-ignore[56]
    @parameterized.expand(
        [
            # score func, window, point_based, seasonality
            ("z_score", 10, False, True),
            ("z_score", 10, True, True),
            ("z_score", 20, True, True),
            ("z_score", 20, False, True),
            ("mad", 10, True, True),
            ("mad", 10, False, True),
            ("iqr", 20, True, True),
            ("iqr", 20, False, True),
            ("modified_z_score_mad", 10, True, False),
            ("modified_z_score_mad", 10, False, False),
            ("modified_z_score_iqr", 20, True, False),
            ("modified_z_score_iqr", 20, False, False),
            ("iqr_median_deviation", 20, True, False),
            ("iqr_median_deviation", 20, False, False),
        ]
    )
    def test_data_with_individual_missing_datapoints(
        self,
        score_func: str,
        window_size: int,
        point_based: bool,
        remove_seasonality: bool,
    ) -> None:
        ig_ts = generate_data_with_individual_missing_datapoints()

        window_size_points = window_size
        if not point_based:
            window_size *= 60

        model = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:window_size_points],
            data=ig_ts[window_size_points:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[window_size_points:]))

        model1 = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
            allow_expanding_window=True,
        )

        anom1 = model1.fit_predict(
            data=ig_ts,
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))

    # pyre-ignore[56]
    @parameterized.expand(
        [
            # score func, window, point_based, seasonality
            ("z_score", 10, False, False),
            ("z_score", 10, True, False),
            ("z_score", 20, True, False),
            ("z_score", 20, False, False),
            ("mad", 10, True, False),
            ("mad", 10, False, False),
            ("iqr", 20, True, False),
            ("iqr", 20, False, False),
            ("modified_z_score_mad", 10, True, False),
            ("modified_z_score_mad", 10, False, False),
            ("modified_z_score_iqr", 20, True, False),
            ("modified_z_score_iqr", 20, False, False),
            ("iqr_median_deviation", 20, True, False),
            ("iqr_median_deviation", 20, False, False),
        ]
    )
    def test_data_with_sudden_granularity_changes(
        self,
        score_func: str,
        window_size: int,
        point_based: bool,
        remove_seasonality: bool,
    ) -> None:
        ig_ts = generate_data_with_sudden_granularity_changes()

        window_size_points = window_size
        if not point_based:
            window_size *= 3600

        model = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:window_size_points],
            data=ig_ts[window_size_points:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[window_size_points:]))

        model1 = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
            allow_expanding_window=True,
        )

        anom1 = model1.fit_predict(
            data=ig_ts,
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))

    # pyre-ignore[56]
    @parameterized.expand(
        [
            # score func, window, point_based, seasonality
            ("z_score", 9, False, False),
            ("z_score", 9, True, False),
            ("mad", 10, True, False),
            ("mad", 10, False, False),
            ("iqr", 10, True, False),
            ("iqr", 10, False, False),
            ("modified_z_score_mad", 9, True, False),
            ("modified_z_score_mad", 9, False, False),
            ("modified_z_score_iqr", 9, True, False),
            ("modified_z_score_iqr", 9, False, False),
            ("iqr_median_deviation", 9, True, False),
            ("iqr_median_deviation", 9, False, False),
        ]
    )
    def test_score_computation_func(
        self,
        score_func: str,
        window_size: int,
        point_based: bool,
        remove_seasonality: bool,
    ) -> None:
        ts = generate_ts_data(length=10)

        if not point_based:
            window_size *= 60

        model = RollingStatsModel(
            rolling_window=window_size,
            statistics=score_func,
            remove_seasonality=remove_seasonality,
            point_based=point_based,
        )

        anom = model.fit_predict(
            data=ts[9:],  # only 1 dp -> reorg_data has shape (1, X)
            historical_data=ts[:9],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), 1)

    def test_regular_ts_data(self) -> None:
        # enough historical data
        ts = generate_ts_data()
        model = RollingStatsModel(
            rolling_window=10,
            statistics="mad",
            point_based=True,
        )
        s = model.fit_predict(historical_data=ts[:10], data=ts[10:]).scores
        model2 = RollingStatsModel(
            rolling_window=600,
            statistics="mad",
            point_based=False,
        )
        s2 = model2.fit_predict(historical_data=ts[:10], data=ts[10:]).scores
        self.assertTrue(
            np.array_equal(
                np.round(s.value.values, 5),
                np.round(s2.value.values, 5),
                equal_nan=True,
            )
        )

        # No historical data
        ts = generate_ts_data()
        model = RollingStatsModel(
            rolling_window=10,
            point_based=True,
            allow_expanding_window=True,
        )
        s = model.fit_predict(data=ts).scores
        model2 = RollingStatsModel(
            rolling_window=600,
            point_based=False,
            allow_expanding_window=True,
        )
        s2 = model2.fit_predict(data=ts).scores
        self.assertTrue(
            np.array_equal(
                np.round(s.value.values, 5),
                np.round(s2.value.values, 5),
                equal_nan=True,
            )
        )


class TestRollingStatsModelsError(TestCase):
    def setUp(self) -> None:
        self.ts = generate_ts_data()
        time_index = pd.date_range(start="2018-01-01", freq="D", periods=2)
        ts_pd = pd.DataFrame({"time": time_index, "val_1": [1, 2], "val_2": [1, 2]})
        self.multi_ts = TimeSeriesData(ts_pd)

    def test_window_size_error(self) -> None:
        model = RollingStatsModel(
            rolling_window=60,
            point_based=False,
        )
        with self.assertRaises(ParameterError):
            _ = model.fit_predict(data=self.ts)

    def test_internal_error(self) -> None:
        model = RollingStatsModel(
            rolling_window=60,
            point_based=False,
        )
        with self.assertRaises(InternalError):
            _ = model.fit(data=self.ts)

        with self.assertRaises(InternalError):
            _ = model.predict(data=self.ts)

    def test_data_error(self) -> None:
        model = RollingStatsModel(
            rolling_window=10,
        )
        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.multi_ts)

        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.ts, historical_data=self.multi_ts)

    # pyre-ignore[56]
    @parameterized.expand(
        [
            # score func, rolling_window
            ("z_score", 9),
            ("mad", 10),
            ("iqr", 10),
            ("modified_z_score_mad", 9),
            ("modified_z_score_iqr", 9),
            ("iqr_median_deviation", 9),
        ]
    )
    def test_insufficient_data_error(
        self,
        score_func: str,
        window_size: int,
    ) -> None:
        model = RollingStatsModel(
            rolling_window=10,
            point_based=True,
            allow_expanding_window=False,
        )
        with self.assertRaises(DataInsufficientError):
            _ = model.fit_predict(data=self.ts)

        with self.assertRaises(DataInsufficientError):
            _ = model.fit_predict(data=self.ts[9:], historical_data=self.ts[:9])
