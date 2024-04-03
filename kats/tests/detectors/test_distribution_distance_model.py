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
    DataIrregularGranularityError,
    InternalError,
    ParameterError,
    TimeSeriesData,
)
from kats.detectors.distribution_distance_model import DistributionDistanceModel

from parameterized.parameterized import parameterized


def generate_multi_ts_data(
    granularities_sec: int = 60,
    length: int = 100,
    date_start_str: str = "2020-01-01 00:00:00",
    include_start_point: bool = True,
) -> TimeSeriesData:
    np.random.seed(0)
    date_start = datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
    multi_ts_val = np.random.normal(10, 3, [length, 10])
    multi_ts_val = np.apply_along_axis(sorted, 1, multi_ts_val)
    if include_start_point:
        ts_time = pd.date_range(
            start=date_start, freq=str(granularities_sec) + "s", periods=length
        )
    else:
        ts_time = pd.date_range(
            start=date_start, freq=str(granularities_sec) + "s", periods=length + 1
        )[1:]

    multi_ts_df = pd.DataFrame(multi_ts_val)
    multi_ts_df.columns = ["val_" + str(i) for i in range(10)]
    multi_ts_df = pd.concat([pd.Series(ts_time, name="time"), multi_ts_df], 1)
    # pyre-fixme[6]: For 1st argument expected `Optional[DataFrame]` but got
    #  `Union[DataFrame, Series]`.
    ts = TimeSeriesData(multi_ts_df)
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
            ts = generate_multi_ts_data(
                granularities_sec=gran_list[i],
                length=len_list[i],
                date_start_str="2020-01-01 00:00:00",
            )
        else:
            ts0 = generate_multi_ts_data(
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

    ts = generate_multi_ts_data(
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

    multi_ts_val = np.random.normal(10, 3, [n, 10])
    multi_ts_val = np.apply_along_axis(sorted, 1, multi_ts_val)

    multi_ts_df = pd.DataFrame(multi_ts_val)
    multi_ts_df.columns = ["val_" + str(i) for i in range(10)]

    ts_time = pd.date_range(
        start="2020-01-01", freq=str(granularities_sec) + "s", periods=length
    )[np.random.choice(list(range(length)), n, replace=False)]

    multi_ts_df = pd.concat([pd.Series(ts_time, name="time"), multi_ts_df], 1)
    # pyre-fixme[6]: For 1st argument expected `Optional[DataFrame]` but got
    #  `Union[DataFrame, Series]`.
    ts = TimeSeriesData(multi_ts_df)
    return ts


class TestDistributionDistanceModels(TestCase):
    # pyre-ignore
    @parameterized.expand(
        [
            # distance_metric, window_size_sec
            ("braycurtis", 3600),
            ("canberra", 3600),
            ("chebyshev", 3600),
            ("cityblock", 3600),
            ("correlation", 3600),
            ("cosine", 3600),
            ("dice", 3600),
            ("euclidean", 3600),
            ("hamming", 3600),
            ("jaccard", 3600),
            ("jensenshannon", 3600),
            ("minkowski", 3600),
            ("rogerstanimoto", 3600),
            ("russellrao", 3600),
            ("seuclidean", 3600),
            ("sokalmichener", 3600),
            ("sokalsneath", 3600),
            ("sqeuclidean", 3600),
            ("yule", 3600),
        ]
    )
    def test_regular_granularity_data(
        self,
        distance_metric: str,
        window_size_sec: int,
    ) -> None:
        ig_ts = generate_multi_ts_data()

        # case 1: historical data is not none, but can not cover window size
        model = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:15],
            data=ig_ts[15:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[15:]))
        # 0.9 is the tolerance fraction
        n = int(window_size_sec / 60 * 0.9 - 1 - 15)
        self.assertEqual(anom.scores.value.iloc[:n].sum(), 0)

        # case 2: no historical data
        model1 = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom1 = model1.fit_predict(data=ig_ts)
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))
        # 0.9 is the tolerance fraction
        n = int(window_size_sec / 60 * 0.9 - 1 - 0)
        self.assertEqual(anom1.scores.value.iloc[:n].sum(), 0)

        # case 3: historical data is not none, and can cover window size
        model2 = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom2 = model2.fit_predict(
            historical_data=ig_ts[:60],
            data=ig_ts[60:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom2.scores), len(ig_ts[60:]))

    # pyre-ignore
    @parameterized.expand(
        [
            # distance_metric, window_size_sec
            ("braycurtis", 3600),
            ("canberra", 3600),
            ("chebyshev", 3600),
            ("cityblock", 3600),
            ("correlation", 3600),
            ("cosine", 3600),
            ("dice", 3600),
            ("euclidean", 3600),
            ("hamming", 3600),
            ("jaccard", 1800),
            ("jensenshannon", 1800),
            ("minkowski", 1800),
            ("rogerstanimoto", 1800),
            ("russellrao", 1800),
            ("seuclidean", 1800),
            ("sokalmichener", 1800),
            ("sokalsneath", 1800),
            ("sqeuclidean", 1800),
            ("yule", 1800),
        ]
    )
    def test_data_with_individual_missing_datapoints(
        self,
        distance_metric: str,
        window_size_sec: int,
    ) -> None:
        ig_ts = generate_data_with_individual_missing_datapoints()

        model = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:15],
            data=ig_ts[15:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[15:]))

        model1 = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom1 = model1.fit_predict(data=ig_ts)
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))

    # pyre-ignore
    @parameterized.expand(
        [
            # distance_metric, window_size_sec
            ("braycurtis", 3600),
            ("canberra", 3600),
            ("chebyshev", 3600),
            ("cityblock", 3600),
            ("correlation", 3600),
            ("cosine", 3600),
            ("dice", 3600),
            ("euclidean", 3600),
            ("hamming", 3600),
        ]
    )
    def test_irregular_granularity_data(
        self,
        distance_metric: str,
        window_size_sec: int,
    ) -> None:
        ig_ts = generate_irregular_granularity_data(percentage=0.8, seed=3)

        model = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom = model.fit_predict(
            historical_data=ig_ts[:15],
            data=ig_ts[15:],
        )
        # prediction returns scores of same length
        self.assertEqual(len(anom.scores), len(ig_ts[15:]))

        model1 = DistributionDistanceModel(
            distance_metric=distance_metric,
            window_size_sec=window_size_sec,
        )

        anom1 = model1.fit_predict(data=ig_ts)
        # prediction returns scores of same length
        self.assertEqual(len(anom1.scores), len(ig_ts))


class TestDistributionDistanceModelError(TestCase):
    def setUp(self) -> None:
        self.multi_ts = generate_multi_ts_data()
        self.ts = TimeSeriesData(
            time=self.multi_ts.time, value=self.multi_ts.value.iloc[:, 0]
        )
        self.ts_low = TimeSeriesData(
            time=self.multi_ts.time, value=self.multi_ts.value.iloc[:, :3]
        )

        val = self.multi_ts.value.iloc[:, :5]
        val.iloc[0, :] = [5, 4, 3, 2, 1]
        self.ts_random = TimeSeriesData(time=self.multi_ts.time, value=val)

    def test_metric_error(self) -> None:
        with self.assertRaises(ParameterError):
            _ = DistributionDistanceModel(
                distance_metric="two_sum",
                window_size_sec=100,
            )

    def test_data_error(self) -> None:
        model = DistributionDistanceModel(
            window_size_sec=3600,
        )

        # This algorithm is supporting multivariate time series data only.
        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.ts)
        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.multi_ts, historical_data=self.ts)

        # Unmatched dimension of historical data and data
        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.multi_ts, historical_data=self.ts_low)

        # Each row of input data must be non-decreasing.
        with self.assertRaises(DataError):
            _ = model.fit_predict(data=self.ts_random)

    def test_data_insufficient_error(
        self,
    ) -> None:
        # case 1: historical data and data together are < window size
        model = DistributionDistanceModel(
            window_size_sec=3600,
        )

        with self.assertRaises(DataInsufficientError):
            _ = model.fit_predict(
                historical_data=self.multi_ts[:15],
                data=self.multi_ts[15:30],
            )

    def test_internal_error(self) -> None:
        model = DistributionDistanceModel(
            window_size_sec=3600,
        )

        with self.assertRaises(InternalError):
            _ = model.fit(data=self.ts)

        with self.assertRaises(InternalError):
            _ = model.predict(data=self.ts)

    def test_data_irregular_granularity_error(self) -> None:
        for distance_metric, window_size_sec in [
            ("euclidean", 3600),
            ("jensenshannon", 3600),
        ]:
            ig_ts = generate_data_with_sudden_granularity_changes()
            model = DistributionDistanceModel(
                distance_metric=distance_metric,
                window_size_sec=window_size_sec,
            )
            with self.assertRaises(DataIrregularGranularityError):
                _ = model.fit_predict(
                    historical_data=ig_ts[:15],
                    data=ig_ts[15:],
                )
            model1 = DistributionDistanceModel(
                distance_metric=distance_metric,
                window_size_sec=window_size_sec,
            )
            with self.assertRaises(DataIrregularGranularityError):
                _ = model1.fit_predict(data=ig_ts)
