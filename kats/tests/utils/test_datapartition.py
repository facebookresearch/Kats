# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, NamedTuple, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.utils.datapartition import (
    RollingOriginDataParition,
    SimpleDataPartition,
    SimpleTimestampDataPartition,
)
from parameterized.parameterized import parameterized


DataPartition = Union[
    List[TimeSeriesData],
    Dict[Union[int, str], TimeSeriesData],
    TimeSeriesData,
]


class TrainTestData(NamedTuple):
    train: DataPartition
    test: DataPartition


DF: pd.DataFrame = pd.DataFrame(
    {"time": pd.date_range("2021-05-06", periods=10), "value": np.arange(10)}
)
TS: TimeSeriesData = TimeSeriesData(DF)
TS_DICT: Dict[Union[int, str], TimeSeriesData] = {0: TS, 1: TS}
TS_LIST: List[TimeSeriesData] = [TS, TS]

ALL_DATASETS: List[DataPartition] = [TS, TS_DICT, TS_LIST]
# ground-truth for simple data partition

RAW_SDP_RES: List[List[TrainTestData]] = [
    [TrainTestData(train=TS[:9], test=TS[9:])],
    [TrainTestData(train={0: TS[:9], 1: TS[:9]}, test={0: TS[9:], 1: TS[9:]})],
    [TrainTestData(train=[TS[:9], TS[:9]], test=[TS[9:], TS[9:]])],
]

RAW_STDP_RES: List[List[TrainTestData]] = [
    [TrainTestData(train=TS[:5], test=TS[7:])],
    [
        TrainTestData(
            train={0: TimeSeriesData(), 1: TimeSeriesData()}, test={0: TS, 1: TS}
        )
    ],
    [TrainTestData(train=[TimeSeriesData(), TimeSeriesData()], test=[TS[7:], TS[7:]])],
]

RAW_RODP_RES: List[List[TrainTestData]] = [
    [
        TrainTestData(train=TS[:5], test=TS[5:7]),
        TrainTestData(train=TS[:8], test=TS[8:]),
    ],
    [TrainTestData(train={0: TS[:5], 1: TS[:5]}, test={0: TS[5:], 1: TS[5:]})],
    [
        TrainTestData(train=[TS[:3]] * 2, test=[TimeSeriesData()] * 2),
        TrainTestData(train=[TS[3:6]] * 2, test=[TimeSeriesData()] * 2),
        TrainTestData(train=[TS[7:10]] * 2, test=[TimeSeriesData()] * 2),
    ],
    [
        TrainTestData(train=TS[:3], test=TS[4:6]),
        TrainTestData(train=TS[2:5], test=TS[6:8]),
        TrainTestData(train=TS[4:7], test=TS[8:10]),
    ],
]


class TestSimpleDataPartition(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            [{"train_frac": 0.9, "test_frac": 0.1}],
            [{"train_frac": 0.9, "test_size": 2}],
            [{"train_size": 9, "test_frac": 0.1}],
            [{"train_size": 9}],
        ]
    )
    def test_split(self, params: Dict[str, Any]) -> None:
        sdp = SimpleDataPartition(**params)
        for raw_data, truth in zip(ALL_DATASETS, RAW_SDP_RES):
            res = sdp.split(raw_data)
            self.assertTrue(res == truth)


class TestSimpleTimestampDataPartition(TestCase):
    # pyre-fixme
    @parameterized.expand(
        [
            (
                {"train_end": "2021-05-10", "test_start": "2021-05-13"},
                ALL_DATASETS[0],
                RAW_STDP_RES[0],
            ),
            (
                {"train_end": "2020-05-10", "test_start": "2020-05-13"},
                ALL_DATASETS[1],
                RAW_STDP_RES[1],
            ),
            (
                {"train_end": "2020-05-10", "test_start": "2021-05-13"},
                ALL_DATASETS[2],
                RAW_STDP_RES[2],
            ),
        ]
    )
    def test_split(
        self, params: Dict[str, Any], data: DataPartition, truth: List[TrainTestData]
    ) -> None:
        stdp = SimpleTimestampDataPartition(**params)
        res = stdp.split(data)
        self.assertTrue(res == truth)


class TestRollingOriginDataParition(TestCase):

    # pyre-fixme
    @parameterized.expand(
        [
            (
                {
                    "start_train_frac": 0.5,
                    "test_frac": 0.2,
                    "expanding_steps": 2,
                },
                ALL_DATASETS[0],
                RAW_RODP_RES[0],
            ),
            (
                {
                    "start_train_frac": 0.5,
                    "test_frac": 0.5,
                    "expanding_steps": 1,
                },
                ALL_DATASETS[1],
                RAW_RODP_RES[1],
            ),
            (
                {
                    "start_train_frac": 0.3,
                    "test_frac": 0.05,
                    "expanding_steps": 3,
                    "constant_train_size": True,
                },
                ALL_DATASETS[2],
                RAW_RODP_RES[2],
            ),
            (
                {
                    "start_train_frac": 0.3,
                    "test_frac": 0.2,
                    "expanding_steps": 3,
                    "constant_train_size": True,
                    "window_frac": 0.1,
                },
                ALL_DATASETS[0],
                RAW_RODP_RES[3],
            ),
        ]
    )
    def test_split(
        self, params: Dict[str, Any], data: DataPartition, truth: List[TrainTestData]
    ) -> None:
        rodp = RollingOriginDataParition(**params)
        res = rodp.split(data)
        self.assertTrue(res == truth)
