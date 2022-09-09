# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the data-partition classes for Kats.

Kats supports multiple types of data-partition classes, including:
    - :class: `SimpleDataPartition` (basic train & test data partition).
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import Any, cast, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData

Timestamp = Union[str, pd.Timestamp, datetime]
DataPartition = Union[
    TimeSeriesData, List[TimeSeriesData], Dict[Union[str, int], TimeSeriesData]
]


class TrainTestData(NamedTuple):
    train: DataPartition
    test: DataPartition


def _raise_error(info: str) -> None:
    logging.error(info)
    raise ValueError(info)


def _get_absolute_size(size: int, frac: float) -> int:
    return int(size * frac)


class DataPartitionBase(ABC):
    """
    This class defines the parent functions for various data partition classes.

    Attributes:
        multi: A boolean indicating whether to use multiprocessing or not.
        max_core: Optional; An integer representing the number of cores for multiprocessing.
                  Default is None, which sets `max_core = max((total_cores - 1) // 2, 1)`.

    Raises:
        ValueError: An invalide data type is passed.
    """

    def __init__(
        self, multi: bool, max_core: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.multi = multi
        if multi:  # using multi-processing
            total_cores = cpu_count()
            if max_core is None:
                self.max_core: int = max((total_cores - 1) // 2, 1)
            elif isinstance(max_core, int) and max_core > 0 and max_core < total_cores:
                self.max_core = max_core
            elif isinstance(max_core, int) and max_core > total_cores:
                logging.warn(
                    f"`max_core` is larger than maximum available cores ({total_cores}) and sets `max_core = {total_cores}`."
                )
            else:
                msg = f"max_core should be a positive integer in [1, {total_cores}] but receives {max_core}."
                logging.error(msg)
                raise ValueError(msg)
        else:
            self.max_core = 1
        self.split_num: int = -1

    @abstractmethod
    def _single_train_test_split(
        self,
        tag: Union[int, str],
        data: TimeSeriesData,
    ) -> Tuple[Union[int, str], List[TrainTestData]]:
        raise NotImplementedError()

    def _data_check(
        self,
        data: Union[
            List[TimeSeriesData],
            Dict[Union[int, str], TimeSeriesData],
            TimeSeriesData,
        ],
    ) -> List[Union[int, str]]:
        keys = []
        if isinstance(data, TimeSeriesData):
            keys = [0]
            data = [data]
        # input is a dictionary
        elif isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list):
            keys = list(range(len(data)))
        else:
            info = (
                "The input data should be a single (or a list or a dictionary of)"
                f"`pd.DataFrame`, `np.ndarray`, or `TimeSeriesData` object, but receives {type(data)}"
            )
            _raise_error(info)
        for k in keys:
            # pyre-fixme
            if not isinstance(data[k], TimeSeriesData):
                info = f"The input data should be a single (or a list or a dictionary of) `TimeSeriesData` object, but receives {type(data[k])}."
                _raise_error(info)
        # pyre-fixme
        return keys

    def split(self, data: DataPartition) -> List[TrainTestData]:
        """Splits the inputs data.

        Args:
            data: A single or a list or a dictionary of `TimeSeriesData` or `pd.DataFrame` or `np.ndarray` objects.

        Returns:
            A List of tuples and each tuple contrains two components, and the first component representing the training time series and the second component representing the test time series.
            More specifically, the returned data type is `List[tuple[train_ts, test_ts]]`.
        """

        # check data type
        keys = self._data_check(data)
        n = len(keys)
        # Only one time series
        if (not isinstance(data, list)) and (not isinstance(data, dict)):
            _, res = self._single_train_test_split(0, data)
            return res
        # Multiple time series
        if self.multi:
            # multi-processing
            # pyre-fixme
            params = [(k, data[k]) for k in keys]
            pool = Pool(self.max_core)
            raw_res = pool.starmap(self._single_train_test_split, params)
            pool.close()
            pool.join()
        else:
            # pyre-fixme
            raw_res = [self._single_train_test_split(k, data[k]) for k in keys]

        res = []
        return_list = isinstance(data, list)
        for i in range(self.split_num):
            tmp_train = {raw_res[k][0]: (raw_res[k][1][i]).train for k in range(n)}
            tmp_test = {raw_res[k][0]: (raw_res[k][1][i]).test for k in range(n)}
            if return_list:
                tmp_train = [tmp_train[k] for k in keys]
                tmp_test = [tmp_test[k] for k in keys]
            res.append(TrainTestData(train=tmp_train, test=tmp_test))
        return res


class SimpleDataPartition(DataPartitionBase):
    """SimpleDataPartition Class provides method for spliting time series into train/test sets given train/test data fraction or data number.

    Attributes:
        train_frac: Optional; A float in (0,1) representing the fraction of data for training, i.e., the first `train_frac*len(data)` data will be training data. Default is None.
        test_frac: Optional; A float in (0,1) representing the fraction of data for testing, i.e., the next `test_frac*len(data)` data after training data will be testing data. Default is None.
        train_size: Optional; An integer represening the number of data for training, i.e., the first `train_size` data will be training data. Default is None.
        test_size: Optional; An integer represening the number of data for testing, i.e., the next `test_size` data after training data will be testing data. Default is None.
        multi: Optional; A boolean representing whether using multi-processing or not. Default is True.
        max_core: Optional; An integer representing the number of cores for multiprocessing. Default is None, which sets `max_core = max((total_cores - 1) // 2, 1)`.

    Sample Usage:
        >>> sdp = SimpleDataPartition(train_frac = 0.8, test_frac = 0.2)
        >>> res = sdp.split(data)
    """

    def __init__(
        self,
        train_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        multi: bool = True,
        max_core: Optional[int] = None,
    ) -> None:

        super(SimpleDataPartition, self).__init__(multi=multi, max_core=max_core)
        self.split_num = 1
        if (train_frac is None) and (train_size is None):
            info = "Either `train_frac` or `train_size` should not be None."
            _raise_error(info)

        self._param_check(train_frac, train_size, "train_frac", "train_size")
        self._param_check(test_frac, test_size, "test_frac", "test_size")

        self.train_frac = train_frac
        self.train_size = train_size
        self.test_frac = test_frac
        self.test_size = test_size

    def _param_check(
        self, frac: Optional[float], size: Optional[int], frac_name: str, size_name: str
    ) -> None:
        if (frac is not None) and (size is not None):
            info = f"Please specify either `{frac_name}` or `{size_name}`, not both."
            _raise_error(info)
        if (frac is not None) and (frac <= 0 or frac >= 1):
            info = f"`{frac_name}` should be a float in (0, 1), but receives {frac}."
        if (size is not None) and (size <= 0):
            info = f"`{size_name}` should be a positive integer, but receives {size}."
        return

    def _single_train_test_split(
        self,
        tag: Union[str, int],
        data: TimeSeriesData,
    ) -> Tuple[Union[int, str], List[TrainTestData]]:
        size = len(data)
        train_size = (
            _get_absolute_size(size, self.train_frac)
            if self.train_frac
            else self.train_size
        )
        train_size = cast(int, train_size)
        if self.test_frac:
            end_idx = _get_absolute_size(size, self.test_frac) + train_size
        elif self.test_size:
            end_idx = self.test_size + train_size
        else:
            end_idx = size
        return tag, [
            TrainTestData(train=data[:train_size], test=data[train_size:end_idx])
        ]


class SimpleTimestampDataPartition(DataPartitionBase):
    """SimpleDataPartition Class provides method for spliting time series into train/test sets given timestamps.

    Attributes:
        train_end: A timestamp (i.e., string, `pd.Timestamp` object or `datetime.datetime` object) representing the end timestamp (inclusive) of training data.
        test_start: A timestamp (i.e., string, `pd.Timestamp` object or `datetime.datetime` object) representing the start timestamp (inclusive) of test data.
        train_start: Optional; A timestamp (i.e., string, `pd.Timestamp` object or `datetime.datetime` object) representing the start timestamp (inclusive) of training data. Default is None, which takes `pd.Timestamp.min`.
        test_end: Optional; A timestamp (i.e., string, `pd.Timestamp` object or `datetime.datetime` object) representing the end timestamp (inclusive) of test data. Default is None, which takes `pd.Timestamp.max`.
        multi: Optional; A boolean representing whether using multi-processing or not. Default is True.
        max_core: Optional; An integer representing the number of cores for multiprocessing. Default is None, which sets `max_core = max((total_cores - 1) // 2, 1)`.

    Sample Usage:
        >>> stdp = SimpleTimestampDataPartition(train_frac = 0.8, test_frac = 0.2)
        >>> res = stdp.split(data)
    """

    def __init__(
        self,
        train_end: Timestamp,
        test_start: Timestamp,
        train_start: Optional[Timestamp] = None,
        test_end: Optional[Timestamp] = None,
        multi: bool = True,
        max_core: Optional[int] = None,
    ) -> None:
        super(SimpleTimestampDataPartition, self).__init__(
            multi=multi, max_core=max_core
        )
        self.split_num = 1
        train_start, train_end, test_start, test_end = self._timestamp_check(
            train_start, train_end, test_start, test_end
        )

        self.train_start: pd.Timestamp = train_start
        self.train_end: pd.Timestamp = train_end
        self.test_start: pd.Timestamp = test_start
        self.test_end: pd.Timestamp = test_end

    def _timestamp_check(
        self,
        train_start: Optional[Timestamp],
        train_end: Timestamp,
        test_start: Timestamp,
        test_end: Optional[Timestamp],
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        train_end = pd.Timestamp(train_end)
        test_start = pd.Timestamp(test_start)

        train_start = (
            pd.Timestamp.min if train_start is None else pd.Timestamp(train_start)
        )
        test_end = pd.Timestamp.max if test_end is None else pd.Timestamp(test_end)

        if train_end <= train_start:
            msg = "`train_end` should be greater than `train_start`."
            _raise_error(msg)
        if test_end <= test_start:
            msg = "`test_end` should be greater than `test_start`."
            _raise_error(msg)
        return train_start, train_end, test_start, test_end

    def _single_train_test_split(
        self, tag: Union[str, int], data: TimeSeriesData
    ) -> Tuple[Union[str, int], List[TrainTestData]]:

        time_index = data.time
        train_start_idx = np.searchsorted(time_index, self.train_start)
        train_end_idx = np.searchsorted(time_index, self.train_end, side="right")
        test_start_idx = np.searchsorted(time_index, self.test_start)
        test_end_idx = np.searchsorted(time_index, self.test_end, side="right")

        return tag, [
            TrainTestData(
                train=data[train_start_idx:train_end_idx],
                test=data[test_start_idx:test_end_idx],
            )
        ]


class RollingOriginDataParition(DataPartitionBase):
    """Class for rolling-origin data-partition method.

    A rolling-origin data-partition splits data over multiple iterations, wherein each iteration, the end of train set "slides" forward
    by a fixed amount. The size of the train data can either increase or keeps constant, and size of test data keeps constant.
    Moreover, we allow a gap between the train and test data sets. The purpose of the gap is to focus on the long range forecasting ability of the model.

    Attributes:
        start_train_frac: A float in (0, 1) for the initial fraction of data used for training.
        test_frac: A float in (0, 1) for the fraction of data used for testing. The test set is taken at sliding positions from `start_train_frac` up to the end of the dataset.
        window_frac: A float in [0, 1) for the fraction of window between train and test data, i.e., the start of test set is `window_frac` away from the end of train set. Default is 0 (i.e., no gap between train and test set).
        expanding_steps: Optional; An integer for the number of expanding steps (i.e., number of folds). Default is 1.
        constant_train_size: Optional; A boolean for whether training data size should be constant. Default is False.
        multi: Optional; A boolean representing whether using multi-processing or not. Default is True.
        max_core: Optional; An integer representing the number of cores for multiprocessing. Default is None, which sets `max_core = max((total_cores - 1) // 2, 1)`.

    Sample Usage:
        >>> stdp = RollingOriginDataParition(start_train_frac = 0.5, test_frac = 0.2, expanding_steps = 3)
        >>> res = stdp.split(data)
    """

    def __init__(
        self,
        start_train_frac: float,
        test_frac: float,
        window_frac: float = 0.0,
        expanding_steps: int = 1,
        constant_train_size: bool = False,
        multi: bool = True,
        max_core: Optional[int] = None,
    ) -> None:
        super(RollingOriginDataParition, self).__init__(multi=multi, max_core=max_core)

        start_train_frac, test_frac, window_frac, expanding_steps = self._param_check(
            start_train_frac, test_frac, window_frac, expanding_steps
        )
        self.start_train_frac = start_train_frac
        self.test_frac = test_frac
        self.window_frac = window_frac
        self.expanding_steps = expanding_steps
        self.constant_train_size = constant_train_size
        self.split_num = expanding_steps

    def _param_check(
        self,
        start_train_frac: float,
        test_frac: float,
        window_frac: float,
        expanding_steps: int,
    ) -> Tuple[float, float, float, int]:

        if start_train_frac < 0.0 or start_train_frac > 1.0:
            msg = f"`start_frain_frac` should be a float in (0, 1) but receives {start_train_frac}."
            _raise_error(msg)
        if test_frac < 0.0 or test_frac > 1.0:
            msg = f"`test_frac` should be a float in (0, 1) but receives {test_frac}."
            _raise_error(msg)
        if window_frac < 0.0 or window_frac > 1.0:
            msg = (
                f"`window_frac` should be a float in [0, 1) but receives {window_frac}."
            )
            _raise_error(msg)
        if start_train_frac + test_frac + window_frac > 1.0:
            msg = "The sum of `start_frain_frac` and `test_frac` should be no larger than 1."
            _raise_error(msg)
        if start_train_frac + test_frac + window_frac == 1 and expanding_steps > 1:
            msg = "When the sum of `start_frain_frac` and `test_frac` is 1, `expanding_steps` should be 1."
            _raise_error(msg)
            msg = "When the sum of `start_frain_frac` and `test_frac` is 1, `expanding_steps` should be 1."
            _raise_error(msg)
        if expanding_steps < 1:
            msg = f"`expanding_steps` should be a positive integer but receives {expanding_steps}."
        return start_train_frac, test_frac, window_frac, expanding_steps

    def _single_train_test_split(
        self, tag: Union[str, int], data: TimeSeriesData
    ) -> Tuple[Union[str, int], List[TrainTestData]]:
        n = len(data)
        start_train_size = _get_absolute_size(n, self.start_train_frac)
        test_size = _get_absolute_size(n, self.test_frac)
        window_size = _get_absolute_size(n, self.window_frac)
        offsets = np.linspace(
            0,
            n - start_train_size - test_size - window_size,
            self.expanding_steps,
            dtype=int,
        )
        if self.constant_train_size:  # does not increase training set length
            res = [
                TrainTestData(
                    train=data[offset : (offset + start_train_size)],
                    test=data[
                        (offset + start_train_size + window_size) : (
                            offset + start_train_size + test_size + window_size
                        )
                    ],
                )
                for offset in offsets
            ]
        else:
            res = [
                TrainTestData(
                    train=data[: (offset + start_train_size)],
                    test=data[
                        (offset + start_train_size + window_size) : (
                            offset + start_train_size + test_size + window_size
                        )
                    ],
                )
                for offset in offsets
            ]
        return tag, res
