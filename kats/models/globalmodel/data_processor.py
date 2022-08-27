# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kats.consts import TimeSeriesData
from kats.models.globalmodel.utils import GMParam
from torch import Tensor

"""
This module provides two Classes for data processing for global models: :class:`GMDataLoader` and :class:`GMBatch`.
"""


class GMDataLoader:
    """

    This is the data loader class for global model, which groups time series with similar lengths into the same batch and generate the batches of time series ids.

    Attributes:
        dataset: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the time series.
        test_dataset: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the time series for testing or validation.
        magnitude: Optional; A float representing the magnitude of randomness added to lengths of time series.

    Sample Usage:
        >>> dl = GMDataLoader(data)
        >>> # generate a batch of size 5
        >>> batch_ids = dl.get_batch(5)
    """

    def __init__(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        dataset: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_dataset: Union[
            None, Dict[Any, TimeSeriesData], List[TimeSeriesData]
        ] = None,
        magnitude: float = 5.0,
    ) -> None:
        keys, lengths = self._valid_dataset(dataset)
        if test_dataset is not None:
            test_keys, test_lengths = self._valid_dataset(test_dataset)
            if len(keys) != len(test_keys) or len(
                set(keys).intersection(test_keys)
            ) != len(keys):
                msg = "The keys of dataset and test_dataset are not the same."
                raise ValueError(msg)
            # pyre-fixme[4]: Attribute must be annotated.
            self.test_lengths = test_lengths
        else:
            self.test_lengths = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.keys = keys
        # pyre-fixme[4]: Attribute must be annotated.
        self.lengths = lengths
        # pyre-fixme[4]: Attribute must be annotated.
        self._batch_ids = None
        self._batch_size = -1
        self._batch_num = -1
        # pyre-fixme[4]: Attribute must be annotated.
        self._last_batch = None
        self._idx = -1
        # pyre-fixme[4]: Attribute must be annotated.
        self.num = len(dataset)
        if magnitude > 0:
            # pyre-fixme[4]: Attribute must be annotated.
            self.magnitude = magnitude

    def _valid_dataset(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        dataset: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(dataset) < 1:
            msg = "Input dataset should be non-empty."
            logging.error(msg)
            raise ValueError(msg)

        if isinstance(dataset, list):
            keys = np.arange(len(dataset))
        elif isinstance(dataset, dict):
            keys = np.array(list(dataset.keys()))
        else:
            msg = f"dataset should be either a list or a dictionary, but receives {type(dataset)}."
            logging.error(msg)
            raise ValueError(msg)
        lengths = []
        for k in keys:
            if not isinstance(dataset[k], TimeSeriesData):
                msg = f"Every element in dataset should be a TimeSeriesData but receives {type(dataset[k])}."
                logging.error(msg)
                raise ValueError(msg)
            lengths.append(len(dataset[k]))
        lengths = np.array(lengths)
        return keys, lengths

    def _get_new_order(self) -> np.ndarray:
        """Generate new orders of time series."""

        if self.test_lengths is None:
            # add some randomness to TSs with the same length.
            new_length = self.lengths + np.random.uniform(0, 1, self.num)
            orders = np.argsort(new_length)
        else:
            # add some randomness to test TSs with the same length.
            new_test_lengths = (
                self.test_lengths + np.random.uniform(0, 1, self.num) * self.magnitude
            )
            # first sort according to lengths of train TSs, then sort according to lengths of test TSs.
            orders = np.lexsort((new_test_lengths, self.lengths))
        return orders

    def _shuffle_batch_ids(self, batch_size: int) -> None:
        """
        Regrouping and shuffling batches for the given batch_size.

        """
        if not isinstance(batch_size, int) or batch_size < 1:
            msg = f"batch_size should be a positive integer but receive {batch_size}."
            logging.error(msg)
            raise ValueError(msg)

        orders = self._get_new_order()
        n = (self.num // batch_size) * batch_size
        batch_ids = orders[:n].reshape(-1, batch_size)
        np.random.shuffle(batch_ids)

        # group the last several TSs into last_batch
        if n < self.num:
            last_batch = orders[n:]
        else:
            last_batch = None

        # convert ids to keys
        if self.keys is not None:
            batch_ids = self.keys[batch_ids]
            if last_batch is not None:
                last_batch = self.keys[last_batch]

        # re-initiate attributes
        self._idx = -1
        self._batch_ids = batch_ids
        self._batch_num = len(batch_ids)
        self._batch_size = batch_size
        self._last_batch = last_batch

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def get_batch(self, batch_size: int) -> List[Any]:
        """Generate a batch of ids of batch_size.

        Args:
            batch_size: A positive integer representing the size of the batch.

        Returns:
            A list of items representing the time series ids.
        """

        self._idx += 1
        if batch_size == self._batch_size:  # same batch_size as the previous query
            if self._idx < self._batch_num:
                return list(self._batch_ids[self._idx])
            elif self._idx == self._batch_num and (self._last_batch is not None):
                return list(self._last_batch)
        # re-grouping and re-shuffle
        self._shuffle_batch_ids(batch_size)
        return self.get_batch(batch_size)


class GMBatch:
    """
    This is the class for
        1) computing indices for training data and validation data.
        2) transforming time series data into `torch.Tensor`.
        3) computing some time series features.

    Attributes:
        params: A :class:`kats.models.globalmodel.utils.GMParam` object for global model.
        batch_ids: A list of items representing the time series ids.
        train_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the training data.
        valid_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the validation data.
        mode: Optional; A string representing the mode. Can be 'train' or 'test'. Default is 'train'.

    Sample Usage:
        >>> batch = GMBatch(gmparam, batch_ids, train_TSs, valid_TSs)
    """

    def __init__(
        self,
        params: GMParam,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        batch_ids: List[Any],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid_TSs: Optional[
            Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        ] = None,
        mode: str = "train",
    ) -> None:

        if not isinstance(params, GMParam):
            msg = f"params should be a GMParam object but receives {type(params)}."
            logging.error(msg)
            raise ValueError(msg)

        train = {idx: train_TSs[idx] for idx in batch_ids}
        valid = (
            None if valid_TSs is None else {idx: valid_TSs[idx] for idx in batch_ids}
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.train = train
        # pyre-fixme[4]: Attribute must be annotated.
        self.valid = valid

        # pyre-fixme[4]: Attribute must be annotated.
        self.training = mode == "train"
        # pyre-fixme[4]: Attribute must be annotated.
        self.batch_size = len(train)
        self.batch_ids = batch_ids
        self.training_encoder_step_num = 1
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_encoder_step_num = params.fcst_step_num

        (
            reduced_length,
            reduced_valid_length,
            train_indices,
            valid_indices,
        ) = self._get_indices(train, valid, params)

        (
            train_x,
            train_time,
            offset,
            valid_x,
            valid_time,
        ) = self._get_array(train, valid, params, reduced_length, reduced_valid_length)

        self.batch_size = len(train)

        # pyre-fixme[4]: Attribute must be annotated.
        self.train_length = reduced_length
        # pyre-fixme[4]: Attribute must be annotated.
        self.valid_length = reduced_valid_length

        # pyre-fixme[4]: Attribute must be annotated.
        self.train_indices = train_indices
        # pyre-fixme[4]: Attribute must be annotated.
        self.valid_indices = valid_indices

        tdtype = torch.get_default_dtype()

        if params.model_type == "rnn" and params.seasonality > 1:
            init_seasonality = self._get_seasonality(train_x, params.seasonality)
            # bound initial seasonalities
            init_seasonality[
                init_seasonality < params.init_seasonality[0]
            ] = params.init_seasonality[0]
            init_seasonality[
                init_seasonality > params.init_seasonality[1]
            ] = params.init_seasonality[1]
            # pyre-fixme[4]: Attribute must be annotated.
            self.init_seasonality = torch.tensor(init_seasonality, dtype=tdtype)
        else:
            self.init_seasonality = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.offset = torch.tensor(offset, dtype=tdtype).view(-1, 1)
        # pyre-fixme[4]: Attribute must be annotated.
        self.indices = train_indices + valid_indices

        if valid or (not self.training):
            x = np.column_stack([train_x, valid_x])
            time = np.column_stack([train_time, valid_time])
        else:
            x = train_x
            time = train_time
        # store info for gmfeature
        # pyre-fixme[4]: Attribute must be annotated.
        self.gmfeature = params.gmfeature
        # pyre-fixme[4]: Attribute must be annotated.
        self.base_features = (
            params.gmfeature.get_base_features(x, time)
            if params.gmfeature is not None
            else None
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.x_array = (
            x  # storing a np.ndarray copy of x for on-the-fly feature computing
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.x = torch.tensor(x, dtype=tdtype)
        # pyre-fixme[4]: Attribute must be annotated.
        self.time = time

    def _get_indices(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train: Dict[Any, TimeSeriesData],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid: Optional[Dict[Any, TimeSeriesData]],
        params: GMParam,
    ) -> Tuple[int, int, List[int], List[int]]:
        """
        Helper function for training and validation indices.

        :Parameters:
        train: Dict[Any, TimeSeriesData]
            Dictionary of training TSs.
        valid: Optional[Dict[Any, TimeSeriesData]]
            Dictionary of validation TSs.
        params: GMParam
            The object for parameters.

        :Returns:
        reduced_length: int
            Reduced length of training TSs and the training tensor would be of shape (batch_size, reduced_length)
        reduced_valid_length: int
            Reduced length of validation TSs and the validation tensor would be of shape (batch_size, reduced_valid_length)
        train_indices: List[int]
            A list of eligible training indicies (i.e., train_x[i-input_window: i] would be feeded to NN where i is in train_indices).
        valid_indices: List[int]
            A list of eligible validation indicies (i.e., valid_x[i-input_window: i] would be feeded to NN where i is in valid_indices).
        """

        input_window = params.input_window
        fcst_window = params.fcst_window
        min_training_step_num = params.min_training_step_num
        min_training_step_length = params.min_training_step_length
        min_warming_up_step_num = params.min_warming_up_step_num
        validation_step_num = params.validation_step_num

        lengths = [len(train[t]) for t in train]
        if np.min(lengths) < params.seasonality:
            msg = "Minimum length of input time series is shorter than seasonality, please pad the time series first or remove it."
            logging.error(msg)
            raise ValueError(msg)
        max_length = np.max(lengths)

        # training mode
        if self.training:
            train_basic_length = (
                input_window + min_training_step_num * min_training_step_length
            )
            # calculate minimum length of a training TS
            if params.model_type == "rnn":
                basic_length = train_basic_length + fcst_window
            else:
                self.training_encoder_step_num = np.min(
                    [
                        int(params.fcst_step_num * 2),
                        np.random.randint(
                            1, max(2, (max_length - train_basic_length) // fcst_window)
                        ),
                    ]
                )
                basic_length = (
                    train_basic_length + fcst_window * self.training_encoder_step_num
                )
            if basic_length > max_length:
                msg = f"TSs for batching are too short! (i.e., the length of the longest TS are recommended be at least {basic_length})."
                logging.warning(msg)

            eligible_length = max(0, max_length - basic_length)

            max_eligible_step_num = eligible_length // min_training_step_length

            reduced_length = (
                basic_length
                + np.random.randint(0, max_eligible_step_num + 1)
                * min_training_step_length
            )

            train_indices = list(
                np.arange(
                    input_window,
                    reduced_length - fcst_window * self.training_encoder_step_num + 1,
                    min_training_step_length,
                )
            )

            if valid is not None:
                if params.model_type == "rnn":
                    max_valid_length = np.max([len(valid[t]) for t in valid])
                    last_valid_index = reduced_length + np.min(
                        [max_valid_length, fcst_window * validation_step_num]
                    )
                else:
                    last_valid_index = (
                        reduced_length + self.training_encoder_step_num * fcst_window
                    )
                    self.test_encoder_step_num = self.training_encoder_step_num
                valid_indices = list(
                    np.arange(
                        reduced_length,
                        last_valid_index,
                        fcst_window,
                    )
                )
                reduced_valid_length = len(valid_indices) * fcst_window
            else:
                valid_indices = []
                reduced_valid_length = 0
        # testing mode
        else:
            min_warming_up_step_num = params.min_warming_up_step_num
            basic_length = (
                input_window + min_warming_up_step_num * min_training_step_length
            )
            # we use all possible data for testing
            reduced_length = np.max(
                (
                    basic_length,
                    max_length - (max_length - input_window) % min_training_step_length,
                )
            )
            train_indices = list(
                np.arange(input_window, reduced_length, min_training_step_length)
            )
            reduced_valid_length = fcst_window * params.fcst_step_num
            valid_indices = list(
                np.arange(
                    reduced_length, reduced_length + reduced_valid_length, fcst_window
                )
            )

        return reduced_length, reduced_valid_length, train_indices, valid_indices

    def _get_array(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train: Dict[Any, TimeSeriesData],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid: Optional[Dict[Any, TimeSeriesData]],
        params: GMParam,
        reduced_length: int,
        reduced_valid_length: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:

        """

        Helper function for transforming TS to arrays, including truncating/padding values,
        extracting dates, seasonalities and offsets.

        :Parameters:
        train: Dict[Any, TimeSeriesData]
            Dictionary of training TSs.
        valid: Optional[Dict[Any, TimeSeriesData]]
            Dictionary of validation TSs.
        params: GMParam
            The object for parameters.
        reduced_length: int
            Reduced length of training TSs and the training tensor would be of shape (batch_size, reduced_length).
        reduced_valid_length: int
            Reduced length of validation TSs and the validation tensor would be of shape (batch_size, reduced_valid_length).

        :Returns:
        train_x: np.ndarray
            Training array of shape (batch_size, reduced_length).
        train_time: np.ndarray
            Training time array of shape (batch_size, reduced_length).
        offset: np.ndarray
            Offset array of shape (batch_size,).
        valid_x: Optional[np.ndarray]
            Validation array of shape (batch_size, reduced_valid_length).
        valid_time: Optional[np.ndarray]
            Valid time array of shape (batch_size, reduced_valid_length).
        """

        uplifting_ratio = params.uplifting_ratio
        seasonality = params.seasonality
        freq = params.freq
        min_init_seasonality, max_init_seasonality = params.init_seasonality

        train_x = []
        train_time = []
        offset = []

        if valid is not None or (not self.training):
            valid_x = []
            valid_time = []

        for idx in train:
            train_ts = train[idx]
            train_val = train_ts.value.values.copy()
            train_timestamp = train_ts.time.values
            # calculate offset_val for uplifting negative value
            min_val = (
                train_ts.min
                if valid is None
                else np.min([train_ts.min, valid[idx].min])
            )
            if min_val <= 0:
                max_val = (
                    train_ts.max
                    if valid is None
                    else np.max([train_ts.max, valid[idx].max])
                )
                if min_val == max_val:  # receives a constant TS
                    offset_val = 1.0 - min_val
                else:
                    offset_val = (max_val - uplifting_ratio * min_val) / (
                        uplifting_ratio - 1
                    )
            else:
                offset_val = 0.0

            offset.append(offset_val)

            # truncate/pad train TS
            train_length = len(train_ts)
            train_val += offset_val
            if train_length >= reduced_length:  # truncate
                tmp_train_x = train_val[-reduced_length:]
                train_time.append(train_timestamp[-reduced_length:])
            else:  # pad
                pad_length = reduced_length - train_length
                pad_rep = pad_length // seasonality + 1
                pad_val = np.tile(train_val[:seasonality], pad_rep)[-pad_length:]
                pad_time = np.tile(train_timestamp[:seasonality], pad_rep)[-pad_length:]

                tmp_train_x = np.concatenate((pad_val, train_val))
                train_time.append(np.concatenate((pad_time, train_timestamp)))

            # fillin the first element if it is NaN (just for safe)
            if np.isnan(tmp_train_x[0]):
                tmp_train_x[0] = tmp_train_x[~np.isnan(tmp_train_x)][0]

            # fillin NaNs in within the first seasonality period
            tmp_train_x = self._fillin_nan_values(tmp_train_x, seasonality)
            train_x.append(tmp_train_x)
            if valid is not None:
                valid_val = valid[idx].value.values + offset_val
                valid_timestamp = valid[idx].time.values
                valid_length = len(valid_val)
                tmp_valid_x = np.full(reduced_valid_length, np.nan)

                if valid_length < reduced_valid_length:
                    tmp_valid_x[:valid_length] = valid_val
                    pad_length = reduced_valid_length - valid_length
                    valid_time.append(
                        np.concatenate(
                            [
                                valid_timestamp,
                                pd.date_range(
                                    valid_timestamp[-1] + freq,
                                    freq=freq,
                                    periods=pad_length,
                                ).values,
                            ]
                        )
                    )
                else:
                    tmp_valid_x = valid_val[:reduced_valid_length]
                    valid_time.append(valid_timestamp[:reduced_valid_length])
                valid_x.append(tmp_valid_x)

            elif (
                not self.training
            ):  # prepare testing data for testing mode when valid is None
                valid_x.append(np.full(reduced_valid_length, np.nan))
                valid_time.append(
                    pd.date_range(
                        train_timestamp[-1] + freq,
                        freq=freq,
                        periods=reduced_valid_length,
                    ).values
                )

        train_x = np.row_stack(train_x)
        train_time = np.row_stack(train_time)
        offset = np.array(offset)

        if valid is not None or not self.training:
            valid_x = np.row_stack(valid_x)
            valid_time = np.row_stack(valid_time)
        else:
            valid_x = None
            valid_time = None

        return train_x, train_time, offset, valid_x, valid_time

    def _get_seasonality(self, x: np.ndarray, seasonality: int) -> np.ndarray:
        """
        Helper function for calculate initial seasonality.

        Args:
            x: A np.ndarray object representing the raw data array.
            seasonality: An integer representing the seasonality period length.

        Returns:
            season: A np.ndarray object representing the seasonality array.
        """
        season = x[:, :seasonality].copy()  # avoid changing values in the x.
        season = season / np.mean(season, axis=1)[:, None]
        return season

    def _fillin_nan_values(self, x: np.ndarray, seasonality: int) -> np.ndarray:
        if seasonality == 1:
            return x
        for i in range(1, seasonality):
            if np.isnan(x[i]):
                x[i] = x[i - 1]
        return x

    def get_features(self, start_idx: int, end_idx: int) -> Tensor:
        """Function for computing time series features.

        Args:
            start_idx: An integer representing the starting index (inclusive) of the processed time series data.
            end_idx: An integer representing the end index (exclusive) of the processed time series data.

        Returns:
            A :class:`torch.Tensor` object representing the feature tensor.
        """

        features = self.gmfeature.get_on_the_fly_features(
            self.x_array[:, start_idx:end_idx], self.time[:, start_idx:end_idx]
        )
        if self.base_features is not None:
            return torch.cat([self.base_features, features], dim=1)
        return features
