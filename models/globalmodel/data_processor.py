import logging
from typing import List, Union, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models.globalmodel.utils import GMParam
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
from torch import Tensor


class GMDataLoader:

    """
    A class for grouping TSs with similar lengths into the same batch and generating batches of ids.

    :Parameters:
    dataset: Union[Dict[int, TimeSeriesData], List[TimeSeriesData]].
        A list or a dictionary of TSs.
    """

    def __init__(
        self, dataset: Union[Dict[int, TimeSeriesData], List[TimeSeriesData]]
    ) -> None:
        if len(dataset) < 1:
            msg = "Input dataset should be non-empty."
            logging.error(msg)
            raise ValueError(msg)

        if isinstance(dataset, list):
            self.keys = None
        elif isinstance(dataset, dict):
            self.keys = np.array(list(dataset.keys()))
        else:
            msg = f"dataset should be either a list or a dictionary, but receives {type(dataset)}."
            logging.error(msg)
            raise ValueError(msg)

        keys = self.keys if self.keys is not None else range(len(dataset))
        lengths = []
        for k in keys:
            if not isinstance(dataset[k], TimeSeriesData):
                msg = f"Every element in dataset should be a TimeSeriesData but receives {type(dataset[k])}."
                logging.error(msg)
                raise ValueError(msg)
            lengths.append(len(dataset[k]))

        self.lengths = np.array(lengths)
        self._batch_ids = None
        self._batch_size = -1
        self._batch_num = -1
        self._last_batch = None
        self._idx = -1
        self.num = len(dataset)

    def _shuffle_batch_ids(self, batch_size: int) -> None:
        """
        Regrouping and shuffling batches for the given batch_size.

        """
        if not isinstance(batch_size, int) or batch_size < 1:
            msg = f"batch_size should be a positive integer but receive {batch_size}."
            logging.error(msg)
            raise ValueError(msg)
        # add some randomness to TSs with the same length.
        new_length = self.lengths + np.random.uniform(0, 1, self.num)
        orders = np.argsort(new_length)
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

    def get_batch(self, batch_size: int) -> List[int]:
        """
        Generate batch of ids of batch_size

        :Parameters:
        batch_size: int
            The size of the batch, should be a positive integer.
        :Returns:
        List[int]
            A list of ids.
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
    A class for transforming TS data into tensor.

    :Parameters:
    params: GMParam
        A GMParam object for global model.
    batch_ids: List[Any]
        A list of TS ids.
    train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        A dictionary or a list of all training TSs.
    valid_TSs: Optional[Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]]=None
        A dictionary or a list of all validation TSs.
    mode: str = 'train'
        Mode for batch. If not 'train', then mode is taken as 'test'.

    """

    def __init__(
        self,
        params: GMParam,
        batch_ids: List[Any],
        train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
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

        self.train = train
        self.valid = valid

        self.training = mode == "train"
        self.batch_size = len(train)
        self.batch_ids = batch_ids

        (
            reduced_length,
            reduced_valid_length,
            train_indices,
            valid_indices,
        ) = self._get_indices(train, valid, params)

        (
            train_x,
            train_time,
            init_seasonality,
            offset,
            valid_x,
            valid_time,
        ) = self._get_array(train, valid, params, reduced_length, reduced_valid_length)

        self.batch_size = len(train)

        self.train_length = reduced_length
        self.valid_length = reduced_valid_length

        self.train_indices = train_indices
        self.valid_indices = valid_indices

        tdtype = torch.get_default_dtype()

        self.init_seasonality = (
            torch.tensor(init_seasonality, dtype=tdtype)
            if params.seasonality > 1
            else None
        )
        self.offset = torch.tensor(offset, dtype=tdtype).view(-1, 1)

        self.indices = train_indices + valid_indices

        if valid or (not self.training):
            x = np.column_stack([train_x, valid_x])
            time = np.column_stack([train_time, valid_time])
        else:
            x = train_x
            time = train_time
        # store info for gmfeature
        self.gmfeature = params.gmfeature
        self.base_features = (
            params.gmfeature.get_base_features(x, time)
            if params.gmfeature is not None
            else None
        )
        self.feature_size = (
            params.gmfeature.feature_size if params.gmfeature is not None else None
        )

        self.x_array = (
            x  # storing a np.ndarray copy of x for on-the-fly feature computing
        )
        self.x = torch.tensor(x, dtype=tdtype)
        self.time = time

    def _get_indices(
        self,
        train: Dict[Any, TimeSeriesData],
        valid: Optional[Dict[Any, TimeSeriesData]],
        params: GMParam,
    ) -> Tuple[int, int, List[int], Optional[List[int]]]:
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

        max_length = np.max([len(train[t]) for t in train])

        # training mode
        if self.training:
            # minimum length of a training TS
            basic_length = (
                input_window
                + fcst_window
                + min_training_step_num * min_training_step_length
            )

            if basic_length > max_length:
                msg = f"TSs for batching are too short! (i.e., the length of the longest TS should be at least {basic_length})."
                logging.error(msg)
                raise ValueError(msg)

            eligible_length = max_length - basic_length

            max_eligible_step_num = eligible_length // min_training_step_length

            reduced_length = (
                basic_length
                + np.random.randint(0, max_eligible_step_num + 1)
                * min_training_step_length
            )

            train_indices = list(
                np.arange(
                    input_window,
                    reduced_length - fcst_window + 1,
                    min_training_step_length,
                )
            )

            if valid is not None:
                max_valid_length = np.max([len(valid[t]) for t in valid])
                valid_indices = list(
                    np.arange(
                        reduced_length,
                        reduced_length
                        + np.min([max_valid_length, fcst_window * validation_step_num]),
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
        train: Dict[Any, TimeSeriesData],
        valid: Optional[Dict[Any, TimeSeriesData]],
        params: GMParam,
        reduced_length: int,
        reduced_valid_length: int,
    ) -> Tuple[
        np.array,
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
        init_seasonality: np.ndarray
            Initial seasonality tensor of shape (batch_size, seasonality).
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
        init_seasonality = []

        if valid is not None or (not self.training):
            valid_x = []
            valid_time = []

        for idx in train:
            train_ts = train[idx]
            train_val = train_ts.value.values
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
                pad_time = np.tile(train_timestamp, pad_rep)[-pad_length:]

                tmp_train_x = np.concatenate((pad_val, train_val))
                train_time.append(np.concatenate((pad_time, train_timestamp)))

            # fillin the first element if it is NaN (just for safe)
            if np.isnan(tmp_train_x[0]):
                tmp_train_x[0] = tmp_train_x[~np.isnan(tmp_train_x)][0]

            # get initial seasonality
            tmp_train_x, season = self._get_seasonality(tmp_train_x, seasonality)

            train_x.append(tmp_train_x)
            init_seasonality.append(season)

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
        init_seasonality = np.row_stack(init_seasonality)
        init_seasonality[init_seasonality < min_init_seasonality] = min_init_seasonality
        init_seasonality[init_seasonality > max_init_seasonality] = max_init_seasonality
        offset = np.array(offset)

        if valid is not None or not self.training:
            valid_x = np.row_stack(valid_x)
            valid_time = np.row_stack(valid_time)
        else:
            valid_x = None
            valid_time = None

        return train_x, train_time, init_seasonality, offset, valid_x, valid_time

    def _get_seasonality(
        self, x: np.ndarray, seasonality: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function for calculate initial seasonality.

        :Parameters:
        x: np.ndarray
            Raw data array.
        seasonality: int
            Seasonality period length.
        : Returns:
        x: np.ndarray
            Raw data array after filling NaNs with values within the first period
        season: np.ndarray
            Seasonality array.
        """
        if seasonality == 1:
            return x, np.array([1.0])
        for i in range(1, seasonality):
            if np.isnan(x[i]):
                x[i] = x[i - 1]
        season = x[:seasonality].copy()  # avoid changing values in the x.
        season /= np.mean(season)
        return x, season

    def get_features(self, start_idx: int, end_idx: int) -> Tensor:
        """
        Get feature tensor when input_x is computed with self.x[:, start_idx: end_idx], where start_idx is inclusive and end_idx is exclusive.

        :Parameters:
        start_idx: int
            Starting index.
        end_idx: int
            End index.
        :Returns:
        features: Tensor
            Feature tensor.
        """
        features = self.gmfeature.get_on_the_fly_features(
            self.x_array[:, start_idx:end_idx], self.time[:, start_idx:end_idx]
        )
        if self.base_features is not None:
            return torch.cat([self.base_features, features], dim=1)
        return features


class GMFeature:
    """
    Feature extraction module for global model

    We currently support three types of features
    - simple date feature: such as date of week/month/year, etc
    - tsfeatures: features defined in Kats tsfeature module
    - time series embedding: embedding from Kats time2vec model # TODO

    :Parameters:
    feature_type: str in ["simple_date", "tsfeatures", "ts2vec"]

    :methods:
    get_features: get time series feature by given feature_type
    _get_date_features: method to get simple date features
    _get_tsfeatures: method to get Kats tsfeatures
    _get_ts2vec: method to get Kats ts2vec embeddings
    """

    def __init__(
        self,
        feature_size: int,
        feature_type: Tuple[str] = ("simple_date"),
    ) -> None:
        if not set(feature_type).issubset(set("simple_date", "tsfeatures", "ts2vec")):
            msg = "feature_type must from simple_date, tsfeatures, or ts2vec."
            logging.error(msg)
            raise ValueError(msg)
        self.feature_type = feature_type
        self.feature_size = feature_size

    @staticmethod
    def _get_tsfeatures(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """
        private method to get Kats tsfeatures
        please refer kats.tsfeatures for more details
        """
        features = []

        for i in range(len(x)):
            print(i)
            features.append(
                TsFeatures().transform(
                    TimeSeriesData(
                        pd.DataFrame({"time": time[i], "value": x[i].numpy()})
                    )
                )
            )
        features = torch.tensor(features)
        return features

    @staticmethod
    def _get_date_features(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """
        private method to get simple date features
        We leverage the properties from `pandas.DatetimeIndex`

        It includes:
            - minute
            - hour
            - day
            - month
            - dayofweek
            - dayofyear
        """
        feature = []

        for i in range(len(x)):
            feature.append(
                sum(
                    [
                        pd.to_datetime(time[i]).day.to_list(),
                        pd.to_datetime(time[i]).month.to_list(),
                        pd.to_datetime(time[i]).dayofweek.to_list(),
                        pd.to_datetime(time[i]).dayofyear.to_list(),
                    ],
                    [],
                )
            )
        feature = torch.tensor(feature)
        return feature

    @staticmethod
    def _get_ts2vec(
        x: np.ndarray,
        time: np.ndarray,
    ):
        # TODO after ts2vec model lands
        pass

    def get_features(
        self,
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        funcs = {
            "simple_date": self._get_date_features,
            "tsfeatures": self._get_tsfeatures,
            "ts2vec": self._get_ts2vec,
        }

        # get features by given feature types
        features = []
        for ft in self.feature_type:
            features.append(funcs[ft](x, time))

        return torch.cat(features, 1)
