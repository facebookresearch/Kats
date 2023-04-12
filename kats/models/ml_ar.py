# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Any, cast, Dict, List, Optional, Tuple, Union

import lightgbm as gbm  # @manual
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.metrics.metrics import smape
from kats.tsfeatures.tsfeatures import TsCalenderFeatures, TsFourierFeatures
from kats.utils.parameter_tuning_utils import (
    get_default_lightgbm_parameter_search_space,
)

from numba import njit
from numpy.random import RandomState


def find_length(curr_series_data: pd.DataFrame, target_variable: List[str]) -> int:
    return int(find_length_array(curr_series_data[target_variable].values) + 1)


@njit
def find_length_array(nums: np.ndarray) -> int:
    if len(nums.shape) == 1:
        for i in range(len(nums) - 1, -1, -1):
            if ~np.isnan(nums[i]):
                return i
    else:
        for i in range(len(nums) - 1, -1, -1):
            if np.any(~np.isnan(nums[i])):
                return i
    return -1


def normalize(
    x: pd.DataFrame,
    normalizer: pd.Series,
    normalizer2: Optional[pd.Series] = None,
    use_default_min: bool = False,
    default_min: float = 1e-8,
    sub_div: str = "div",
) -> pd.DataFrame:
    """Row-wise normalization applied to the target `pd.DataFrame`."""
    if use_default_min:
        x = x + default_min
        normalizer = normalizer + default_min
        if normalizer2 is not None:
            normalizer2 = normalizer2 + default_min

    if sub_div == "sub":
        norm_data = x.sub(normalizer, axis=0)
    elif sub_div == "div":
        norm_data = x.div(normalizer, axis=0)
    elif sub_div == "sub_div":
        norm_data = (x.sub(normalizer, axis=0)).div(normalizer2, axis=0)
    else:
        raise ValueError(f"`sub_dive` method {sub_div} is invalid.")

    return norm_data


def denormalize(
    x: pd.DataFrame,
    normalizer: pd.Series,
    normalizer2: Optional[pd.Series] = None,
    use_default_min: bool = False,
    default_min: float = 1e-8,
    sub_div: str = "div",
) -> pd.DataFrame:
    """Denormalization applied to the target `pd.DataFrame`."""

    if use_default_min:
        x = x - default_min
        normalizer = normalizer - default_min
        if normalizer2 is not None:
            normalizer2 = normalizer2 - default_min

    if sub_div == "sub":
        norm_data = x.add(normalizer, axis=0)
    elif sub_div == "div":
        norm_data = x.mul(normalizer, axis=0)
    elif sub_div == "sub_div":
        norm_data = (x.mul(normalizer2, axis=0)).add(normalizer, axis=0)
    else:
        raise ValueError(f"`sub_dive` method {sub_div} is invalid.")
    return norm_data


# TODO: Fourier terms related computation should be wrapped in TsFourierFeatures
def fourier_terms(
    dates: pd.Series, period: Union[float, int], series_order: int
) -> np.ndarray:
    """Provides Fourier series components with the specified frequency
    and order. The starting time is always the epoch.

    Args:
        dates: pd.Series containing timestamps.
        period: Number of hours of the period.
        series_order: Number of components.
    Returns:
        A `np.ndarray` representing the Fourier Features.
    """
    # convert to days since epoch
    t = np.array(dates.astype("int") // 10**9) / 3600.0
    return np.column_stack(
        [
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ]
    )


def categorical_encode(
    df: pd.DataFrame,
    categoricals: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """create one-hot encoding for categorical columns
    Args:
        df: pd.DataFrame with categorical features to be embedded
        categoricals: List of columns of categorical features
    Returns:
        Tuple[pd.DataFrame, List[str]] of data after one-hot encoding and the columns name
    """

    result_list = []
    for feature in categoricals:
        dummy = pd.get_dummies(df[feature])
        dummy.columns = [f"{feature}_is_{column}" for column in dummy.columns]
        result_list.append(dummy)
    result = pd.concat(result_list, axis=1, copy=False)
    return result, list(result.columns)


@njit
def embed(series: np.ndarray, lags: int, horizon: int, max_lags: int) -> np.ndarray:
    # slight speed-up against the earlier version
    result = np.full(
        shape=(series.size - max_lags + 1, lags + horizon), fill_value=np.nan
    )

    # do the windowing for the bulk of the series
    for i in range((max_lags - lags), series.size - (lags + horizon)):
        result[i - (max_lags - lags), :] = series[i : i + lags + horizon]

    # do the windowing for the final part of the series, and fill missing horizons with nan
    for i in range(series.size - (lags + horizon), series.size - lags + 1):
        curr_wind = series[i : i + lags + horizon]
        result[i - (max_lags - lags), 0 : curr_wind.size] = curr_wind

    return result


class MLARParams:
    """Parameter class for time series MLAR model

    This is the parameter class for time series LightGBM model. Details of lightgbm inherented params
    can be found https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

    [TODO] Parameters needs updates
    Args:
        target_variable: either a string or a list of strings for target variables columns.
        horizon: a positive integer for output window, default is 10.
        input_window: a positive integer for input windows, i.e., the number of lags. Default is 10.
        freq: a string representing the frequency of time series. Default is "D".
        cov_history_input_windows: a dictionary representing historical covariate information, whose keys are strings for covariate names and values are positive integers for number of lags. Default is None.
        cov_future_input_windows: a dictionary representing future covariate information, whose keys are strings for covariate names and values are positive integers for number of leaps. Default is None.
        categoricals: a list of strings for the column names of categorical columns.
        one_hot_encode: a boolean for whether using one-hot encoding for categorical covariates. Default is True.
        calendar_features: a string representing calendar feature type. Can be 'cal_own_cal' or 'cal_own_ft'. Default is 'cal_own_cal'.

        n_jobs: integer Number of jobs in LGBMRegressor, default = 1
        max_depth: integer max_depth in LGBMRegressor -- (int, optional (default=-1)) – Maximum tree depth for base learners, <=0 means no limit
        min_data_in_leaf: integer min_data_in_leaf in LGBMRegressor -- min_child_samples (int, optional (default=20)) – Minimum number of data needed in a child (leaf)
        subsample: integer min_data_in_leaf in LGBMRegressor -- (float, optional (default=1.)) – Subsample ratio of the training instance.
        n_estimators: integer n_estimators in LGBMRegressor -- (int, optional (default=100)) – Number of boosted trees to fit.
        learning_rate: float learning_rate in LGBMRegressor -- (float, optional (default=0.1)) – Boosting learning rate.
                    You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback.
                    Note, that this will ignore the learning_rate argument in training.
        colsample_bytree: float colsample_bytree in LGBMRegressor -- (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.
        min_split_gain: float min_split_gain in LGBMRegressor -- (float, optional (default=0.)) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
        reg_alpha: float reg_alpha in LGBMRegressor -- (float, optional (default=0.)) – L1 regularization term on weights.
        num_leaves: integer num_leaves in LGBMRegressor -- (int, optional (default=31)) – Maximum tree leaves for base learners.
        boosting_type: str for boosting_type in LGBMRegressor –- ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive
                    Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
        subsample_for_bin: integer subsample_for_bin in LGBMRegressor -- (int, optional (default=200000)) – Number of samples for constructing bins.
        objective: str as objective function for LGBMRegressor, we set default as quantile
            as the objective function of the algorithm, i.e. what it's trying to maximize or minimize, e.g. "regression" means it's minimizing squared residuals.
        alpha: float for alpha in LGMRegressor default = 0.9, type = double, constraints: alpha > 0.0, used only in huber and quantile regression applications.
        random_state: int or RandomState or None type random_state argument in LGBMRegressor function. It is used to seed the random number generator during the training process.
    """

    def __init__(
        self,
        target_variable: Union[List[str], str] = "y",
        horizon: int = 10,
        input_window: int = 10,
        freq: Optional[str] = None,
        cov_history_input_windows: Optional[Dict[str, int]] = None,
        cov_future_input_windows: Optional[Dict[str, int]] = None,
        categoricals: Optional[List[str]] = None,
        one_hot_encode: bool = True,
        calendar_features: Union[str, List[str]] = "auto",
        fourier_features_period: Union[str, List[Union[float, int]]] = "auto",
        fourier_features_order: Union[str, List[int]] = "auto",
        fourier_features_offset: Union[str, int] = "auto",
        n_jobs: int = 1,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        subsample: float = 1,
        n_estimators: int = 150,
        learning_rate: float = 0.1,
        colsample_bytree: float = 1.0,
        min_split_gain: float = 0.0,
        reg_alpha: float = 0.0,
        num_leaves: int = 31,
        boosting_type: str = "gbdt",
        subsample_for_bin: float = 200000,
        objective: str = "quantile",
        alpha: float = 0.9,
        verbose: int = -1,
        random_state: Union[int, RandomState, None] = None,
        norm_window_size: Optional[int] = None,
        norm_sum_stat: str = "mean",
        sub_div: str = "div",
        use_default_min: bool = False,
        default_min: float = 1e-8,
        transform_data: Optional[str] = None,
        cov_history_norm: bool = False,
        cov_future_norm: bool = False,
        use_sum_stats: bool = True,
    ) -> None:
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.min_split_gain = min_split_gain
        self.reg_alpha = reg_alpha
        self.num_leaves = num_leaves
        self.boosting_type = boosting_type
        self.objective = objective
        self.alpha = alpha
        self.verbose = verbose

        self.target_variable: List[str] = (
            target_variable if isinstance(target_variable, List) else [target_variable]
        )
        self.horizon = horizon
        self.input_window = input_window
        self.cov_history_input_windows: Dict[str, int] = (
            {}
            if not isinstance(cov_history_input_windows, dict)
            else cov_history_input_windows
        )
        self.cov_future_input_windows: Dict[str, int] = (
            {}
            if not isinstance(cov_future_input_windows, dict)
            else cov_future_input_windows
        )

        self.freq = freq

        self.expand_feature_space: List[str] = list(
            set(self.cov_history_input_windows.keys())
            | set(self.cov_future_input_windows.keys())
        )
        self.categoricals: List[str] = [] if not categoricals else categoricals
        self.one_hot_encode = one_hot_encode
        self.random_state = random_state
        self.forecast_ds_index = 0

        self.norm_window_size: int = (
            input_window if norm_window_size is None else norm_window_size
        )
        self.norm_sum_stat = norm_sum_stat
        self.sub_div = sub_div
        self.use_default_min = use_default_min
        self.default_min = default_min
        self.transform_data = transform_data

        self.cov_history_norm = cov_history_norm
        self.cov_future_norm = cov_future_norm

        self.calendar_features = calendar_features
        self.fourier_features_order = fourier_features_order
        self.fourier_features_period = fourier_features_period
        self.fourier_features_offset = fourier_features_offset

        self.use_sum_stats = use_sum_stats

        # get all necessary column names
        self.all_vars: set[str] = set()
        if self.cov_history_input_windows:
            self.all_vars.update(list(self.cov_history_input_windows.keys()))
        if self.cov_future_input_windows:
            self.all_vars.update(list(self.cov_future_input_windows.keys()))
        self.all_vars.update(self.target_variable)
        self.all_vars.update(self.categoricals)

        # set up lag/horizon names
        self.lag_names: List[str] = [
            "lag_" + str(i) for i in range(self.input_window - 1, -1, -1)
        ]
        self.hor_names: List[str] = ["hor_" + str(i + 1) for i in range(self.horizon)]
        # set up max_lags
        self.max_lags: int = (
            max(
                np.max(list(self.cov_history_input_windows.values())),
                self.input_window,
            )
            if self.cov_history_input_windows
            else self.input_window
        )

        self._validate_params()

    # [TODO]: more comprehensive parameter check.
    def _validate_params(self) -> None:
        """validate boosting_type of type string is of the following:
        'gbdt', traditional Gradient Boosting Decision Tree.
        'dart', Dropouts meet Multiple Additive Regression Trees.
        'goss', Gradient-based One-Side Sampling.
        'rf', Random Forest.
            validate target_variable exists
        Returns:
            None
        """
        if self.boosting_type not in ["gbdt", "dart", "goss", "rf"]:
            msg = f"boosting_type should be in 'gbdt', 'dart', 'goss', 'rf' but receives {self.boosting_type}."
            logging.error(msg)
            raise ValueError(msg)

        for k, v in self.cov_history_input_windows.items():
            if v < 0:
                msg = f"the value of the key '{k}' in cov_history_input_windows must be non-negative but receives {v})."
                logging.error(msg)
                raise ValueError(msg)

        for k, v in self.cov_future_input_windows.items():
            if v < 0 or v > self.horizon:
                msg = f"the value of the key '{k}' in cov_future_input_windows must be between 0 and {self.horizon}, but receives {v}."
                logging.error(msg)
                raise ValueError(msg)

        # [TODO] add more verififcation for calendar features
        if isinstance(self.calendar_features, str) and self.calendar_features != "auto":
            msg = f"`calendar_features` only accepts `auto` or a list of calendar features. Got {self.calendar_features}."
            raise ValueError(msg)
        # [TODO] add more verififcation for fourier features


class MLARModel:
    """The class for building LightGBM model for time series modeling.
    This class provides functions including train, predict, and evaluate and save_model.

    Attributes:
        params:class:`kats.models.lightgbm_ts.LightgbmParams` object for building the LightGBM model.

    Sample Usage:
        >>> mlarp = MLARParams()
        >>> # create an object of MLARModel with parameters
        >>> mlarmodel = MLARModel(mlarp)
        >>> # Train the LightgbmTS model.
        >>> mlarmodel.train(train_TSs)
        >>> # Generate forecasts.
        >>> mlarmodel = gbm_model.predict(steps=10)

    """

    def __init__(
        self,
        params: MLARParams,
    ) -> None:
        if not isinstance(params, MLARParams):
            msg = (
                f"params should be a LightgbmParams object but receives {type(params)}."
            )
            logging.error(msg)
            raise ValueError(msg)
        self.params = params
        self.debug = False
        # pyre-fixme
        self.model = None
        self.train_data: pd.DataFrame = pd.DataFrame([])
        self.train_data_in: np.ndarray = np.array([])
        self.forecast_data: pd.DataFrame = pd.DataFrame([])
        self.forecast_data_in: np.ndarray = np.array([])

        self.full_mat: np.ndarray = np.array([])
        self.curr_idx_full_mat = 0
        self.num_rows_full_mat = 0
        self.num_cols_full_mat = 0

        self.all_dates: List[str] = []
        self.feature_columns: List[str] = []
        self.keys: Dict[Union[Tuple[str, str], str], str] = {}

    def _infer_freq(self, data: pd.Series, freq: Optional[str]) -> str:

        if freq is None:
            freq = pd.infer_freq(data)
            if freq is None:
                msg = f"Fail to infer data frequency. Please specify `freq` in `MLARParams`. Got {freq}."
                logging.error(msg)
                raise ValueError(msg)
            elif freq in {"BM", "CBM", "MS", "BMS", "CBMS"}:
                freq = "M"
            elif freq in {"BQ", "QS", "BQS"}:
                freq = "Q"
            elif freq in {"A", "BA", "BY", "AS", "YS", "BAS", "BYS"}:
                freq = "Y"
        return freq

    def _check_single_ts(
        self, data: TimeSeriesData
    ) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
        """Validate input dataset columns"""
        if not isinstance(data, TimeSeriesData):
            msg = f"Every element in dataset should be a TimeSeriesData but received {type(data)}."
            logging.error(msg)
            raise ValueError(msg)

        cols_set = (
            set(data.value.columns)
            if isinstance(data.value, pd.DataFrame)
            else {data.value.name}
        )
        # check if all information is available.
        if not self.params.all_vars.issubset(cols_set):
            msg = f"Not all required columns are available! (required columns are {self.params.all_vars})."
            raise ValueError(msg)
        curr_series_data = data.to_dataframe().set_index(data.time_col_name)
        # fillna
        idx = pd.date_range(
            min(curr_series_data.index),
            max(curr_series_data.index),
            freq=self.params.freq,
        )
        curr_series_data = curr_series_data.reindex(idx, fill_value=np.nan)

        # calculate min and max timestamp
        min_ts_idx = curr_series_data.index.min()
        max_ts_idx = curr_series_data[self.params.target_variable].last_valid_index()

        # get valid length of timeseries
        length = find_length(curr_series_data, self.params.target_variable)
        return curr_series_data, min_ts_idx, max_ts_idx, length

    def _valid_and_fillna(
        self,
        data: Union[Dict[str, TimeSeriesData], List[TimeSeriesData]],
    ) -> Tuple[Dict[str, int], pd.Timestamp, pd.Timestamp, Dict[str, pd.DataFrame]]:
        """This is a function to validate dataset before training and prediction
            For both input as :class:`kats.consts.TimeSeriesData` or :class:`kats.consts.TimeSeriesData`
            categorical features is optional, and if `categoricals` is initiated, all `categoricals` must exist in TimeSeriesData's columns
        Args:
            data: a list of :class:`kats.consts.TimeSeriesData` or a dictionary of :class:`kats.consts.TimeSeriesData`
            where key is the unique category for the :class:`kats.consts.TimeSeriesData`.
        Returns:
            lengths: a dictionary that stores the length of the dataset
            data: a dictionary of :class:`kats.consts.TimeSeriesData`
            [min_ts, max_ts]: the minimum and maximum valid timestamp of all input time series.
        """
        if len(data) < 1:
            msg = "Input dataset should be non-empty."
            logging.error(msg)
            raise ValueError(msg)

        if isinstance(data, list):
            keys = np.array(range(len(data)))
        elif isinstance(data, dict):
            keys = list(data.keys())
        else:
            msg = f"dataset should be either a list or a dictionary, but received {type(data)}."
            logging.error(msg)
            raise ValueError(msg)

        # infer freq
        self.params.freq = self._infer_freq(data[keys[0]].time, self.params.freq)

        data_dict = {}
        lengths = {}
        min_ts = None
        max_ts = None

        for k in keys:
            curr_series_data, min_ts_idx, max_ts_idx, length = self._check_single_ts(
                data[k]
            )

            data_dict[k] = curr_series_data

            lengths[k] = length

            min_ts = min(min_ts, min_ts_idx) if min_ts is not None else min_ts_idx
            max_ts = max(max_ts, max_ts_idx) if max_ts is not None else max_ts_idx

        if np.max(list(lengths.values())) < self.params.input_window:
            msg = "Not enough data for direct modeling, please include more history or decrease input window length"
            logging.error(msg)
            raise ValueError(msg)

        # pyre-ignore
        return lengths, min_ts, max_ts, data_dict

    def _normalize_data(
        self,
        emb_ts: pd.DataFrame,
        feature_names: Optional[List[str]],
        norm_window_size: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if feature_names is None:
            in_data = emb_ts
        else:
            in_data = emb_ts[feature_names]

        normalizer2 = None

        if self.params.norm_sum_stat == "mean":
            normalizer = in_data.iloc[:, -norm_window_size:].mean(axis=1)
        elif self.params.norm_sum_stat == "median":
            normalizer = in_data.iloc[:, -norm_window_size:].median(axis=1)
        elif self.params.norm_sum_stat == "max":
            normalizer = in_data.iloc[:, -norm_window_size:].max(axis=1)
        elif self.params.norm_sum_stat == "std":
            normalizer = in_data.iloc[:, -norm_window_size:].std(axis=1)
        else:  # self.params.norm_sum_stat == "z-score"
            normalizer = in_data.iloc[:, -norm_window_size:].mean(axis=1)
            normalizer2 = in_data.iloc[:, -norm_window_size:].std(axis=1)
            self.params.sub_div = "sub_div"
        norm_emb_ts = normalize(
            emb_ts,
            normalizer,
            normalizer2,
            use_default_min=self.params.use_default_min,
            default_min=self.params.default_min,
            sub_div=self.params.sub_div,
        )

        normalizer = pd.DataFrame(normalizer)
        normalizer.columns = ["normalizer"]

        if normalizer2 is not None:
            normalizer["normalizer2"] = normalizer2

        return norm_emb_ts, normalizer

    def _embed_regressor(
        self, data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Embedding external regressors.

        TODO:
        Could also normalize across their sum
        normalizing with the sum should be easy to do:
        just embed first, then call norm func with feature names only the lags of the target TS.
        """
        cov_hist_data = []
        cov_hist_data_cols = []
        if self.params.cov_history_input_windows:

            for curr_cov in self.params.cov_history_input_windows.keys():
                curr_cov_lags = self.params.cov_history_input_windows[curr_cov]

                emb_ts_curr_cov = pd.DataFrame(
                    embed(
                        data[curr_cov].to_numpy(),
                        curr_cov_lags,
                        0,
                        self.params.max_lags,
                    )
                )

                emb_ts_curr_cov = cast(
                    pd.DataFrame, emb_ts_curr_cov.add_prefix(f"{curr_cov}_")
                )

                if self.params.cov_history_norm:
                    norm_emb_ts_curr_cov, normalizer_curr_cov = self._normalize_data(
                        emb_ts_curr_cov, None, curr_cov_lags
                    )

                    cov_hist_data.append(norm_emb_ts_curr_cov.values)

                    cov_hist_data_cols.extend(list(norm_emb_ts_curr_cov.columns))
                else:
                    cov_hist_data.append(emb_ts_curr_cov.values)
                    cov_hist_data_cols.extend(list(emb_ts_curr_cov.columns))
        return cov_hist_data, cov_hist_data_cols

    def _embed_categorical(
        self, data: pd.DataFrame, cat_labels: Dict[str, pd.CategoricalDtype]
    ) -> Tuple[np.ndarray, List[str]]:
        cat_encoded_data = pd.DataFrame()
        if self.params.categoricals:

            cat_data = pd.DataFrame()
            for curr_cat in self.params.categoricals:
                cat_data[curr_cat] = data[curr_cat].astype(cat_labels[curr_cat])

            if self.params.one_hot_encode:
                cat_encoded, cat_names = categorical_encode(
                    cat_data, self.params.categoricals
                )
            else:
                # convert to numeric
                cat_encoded = pd.DataFrame(cat_data.to_numpy())

            # As the categorical features are assumed to be constant,
            # it doesn't matter which ones we pick, but these ones should be the correct ones
            cat_encoded_data = cat_encoded.iloc[(self.params.max_lags - 1) :]
        return cat_encoded_data.values, list(cat_encoded_data.columns)

    def _embed_and_gen_past_features_single_series(
        self,
        series_data: pd.DataFrame,
        horizon: int,
        lags: int,
        target_vars: Union[List[str], str],
        cat_labels: Dict[str, pd.CategoricalDtype],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame], List[str]]:

        data = series_data.loc[: series_data[target_vars].last_valid_index()]

        lag_names = self.params.lag_names
        hor_names = self.params.hor_names
        max_lags = self.params.max_lags
        # Transform the data
        if self.params.transform_data == "log":
            data[target_vars] = np.log(data[target_vars])

        # Get time stamps
        emb_time_stamps = pd.DataFrame(data.index[max_lags - 1 :])
        emb_time_stamps.columns = ["origin_time"]

        # Embed external regressor variables
        cov_hist_data, cov_hist_data_cols = self._embed_regressor(data)
        # Embed categorical variables
        cat_encoded_data, cat_encoded_data_cols = self._embed_categorical(
            data, cat_labels
        )

        all_norm_in_data_dict = {}
        all_norm_out_data_dict = {}
        all_col_names = []

        for i, target_var in enumerate(target_vars):
            # Embed the target var
            emb_ts = pd.DataFrame(
                embed(data[target_var].values, lags, horizon, max_lags)
            )
            emb_ts.columns = lag_names + hor_names
            # Normalize the data
            norm_emb_ts, normalizer = self._normalize_data(
                emb_ts, lag_names, self.params.norm_window_size
            )

            # separate out the input and output data
            norm_in_data = norm_emb_ts[lag_names]
            norm_out_data = norm_emb_ts[hor_names]

            all_norm_in_data = []
            norm_window_stats = pd.DataFrame()

            all_norm_in_data.append(norm_in_data.values)

            # Calculate summary stats over the windows

            if self.params.use_sum_stats:

                norm_window_stats["min"] = norm_in_data.min(axis=1)
                norm_window_stats["max"] = norm_in_data.max(axis=1)
                norm_window_stats["mean"] = norm_in_data.mean(axis=1)
                norm_window_stats["median"] = norm_in_data.median(axis=1)
                norm_window_stats["std"] = norm_in_data.std(axis=1)

                all_norm_in_data.append(norm_window_stats.values)

            # Add cov_history, if present
            if self.params.cov_history_input_windows:
                all_norm_in_data.extend(cov_hist_data)

            # Add categoricals, if present
            if self.params.categoricals:
                all_norm_in_data.append(cat_encoded_data)

            # get one-hot encoded version of which target variable we currently have
            target_var_one_hot = np.zeros((len(emb_time_stamps), len(target_vars)))
            target_var_one_hot[:, i] = 1
            all_norm_in_data.append(target_var_one_hot)

            all_norm_in_data = np.column_stack(all_norm_in_data)
            all_norm_out_data = pd.concat(
                [norm_out_data, normalizer, emb_time_stamps], axis=1, copy=False
            )
            all_norm_out_data.set_index("origin_time", inplace=True)

            all_norm_in_data_dict[target_var] = all_norm_in_data
            all_norm_out_data_dict[target_var] = all_norm_out_data

        # All DataFrames have the same columns.
        all_col_names = (
            list(norm_in_data.columns)
            + list(norm_window_stats.columns)
            + cov_hist_data_cols
            + cat_encoded_data_cols
            + ["TV_" + t for t in target_vars]
        )

        return all_norm_in_data_dict, all_norm_out_data_dict, all_col_names

    def _generate_auto_calendar_feature(self, freq: Optional[str]) -> List[str]:
        """Generate default calender featureas based on data frequency."""
        calendar_features = []
        if freq == "Q":
            calendar_features = ["year"]
        elif freq == "M":
            calendar_features = ["year", "quarter"]
        elif freq == "W":
            calendar_features = ["year", "quarter", "month", "weekofyear"]
        elif freq == "D":
            calendar_features = ["year", "quarter", "month", "day", "weekday"]
        elif freq == "H":
            calendar_features = ["year", "quarter", "month", "weekday", "hour"]
        elif freq == "min":
            calendar_features = [
                "year",
                "quarter",
                "month",
                "weekday",
                "hour",
                "minuteofday",
            ]
        elif freq == "S":
            calendar_features = [
                "year",
                "quarter",
                "month",
                "weekday",
                "hour",
                "minuteofday",
            ]
        return calendar_features

    def _generate_auto_fourier_features(
        self, freq: Optional[str]
    ) -> Tuple[List[int], List[int], int]:
        """Generate default fourier featureas based on data frequency."""
        if freq == "W":
            fperiod = [365]
            forder = [7]
            foffset = 60 * 60 * 24  # offset for day
        elif freq == "D":
            fperiod = [7, 365]
            forder = [3, 7]
            foffset = 60 * 60 * 24  # offset for day
        elif freq == "H":
            fperiod = [24, 24 * 7, 24 * 365]
            forder = [3, 5, 7]
            foffset = 60 * 60  # offset for hour
        elif freq == "min":
            fperiod = [60]
            forder = [7]
            foffset = 60  # offset for minute
        elif freq == "S":
            fperiod = [60]
            forder = [7]
            foffset = 1  # offset for second
        else:
            fperiod, forder, foffset = [], [], 1
        return fperiod, forder, foffset

    def _gen_cal_feat(
        self,
        min_ts: pd.Timestamp,
        max_ts: pd.Timestamp,
        horizon: int,
        calendar_features: Union[str, List[str]],
    ) -> pd.DataFrame:

        offset = pd.tseries.frequencies.to_offset(self.params.freq)

        timestamps = pd.Series(
            pd.date_range(
                min_ts,
                # pyre-ignore
                max_ts + offset * horizon,
                freq=self.params.freq,
            )
        )

        # Compute calendar features
        calendar_features = (
            self.params.calendar_features
            if self.params.calendar_features != "auto"
            else self._generate_auto_calendar_feature(self.params.freq)
        )
        if calendar_features != []:
            calendar_features_df = TsCalenderFeatures(
                cast(List[str], calendar_features)
            ).get_features(timestamps)
            calendar_features_df = cast(pd.DataFrame, calendar_features_df)
        else:
            calendar_features_df = pd.DataFrame()

        # compute Fourier features
        if self.params.fourier_features_order == "auto":
            fperiod, forder, foffset = self._generate_auto_fourier_features(
                self.params.freq
            )
        else:
            fperiod, forder, foffset = (
                self.params.fourier_features_period,
                self.params.fourier_features_order,
                self.params.fourier_features_offset,
            )
        if len(fperiod) > 0:

            fourier_features_df = TsFourierFeatures(
                # pyre-fixme
                fperiod,
                # pyre-fixme
                forder,
                # pyre-fixme
                foffset,
            ).get_features(timestamps)
            fourier_features_df = cast(pd.DataFrame, fourier_features_df)
        else:
            fourier_features_df = pd.DataFrame()

        features = pd.concat([calendar_features_df, fourier_features_df], axis=1)
        if len(features) > 0:
            features.set_index(timestamps, inplace=True)

        return features

    def _embed_future_cov(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        emb_ts_curr_cov_dict = {}

        for curr_cov in self.params.cov_future_input_windows.keys():
            curr_cov_lags = self.params.cov_future_input_windows[curr_cov]

            # TODO: Currently we go the same amount of lags into the past
            # and into the future for the future regressors
            emb_ts_curr_cov = pd.DataFrame(
                embed(
                    data[curr_cov].to_numpy(),
                    curr_cov_lags,
                    curr_cov_lags,
                    curr_cov_lags,
                )
            )

            emb_ts_curr_cov = cast(
                pd.DataFrame, emb_ts_curr_cov.add_prefix(f"Fut_Cov_{curr_cov}_")
            )

            emb_ts_curr_cov.set_index(data.index[curr_cov_lags - 1 :], inplace=True)

            if self.params.cov_future_norm:
                norm_emb_ts_curr_cov, normalizer_curr_cov = self._normalize_data(
                    emb_ts_curr_cov, None, curr_cov_lags
                )

                emb_ts_curr_cov_dict[curr_cov] = norm_emb_ts_curr_cov
            else:
                emb_ts_curr_cov_dict[curr_cov] = emb_ts_curr_cov

        cov_future_data = list(emb_ts_curr_cov_dict.values())

        all_cov_future_data = pd.DataFrame()

        if len(cov_future_data) > 0:
            all_cov_future_data = cov_future_data[0]

        if len(cov_future_data) > 1:
            all_cov_future_data = all_cov_future_data.join(
                cov_future_data[1:], how="outer"
            )

        return all_cov_future_data

    def _merge_past_and_future_reg(
        self,
        norm_in_data: Dict[str, np.ndarray],
        norm_out_data: Dict[str, pd.DataFrame],
        horizon: int,
        cal_feat: pd.DataFrame,
        emb_fut_cov: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:

        first_norm_in_data = norm_in_data[list(norm_in_data.keys())[0]]

        if self.num_cols_full_mat == 0:
            self.num_cols_full_mat = (
                first_norm_in_data.shape[1]
                + cal_feat.shape[1]
                + emb_fut_cov.shape[1]
                + 1
            )
            self.full_mat = np.full(
                shape=(self.num_rows_full_mat, self.num_cols_full_mat),
                fill_value=np.nan,
            )

        all_out_data_list = []

        for target_var in norm_out_data.keys():

            for curr_horizon in range(1, self.params.horizon + 1):

                # find offset between our current dataset and the precomputed Fourier terms, then select the right cal features
                offset_cf = np.where(
                    cal_feat.index == norm_out_data[target_var].index.min()
                )[0][0]
                curr_cal_feat = cal_feat.iloc[
                    (offset_cf + curr_horizon) : (
                        offset_cf + curr_horizon + norm_out_data[target_var].shape[0]
                    ),
                    :,
                ]

                if not emb_fut_cov.empty:
                    offset_fut_cov = np.where(
                        emb_fut_cov.index == norm_out_data[target_var].index.min()
                    )[0][0]
                    curr_fut_cov = emb_fut_cov.iloc[
                        (offset_fut_cov + curr_horizon) : (
                            offset_fut_cov
                            + curr_horizon
                            + norm_out_data[target_var].shape[0]
                        ),
                        :,
                    ]
                else:
                    curr_fut_cov = pd.DataFrame()

                curr_out_data = pd.DataFrame()
                curr_out_data["origin_time"] = norm_out_data[target_var].index
                curr_out_data["target_time"] = curr_cal_feat.index
                curr_out_data["output"] = norm_out_data[target_var][
                    f"hor_{curr_horizon}"
                ].to_numpy()
                curr_out_data["normalizer"] = norm_out_data[target_var][
                    "normalizer"
                ].to_numpy()
                curr_out_data["variable"] = target_var

                if "normalizer2" in norm_out_data[target_var]:
                    curr_out_data["normalizer2"] = norm_out_data[target_var][
                        "normalizer2"
                    ].to_numpy()

                hor = np.repeat(curr_horizon, norm_in_data[target_var].shape[0])

                tv_idx = self.curr_idx_full_mat
                num_rows_dat = norm_in_data[target_var].shape[0]
                num_cols_dat = norm_in_data[target_var].shape[1]
                num_cols_cal_feat = curr_cal_feat.shape[1]
                num_cols_fut_cov = curr_fut_cov.shape[1]

                self.full_mat[
                    tv_idx : (tv_idx + num_rows_dat), 0:num_cols_dat
                ] = norm_in_data[target_var]

                if num_cols_cal_feat != 0:
                    self.full_mat[
                        tv_idx : (tv_idx + num_rows_dat),
                        num_cols_dat : (num_cols_dat + num_cols_cal_feat),
                    ] = curr_cal_feat.to_numpy()

                if num_cols_fut_cov != 0:
                    self.full_mat[
                        tv_idx : (tv_idx + num_rows_dat),
                        (num_cols_dat + num_cols_cal_feat) : (
                            num_cols_dat + num_cols_cal_feat + num_cols_fut_cov
                        ),
                    ] = curr_fut_cov.to_numpy()

                self.full_mat[
                    tv_idx : (tv_idx + num_rows_dat),
                    num_cols_dat + num_cols_cal_feat + num_cols_fut_cov,
                ] = hor

                all_out_data_list.append(curr_out_data)

                self.curr_idx_full_mat = (
                    self.curr_idx_full_mat + norm_in_data[target_var].shape[0]
                )

        all_out_data = pd.concat(all_out_data_list, copy=False)

        col_names = list(curr_cal_feat.columns)
        col_names.extend(curr_fut_cov.columns)
        col_names.append("horizon")

        return all_out_data, col_names

    def _get_all_cat_labels(
        self, all_series: Dict[str, pd.DataFrame], categoricals: List[str]
    ) -> Dict[str, pd.CategoricalDtype]:
        """
        Generate categorical 1-hot-encoding necessary information.
        """

        all_cat_list = []
        for curr_series_name in all_series.keys():
            # TODO: Assuming that categoricals are constant per series
            all_cat_list.append(all_series[curr_series_name][categoricals].head(1))

        all_cat = pd.concat(all_cat_list)

        cat_labels = {}
        for cat in all_cat.columns:
            cat_labels[cat] = pd.CategoricalDtype(
                categories=all_cat[cat].unique(), ordered=False
            )

        return cat_labels

    def _embed_and_gen_features(
        self,
        data: Union[Dict[str, TimeSeriesData], List[TimeSeriesData]],
        horizon: int,
        lags: int,
        fillna: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:

        lengths, min_ts, max_ts, all_series = self._valid_and_fillna(data)

        # amount of rows the large matrix will have
        lens = [
            (t - lags + 1) * horizon * len(self.params.target_variable)
            for t in lengths.values()
        ]
        self.num_rows_full_mat = np.array(lens).sum()

        all_meta_data_list = []

        all_res_data_np = np.array([])
        curr_idx = 0

        cat_labels = {}
        all_col_names = []
        if self.params.categoricals:
            cat_labels = self._get_all_cat_labels(all_series, self.params.categoricals)

        cal_feat = self._gen_cal_feat(
            min_ts, max_ts, horizon, self.params.calendar_features
        )

        for curr_series_name in all_series:

            logging.info(f"Current time series to be preprocessed: {curr_series_name}")

            (
                norm_in_data,
                norm_out_data,
                in_data_col_names,
            ) = self._embed_and_gen_past_features_single_series(
                all_series[curr_series_name],
                horizon=horizon,
                lags=lags,
                target_vars=self.params.target_variable,
                cat_labels=cat_labels,
            )

            emb_fut_cov = self._embed_future_cov(all_series[curr_series_name])

            logging.info("_embed_and_gen_past_features_single_series finished")

            meta_data, add_col_names = self._merge_past_and_future_reg(
                norm_in_data,
                norm_out_data,
                horizon,
                cal_feat,
                emb_fut_cov,
            )

            # add the series name dummy
            meta_data["series_name"] = curr_series_name

            all_meta_data_list.append(meta_data)

            logging.info(f"curr_idx: {curr_idx}")
            if not all_col_names:
                all_col_names.extend(in_data_col_names)
                all_col_names.extend(add_col_names)

        logging.info(f"all_res_data_np_shape: {all_res_data_np.shape}")

        all_meta_data = pd.concat(all_meta_data_list, copy=False)

        logging.info("Finished _embed_and_gen_features")

        return all_meta_data, all_col_names

    def _post_process(self) -> Dict[str, pd.DataFrame]:

        fc_groups = self.forecast_data.groupby("series_name")

        # we rename target_time and series_name to time and dummy, for compatibility with how the old version works
        fc_postproc = fc_groups.apply(
            lambda grp: grp[grp["origin_time"] == grp["origin_time"].max()][
                ["target_time", "series_name", "variable", "forecast"]
            ].rename(columns={"target_time": "time", "series_name": "dummy"})
        )
        fc_postproc.reset_index(drop=True, inplace=True)

        res = {dummy: group for dummy, group in fc_postproc.groupby("dummy")}
        # res = {series_name: group for series_name, group in fc_postproc.groupby("series_name")}
        return res

    def _train(
        self,
        in_data: np.ndarray,
        meta_and_out_data: pd.DataFrame,
    ) -> None:
        """Train session with (x, y) matrix input and output sMAPE in logging
        Args:
            dataset: pd.DataFrame for training lightgbm returned by function _ts_to_xy

        Returns:
            None
        """

        logging.info("feature columns:" + str(self.feature_columns))
        # logging.info("forecast:")
        forecast_index = meta_and_out_data["output"].apply(np.isnan)
        train_index = ~forecast_index

        self.train_data_in = in_data[train_index, :]
        self.forecast_data_in = in_data[forecast_index, :]

        self.train_data = meta_and_out_data.loc[train_index]
        self.forecast_data = meta_and_out_data.loc[forecast_index]

        normalized_train_y = meta_and_out_data["output"].loc[train_index]

        # optimized parameter from backtesting (model performance not very sensitive to these parameters)
        lgb_params = {
            "n_jobs": self.params.n_jobs,
            "max_depth": self.params.max_depth,
            "min_data_in_leaf": self.params.min_data_in_leaf,
            "num_leaves": self.params.num_leaves,
            "subsample": self.params.subsample,
            "n_estimators": self.params.n_estimators,
            "learning_rate": self.params.learning_rate,
            "colsample_bytree": self.params.colsample_bytree,
            "boosting_type": self.params.boosting_type,
            "alpha": self.params.alpha,
            "random_state": self.params.random_state,
            "verbose": self.params.verbose,
        }
        # pyre-fixme
        regr = gbm.LGBMRegressor(
            objective=self.params.objective,
            **lgb_params,
        )

        train_rows = self.train_data_in.shape[0]
        train_cols = self.train_data_in.shape[1]
        logging.info(f"Training dataset size: {train_rows}x{train_cols}")

        # fit lightgbm
        regr.fit(
            self.train_data_in, normalized_train_y, feature_name=self.feature_columns
        )
        # predict lightgbm

        self.train_data.loc[:, "forecast"] = regr.predict(self.train_data_in)

        if "normalizer2" in self.train_data.columns:
            normalizer2 = self.train_data["normalizer2"]
        else:
            normalizer2 = None

        self.train_data.loc[:, "forecast"] = denormalize(
            self.train_data[["forecast"]],
            self.train_data["normalizer"],
            normalizer2,
            use_default_min=self.params.use_default_min,
            default_min=self.params.default_min,
            sub_div=self.params.sub_div,
        )

        if self.params.transform_data == "log":
            self.train_data.loc[:, "forecast"] = np.exp(
                self.train_data.loc[:, "forecast"]
            )

        logging.info(
            "train sMAPE: "
            + str(
                smape(
                    self.train_data["output"].values, self.train_data["forecast"].values
                )
            )
        )
        self.model = regr
        return

    def _predict(
        self,
    ) -> None:
        """forecast session
        Args: None
        Returns: None
        """

        self.forecast_data.loc[:, "forecast"] = self.model.predict(
            self.forecast_data_in
        )

        if "normalizer2" in self.forecast_data.columns:
            normalizer2 = self.forecast_data["normalizer2"]
        else:
            normalizer2 = None

        self.forecast_data.loc[:, "forecast"] = denormalize(
            self.forecast_data[["forecast"]],
            self.forecast_data["normalizer"],
            normalizer2,
            use_default_min=self.params.use_default_min,
            default_min=self.params.default_min,
            sub_div=self.params.sub_div,
        )

        if self.params.transform_data == "log":
            self.forecast_data.loc[:, "forecast"] = np.exp(
                self.forecast_data.loc[:, "forecast"]
            )

        # TODO CB: Need to check what this is doing!
        # self.forecast_data.dropna(
        #     subset=[c for c in self.forecast_data.columns if "ts" in c],
        #     how="all",
        #     inplace=True,
        # )

        logging.info(
            "sMAPE between forecast and median of most renct three weeks actual: "
            + str(
                smape(
                    self.forecast_data["normalizer"].values,
                    self.forecast_data["forecast"].values,
                )
            )
        )
        return None

    def train(
        self,
        data: Union[Dict[str, TimeSeriesData], List[TimeSeriesData]],
    ) -> None:
        """Full workflow of train and forecast
        Args:
        data: a list of :class:`kats.consts.TimeSeriesData` or a dictionary of :class:`kats.consts.TimeSeriesData`
            where key is the unique category for the :class:`kats.consts.TimeSeriesData`.
        Returns: None
        """

        self.curr_idx_full_mat = 0
        self.num_rows_full_mat = 0
        self.num_cols_full_mat = 0

        meta_data, all_col_names = self._embed_and_gen_features(
            data,
            horizon=self.params.horizon,
            lags=self.params.input_window,
            fillna=np.nan,
        )

        self.feature_columns = all_col_names

        self._train(self.full_mat, meta_data)
        return None

    def predict(
        self,
        steps: int,
    ) -> Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
        """Full workflow of train and forecast with post process
        Args:
            steps: int for overwriting horizon, which should be no more than horizon.
        Returns: a dictionary of pd.DataFrame with forecasting where key is the unique category or
            a list of of pd.DataFrame forecasting without categorical information
        """

        self._predict()

        forecast_result = self._post_process()
        return forecast_result

    def __str__(self) -> str:
        return "ML_AR_Model"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default LIGHTGBM parameter search space.

        Args:
            None

        Returns:
            A dictionary with the default LIGHTGBM parameter search space
        """

        return get_default_lightgbm_parameter_search_space()
