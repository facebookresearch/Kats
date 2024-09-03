# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

import lightgbm as gbm  # @manual
import numpy as np
import numpy.typing as npt
import pandas as pd
from kats.consts import TimeSeriesData
from kats.metrics.metrics import smape
from kats.tsfeatures.tsfeatures import TsCalenderFeatures, TsFourierFeatures
from kats.utils.parameter_tuning_utils import (
    get_default_lightgbm_parameter_search_space,
)

from numpy.random import RandomState

try:
    from numba import jit  # @manual
except ImportError:
    logging.warning("numba is not installed. jit compilation of tsfeatures is disabled")

    # pyre-fixme
    def jit(func):  # type ignore
        def tmp(*args, **kwargs):  # type: ignore
            return func(*args, **kwargs)

        return tmp


# @njit
def find_first_missing_number(nums: npt.NDArray) -> int:
    missing_numbers = np.sort(np.setdiff1d(np.arange(1, len(nums) + 2), nums))

    if len(missing_numbers) == 0:
        missing = np.nan
    else:
        missing = missing_numbers[0]
    return missing


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
        raise ValueError(f"`sub_div` method {sub_div} is invalid.")

    return norm_data


def denormalize(
    x: Union[pd.DataFrame, pd.Series],
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


@jit
def embed(series: npt.NDArray, lags: int, horizon: int, max_lags: int) -> npt.NDArray:

    result = np.full(
        shape=(series.size - max_lags + 1, lags + horizon), fill_value=np.nan
    )
    for i in range((max_lags - lags), (series.size - lags + 1)):
        j = i - (max_lags - lags)
        curr_wind = series[i : i + lags + horizon]
        result[j, 0 : curr_wind.size] = curr_wind

    return result


class MLARParams:
    """Parameter class for time series MLAR model

    This is the parameter class for time series LightGBM model. Details of lightgbm inherented params
    can be found https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

    Args:
        target_variable: either a string or a list of strings for target variables columns.
        horizon: a list of horizons to train with. Alternatively, a positive integer i that gets expanded to the range from 1 to i. The default is 10.
        input_window: a positive integer for input windows, i.e., the number of lags. Default is 10.
        freq: a string representing the frequency of time series. Default is "D". If missing, the algorithm will try to infer the frequency.
        cov_history_input_windows: a dictionary representing historical covariate information, whose keys are strings for covariate names and values are positive integers for number of lags. Default is None.
        cov_future_input_windows: a dictionary representing future covariate information, whose keys are strings for covariate names and values are positive integers for number of leaps. Default is None.
        categoricals: a list of strings for the column names of categorical columns.
        one_hot_encode: a boolean for whether to use one-hot encoding for categorical covariates or not. Default is True.
        calendar_features: a list of strings for the calendar features or "auto" for default calendar features. Default is None, which is for no calendar features (as by default Fourier terms are used). Possible values are "year", "quarter", "month", "weekday", "hour", "minuteofday"
        fourier_features_period: Can be None for no Fourier terms, or "auto" to automatically determine the features. Otherwise, a list of numbers for the periods. For example, for hourly series we can use [24, 24 * 7, 24 * 365] to model a daily, weekly, and yearly seasonality.
        fourier_features_order: Can be "auto" to automatically determine features. Otherwise a list of ints to determine the order of Fourier terms (parameter k in many notations). For example, for hourly series we can use [3, 5, 7] for daily, weekly, yearly orders.
        fourier_features_offset: Can be "auto", otherwise an int. Determines by which factor all Fourier series need to be divided, relative to seconds. For example, for hourly series set to 3600, for daily series to 86400.
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
        alpha: float for alpha in LGBMRegressor default = 0.9, type = double, constraints: alpha > 0.0, used only in huber and quantile regression applications.
        verbose: verbosity level of LGBMRegressor, default is -1. Level 3 is full verbosity.
        random_state: int or RandomState or None type random_state argument in LGBMRegressor function. It is used to seed the random number generator during the training process.
        norm_window_size: integer for the window size of the normalization window (can be smaller than the input window), default is same size as input window
        norm_sum_stat: string to determine how to calculate the window normalizer. Default is "mean", other options are: "median", "max", "std", "z-score"
        sub_div: string to determine how to apply the normalizer. "div" for division, "sub" for subtraction. "sub_div" to subtract normalizer1 and then divide by normalizer2 (currently only used for z-score)
        use_default_min: bool to determine if default_min is added to series and normalizer before normalization
        default_min: Default is 1e-8. It is a small value that is added to both serise and normalizer, e.g., to avoid division by zero
        transform_data: string to determine if target_vars should be transformed before processing (i.e., before normalization and training). After prediction they are back-transformed. Currently, only transform implemented is "log". Default is None.
        cov_history_norm: bool to determine if historic covariates should be normalized. If true, the same per-window normalization is applied as for the the target vars. Default is False.
        cov_future_norm: bool to determine if future covariates should be normalized. If true, the same per-window normalization is applied as for the the target vars. Default is False.
        use_sum_stats: bool to determine if min, max, mean, median, std should be calculated over the input window, and added as a feature. Default is True.
        calculate_fit: bool to determine if the predict function should be called finally over the full training set to calculate the fit on the training set. For large datasets, can be slow, and will mean that the training dataset is fully stored in the model object. Default is False.
    """

    def __init__(
        self,
        target_variable: Union[List[str], str] = "y",
        horizon: Union[List[int], int] = 10,
        input_window: int = 10,
        freq: Optional[str] = None,
        cov_history_input_windows: Optional[Dict[str, int]] = None,
        cov_future_input_windows: Optional[Dict[str, int]] = None,
        categoricals: Optional[List[str]] = None,
        one_hot_encode: bool = True,
        calendar_features: Union[None, str, List[str]] = None,
        fourier_features_period: Union[None, str, List[Union[float, int]]] = "auto",
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
        calculate_fit: bool = False,
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

        if isinstance(horizon, int) and horizon > 0:
            self.horizon: List[int] = list(range(1, horizon + 1))
            self.max_horizon: int = horizon
        elif (
            isinstance(horizon, list)
            and all([isinstance(t, int) for t in horizon])
            and all([t > 0 for t in horizon])
        ):
            self.horizon: List[int] = horizon
            self.max_horizon: int = max(horizon)
        else:
            msg = f"`horizon` is invalid. Got {horizon}."
            raise ValueError(msg)

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

        self.calculate_fit = calculate_fit

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
        self.hor_names: List[str] = [
            "hor_" + str(i) for i in range(1, self.max_horizon + 1)
        ]
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
            if v < 0 or v > self.max_horizon:
                msg = f"the value of the key '{k}' in cov_future_input_windows must be between 0 and {self.max_horizon}, but receives {v}."
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
        self.train_data_in: npt.NDArray = np.array([])
        self.forecast_data: pd.DataFrame = pd.DataFrame([])
        self.forecast_data_in: npt.NDArray = np.array([])

        self.full_mat: npt.NDArray = np.array([])

        self.num_hist_reg = 0

        self.all_series: Dict[str, pd.DataFrame] = {}

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

    def _check_single_ts(self, data: TimeSeriesData) -> pd.DataFrame:
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

        return curr_series_data

    def _valid_and_fillna(
        self,
        data: Union[Dict[str, TimeSeriesData], List[TimeSeriesData]],
        # pyre-fixme[11]: Annotation `Timestamp` is not defined as a type.
    ) -> Tuple[Dict[str, pd.DataFrame], Set[pd.Timestamp]]:
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
        timestamps = set()
        offset = pd.tseries.frequencies.to_offset(self.params.freq)

        for k in keys:

            curr_series_data = self._check_single_ts(data[k])

            data_dict[k] = curr_series_data
            timestamps.update(curr_series_data.index)
            # add forecast timestamps
            timestamps.update(
                # pyre-fixme
                [curr_series_data.index[-1] + offset * i for i in self.params.horizon]
            )
        return data_dict, timestamps

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

                # pyre-fixme[22]: The cast is redundant.
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
        self.num_hist_reg = len(all_col_names)
        return all_norm_in_data_dict, all_norm_out_data_dict, all_col_names

    def _generate_auto_calendar_feature(self, freq: Optional[str]) -> List[str]:
        """Generate default calendar featureas based on data frequency."""
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
        timestamps: Set[pd.Timestamp],
        calendar_features: Union[None, str, List[str]],
    ) -> pd.DataFrame:

        ts = pd.Series(list(timestamps))
        # Compute calendar features
        if calendar_features is None:
            calendar_features = []
        elif self.params.calendar_features == "auto":
            calendar_features = self._generate_auto_calendar_feature(self.params.freq)
        else:
            calendar_features = cast(List[str], self.params.calendar_features)

        if calendar_features != []:
            calendar_features_df = TsCalenderFeatures(calendar_features).get_features(
                ts
            )
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
        if fperiod is not None and len(fperiod) > 0:

            fourier_features_df = TsFourierFeatures(
                # pyre-fixme
                fperiod,
                # pyre-fixme
                forder,
                # pyre-fixme
                foffset,
            ).get_features(ts)
            fourier_features_df = cast(pd.DataFrame, fourier_features_df)
        else:
            fourier_features_df = pd.DataFrame()

        features = pd.concat([calendar_features_df, fourier_features_df], axis=1)
        if len(features) > 0:
            features.set_index(ts, inplace=True)

        return features

    def _embed_future_cov(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.params.cov_future_input_windows:
            return pd.DataFrame()
        emb_ts_curr_cov_list = []

        for curr_cov in self.params.cov_future_input_windows:
            curr_cov_lags = self.params.cov_future_input_windows[curr_cov]
            emb_ts_curr_cov = pd.DataFrame(
                embed(
                    data[curr_cov].to_numpy(),
                    curr_cov_lags,
                    curr_cov_lags,
                    curr_cov_lags,
                )
            )

            # pyre-fixme[22]: The cast is redundant.
            emb_ts_curr_cov = cast(
                pd.DataFrame, emb_ts_curr_cov.add_prefix(f"Fut_Cov_{curr_cov}_")
            )

            emb_ts_curr_cov.set_index(data.index[curr_cov_lags - 1 :], inplace=True)

            if self.params.cov_future_norm:
                norm_emb_ts_curr_cov, normalizer_curr_cov = self._normalize_data(
                    emb_ts_curr_cov, None, curr_cov_lags
                )

                emb_ts_curr_cov_list.append(norm_emb_ts_curr_cov)
            else:
                emb_ts_curr_cov_list.append(emb_ts_curr_cov)

        all_cov_future_data = pd.concat(emb_ts_curr_cov_list, axis=1, join="outer")
        return all_cov_future_data

    def _merge_past_and_future_reg(
        self,
        norm_in_data: Dict[str, np.ndarray],
        norm_out_data: Dict[str, pd.DataFrame],
        horizons: List[int],
        cal_feat: pd.DataFrame,
        emb_fut_cov: pd.DataFrame,
        gen_meta_data: bool = True,
    ) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:

        offset = pd.tseries.frequencies.to_offset(self.params.freq)
        num_cols = self.num_hist_reg + cal_feat.shape[1] + emb_fut_cov.shape[1] + 1
        tv_idx = 0
        num_rows_full_mat = int(
            np.sum([len(norm_out_data[t]) for t in norm_out_data]) * len(horizons)
        )

        full_mat = np.full(
            shape=(num_rows_full_mat, num_cols),
            fill_value=np.nan,
        )

        all_out_data_list = []

        for target_var in norm_out_data.keys():

            for curr_horizon in horizons:

                target_timestamps = pd.DataFrame(
                    index=norm_out_data[target_var].index
                    # pyre-fixme
                    + offset * curr_horizon
                )

                curr_feat = target_timestamps[[]].merge(
                    cal_feat, left_index=True, right_index=True, how="left"
                )

                curr_feat = curr_feat.merge(
                    emb_fut_cov, left_index=True, right_index=True, how="left"
                )

                curr_out_data = pd.DataFrame()
                curr_out_data["origin_time"] = norm_out_data[target_var].index

                curr_out_data["output"] = norm_out_data[target_var][
                    f"hor_{curr_horizon}"
                ].values

                if gen_meta_data:
                    curr_out_data["target_time"] = curr_feat.index
                    curr_out_data["normalizer"] = norm_out_data[target_var][
                        "normalizer"
                    ].to_numpy()
                    curr_out_data["variable"] = target_var

                    if "normalizer2" in norm_out_data[target_var]:
                        curr_out_data["normalizer2"] = norm_out_data[target_var][
                            "normalizer2"
                        ].to_numpy()

                num_rows_dat = norm_in_data[target_var].shape[0]
                num_cols_dat = norm_in_data[target_var].shape[1]

                full_mat[tv_idx : (tv_idx + num_rows_dat), 0:num_cols_dat] = (
                    norm_in_data[target_var]
                )

                full_mat[
                    tv_idx : (tv_idx + num_rows_dat),
                    num_cols_dat : num_cols_dat + curr_feat.shape[1],
                ] = curr_feat.values

                full_mat[
                    tv_idx : (tv_idx + num_rows_dat), num_cols_dat + curr_feat.shape[1]
                ] = curr_horizon

                all_out_data_list.append(curr_out_data)

                tv_idx = tv_idx + norm_in_data[target_var].shape[0]

        all_out_data = pd.concat(all_out_data_list, copy=False)

        col_names = list(curr_feat.columns) + ["horizon"]

        return full_mat, all_out_data, col_names

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
        timestamps: Set[pd.Timestamp],
        curr_all_series: Dict[str, pd.DataFrame],
        horizons: List[int],
        lags: int,
        fillna: Optional[float] = None,
        gen_meta_data: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:

        all_in_data_list = []
        all_meta_data_list = []

        all_res_data_np = np.array([])

        cat_labels = {}
        all_col_names = []
        if self.params.categoricals:
            cat_labels = self._get_all_cat_labels(
                curr_all_series, self.params.categoricals
            )

        cal_feat = self._gen_cal_feat(
            timestamps,
            self.params.calendar_features,
        )

        for curr_series_name in curr_all_series:

            # logging.info(f"Current time series to be preprocessed: {curr_series_name}")

            (
                norm_in_data,
                norm_out_data,
                in_data_col_names,
            ) = self._embed_and_gen_past_features_single_series(
                curr_all_series[curr_series_name],
                horizon=self.params.max_horizon,
                lags=lags,
                target_vars=self.params.target_variable,
                cat_labels=cat_labels,
            )

            emb_fut_cov = self._embed_future_cov(curr_all_series[curr_series_name])

            # logging.info("_embed_and_gen_past_features_single_series finished")

            in_data, meta_data, add_col_names = self._merge_past_and_future_reg(
                norm_in_data,
                norm_out_data,
                horizons,
                cal_feat,
                emb_fut_cov,
                gen_meta_data,
            )

            # add the series name dummy
            meta_data["series_name"] = curr_series_name
            all_in_data_list.append(in_data)
            all_meta_data_list.append(meta_data)

            if not all_col_names:
                all_col_names.extend(in_data_col_names)
                all_col_names.extend(add_col_names)

        logging.info(f"all_res_data_np_shape: {all_res_data_np.shape}")

        all_meta_data = pd.concat(all_meta_data_list, copy=False)
        self.full_mat = np.row_stack(all_in_data_list)

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
        in_data: npt.NDArray,
        meta_and_out_data: pd.DataFrame,
    ) -> None:
        """Train session with (x, y) matrix input and output sMAPE in logging
        Args:
            dataset: pd.DataFrame for training lightgbm returned by function _ts_to_xy

        Returns:
            None
        """

        logging.info("feature columns:" + str(self.feature_columns))
        forecast_index = meta_and_out_data["output"].apply(np.isnan)
        train_index = ~forecast_index

        self.train_data_in = in_data[train_index, :]
        self.train_data = meta_and_out_data.loc[train_index]
        normalized_train_y = meta_and_out_data["output"].loc[train_index]
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

        if self.params.calculate_fit:
            self.train_data.loc[:, "forecast"] = regr.predict(self.train_data_in)

            if "normalizer2" in self.train_data.columns:
                normalizer2 = self.train_data["normalizer2"]
            else:
                normalizer2 = None

            self.train_data.loc[:, "forecast"] = denormalize(
                self.train_data["forecast"],
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
                        denormalize(
                            self.train_data["output"],
                            self.train_data["normalizer"],
                            normalizer2,
                            use_default_min=self.params.use_default_min,
                            default_min=self.params.default_min,
                            sub_div=self.params.sub_div,
                        ).values,
                        self.train_data["forecast"].values,
                    )
                )
            )

        self.model = regr

        return

    def _predict(
        self,
        fill_missing: bool = False,
        new_data: Optional[Dict[Union[str, int], pd.DataFrame]] = None,
        new_data_is_forecast: bool = True,
    ) -> None:

        new_data_dict = {}
        offset = pd.tseries.frequencies.to_offset(self.params.freq)

        for curr_series in self.all_series.keys():
            in_data = self.all_series[curr_series]

            if new_data is None:
                fc_origin = in_data[self.params.target_variable].last_valid_index()
                max_ts = max(fc_origin, max(in_data.index))
            else:
                if new_data_is_forecast:
                    fc_wide = new_data[curr_series].pivot(
                        index="time", columns="variable", values="forecast"
                    )
                    fc_origin = np.max(fc_wide.index)

                    # find the first gap in the horizon, and set fc_origin to the value before, so that we can fill the gap with forecasts
                    if fill_missing:

                        min_ts = np.min(fc_wide.index)
                        all_missing = pd.date_range(
                            min_ts,
                            fc_origin,
                            freq=self.params.freq,
                        ).difference(fc_wide.index)

                        if len(all_missing) != 0:
                            fc_origin = all_missing[0] - offset
                    max_ts = max(fc_origin, max(in_data.index))
                else:
                    curr_new_data = new_data[curr_series].to_dataframe()
                    time_col = new_data[curr_series].time_col_name
                    last_index = curr_new_data[
                        self.params.target_variable
                    ].last_valid_index()
                    if last_index is None:
                        fc_origin = min(curr_new_data[time_col]) - offset
                    else:
                        fc_origin = curr_new_data.loc[last_index, time_col]
                    max_ts = max(fc_origin, max(curr_new_data[time_col]))

            # get needed time stamps, the full input window
            # and any future values that may be there in the data
            timestamps = pd.Series(
                pd.date_range(
                    # pyre-ignore
                    fc_origin - offset * (self.params.max_lags - 1),
                    max_ts,
                    freq=self.params.freq,
                )
            )

            # get actual data that is available
            in_window = pd.DataFrame(index=timestamps)
            in_window = in_window.join(in_data, how="left")

            if new_data is not None:
                # fill rest of the data with forecasts
                if new_data_is_forecast:
                    in_window.update(
                        # pyre-fixme[61]: `fc_wide` is undefined, or not always defined.
                        fc_wide[fc_wide.index <= fc_origin],
                        overwrite=False,
                    )
                else:
                    # in case the new_data is actually earlier than the training data, we need to remove any data
                    # that we want to predict but that may have actuals in the training data
                    in_window.loc[
                        in_window.index > fc_origin, self.params.target_variable
                    ] = np.nan

                    curr_new_data = new_data[curr_series].to_dataframe()
                    time_col = new_data[curr_series].time_col_name
                    curr_new_data.set_index(time_col, inplace=True)

                    in_window.update(curr_new_data, overwrite=True)

            new_data_dict[curr_series] = TimeSeriesData(
                in_window.reset_index(),
                time_col_name="index",
                categorical_var=self.params.categoricals,
            )

        curr_all_series, timestamps = self._valid_and_fillna(new_data_dict)

        # if we are running with actual new data, save the new data as the all_series object for later forecast iteration
        if (new_data is not None) and (not new_data_is_forecast):
            self.all_series = curr_all_series

        meta_data, all_col_names = self._embed_and_gen_features(
            timestamps,
            curr_all_series,
            horizons=self.params.horizon,
            lags=self.params.input_window,
            fillna=np.nan,
        )

        self.forecast_data_in = self.full_mat
        self.forecast_data = meta_data

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

        logging.info(
            "sMAPE between forecast and median of most recent three weeks actual: "
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

        curr_all_series, timestamps = self._valid_and_fillna(data)

        self.all_series = curr_all_series
        meta_data, all_col_names = self._embed_and_gen_features(
            timestamps,
            curr_all_series,
            horizons=self.params.horizon,
            lags=self.params.input_window,
            fillna=np.nan,
            gen_meta_data=self.params.calculate_fit,
        )

        self.feature_columns = all_col_names

        self._train(self.full_mat, meta_data)
        return None

    def predict(
        self,
        steps: Union[List[int], int, None] = None,
        new_data: Optional[Dict[Union[str, int], pd.DataFrame]] = None,
    ) -> Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
        """Full workflow of train and forecast with post process
        Args:
            steps: int for overwriting horizon, which should be no more than horizon.
        Returns: a dictionary of pd.DataFrame with forecasting where key is the unique category or
            a list of of pd.DataFrame forecasting without categorical information
        """

        self._predict(fill_missing=False, new_data=new_data, new_data_is_forecast=False)
        forecast_result = self._post_process()

        if steps is not None:

            horizons = np.array(self.params.horizon)

            if 1 not in horizons:
                logging.warning(
                    "Model not trained for 1-step, so iterating the model may not work properly."
                )

            if type(steps) == int:
                steps = np.arange(1, steps + 1)

            cont_steps_in_hor = find_first_missing_number(horizons) - 1

            # TODO: Steps needs to be an int, otherwise it all gets too complicated and not very useful?

            i = 1
            iterate_horizons = np.setdiff1d(steps, horizons)

            while len(iterate_horizons) != 0:
                # pyre-fixme
                self._predict(fill_missing=True, new_data=forecast_result)
                fcr = self._post_process()
                for curr_series in forecast_result.keys():
                    # forecast_result[curr_series] = pd.concat([forecast_result[curr_series], fcr[curr_series]], axis=0)

                    forecast_result[curr_series].set_index("time", inplace=True)
                    fcr[curr_series].set_index("time", inplace=True)
                    forecast_result[curr_series] = forecast_result[
                        curr_series
                    ].combine_first(fcr[curr_series])
                    forecast_result[curr_series].reset_index(inplace=True, drop=False)
                    # pyre-fixme[6]: For 2nd argument expected `DataFrame` but got
                    #  `Optional[DataFrame]`.
                    forecast_result[curr_series] = forecast_result[
                        curr_series
                    ].drop_duplicates()

                horizons = np.union1d(
                    horizons, np.array(self.params.horizon) + i * cont_steps_in_hor
                )
                i += 1
                iterate_horizons = np.setdiff1d(steps, horizons)

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
