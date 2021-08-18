# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A module for meta-learner hyper-parameter selection.

This module contains two classes, including:
    - :class:`MetaLearnHPT` for recommending hyper-parameters of forecasting models;
    - :class:`MultitaskNet` for multi-task neural network built with pytorch.
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
from sklearn.model_selection import train_test_split

default_model_params = {
    "holtwinters": {
        "categorical_idx": ["trend", "damped", "seasonal", "seasonal_periods"],
        "numerical_idx": [],
    },
    "arima": {"categorical_idx": ["p", "d", "q"], "numerical_idx": []},
    "sarima": {
        "categorical_idx": ["seasonal_order", "trend", "p", "d", "q"],
        "numerical_idx": [],
    },
    "theta": {"categorical_idx": ["m"], "numerical_idx": []},
    "stlf": {"categorical_idx": ["method", "m"], "numerical_idx": []},
    "prophet": {
        "categorical_idx": [
            "yearly_seasonality",
            "weekly_seasonality",
            "daily_seasonality",
            "seasonality_mode",
            "seasonality_prior_scale",
            "changepoint_prior_scale",
            "changepoint_range",
        ],
        "numerical_idx": [],
    },
    "cusum": {
        "categorical_idx": ["score_func"],
        "numerical_idx": ["delta_std_ratio", "scan_window", "historical_window"],
    },
    "statsig": {
        "categorical_idx": [],
        "numerical_idx": ["n_control", "n_test"],
    },
}

default_model_networks = {
    "holtwinters": {
        "n_hidden_shared": [20],
        "n_hidden_cat_combo": [[2], [3], [5], [3]],
        "n_hidden_num": [],
    },
    "arima": {
        "n_hidden_shared": [40],
        "n_hidden_cat_combo": [[5], [5], [5]],
        "n_hidden_num": [],
    },
    "sarima": {
        "n_hidden_shared": [40],
        "n_hidden_cat_combo": [[5], [5], [5], [5], [5]],
        "n_hidden_num": [],
    },
    "theta": {"n_hidden_shared": [40], "n_hidden_cat_combo": [[5]], "n_hidden_num": []},
    "stlf": {
        "n_hidden_shared": [20],
        "n_hidden_cat_combo": [[5], [5]],
        "n_hidden_num": [],
    },
    "prophet": {
        "n_hidden_shared": [40],
        "n_hidden_cat_combo": [[5], [5], [2], [3], [5], [5], [5]],
        "n_hidden_num": [],
    },
    "cusum": {
        "n_hidden_shared": [20],
        "n_hidden_cat_combo": [[3]],
        "n_hidden_num": [5, 5, 5],
    },
    "statsig": {
        "n_hidden_shared": [20],
        "n_hidden_cat_combo": [],
        "n_hidden_num": [5, 5],
    },
}


def _log_error(msg: str) -> ValueError:
    logging.error(msg)
    return ValueError(msg)


class MetaLearnHPT:
    """A class for meta-learner framework on hyper-parameters tuning.

    MetaLearnHPT is a framework for choosing hyper-parameter for forecasting models. It uses a multi-task neural network for recommendation, with time series features as inputs, and the hyper-parameters of a given model as outputs.
    For training, it uses time series features as inputs and the corresponding best hyper-parameters as labels. For prediction, it takes time series or time series features to predict the best hyper-parameters.
    MetaLearnHPT provides get_default_model, build_network, train, pred, pred_by_feature, save_model, load_model and plot.

    Attributes:
        data_x: Optional; A `pandas.DataFrame` object of time series features. data_x should not be None unless load_model is True. Default is None.
        data_y: Optional; A `pandas.DataFrame` object of the corresponding best hyper-parameters. data_y should not be None unless load_model is True. Default is None.
        categorical_idx: Optional; A list of strings of the names of the categorical hyper-parameters. Default is None.
        numerical_idx: Optional; A list of strings of the names of the numerical hyper-parameters. Default is None.
        default_model: Optional; A string of the name of the forecast model whose default settings will be used.
                       Can be 'arima', 'sarima', 'theta', 'prophet', 'holtwinters', 'stlf' or None. Default is None.
        scale: Optional; A boolean to specify whether or not to normalize time series features to zero mean and unit variance. Default is True.
        load_model: Optional; A boolean to specify whether or not to load a trained model. Default is False.

    Sample Usage:
        >>> mlhpt_hw = MetaLearnHPT(X, Y, default_model='holtwinters') # Use default Holt-Winter's model as an example.
        >>> mlhpt_hw.build_network()
        >>> mlhpt_hw.train()
        >>> mlhpt_hw.pred(ts=TSdata) # Recommend hyper-parameters for TSdata.
        >>> mlhpt_hw.save_model('my_model_binary.pkl') # Save trained model to a binary
        >>> mlhpt_hw2=MetaLearnHPT(load_model=True) # Load a trained model
        >>> mlhpt_hw2.load_model('my_model_binary.pkl')
        >>> mlhpt_hw = MetaLearnHPT(X, Y_holtwinters, ['trend','damped', 'seasonal'], ['seasonal_periods']) # Example for building customized MetaLearnHPT object.
        >>> mlhpt_hw.build_network(n_hidden_shared=[30], n_hidden_cat_combo=[[2], [3], [5]],n_hidden_num=[3])
        >>> mlhpt_hw.train(loss_scale=30, lr=0.001)
    """

    def __init__(
        self,
        data_x: Optional[pd.DataFrame] = None,
        data_y: Optional[pd.DataFrame] = None,
        categorical_idx: Optional[List[str]] = None,
        numerical_idx: Optional[List[str]] = None,
        default_model: Optional[str] = None,
        scale: bool = True,
        load_model: bool = False,
        **kwargs,
    ) -> None:
        if not load_model:
            if data_x is None:
                raise _log_error("data_x is necessary to initialize a new model!")
            if data_y is None:
                raise _log_error("data_y is necessary to initialize a new model!")

            data_x.fillna(0, inplace=True)
            self.dataX = np.asarray(data_x)
            self.dataY = data_y.copy()
            self.dim_input = self.dataX.shape[1]
            self.model = None

            # Record loss path for validation/trainin set and for both classification and regression.
            self._loss_path = collections.defaultdict(list)

            if isinstance(default_model, str):
                default_model = default_model.lower()
            self.__default_model = default_model

            if default_model is not None:

                if (categorical_idx is not None) or (numerical_idx is not None):
                    msg = """
                         Default model cannot accept customized categorical_idx or customized numerical_idx! Please set
                         'categorical_idx=None' and 'numerical_idx=None' to initialize a default model,
                         or set 'default_model=None' to initialize a customized model!
                         """
                    raise _log_error(msg)

                if default_model in default_model_params:
                    categorical_idx = default_model_params[default_model][
                        "categorical_idx"
                    ]
                    numerical_idx = default_model_params[default_model]["numerical_idx"]

                else:
                    msg = f"default_model={default_model} is not available! Please choose one from 'prophet', 'arima', 'sarima', 'holtwinters', 'stlf', 'theta', 'cusum', 'statsig'"
                    raise _log_error(msg)

            if (not numerical_idx) and (not categorical_idx):
                msg = "At least one of numerical_idx and categorical_idx should be a non-empty list."
                raise _log_error(msg)

            self.categorical_idx = categorical_idx
            self.numerical_idx = numerical_idx
            self._target_num = (
                np.asarray(self.dataY[self.numerical_idx])
                if self.numerical_idx
                else None
            )
            self._dim_output_num = (
                self._target_num.shape[1] if self.numerical_idx else 0
            )
            self._get_target_cat()
            self._validate_data()
            self.scale = scale
            if self.scale:
                self.x_mean = self.dataX.mean(0)
                x_std = self.dataX.std(0)
                x_std[x_std == 0.0] = 1.0
                self.x_std = x_std
                self.dataX = (self.dataX - self.x_mean) / self.x_std

        self.n_hidden_shared = [0]
        self.n_hidden_cat_combo = [0]
        self.n_hidden_num = [0]

    def _get_target_cat(self) -> None:
        # List of number of classes (dim of output) of each categorical variable
        if self.categorical_idx is None:
            self.target_cat = None
            self.dim_output_cat = []
            return
        n_cat = []
        # Dict for encoder, categories --> int
        self.cat_code_dict = {}
        for col in self.categorical_idx:
            n_cat.append(self.dataY[col].nunique())
            self.dataY[col] = self.dataY[col].astype("category")
            self.cat_code_dict[col] = dict(enumerate(self.dataY[col].cat.categories))
            self.dataY[col] = self.dataY[col].cat.codes.values

        self.target_cat = (
            np.asarray(self.dataY[self.categorical_idx])
            if self.categorical_idx
            else None
        )
        self.dim_output_cat = n_cat

    def get_default_model(self) -> Optional[str]:
        """Get the name of default_model. It the instance is a customized model, return None.

        Returns:
            A string reprsenting the default model or None.
        """

        return self.__default_model

    def _validate_data(self) -> None:
        """Validate input data."""

        n_cat = len(self.categorical_idx) if self.categorical_idx is not None else 0
        n_num = len(self.numerical_idx) if self.numerical_idx is not None else 0
        dim = self.dataY.shape[1]
        if n_cat + n_num != dim:
            msg = f"Dimensions of data_y (dim={dim}) and the input variables (dim={n_cat}+{n_num}) do not agree!"
            raise _log_error(msg)

        for i, var in enumerate(self.categorical_idx):
            if self.dim_output_cat[i] == 1:
                msg = f"Column {var} only has one class, not able to train a model!"
                raise _log_error(msg)

        if self.dataX.shape[0] <= 30:
            msg = "Dataset is too small to train a model!"
            raise _log_error(msg)

    @staticmethod
    def _get_hidden_and_output_cat_combo(
        n_hidden_cat_combo: List[List[int]], out_dim_cat: List[int]
    ) -> List[List[int]]:
        # The length of n_hidden_cat_combo and out_dim_cat should be same
        # If there is no categorical variable, out_dim_cat = []
        if not out_dim_cat:
            return []
        res = []
        for i in range(len(n_hidden_cat_combo)):
            res.append(n_hidden_cat_combo[i] + [out_dim_cat[i]])
        return res

    @staticmethod
    def _get_hidden_and_output_num(
        n_hidden_num: List[int], out_dim_num: int
    ) -> List[int]:
        # If there is no numerical variable, out_dim_num = []
        if not out_dim_num:
            return []
        return n_hidden_num + [out_dim_num]

    def build_network(
        self,
        n_hidden_shared: Optional[List[int]] = None,
        n_hidden_cat_combo: Optional[List[int]] = None,
        n_hidden_num: Optional[List[int]] = None,
    ) -> None:
        """Build a multi-task neural network.

        This function builds a multi-task neural network according to given neural network structure (i.e., n_hidden_shared, n_hidden_cat_combo, n_hidden_num).
        If the MetaLearnHPT object is initiated as a default_model (i.e., default_model is not None), then a default neural network structure will be built and cannot accept customized.

        Args:
            n_hidden_shared: Optional; A list of numbers of hidden neurons in each shared hidden layer.
                             For example, n_hidden_shared = [first_shared_hid_dim, second_shared_hid_dim, ....]. Default is None.

            n_hidden_cat_combo: Optional; A list of lists of task-specific hidden layers’ sizes of each categorical response variables.
                                For example, if we have 3 categorical y, then
                                n_hidden_cat_combo = [[first_spec_hid_dim_cat1, second_spec_hid_dim_cat1, ...],
                                [first_spec_hid_dim_cat2, second_spec_hid_dim_cat2, ...],
                                [first_spec_hid_dim_cat3, second_spec_hid_dim_cat3, ...]].
                                Length of n_hidden_cat_combo must match the number of categorical y. Default is None.

            n_hidden_num: Optional; A list of task-specific hidden layers’ sizes of numerical response variables.
                          For example, n_hidden_num = [first_spec_hid_dim, second_spec_hid_dim, ...]. Default is None.

        Returns:
            None.
        """

        network_structure = (
            (n_hidden_shared is None)
            and (n_hidden_cat_combo is None)
            and (n_hidden_num is None)
        )

        default_model = self.__default_model

        if default_model is not None:
            if not network_structure:
                msg = f"A default model structure ({default_model}) is initiated and cannot accept the customized network structure!"
                raise _log_error(msg)

            if default_model in default_model_networks:
                n_hidden_shared = default_model_networks[default_model][
                    "n_hidden_shared"
                ]
                n_hidden_cat_combo = default_model_networks[default_model][
                    "n_hidden_cat_combo"
                ]
                n_hidden_num = default_model_networks[default_model]["n_hidden_num"]
            else:
                msg = f"Default neural network for model {default_model} is not implemented!"
                raise _log_error(msg)
            msg = f"Default neural network for model {default_model} is built."
            logging.info(msg)
        elif n_hidden_shared is None:
            msg = "n_hidden_shared is missing!"
            raise _log_error(msg)
        elif n_hidden_cat_combo is None:
            msg = "n_hidden_cat_combo is missing!"
            raise _log_error(msg)
        elif n_hidden_num is None:
            msg = "n_hidden_num is missing!"
            raise _log_error(msg)

        if len(n_hidden_cat_combo) != len(self.dim_output_cat):
            msg = "Unmatched dimension!"
            raise _log_error(msg)
        # Add input dim before n_hidden_shared.
        # Add output dim at the end of n_hidden_cat_combo.
        # Add output dim at the end of n_hidden_num.
        self.n_hidden_shared = n_hidden_shared
        self.n_hidden_cat_combo = n_hidden_cat_combo
        self.n_hidden_num = n_hidden_num

        self.model = MultitaskNet(
            input_and_n_hidden_shared=[self.dim_input] + n_hidden_shared,
            n_hidden_and_output_cat_combo=self._get_hidden_and_output_cat_combo(
                n_hidden_cat_combo, self.dim_output_cat
            ),
            n_hidden_and_output_num=self._get_hidden_and_output_num(
                n_hidden_num, self._dim_output_num
            ),
        )
        print("Multi-task neural network structure:")
        print(self.model)

    def _prepare_data(
        self, val_size: float
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.LongTensor],
        Optional[torch.FloatTensor],
        torch.FloatTensor,
        Optional[torch.LongTensor],
        Optional[torch.FloatTensor],
    ]:
        # split to train and validation sets
        train_idx, val_idx = train_test_split(
            np.arange(len(self.dataX)), test_size=val_size
        )

        # change training set to tensors
        x_fs = torch.from_numpy(self.dataX[train_idx, :]).float()
        y_cat = (
            torch.from_numpy(self.target_cat[train_idx, :]).long()
            if self.categorical_idx
            else None
        )
        y_num = (
            torch.from_numpy(self._target_num[train_idx, :].astype("float")).float()
            if self.numerical_idx
            else None
        )

        # change validation set to tensors
        x_fs_val = torch.from_numpy(self.dataX[val_idx, :]).float()
        y_cat_val = (
            torch.from_numpy(self.target_cat[val_idx, :]).long()
            if self.categorical_idx
            else None
        )
        y_num_val = (
            torch.from_numpy(self._target_num[val_idx, :].astype("float")).float()
            if self.numerical_idx
            else None
        )
        return x_fs, y_cat, y_num, x_fs_val, y_cat_val, y_num_val

    def _loss_function(
        self,
        o1: Optional[torch.Tensor],
        o2: Optional[torch.Tensor],
        y_cat: Optional[torch.Tensor],
        y_num: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loss function for multi-task neural network."""

        loss_func_num = nn.MSELoss()
        loss_func_cat = nn.CrossEntropyLoss()
        loss_cat_train = torch.tensor([0.0])
        # Loss of classification.
        if o1 is not None:
            assert y_cat is not None
            batch_y1 = y_cat
            for col in range(batch_y1.shape[1]):
                loss_cat_train += loss_func_cat(o1[col], batch_y1[:, col])

        # Loss of regression.
        loss_num_train = torch.tensor([0.0])
        if o2 is not None:
            batch_y2 = y_num
            loss_num_train += loss_func_num(o2, batch_y2)

        return loss_cat_train, loss_num_train

    def train(
        self,
        loss_scale: float = 1.0,
        lr: float = 0.001,
        n_epochs: int = 1000,
        batch_size: int = 128,
        method: str = "SGD",
        val_size: float = 0.1,
        momentum: float = 0.9,
        n_epochs_stop: Union[int, float] = 20,
    ) -> None:
        """Train the pre-built multi-task neural network.

        Args:
            loss_scale: Optional; A float to specify the hyper-parameter to scale regression loss and classification loss, which controls the trade-off between the accuracy of regression task and classification task.
                        A larger loss_scale value gives a more accurate prediction for classification part, and a lower value gives a more accurate prediction for regression part. Default is 1.0.
            lr: Optional; A float for learning rate. Default is 0.001.
            n_epochs: Optional; An integer for the number of epochs. Default is 1000.
            batch_size: Optional; An integer for the batch size. Default is 128.
            method: Optional; A string for the name of optimizer. Can be 'SGD' or 'Adam'. Default is 'SGD'.
            val_size: Optional; A float for the proportion of validation set of. It should be within (0, 1). Default is 0.1.
            momentum: Optional; A fload for the momentum for SGD. Default value is 0.9.
            n_epochs_stop: Optional; An integer or a float for early stopping condition. If the number of epochs is larger than n_epochs_stop and there is no improvement on validation set, we stop training.
                           One can turn off the early stopping feature by setting n_epochs_stop = np.inf. Default is 20.

        Returns:
            None.
        """

        if self.model is None:
            raise _log_error("Haven't built a model. Please build a model first!")

        if method == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum
            )
        elif method == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise _log_error(
                "Only support SGD and Adam optimaizer. Please use 'SGD' or 'Adam'."
            )

        if val_size <= 0 or val_size >= 1:
            raise _log_error("Illegal validation size.")

        # get training tensors
        x_fs, y_cat, y_num, x_fs_val, y_cat_val, y_num_val = self._prepare_data(
            val_size
        )

        # validate batch size
        if batch_size >= x_fs.size()[0]:
            raise _log_error(
                f"batch_size {batch_size} is larger than training data size {x_fs.size()[0]}!"
            )

        # variables for early stopping
        min_val_loss = np.inf
        epochs_no_improve = 0.0

        for epoch in range(n_epochs):
            total_loss = 0
            permutation = torch.randperm(x_fs.size()[0])
            for i in range(0, x_fs.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_x = x_fs[indices]

                # two outputs, o1: classification, o2: regression
                o1, o2 = self.model.forward(batch_x)
                tmp_y_cat = y_cat[indices] if y_cat is not None else None
                tmp_y_num = y_num[indices] if y_num is not None else None
                loss_cat_train, loss_num_train = self._loss_function(
                    o1, o2, tmp_y_cat, tmp_y_num
                )
                cur_loss = loss_cat_train + loss_num_train / loss_scale
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

                total_loss += cur_loss

            # Record loss of training set at the last iteration of each epoch.
            self._loss_path["LOSS_train_cat"].append(loss_cat_train.item())
            self._loss_path["LOSS_train_num"].append(loss_num_train.item())

            # Record loss of validatiaon set for each epoch.
            o1_val, o2_val = self.model.forward(x_fs_val)
            loss_cat_val, loss_num_val = self._loss_function(
                o1_val, o2_val, y_cat_val, y_num_val
            )
            self._loss_path["LOSS_val_cat"].append(loss_cat_val.item())
            self._loss_path["LOSS_val_num"].append(loss_num_val.item())
            loss_sum_val = loss_cat_val + loss_num_val / loss_scale

            # early stopping variables update
            if loss_sum_val < min_val_loss:
                min_val_loss = loss_sum_val
                epochs_no_improve = 0.0
            else:
                epochs_no_improve += 1

            # check early stopping condition
            if epoch > 20 and epochs_no_improve >= n_epochs_stop:
                logging.info(f"Early stopping! Stop at epoch {epoch + 1}.")
                break

    def pred(self, source_ts: TimeSeriesData, ts_scale: bool = True) -> pd.DataFrame:
        """Predict hyper-parameters for a new time series data.

        Args:
            source_ts: :class:`kats.consts.TimeSeriesData` object representing the time series for which to generate hyper-parameters
            ts_scale: A boolean to specify whether or not to rescale time series data (i.e., divide its value by its maximum value) before calculating its features. Default is True.

        Returns:
            A `pandas.DataFrame` object storing the recommended hyper-parameters.
        """

        ts = TimeSeriesData(pd.DataFrame(source_ts.to_dataframe().copy()))

        if self.model is None:
            raise _log_error(
                "Haven't trained a model. Please train a model or load a model before predicting."
            )

        if ts_scale:
            # scale time series to make ts features more stable
            ts.value /= ts.value.max()
            msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
            logging.info(msg)

        self.model.eval()
        new_feature = TsFeatures().transform(ts)
        # pyre-fixme[16]: `List` has no attribute `values`.
        new_feature_vector = np.asarray(list(new_feature.values()))

        if np.any(np.isnan(new_feature_vector)):
            logging.warning(
                "Time series features contain NaNs!"
                f"Time series features are {new_feature}. "
                "Fill in NaNs with 0."
            )

        pred_res = self.pred_by_feature([new_feature_vector])[0]

        # To have a consistent type with orginal HPT methods' output.
        res = [["0_0", "unknown", 0, 0.0, 0, pred_res]]
        res = pd.DataFrame(res)
        res.columns = [
            "arm_name",
            "metric_name",
            "mean",
            "sem",
            "trial_index",
            "parameters",
        ]
        return res

    def pred_by_feature(
        self, source_x: Union[np.ndarray, List[np.ndarray], pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Predict hyper-parameters for time series features.

        Args:
            source_x: Time series features.

        Returns:
            A list of dictionaries storing the recommended hyper-parameters.
        """

        if self.model is None:
            raise _log_error(
                "Haven't trained a model. Please train a model or load a model before predicting."
            )
        if isinstance(source_x, List):
            x = np.row_stack(source_x)
        elif isinstance(source_x, pd.DataFrame):
            x = source_x.values.copy()
        elif isinstance(source_x, np.ndarray):
            x = source_x.copy()
        else:
            raise _log_error(f"In valid source_x type: {type(source_x)}.")
        x[np.isnan(x)] = 0.0
        if self.scale:
            x = (x - self.x_mean) / self.x_std

        n = len(x)
        x = torch.from_numpy(x).float()

        self.model.eval()
        cats, nums = self.model(x)
        cats = (
            [torch.argmax(t, dim=1).detach().numpy() for t in cats]
            if cats is not None
            else []
        )
        nums = nums.detach().numpy() if nums is not None else []

        ans = [{} for _ in range(n)]
        for j, c in enumerate(self.categorical_idx):
            vals = cats[j]
            for i in range(n):
                ans[i][c] = self.cat_code_dict[c][vals[i]]
        for j, c in enumerate(self.numerical_idx):
            vals = nums[:, j]
            for i in range(n):
                ans[i][c] = vals[i]
        return ans

    def save_model(self, file_path: str) -> None:
        """Save trained model to a binary.

        Args:
            file_path: A string representing the path to save a trained model, which should contain either '.p' or '.pkl' file extension.

        Returns:
            None.
        """

        if self.model is None:
            raise _log_error("Haven't trained a model.")
        else:
            joblib.dump(self.__dict__, file_path)
            logging.info("Successfully saved the trained model!")

    def load_model(self, file_path: str) -> None:
        """Load a pre-trained model from a binary.

        Args:
            file_path: A string representing the path to load a pre-trained model.

        Returns:
            None.
        """

        try:
            self.__dict__ = joblib.load(file_path)
        except Exception as e:
            raise _log_error(
                f"Fail to load model from {file_path}, and error message is: {e}"
            )

    def plot(self):
        """Plot loss paths of classification/regression on both training and validation set."""

        if (
            not self._loss_path["LOSS_train_cat"]
            and not self._loss_path["LOSS_train_num"]
        ):
            raise _log_error("Using a loaded model or no trained model!")

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(self._loss_path["LOSS_train_cat"], ".-")
        plt.plot(self._loss_path["LOSS_val_cat"], "o-")
        plt.legend(["training set", "validation set"])
        plt.title("Loss path of classification tasks")
        plt.ylabel("Cross-entropy")
        plt.xlabel("Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(self._loss_path["LOSS_train_num"], ".-")
        plt.plot(self._loss_path["LOSS_val_num"], "o-")
        plt.legend(["training set", "validation set"])
        plt.title("Loss path of regression task")
        plt.ylabel("MSE")
        plt.xlabel("Epoch")


class MultitaskNet(nn.Module):
    """A class for multi-task neural network.

    Build a multi-task neural network used by MetaLearnHPT. It can also be used to single-task learning. Currently only support Relu activation function.

    Attributes:
        input_and_n_hidden_shared: A list of integers contains the dimension of input and numbers of hidden neurons in each shared hidden layer.
                                   The first value in this list is dimension of input, which is dimension of feature vector.
                                   For example, input_and_n_hidden_shared = [input_dim, first_shared_hid_dim, second_shared_hid_dim, ....].
        n_hidden_and_output_cat_combo: A list of lists of task-specific hidden layers’ sizes of each categorical response variables and their dimension of output.
                                       For example, if we have 3 categorical y with 3, 2, 4 classes respectively, then
                                       n_hidden_and_output_cat_combo = [[first_spec_hid_dim_cat1, second_spec_hid_dim_cat1, ..., 3],
                                       [first_spec_hid_dim_cat2, second_spec_hid_dim_cat2, ..., 2],
                                       [first_spec_hid_dim_cat3, second_spec_hid_dim_cat3, ..., 4]].
        n_hidden_and_output_num: A list of integers contains task-specific hidden layers’ sizes of numerical response variables and the dimension of output,
                                 which is the number of numerical response variables in data_y.
                                 For example, if we have three numerical response variables in y, then
                                 n_hidden_and_output_num = [first_spec_hid_dim, second_spec_hid_dim, ..., 3]
    """

    def __init__(
        self,
        input_and_n_hidden_shared: List[int],
        n_hidden_and_output_cat_combo: List[List[int]],
        n_hidden_and_output_num: List[int],
    ) -> None:
        super(MultitaskNet, self).__init__()
        self.shared_layer = nn.ModuleList()
        for i in range(len(input_and_n_hidden_shared) - 1):
            self.shared_layer.append(
                nn.Linear(
                    input_and_n_hidden_shared[i], input_and_n_hidden_shared[i + 1]
                )
            )

        self.cat_layer_combo = nn.ModuleList()
        for n_hidden_and_output_cat in n_hidden_and_output_cat_combo:
            cat_layer = nn.ModuleList()
            # input for task-specific layer is the dim of last shared hidden layer
            curr_input = input_and_n_hidden_shared[-1]
            for i in range(len(n_hidden_and_output_cat)):
                cat_layer.append(nn.Linear(curr_input, n_hidden_and_output_cat[i]))
                curr_input = n_hidden_and_output_cat[i]
            self.cat_layer_combo.append(cat_layer)

        self.num_layer = nn.ModuleList()
        curr_input = input_and_n_hidden_shared[-1]
        for i in range(len(n_hidden_and_output_num)):
            self.num_layer.append(nn.Linear(curr_input, n_hidden_and_output_num[i]))
            curr_input = n_hidden_and_output_num[i]

    def forward(self, x):
        """Forward function in neural networks."""

        # Shared layers.
        for layer in self.shared_layer:
            x = layer(x)
            x = nn.functional.relu(x)

        # Categorical part.
        if not self.cat_layer_combo:
            y_pred_cat_combo = None
        else:
            y_pred_cat_combo = []
            for cat_layer in self.cat_layer_combo:
                y_pred_cat = cat_layer[0](x)
                for i in range(1, len(cat_layer)):
                    # the last layer has no activation function
                    y_pred_cat = nn.functional.relu(y_pred_cat)
                    y_pred_cat = cat_layer[i](y_pred_cat)
                y_pred_cat_combo.append(y_pred_cat)

        # Numerical part.
        if not self.num_layer:
            y_pred_num = None
        else:
            y_pred_num = self.num_layer[0](x)
            for i in range(1, len(self.num_layer)):
                # the last layer has no activation function
                y_pred_num = nn.functional.relu(y_pred_num)
                y_pred_num = self.num_layer[i](y_pred_num)

        return y_pred_cat_combo, y_pred_num
