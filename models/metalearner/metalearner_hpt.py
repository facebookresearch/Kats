#!/usr/bin/env python3

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
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
}


class MetaLearnHPT:
    """
    Meta-learning framework on hyper-parameters tuning.

    This framework is using multi-task neural networks with time series features as inputs, and best hyper-params for a given model as outputs.

    After training a neural network for a given model, it can directly predict best hyper-params for a new time series data.

    :Parameters:
    data_x: Optional[pd.DataFrame]
        Time series features matrix.
    data_y: Optional[pd.DataFrame]
        Best hyper-parameters matrix for a given model.
        For example, for Holt Winter's model, columns list of data_y is ['season', 'trend', 'damped', 'seasonal_periods'].
    categorical_idx: Optional[List]
        List of names of categorical columns.
        For example, for Holt Winter's model, categorical_idx is ['season', 'trend', 'damped'].
        If no categorical variables, then let categorical_idx = [].
    numerical_idx: Optional[List]
        List of names of numerical columns.
        For example, for Holt Winter's model, numerical_idx is ['seasonal_periods'].
        If no numerical variables, then let numerical_idx = [].
    default_model: Optional[str]
        If not none, the default set-up for default_model will be initiated, and categorical_idx and numerical_idx should be None.
        Currently we support ['prophet','arima','sarima','theta','stlf','holtwinters'].
        If None, then a customized model will be initiated.
    scale: bool
        Whether to normailize feature matrix.
        Center to the mean and component wise scale to unit variance. True by default.
    load_model: bool
        Whether one wants to load a trained model.
        If True, then the object can be initiated with data_x, data_y, categorical_idx and numerical_idx as None.
        False by default.


    :Example:
    >>> from infrastrategy.kats.models.metalearner.metalearner_hpt import MetaLearnHPT
    >>> # create a default model, using Holt-Winter's model as an example
    >>> mlhpt_hw = MetaLearnHPT(X, Y_holtwinters, default_model='holtwinters')
    >>> # build a network
    >>> mlhpt_hw.build_network()
    >>> mlhpt_hw.train()
    >>> # plot loss paths
    >>> mlhpt_hw.plot()
    >>> # prediction for a TimeSeriesData
    >>> mlhpt_hw.pred(ts=TSdata)
    >>> # prediction for time series features
    >>> mlhpt_hw.pred_by_feature(features)
    >>> # save trained model to a binary
    >>> mlhpt_hw.save_model('my_model_binary.pkl')
    >>> # load a trained model
    >>> mlhpt_hw2=MetaLearnHPT(load_model=True)
    >>> mlhpt_hw2.load_model('my_model_binary.pkl')
    >>> # Example for a customized object of class MetaLearnHPT, using Holt-Winter's model as an example
    >>> mlhpt_hw = MetaLearnHPT(X, Y_holtwinters, ['trend','damped', 'seasonal'], ['seasonal_periods'])
    >>> mlhpt_hw.build_network(n_hidden_shared=[30], n_hidden_cat_combo=[[2], [3], [5]],n_hidden_num=[3])
    >>> mlhpt_hw.train(loss_scale=30, lr=0.001)

    :Methods:
    """

    def __init__(
        self,
        data_x: Optional[pd.DataFrame] = None,
        data_y: Optional[pd.DataFrame] = None,
        categorical_idx: Optional[List] = None,
        numerical_idx: Optional[List] = None,
        default_model: Optional[str] = None,
        scale: bool = True,
        load_model: bool = False,
        **kwargs,
    ) -> None:
        if not load_model:
            if data_x is None:
                msg = "data_x is necessary to initialize a new model!"
                logging.error(msg)
                raise ValueError(msg)
            if data_y is None:
                msg = "data_y is necessary to initialize a new model!"
                logging.error(msg)
                raise ValueError(msg)

            self.dataX = np.asarray(data_x)
            self.dataY = data_y.copy()
            self.dim_input = self.dataX.shape[1]  # dimension of features
            self.model = None
            self.error_method = kwargs.get("error_method", "unknown")

            # record loss path for validation/trainin set and for both classification and regression
            self.LOSS_PATH = collections.defaultdict(list)

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
                    logging.error(msg)
                    raise ValueError(msg)

                if default_model in default_model_params:
                    categorical_idx = default_model_params[default_model][
                        "categorical_idx"
                    ]
                    numerical_idx = default_model_params[default_model]["numerical_idx"]

                else:
                    msg = f"default_model={default_model} is not available! Please choose one from 'prophet', 'arima', 'sarima', 'holtwinters', stlf, 'theta'"
                    logging.error(msg)
                    raise ValueError(msg)

            if (not numerical_idx) and (not categorical_idx):
                msg = "At least one of numerical_idx and categorical_idx should be a non-empty list."
                logging.error(msg)
                raise ValueError(msg)

            self.categorical_idx = categorical_idx
            self.numerical_idx = numerical_idx

            # numerical output
            self.target_num = (
                np.asarray(self.dataY[self.numerical_idx])
                if self.numerical_idx
                else None
            )
            # dimension of numerical output
            self.dim_output_num = self.target_num.shape[1] if self.numerical_idx else 0

            # assign categorical outputs and corresponding dimensions list
            self._get_target_cat()

            # validate data
            self._validate_data()

            # to scale dataX. Center to the mean and component wise scale to unit variance.
            self.scale = scale
            if self.scale:
                self.x_mean = self.dataX.mean(0)
                x_std = self.dataX.std(0)
                x_std[x_std == 0.0] = 1.0
                self.x_std = x_std
                self.dataX = (self.dataX - self.x_mean) / self.x_std

    def _get_target_cat(self) -> None:
        # list of number of classes (dim of output) of each categorical variable
        if self.categorical_idx is None:
            self.target_cat = None
            self.dim_output_cat = []
            return
        n_cat = []
        # Dict for encoder, categories --> int
        self.cat_code_dict = {}
        for col in self.categorical_idx:
            # dim of output for dataY[col]
            n_cat.append(self.dataY[col].nunique())
            # classes encoder
            self.dataY[col] = self.dataY[col].astype("category")
            self.cat_code_dict[col] = dict(enumerate(self.dataY[col].cat.categories))
            self.dataY[col] = self.dataY[col].cat.codes.values

        # either be None or a np.array, categorical outputs
        self.target_cat = (
            np.asarray(self.dataY[self.categorical_idx])
            if self.categorical_idx
            else None
        )
        # dimension list. n_cat might be empty [], if self.categorical_idx is empty
        self.dim_output_cat = n_cat

    def get_default_model(self):
        """
        Return the name of default_model. It the instance is a customized model, return None.
        """
        return self.__default_model

    def _validate_data(self):
        # check if input dimensions agree
        n_cat = len(self.categorical_idx) if self.categorical_idx is not None else 0
        n_num = len(self.numerical_idx) if self.numerical_idx is not None else 0
        dim = self.dataY.shape[1]
        if n_cat + n_num != dim:
            msg = f"Dimensions of data_y (dim={dim}) and the input variables (dim={n_cat}+{n_num}) do not agree!"
            logging.error(msg)
            raise ValueError(msg)

        for i, var in enumerate(self.categorical_idx):
            if self.dim_output_cat[i] == 1:
                msg = f"Column {var} only has one class, not able to train a model!"
                logging.error(msg)
                raise ValueError(msg)

        if self.dataX.shape[0] <= 30:
            msg = "Dataset is too small to train a model!"
            logging.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _get_hidden_and_output_cat_combo(
        n_hidden_cat_combo: List[List], out_dim_cat: List
    ) -> List[List]:
        # length of n_hidden_cat_combo and out_dim_cat should be same
        # if no categorical variable, out_dim_cat = []
        if not out_dim_cat:
            return []
        res = []
        for i in range(len(n_hidden_cat_combo)):
            res.append(n_hidden_cat_combo[i] + [out_dim_cat[i]])
        return res

    @staticmethod
    def _get_hidden_and_output_num(n_hidden_num: List, out_dim_num: int) -> List:
        # if no numerical variable, out_dim_num = 0
        if not out_dim_num:
            return []
        return n_hidden_num + [out_dim_num]

    def build_network(
        self,
        n_hidden_shared: Optional[List] = None,
        n_hidden_cat_combo: Optional[List] = None,
        n_hidden_num: Optional[List] = None,
    ) -> None:
        """
        Build a multi-task neural network. It can also be used to single-task learning. Only support Relu activation function now.
        If the object is initiated as a default_model, then a default neural network structure will be built and cannot accept customized
        neural network structure (i.e., n_hidden_shared, n_hidden_cat_combo, n_hidden_num should all be None).

        :Parameters:
        n_hidden_shared: Optional[List]
            A list of numbers of hidden neurons in each shared hidden layer.
            For example, n_hidden_shared = [first_shared_hid_dim, second_shared_hid_dim, ....].
            It could be an empty list.

        n_hidden_cat_combo: Optional[List]
            A list of lists of task-specific hidden layers’ sizes of each categorical response variables.
            For example, if we have 3 categorical y, then
            n_hidden_cat_combo = [[first_spec_hid_dim_cat1, second_spec_hid_dim_cat1, ...],
            [first_spec_hid_dim_cat2, second_spec_hid_dim_cat2, ...],
            [first_spec_hid_dim_cat3, second_spec_hid_dim_cat3, ...]].
            Length of n_hidden_cat_combo must match the number of categorical y!
            It could be an empty list.

        n_hidden_num: Optional[List]
            A list of task-specific hidden layers’ sizes of numerical response variables.
            For example, n_hidden_num = [first_spec_hid_dim, second_spec_hid_dim, ...].
            It could be an empty list.
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
                logging.error(msg)
                raise ValueError(msg)

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
                logging.error(msg)
                raise ValueError(msg)
            msg = f"Default neural network for model {default_model} is built."
            logging.info(msg)
        elif n_hidden_shared is None:
            msg = "n_hidden_shared is missing!"
            logging.error(msg)
            raise ValueError(msg)
        elif n_hidden_cat_combo is None:
            msg = "n_hidden_cat_combo is missing!"
            logging.error(msg)
            raise ValueError(msg)
        elif n_hidden_num is None:
            msg = "n_hidden_num is missing!"
            logging.error(msg)
            raise ValueError(msg)

        if len(n_hidden_cat_combo) != len(self.dim_output_cat):
            msg = "Unmatched dimension!"
            logging.error(msg)
            raise ValueError(msg)

        # add input dim before n_hidden_shared
        # add output dim at the end of n_hidden_cat_combo
        # add output dim at the end of n_hidden_num

        self.n_hidden_shared = n_hidden_shared
        self.n_hidden_cat_combo = n_hidden_cat_combo
        self.n_hidden_num = n_hidden_num

        self.model = MultitaskNet(
            input_and_n_hidden_shared=[self.dim_input] + n_hidden_shared,
            n_hidden_and_output_cat_combo=self._get_hidden_and_output_cat_combo(
                n_hidden_cat_combo, self.dim_output_cat
            ),
            n_hidden_and_output_num=self._get_hidden_and_output_num(
                n_hidden_num, self.dim_output_num
            ),
        )
        print("Multi-task neural network structure:")
        print(self.model)

    def _prepare_data(self, val_size: float) -> Tuple:
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
            torch.from_numpy(self.target_num[train_idx, :]).float()
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
            torch.from_numpy(self.target_num[val_idx, :]).float()
            if self.numerical_idx
            else None
        )

        return x_fs, y_cat, y_num, x_fs_val, y_cat_val, y_num_val

    def train(
        self,
        loss_scale: float = 1.0,
        lr: float = 0.001,
        n_epochs: int = 1000,
        batch_size: int = 128,
        method: str = "SGD",
        val_size: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Train the built multi-task neural network.

        :Parameters:
        loss_scale: float
            A hyper-parameter to scale regression loss and classification loss.
            There is a trade off between accuracy of regression and classification.
            A larger loss_scale value gives a more accurate prediction for classification part, and a lower value gives a more accurate prediction for regression part.
            It can be tuned based on errors from validation set.
        lr: float
            A constant learning rate.
        n_epochs: int
            Number of epochs.
        batch_size: int
            Batch size in each iteration and in each epoch.
        method: str
            Objective funtion optimization method. Only support 2 optimizer function, SGD and Adam.
        val_size: float
            Proportion of validation set. It should be within (0, 1).
        momentum: optional
            Momentum for SGD. Default value is 0.9.
        n_epochs_stop: optional
            A early stopping condition. If the number of epochs with no improvement on validation set >= n_epochs_stop, stop training.
            Turn off early stopping by setting n_epochs_stop = np.inf.
        """

        if method not in ["SGD", "S", "Adam", "A"]:
            msg = "Only support SGD and Adam optimaizer. Choose one of SGD, S, Adam, A as the value of method."
            logging.error(msg)
            raise ValueError(msg)

        if self.model is None:
            msg = "Haven't built a model. Please build a model first!"
            logging.error(msg)
            raise ValueError(msg)

        # define optimizer
        if method in ["SGD", "S"]:
            momentum = kwargs.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if val_size <= 0 or val_size >= 1:
            msg = "Illegal validation size."
            logging.error(msg)
            raise ValueError(msg)

        # get training tensors
        x_fs, y_cat, y_num, x_fs_val, y_cat_val, y_num_val = self._prepare_data(
            val_size
        )

        # validate batch size
        if batch_size >= x_fs.size()[0]:
            msg = "Either batch size is too larger or training set is too small!"
            logging.error(msg)
            raise ValueError(msg)

        # regression loss function
        loss_func_num = nn.MSELoss()

        # classification loss function
        loss_func_cat = nn.CrossEntropyLoss()

        # variables for early stopping
        min_val_loss = np.inf
        epochs_no_improve = 0
        n_epochs_stop = kwargs.get("n_epochs_stop", 20)

        for epoch in range(n_epochs):
            total_loss = 0
            # shuffle
            permutation = torch.randperm(x_fs.size()[0])

            for i in range(0, x_fs.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_x = x_fs[indices]

                # two outputs, o1: classification, o2: regression
                o1, o2 = self.model.forward(batch_x)

                # loss of task classification
                loss_cat_train = 0
                if o1 is not None:
                    batch_y1 = y_cat[indices]
                    for col in range(batch_y1.shape[1]):
                        loss_cat_train += loss_func_cat(o1[col], batch_y1[:, col])

                # loss of task regression
                loss_num_train = 0
                if o2 is not None:
                    batch_y2 = y_num[indices]
                    loss_num_train += loss_func_num(o2, batch_y2)

                cur_loss = loss_cat_train + loss_num_train / loss_scale
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

                total_loss += cur_loss

            if (epoch + 1) % 100 == 0:
                print(
                    "Epoch [%d] loss: %.3f"
                    % (epoch + 1, total_loss / (x_fs.size()[0] // batch_size))
                )

            # record loss of training set at the last iteration of each epoch
            if o1 is not None:
                self.LOSS_PATH["LOSS_train_cat"].append(loss_cat_train.item())
            if o2 is not None:
                self.LOSS_PATH["LOSS_train_num"].append(loss_num_train.item())

            # record loss of validatiaon set for each epoch
            o1_val, o2_val = self.model.forward(x_fs_val)

            loss_cat_val = 0
            if o1_val is not None:
                for col in range(y_cat_val.shape[1]):
                    loss_cat_val += loss_func_cat(o1_val[col], y_cat_val[:, col])
                self.LOSS_PATH["LOSS_val_cat"].append(loss_cat_val.item())

            loss_num_val = 0
            if o2_val is not None:
                loss_num_val += loss_func_num(o2_val, y_num_val)
                self.LOSS_PATH["LOSS_val_num"].append(loss_num_val.item())

            loss_sum_val = loss_cat_val + loss_num_val / loss_scale

            # early stopping variables update
            if loss_sum_val < min_val_loss:
                min_val_loss = loss_sum_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # check early stopping condition
            if epoch > 20 and epochs_no_improve >= n_epochs_stop:
                logging.info(f"Early stopping! Stop at epoch {epoch + 1}.")
                break

    def pred(self, source_ts: TimeSeriesData, ts_scale: bool = True) -> pd.DataFrame:
        """Predict hyper-parameters for a new time series data.

        :Parameters:
        source_ts: TimeSeriesData
            A new time series data.
        ts_scale: bool
            Whether to scale time series data before calculating features.
        """
        ts = TimeSeriesData(source_ts.to_dataframe().copy())

        if self.model is None:
            msg = "Haven't trained a model. Please train a model or load a model before predicting."
            logging.error(msg)
            raise ValueError(msg)

        if ts_scale:
            # scale time series to make ts features more stable
            ts.value /= ts.value.max()
            msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
            logging.info(msg)

        self.model.eval()

        # calcualte features:
        new_feature = TsFeatures().transform(ts)
        new_feature_vector = np.asarray(list(new_feature.values()))

        if np.sum(np.isnan(new_feature_vector)) > 0:
            logging.warning(
                "Time series features contain NaN, please consider preprocessing it! "
                f"Time series features are {new_feature}. "
                "Return empty dict as predicted hyper-parameters."
            )
            pred_res = {}
        else:
            pred_res = self.pred_by_feature([new_feature_vector])[0]

        # in order to have a consistent type with orginal HPT methods' output
        res = [["0_0", self.error_method, 0, 0.0, 0, pred_res]]
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
        """
        Predict hyper-parameters given time series features.

        :Parameters:
        source_x: Union[np.ndarray, List[np.ndarray], pd.DataFrame]
            Time series features
        """
        if self.model is None:
            msg = "Haven't trained a model. Please train a model or load a model before predicting."
            logging.error(msg)
            raise ValueError(msg)
        if isinstance(source_x, List):
            x = np.row_stack(source_x)
        elif isinstance(source_x, pd.DataFrame):
            x = source_x.values.copy()
        else:
            x = source_x.copy()
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
        """
        Save trained model to a binary.

        :Parameters:
        file_path: str
            Path to save a trained model.
            Should use .p or .pkl  file extension.
        """
        if self.model is None:
            msg = "Haven't trained a model."
            logging.error(msg)
            raise ValueError(msg)
        else:
            joblib.dump(self.__dict__, file_path)
            logging.info("Successfully saved the trained model!")

    def load_model(self, file_path: str) -> None:
        """
        Load a pre-trained model from a binary.

        :Parameters:
        file_path: str
            Path to load a pre-trained model.
        """
        try:
            self.__dict__ = joblib.load(file_path)
        except Exception as e:
            msg = f"Fail to load model from {file_path}, and error message is: {e}"
            logging.error(msg)
            raise ValueError(msg)

    def plot(self):
        """Plot loss paths of classification/regression on both training and validation set."""
        if (
            not self.LOSS_PATH["LOSS_train_cat"]
            and not self.LOSS_PATH["LOSS_train_num"]
        ):
            msg = "Using a loaded model or no trained model!"
            logging.error(msg)
            raise ValueError(msg)

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(self.LOSS_PATH["LOSS_train_cat"], ".-")
        plt.plot(self.LOSS_PATH["LOSS_val_cat"], "o-")
        plt.legend(["training set", "validation set"])
        plt.title("Loss path of classification tasks")
        plt.ylabel("Cross-entropy")
        plt.xlabel("Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(self.LOSS_PATH["LOSS_train_num"], ".-")
        plt.plot(self.LOSS_PATH["LOSS_val_num"], "o-")
        plt.legend(["training set", "validation set"])
        plt.title("Loss path of regression task")
        plt.ylabel("MSE")
        plt.xlabel("Epoch")


class MultitaskNet(nn.Module):
    """
    Build a multi-task neural network. It can also be used to single-task learning. Only support Relu activation function now.

    :Parameters:
    input_and_n_hidden_shared: List[int]
        A list contains the dimension of input and numbers of hidden neurons in each shared hidden layer.
        The first value in this list is dimension of input, which is dimension of feature vector.
        For example, input_and_n_hidden_shared = [input_dim, first_shared_hid_dim, second_shared_hid_dim, ....].

    n_hidden_and_output_cat_combo: List[List]
        A list of lists of task-specific hidden layers’ sizes of each categorical response variables and their dimension of output.
        For example, if we have 3 categorical y with 3, 2, 4 classes respectively, then
        n_hidden_and_output_cat_combo = [[first_spec_hid_dim_cat1, second_spec_hid_dim_cat1, ..., 3],
        [first_spec_hid_dim_cat2, second_spec_hid_dim_cat2, ..., 2],
        [first_spec_hid_dim_cat3, second_spec_hid_dim_cat3, ..., 4]].

    n_hidden_and_output_num: List[int]
        A list contains task-specific hidden layers’ sizes of numerical response variables and the dimension of output,
        which is the number of numerical response variables in data_y.
        For example, if we have three numerical response variables in y, then
        n_hidden_and_output_num = [first_spec_hid_dim, second_spec_hid_dim, ..., 3]

    :Methods:
    """

    def __init__(
        self,
        input_and_n_hidden_shared: List[int],
        n_hidden_and_output_cat_combo: List[List],
        n_hidden_and_output_num: List[int],
    ):
        super(MultitaskNet, self).__init__()
        # shared layer
        self.shared_layer = nn.ModuleList()
        for i in range(len(input_and_n_hidden_shared) - 1):
            self.shared_layer.append(
                nn.Linear(
                    input_and_n_hidden_shared[i], input_and_n_hidden_shared[i + 1]
                )
            )

        # task-specific layer list for categorical tasks
        self.cat_layer_combo = nn.ModuleList()
        for n_hidden_and_output_cat in n_hidden_and_output_cat_combo:
            cat_layer = nn.ModuleList()
            # input for task-specific layer is the dim of last shared hidden layer
            curr_input = input_and_n_hidden_shared[-1]
            for i in range(len(n_hidden_and_output_cat)):
                cat_layer.append(nn.Linear(curr_input, n_hidden_and_output_cat[i]))
                curr_input = n_hidden_and_output_cat[i]
            self.cat_layer_combo.append(cat_layer)

        # task-specific layer list for numerical tasks
        self.num_layer = nn.ModuleList()
        # input for task-specific layer is the dim of last shared hidden layer
        curr_input = input_and_n_hidden_shared[-1]
        for i in range(len(n_hidden_and_output_num)):
            self.num_layer.append(nn.Linear(curr_input, n_hidden_and_output_num[i]))
            curr_input = n_hidden_and_output_num[i]

    def forward(self, x):
        """Forward function in neural networks."""
        # shared layers
        for layer in self.shared_layer:
            x = layer(x)
            x = nn.functional.relu(x)

        # categorical part
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

        # numerical part
        if not self.num_layer:
            y_pred_num = None
        else:
            y_pred_num = self.num_layer[0](x)
            for i in range(1, len(self.num_layer)):
                # the last layer has no activation function
                y_pred_num = nn.functional.relu(y_pred_num)
                y_pred_num = self.num_layer[i](y_pred_num)

        return y_pred_cat_combo, y_pred_num
