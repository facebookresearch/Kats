# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kats.consts import TimeSeriesData
from kats.metrics import metrics
from kats.models.globalmodel.data_processor import GMBatch, GMDataLoader
from kats.models.globalmodel.utils import (
    AdjustedPinballLoss,
    DilatedRNNStack,
    GMParam,
    gmparam_from_string,
    PinballLoss,
)
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam

NoneT = torch.FloatTensor([-1e38])


class GMModel:
    """The class for building single global model.

    This class provides functions including train, predict, and evaluate and save_model.

    Attributes:
        params: A :class:`kats.models.globalmodel.utils.GMParam` object for building the global model.

    Sample Usage:
        >>> # create an object of global model with param
        >>> gmm = GMModel(params)
        >>> # train a model and get training info (e.g., training/validation losses)
        >>> training_info = gmm.train(train_TSs, valid_TSs)
        >>> # make prediction
        >>> gmm.predict(train_TSs)
        >>> # save model
        >>> gmm.save_model("global_model.pickle")
        >>> # Evalute model performance on a given dataset.
        >>> evals = gmm.evalute(test_train, test_test)
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self, params: GMParam):

        if not isinstance(params, GMParam):
            msg = f"params should be a GMParam object but receives {type(params)}."
            logging.error(msg)
            raise ValueError(msg)

        self.params = params
        self.debug = False

        # pyre-fixme[4]: Attribute must be annotated.
        self.rnn = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.decoder = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.encoder = None

    def _reset_nn_states(self) -> None:
        if self.params.model_type == "rnn":
            self.rnn.reset_state()
        elif self.params.model_type == "s2s":
            self.encoder.reset_state()
            self.decoder.reset_state()
        else:
            msg = "Not implemented."
            raise ValueError(msg)

    def _initiate_nn(self) -> None:
        if self.params.model_type == "rnn":
            if self.rnn is None:
                self.build_rnn()
        elif self.params.model_type == "s2s":
            if self.decoder is None or self.encoder is None:
                self.build_s2s()
        else:
            msg = "Not implemented."
            raise ValueError(msg)

    # pyre-fixme[24]: Generic type `Generator` expects 3 type parameters.
    # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
    #  `typing.List` to avoid runtime subscripting errors.
    def _get_nn_parameters(self) -> Union[Generator, List]:
        if self.params.model_type == "rnn":
            return self.rnn.parameters()
        elif self.params.model_type == "s2s":
            return list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            msg = "Not implemented."
            raise ValueError(msg)

    def _set_nn_status(self, mode: str) -> None:
        if self.params.model_type == "rnn":
            if mode == "train":
                self.rnn.train()
            elif mode == "test":
                self.rnn.eval()
        elif self.params.model_type == "s2s":
            if mode == "train":
                self.encoder.train()
                self.decoder.train()
            elif mode == "test":
                self.encoder.eval()
                self.decoder.eval()
        else:
            msg = "Not implemented."
            raise ValueError(msg)

    def build_rnn(self) -> None:
        """Helper function for building RNN."""

        params = self.params
        feature_size = (
            params.gmfeature.get_feature_size(params.input_window)
            if params.gmfeature
            else 0
        )
        input_size = (
            params.input_window + feature_size + 2
        )  # two additional positions for step_num_encode and step_size_encode
        len_quantile = (
            0 if params.quantile is None else len(params.quantile)
        )  # len(params.quantile) if params.quantile is not None else 0
        output_size = (
            params.fcst_window * len_quantile + 1
        )  # one additional position for level smoothing parameter
        if params.seasonality > 1:
            input_size += 2 * params.seasonality
            output_size += (
                1  # one additional position for seasonality smoothing parameter
            )
        # ensure data type for jit
        input_size = int(input_size)
        output_size = int(output_size)
        rnn = DilatedRNNStack(
            params.nn_structure,
            params.cell_name,
            input_size,
            params.state_size,
            output_size,
            params.h_size,
            params.jit,
        )
        self.rnn = rnn
        return

    def build_loss_function(self) -> Union[AdjustedPinballLoss, PinballLoss, _Loss]:
        """Helper function for building loss function.

        Returns:
            A :class:`kats.models.globalmodel.utils.PinballLoss` or a :class:`kats.models.globalmodel.utils.AdjustedPinballLoss` object representing the loss function.
        """

        if self.params.loss_function == "pinball":
            loss_func = PinballLoss(
                quantile=torch.tensor(
                    self.params.training_quantile, dtype=torch.get_default_dtype()
                ),
                weight=torch.tensor(
                    self.params.quantile_weight, dtype=torch.get_default_dtype()
                ),
                reduction="mean",
            )
        elif self.params.loss_function == "adjustedpinball":
            loss_func = AdjustedPinballLoss(
                quantile=torch.tensor(
                    self.params.training_quantile, dtype=torch.get_default_dtype()
                ),
                weight=torch.tensor(
                    self.params.quantile_weight, dtype=torch.get_default_dtype()
                ),
                reduction="mean",
                input_log=True,
            )
        else:
            msg = f"Loss function {self.params.loss_function} cannot be recognized."
            raise ValueError(msg)
        return loss_func

    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def build_validation_function(self) -> Callable:
        """Helper function for building validation function.

        Returns:
            A callable object representing the validation function.
        """

        valid_metric_names = self.params.validation_metric
        quantile = np.array(self.params.quantile[1:])

        # pyre-fixme[53]: Captured variable `quantile` is not annotated.
        # pyre-fixme[53]: Captured variable `valid_metric_names` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def valid_func(fcst, target):
            ans = {}
            if len(fcst.shape) == 1:
                fcst = fcst.reshape(1, -1)
            if len(target.shape) == 1:
                target = target.reshape(1, -1)
            d = target.shape[1]
            for name in valid_metric_names:
                if name == "smape":
                    ans["smape"] = metrics.smape(target, fcst[:, :d])
                elif name == "sbias":
                    ans["sbias"] = metrics.sbias(target, fcst[:, :d])
                elif name == "exceed" and len(quantile) > 1:
                    tmp_val = metrics.mult_exceed(target, fcst[:, d:], quantile)
                    tmp_dict = {
                        f"exceed_{quantile[i]}": tmp_val[i]
                        for i in range(len(quantile))
                    }
                    ans.update(tmp_dict)
            return ans

        return valid_func

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def build_optimizer(self) -> Tuple[Any, Dict[str, Any]]:
        """Helper function for building optimizer.

        Returns:
            A tuple (optimizer, optimizer_param) where optimizer is a torch optimizer; and optimizer_param is a dictionary of optimizer parameters.
        """

        name = self.params.optimizer["name"]
        optimizer_param = (
            self.params.optimizer["params"] if "params" in self.params.optimizer else {}
        )
        if name.lower() == "adam":
            optimizer = Adam

        else:
            msg = f"Optimize method {name} is not available."
            logging.error(msg)
            raise ValueError(msg)
        return optimizer, optimizer_param

    @staticmethod
    @torch.jit.script
    def _process(
        prev_idx: int,
        cur_idx: int,
        x: Tensor,
        x_lt: List[Tensor],
        levels: List[Tensor],
        seasonality: List[Tensor],
        input_window: int,
        fcst_window: int,
        period: int,
        level_sm: Tensor,
        season_sm: Tensor,
        fcst_fill: bool = False,
        fcst_tensor: Tensor = NoneT,
    ) -> Tuple[
        Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], Tensor, Tensor, Tensor
    ]:
        """
        Helper function for on-the-fly preprocessing, including de-trending, de-seasonality and calculating seasonality.

        """
        batch_size = x.size()[0]
        if prev_idx == 0:
            if period <= 1:
                levels.append(x[:, 0].view(batch_size, 1))
            else:
                levels.append(x[:, 0].view(batch_size, 1) / seasonality[0])
            x_lt.append(x[:, 0].view(batch_size, 1))
            prev_idx = 1

        for idx in range(prev_idx, cur_idx):
            new_x0 = x[:, idx].view(batch_size, 1)
            nans = torch.isnan(new_x0)
            if torch.any(nans):
                new_x = new_x0.clone()
                if fcst_fill:
                    # fill-in NaNs with fcst_tensor.
                    new_x[nans] = fcst_tensor[:, idx - prev_idx].view(batch_size, 1)[
                        nans
                    ]
                else:
                    # fill-in NaNs with levels and seasonalities.
                    new_x[nans] = levels[idx - 1][nans]
                    if period > 1:
                        new_x[nans] *= seasonality[idx][nans]
            else:
                new_x = new_x0
            x_lt.append(new_x)

            if period == 1:
                levels.append(level_sm * new_x + (1 - level_sm) * levels[idx - 1])
            else:
                new_level = (
                    level_sm * new_x / seasonality[idx]
                    + (1 - level_sm) * levels[idx - 1]
                )
                levels.append(new_level)
                new_season = (
                    season_sm * new_x / new_level + (1 - season_sm) * seasonality[idx]
                )
                seasonality.append(new_season)
        idx = cur_idx
        anchor_level = levels[idx - 1].view(batch_size, 1)
        xi_t = torch.cat(x_lt[idx - input_window : idx], dim=1)

        x_t = xi_t / anchor_level

        if period > 1:

            input_season = torch.cat(seasonality[idx - input_window : idx], dim=1)
            x_t = x_t / input_season

            next_season = torch.cat(seasonality[idx : idx + period], dim=1)
            prev_season = torch.cat(seasonality[idx - period : idx], dim=1)
            diff_season = next_season - prev_season

            fcst_season = next_season.repeat(1, fcst_window // period + 1)[
                :, :fcst_window
            ]
        else:
            # just for complie, would not be used.
            fcst_season, diff_season, next_season = (
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
            )
        return (
            x_t,
            anchor_level,
            x_lt,
            levels,
            seasonality,
            next_season,
            diff_season,
            fcst_season,
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _valid_tensor(self, input_t, tag):
        """
        Helper function for debug use.
        """
        if self.debug:
            if torch.isnan(input_t).sum() > 0:
                msg = f"{tag} tensor contains NaN (tensor = {input_t})"
                logging.error(msg)
            if torch.isinf(input_t).sum() > 0:
                msg = f"{tag} tensor contains inf (tensor = {input_t})"
                logging.error(msg)

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def train(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid_TSs: Optional[
            Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        ] = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_train_TSs: Optional[
            Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        ] = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_valid_TSs: Optional[
            Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        ] = None,
        fcst_monitor: bool = False,
        debug: bool = False,
    ) -> Dict[str, List[Any]]:
        """Train the global model.

        Args:
            train_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for training.
            valid_TSs: Optional; A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for validation.
            test_train_TSs: Optional; A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the warming-ups for evaluation.
            test_valid_TSs: Optional; A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the evaluation parts for evaluation.
            fcst_monitor: Optional; A boolean representing whether or not to return the forecasts during training for the validation data. Default is False.
            debug: Optional; A boolean representing whether or not to use helper function during training. Default is False.

        Returns:
            A dictionary storing training information. training_info["train_loss_monitor"] is a list of dictionaries storing the averaged traning losses of each epoch.
            training_info["valid_loss_monitor"] is a list of dictionaries storing the averaged validation losses of each epoch. training_info["valid_fcst_monitor"] is a list of dictionaries storing the forecasts generated by global model for validation time series during training.
        """

        self._initiate_nn()
        self.debug = debug
        self._set_nn_status("train")
        loss_func = self.build_loss_function()
        valid_loss_func = self.build_validation_function()
        optimizer, optimizer_param = self.build_optimizer()
        dl = GMDataLoader(train_TSs)
        params = self.params
        len_quantile = len(params.quantile)

        train_loss_monitor = []
        valid_loss_monitor = []
        train_loss_val = []

        is_test = test_train_TSs is not None
        test_eval = []
        valid_fcst_monitor = []
        lr = params.learning_rate[0]
        batch_size = params.batch_size[0]

        for epoch in range(params.epoch_num):
            train_loss_track = 0
            tmp_train_loss_monitor = []
            tmp_valid_loss_monitor = []

            if epoch in params.batch_size:
                batch_size = params.batch_size[epoch]
            if epoch in params.learning_rate:
                lr = params.learning_rate[epoch]

            trainer = optimizer(
                params=self._get_nn_parameters(), lr=lr, **optimizer_param
            )

            logging.info(
                f"Training for epoch {epoch} with batch_size = {batch_size} and learning_rate = {lr}."
            )

            for _ in range(params.epoch_size):
                self._reset_nn_states()

                # fetch batch_ids
                batch_ids = dl.get_batch(batch_size)

                # batch TSs
                batch = GMBatch(params, batch_ids, train_TSs, valid_TSs, mode="train")

                train_loss, train_res, valid_res, fcst_store = self._single_pass(
                    params,
                    batch,
                    training_mode=True,
                    fcst_monitor=fcst_monitor,
                    loss_func=loss_func,
                    valid_loss_func=valid_loss_func,
                )
                # update RNN
                trainer.zero_grad()
                avg_train_loss = sum(train_loss) / len_quantile
                # pyre-fixme[16]: `float` has no attribute `backward`.
                avg_train_loss.backward()
                trainer.step()
                # pyre-fixme[16]: `float` has no attribute `detach`.
                train_loss_track += avg_train_loss.detach().numpy()

                # record training_info

                tmp_train_loss_monitor.extend(train_res)
                tmp_valid_loss_monitor.extend(valid_res)
                if fcst_monitor:
                    # pyre-fixme[6]: Expected `ndarray` for 2nd param but got `int`.
                    fcst_store["epoch"] = epoch
                    valid_fcst_monitor.append(fcst_store)

            train_loss_monitor.append(np.average(tmp_train_loss_monitor))

            valid_res = pd.DataFrame(tmp_valid_loss_monitor).mean(skipna=True).to_dict()
            valid_res["epoch"] = epoch
            valid_loss_monitor.append(valid_res)

            train_loss_val.append(train_loss_track / params.epoch_size)

            if is_test:
                # pyre-fixme[6]: Expected `Union[Dict[typing.Any, TimeSeriesData],
                #  List[TimeSeriesData]]` for 1st param but got `Union[None,
                #  Dict[typing.Any, TimeSeriesData], List[TimeSeriesData]]`.
                tr = self.evaluate(test_train_TSs, test_valid_TSs)
                tr["epoch"] = epoch
                test_eval.append(tr)

            logging.info(
                f"Successfully finished training for epoch {epoch} with train_loss {train_loss_val[-1]}"
            )

        training_info = {
            "train_loss_monitor": train_loss_monitor,
            "valid_loss_monitor": valid_loss_monitor,
            "valid_fcst_monitor": valid_fcst_monitor,
            "train_loss_val": train_loss_val,
        }

        if is_test:
            training_info["test_info"] = test_eval

        self._reset_nn_states()
        return training_info

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _format_fcst(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        ids: List[Any],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        fcst_store: Dict[Any, List[np.ndarray]],
        steps: int,
        first_time: np.ndarray,
    ) -> Dict[Any, pd.DataFrame]:
        """
        Helper function for transforming raw forecast data into pd.DataFrame.

        :Parameters:
        ids: List[Any]
            The list of time series ids that are associated with the forecasts.
        fcst_store: Dict[Any, List[np.ndarray]]
            Raw forecasts generated by global model.
        steps: int
            The step of forecasts.
        first_time:
            The list of the first timestamps of the forecasts.

        :Returns: Dict[str, pd.DataFrame]
        A dictionary of forecasts in format of pd.DataFrame.

        """
        ans = {}
        quantile = self.params.quantile
        n = len(ids)
        n_quantile = len(quantile)
        cols = [f"fcst_quantile_{q}" for q in quantile]
        fcst = np.concatenate(
            [t.reshape(n, n_quantile, -1) for t in fcst_store["fcst"]], axis=2
        )
        if "actual" in fcst_store:
            actual = np.column_stack(fcst_store["actual"])[:, :steps]

        for i, idx in enumerate(ids):

            df = pd.DataFrame(
                fcst[i].transpose()[
                    :steps,
                ],
                columns=cols,
            )
            df["time"] = pd.date_range(
                first_time[i], freq=self.params.freq, periods=steps
            )
            if "actual" in fcst_store:
                df["actual"] = actual[i]
            ans[idx] = df
        return ans

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def predict(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
        steps: int,
        test_batch_size: int = 500,
        raw: bool = False,
    ) -> Dict[Any, Union[pd.DataFrame, List[np.ndarray]]]:
        """Generate forecasts for target time series.

        Args:
            test_TSs: A single, a list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for testing warming-ups.
            steps: A positive integer representing the forecast steps.
            test_batch_size: Optional; An integer representing the batch size for testing. Default is 500.
            raw: Optional; A boolean representing whether or not to return raw forecasts (i.e., `numpy.ndarray` objects). If False, the forecasts are `pandas.DataFrame` objects. Default is False.

        Returns:
            A dictionary of forecasts, whose keys are the ids for time series, and the values are the corresponding forecasts.
        """
        self._set_nn_status("test")
        if isinstance(test_TSs, TimeSeriesData):
            test_TSs = [test_TSs]
        elif isinstance(test_TSs, dict) or isinstance(test_TSs, list):
            pass
        else:
            msg = f"predict function only accepts a TimeSeriesData object, a dictionary or a list of TimeSeriesData objects, but receives {type(test_TSs)}"

        # calculate fcst step num.
        fcst_step_num = steps // self.params.fcst_window + int(
            steps % self.params.fcst_window != 0
        )
        fcst_params = self.params.copy()
        # pyre-fixme[16]: `object` has no attribute `fcst_step_num`.
        fcst_params.fcst_step_num = fcst_step_num
        if not isinstance(test_batch_size, int) or test_batch_size <= 0:
            msg = f"test_batch_size should be a positive integer but receives {test_batch_size}."
            logging.error(msg)
            raise ValueError(msg)

        n = len(test_TSs)
        batch_size = 500  # here we set the maximum batch_size for prediction is 1000.
        m = n // batch_size + (n % batch_size != 0)
        dl = GMDataLoader(test_TSs)
        self._set_nn_status("test")
        fcst_collects = {}
        for i in range(m):

            self._reset_nn_states()

            ids = dl.get_batch(batch_size)
            # pyre-fixme
            batch = GMBatch(fcst_params, batch_ids=ids, train_TSs=test_TSs, mode="test")

            _, _, _, fcst_store = self._single_pass(
                # pyre-fixme
                fcst_params,
                batch,
                training_mode=False,
                fcst_monitor=False,
                loss_func=None,
                valid_loss_func=None,
            )

            if self.params.model_type == "rnn":
                # Adjust predictive interval.
                fcst_store["fcst"] = self._adjust_pi(fcst_store["fcst"])

            if not raw:
                fcst_store = self._format_fcst(
                    ids, fcst_store, steps, batch.time[:, batch.train_length]
                )
                fcst_collects.update(fcst_store)
            else:
                tmp = {
                    ids[i]: [t[i] for t in fcst_store["fcst"]] for i in range(len(ids))
                }
                fcst_collects.update(tmp)

        return fcst_collects

    def save_model(self, file_name: str) -> None:
        """Save global model.

        Args:
            file_name: A string representing the file address and file name.
        """
        self._reset_nn_states()
        info = {
            "gmparam_string": self.params.to_string(),
            "state_dict": self.rnn.state_dict() if self.rnn is not None else None,
            "encoder_state_dict": self.encoder.state_dict()
            if self.encoder is not None
            else None,
            "decoder_state_dict": self.decoder.state_dict()
            if self.decoder is not None
            else None,
        }
        with open(file_name, "wb") as f:
            joblib.dump(info, f)
        logging.info(f"Successfully save model to file {file_name}.")

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _single_pass_rnn(
        self,
        rnn: nn.Module,
        params: GMParam,
        batch: GMBatch,
        training_mode: bool = True,
        fcst_monitor: bool = False,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_func: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        valid_loss_func: Optional[Callable] = None,
    ) -> Tuple[List[Any], List[Any], List[Any], Dict[Any, List[Any]]]:
        """
        Helper function for passing a batch into the seasonal RNN.
        """
        tmp_batch_size = batch.batch_size
        tdtype = torch.get_default_dtype()
        len_quantile = len(params.quantile)

        # initialize data
        train_loss = []
        train_res = []
        valid_res = []
        fcst_store = collections.defaultdict(list)
        level_sm = torch.ones(tmp_batch_size, 1) * params.init_smoothing_params[0]
        season_sm = (
            torch.ones(tmp_batch_size, 1) * params.init_smoothing_params[1]
            if params.init_smoothing_params is not None
            else NoneT
        )

        period = params.seasonality
        x_lt, levels, seasonality = [], [], []
        if params.seasonality > 1:
            seasonality = [
                batch.init_seasonality[:, i].view(tmp_batch_size, 1)
                for i in range(period)
            ]
            seasonality.append(batch.init_seasonality[:, 0].view(tmp_batch_size, 1))
        else:
            seasonality = [Tensor([1.0])]

        total_step_num = len(batch.indices)
        # moving step in terms of indices (would be adjusted for validation)
        step_delta = max(
            1,
            min(
                params.max_step_delta,
                (total_step_num - params.validation_step_num)
                // params.soft_max_training_step_num,
            ),
        )

        # define first test idx
        first_valid_idx = (
            batch.valid_indices[0]
            if len(batch.valid_indices) > 0
            else batch.train_indices[-1] + 1
        )
        last_train_step = len(batch.train_indices) - 1

        cur_step = 0
        prev_idx = 0

        fcst_col_idx = int(params.seasonality > 1) + 1
        fcst_fill = False
        fcst_tensor = NoneT
        while cur_step < total_step_num:
            cur_idx = batch.indices[cur_step]
            is_valid = cur_idx >= first_valid_idx
            step_size_encode = (
                torch.ones((tmp_batch_size, 1), dtype=tdtype)
                * np.log(cur_idx - prev_idx)
                / 2.0
            )
            step_num_encode = torch.ones((tmp_batch_size, 1), dtype=tdtype) * np.log(
                cur_step + 1
            )

            tmp_cur_training_loss = torch.tensor(0, dtype=torch.get_default_dtype())

            (
                x_t,
                anchor_level,
                x_lt,
                levels,
                seasonality,
                next_season,
                diff_season,
                fcst_season,
            ) = self._process(
                prev_idx,
                cur_idx,
                batch.x,
                x_lt,
                levels,
                seasonality,
                params.input_window,
                params.fcst_window,
                period,
                level_sm,
                season_sm,
                fcst_fill,
                fcst_tensor,
            )

            if params.gmfeature is not None:
                features = batch.get_features(cur_idx - params.input_window, cur_idx)
                input_t = torch.cat(
                    [torch.log(x_t), features, step_size_encode, step_num_encode], dim=1
                )
            else:
                input_t = torch.cat(
                    [torch.log(x_t), step_size_encode, step_num_encode], dim=1
                )

            if period > 1:
                input_t = torch.cat([input_t, next_season - 1.0, diff_season], dim=1)

            self._valid_tensor(input_t, "input")

            fcst_all = rnn(input_t)

            self._valid_tensor(fcst_all, "fcst")

            level_sm = torch.sigmoid(fcst_all[:, 0]).view(tmp_batch_size, 1)

            fcst = fcst_all[:, fcst_col_idx:] + torch.log(anchor_level)

            if period > 1:
                season_sm = torch.sigmoid(fcst_all[:, 1]).view(tmp_batch_size, 1)
                fcst = fcst + torch.log(fcst_season).repeat(1, len_quantile)

            if (is_valid) or training_mode:
                tmp_cur_training_loss, valid_res, fcst_store = self._calculate(
                    cur_idx,
                    params,
                    batch,
                    fcst,
                    loss_func,
                    valid_loss_func,
                    is_valid,
                    training_mode,
                    valid_res,
                    fcst_monitor,
                    fcst_store,
                )

                if not isinstance(tmp_cur_training_loss, float):
                    train_loss.append(tmp_cur_training_loss)
                    train_res.append(tmp_cur_training_loss.detach().numpy())

            cur_step += step_delta
            if (
                cur_step >= last_train_step
                and (cur_step - step_delta) < last_train_step
            ):
                cur_step = last_train_step
                step_delta = 1
            prev_idx = cur_idx

        return train_loss, train_res, valid_res, fcst_store

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _single_pass(
        self,
        params: GMParam,
        batch: GMBatch,
        training_mode: bool = True,
        fcst_monitor: bool = False,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_func: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        valid_loss_func: Optional[Callable] = None,
    ) -> Tuple[List[Any], List[Any], List[Any], Dict[Any, List[Any]]]:
        if params.model_type == "rnn":
            return self._single_pass_rnn(
                self.rnn,
                params,
                batch,
                training_mode,
                fcst_monitor,
                loss_func,
                valid_loss_func,
            )
        elif params.model_type == "s2s":
            return self._single_pass_s2s(
                self.encoder,
                self.decoder,
                params,
                batch,
                training_mode,
                fcst_monitor,
                loss_func,
                valid_loss_func,
            )
        else:
            msg = "Not implemented."
            raise ValueError(msg)

    def evaluate(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_train_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_valid_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
    ) -> pd.DataFrame:
        """Evaluate the global model performance on a dataset.

        Args:
            test_train_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for warming-ups.
            test_valid_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for evaluation.

        Returns:
            A `pandas.DataFrame` object representing the evaluation results.
        """

        if type(test_train_TSs) != type(test_valid_TSs):
            msg = (
                "The data type of test_train_TSs and test_valid_TSs should be the same."
            )
            logging.error(msg)
            raise ValueError(msg)

        if isinstance(test_train_TSs, TimeSeriesData):
            test_train_TSs = [test_train_TSs]
            # pyre-fixme [9]
            test_valid_TSs = [test_valid_TSs]

        if len(test_train_TSs) != len(test_valid_TSs):
            msg = "test_train_TSs and test_valid_TSs should be of the same length."
            logging.error(msg)
            raise ValueError(msg)
        keys = (
            test_train_TSs.keys()
            if isinstance(test_train_TSs, dict)
            else range(len(test_train_TSs))
        )
        if len(keys) == 0:
            msg = "The input collection of time series should not be empty."
            logging.error(msg)
            raise ValueError(msg)

        steps = np.max([len(test_valid_TSs[t]) for t in keys])

        fcst = self.predict(test_train_TSs, steps=steps, raw=True)
        logging.info(
            f"Successfully generate forecasts for all test time series with length {steps}."
        )
        eval_func = self.build_validation_function()
        fcst_window = self.params.fcst_window
        ans = []
        keys = (
            test_train_TSs.keys()
            if isinstance(test_train_TSs, dict)
            else range(len(test_train_TSs))
        )
        for k in keys:
            tmp = test_valid_TSs[k].value.values
            tmp_step = len(tmp) // fcst_window + int(len(tmp) % fcst_window != 0)
            tmp_fcst_length = tmp_step * fcst_window
            actuals = np.full(tmp_fcst_length, np.nan, np.float)
            actuals[: len(tmp)] = tmp
            for j in range(tmp_step):
                tmp_actuals = actuals[j * fcst_window : (j + 1) * fcst_window]
                tmp = eval_func(fcst[k][j], tmp_actuals)
                tmp["step"] = j
                tmp["idx"] = k
                ans.append(tmp)
        return pd.DataFrame(ans)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _adjust_pi(self, fcsts):
        # only 1 step fcst and hence no need for adjustment
        if len(fcsts) == 1:
            return fcsts
        fw = self.params.fcst_window
        len_quantile = len(self.params.quantile)
        for i in range(1, len(fcsts)):
            for j in range(1, len_quantile):
                diff = fcsts[i - 1][:, (j + 1) * fw - 1] - fcsts[i][:, j * fw]
                fcsts[i][:, (j * fw) : (j + 1) * fw] += diff[:, None]  # row-wise add
        return fcsts

    def build_s2s(
        self,
    ) -> None:
        params = self.params
        encoder_feature_size = (
            params.gmfeature.get_feature_size(params.input_window)
            if params.gmfeature
            else 0
        )
        # two additional positions for step_num_encode and step_size_encode
        encoder_input_size = int(params.input_window + encoder_feature_size + 2)
        len_quantile = len(params.quantile) if params.quantile is not None else 0

        self.encoder = DilatedRNNStack(
            nn_structure=params.nn_structure,
            cell_name=params.cell_name,
            input_size=encoder_input_size,
            state_size=params.state_size,
            output_size=None,
            h_size=params.h_size,
            jit=params.jit,
        )
        decoder_feature_size = (
            params.gmfeature.get_feature_size(params.fcst_window)
            if params.gmfeature
            else 0
        )
        # one additional position for step_num_encode
        decoder_input_size = int(decoder_feature_size + 1 + self.encoder.out_size)
        decoder_output_size = int(len_quantile * params.fcst_window)

        self.decoder = DilatedRNNStack(
            nn_structure=params.decoder_nn_structure,
            cell_name=params.cell_name,
            input_size=decoder_input_size,
            state_size=params.state_size,
            output_size=decoder_output_size,
            h_size=params.h_size,
            jit=params.jit,
        )

        return

    @staticmethod
    @torch.jit.script
    def _process_s2s(
        prev_idx: int,
        cur_idx: int,
        x: Tensor,
        x_lt: List[Tensor],
        period: int,
        input_window: int,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Helper function for on-the-fly preprocessing, including de-trending, de-seasonality and calculating seasonality.

        """
        batch_size = x.size()[0]
        for idx in range(prev_idx, cur_idx):
            new_x0 = x[:, idx].view(batch_size, 1)
            nans = torch.isnan(new_x0)
            if torch.any(nans):
                new_x = new_x0.clone()
                new_x[nans] = x_lt[max(0, idx - period)][nans]
            else:
                new_x = new_x0
            x_lt.append(new_x)

        xi_t = torch.cat(x_lt[cur_idx - input_window : cur_idx], dim=1)
        anchor_level = torch.median(xi_t[:, -period:], dim=1)[0].view(-1, 1)

        x_t = xi_t / anchor_level

        return (
            x_t,
            anchor_level,
            x_lt,
        )

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _single_pass_s2s(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        params: GMParam,
        batch: GMBatch,
        training_mode: bool = True,
        fcst_monitor: bool = False,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_func: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        valid_loss_func: Optional[Callable] = None,
    ) -> Tuple[List[Any], List[Any], List[Any], Dict[Any, List[Any]]]:
        """
        Helper function for passing a batch into S2S NN.
        """
        tmp_batch_size = batch.batch_size
        tdtype = torch.get_default_dtype()

        # initialize data
        train_loss = []
        train_res = []
        valid_res = []
        fcst_store = collections.defaultdict(list)

        period = params.seasonality

        x_lt = []

        total_step_num = len(batch.indices)
        # moving step in terms of indices (would be adjusted for validation)
        step_delta = max(
            1,
            min(
                params.max_step_delta,
                total_step_num // params.soft_max_training_step_num,
            ),
        )
        # define first test idx
        first_valid_idx = (
            batch.valid_indices[0]
            if len(batch.valid_indices) > 0
            else batch.train_indices[-1] + 1
        )

        first_valid_step = len(batch.train_indices)

        cur_step = 0
        prev_idx = 0

        while cur_step < total_step_num:

            cur_idx = batch.indices[cur_step]
            is_valid = cur_idx >= first_valid_idx

            is_training = (training_mode) and (not is_valid)
            cur_training_loss = torch.tensor(0.0, dtype=torch.get_default_dtype())

            step_size_encode = (
                torch.ones((tmp_batch_size, 1), dtype=tdtype)
                * np.log(cur_idx - prev_idx)
                / 2.0
            )
            step_num_encode = torch.ones((tmp_batch_size, 1), dtype=tdtype) * np.log(
                cur_step + 1
            )

            (x_t, anchor_level, x_lt,) = self._process_s2s(
                prev_idx, cur_idx, batch.x, x_lt, period, params.input_window
            )

            if params.gmfeature is not None:
                features = batch.get_features(cur_idx - params.input_window, cur_idx)
                input_t = torch.cat(
                    [torch.log(x_t), features, step_size_encode, step_num_encode], dim=1
                )
            else:
                input_t = torch.cat(
                    [torch.log(x_t), step_size_encode, step_num_encode], dim=1
                )

            self._valid_tensor(input_t, "input")

            # get encoded tensor
            tmp_encode = encoder(input_t)

            self._valid_tensor(tmp_encode, "encoder_output")

            if training_mode or is_valid:

                # pyre-fixme
                encoder.prepare_decoder(decoder)
                encoder_step = (
                    batch.training_encoder_step_num
                    if not is_valid
                    else batch.test_encoder_step_num
                )
                for decoder_step_num in range(encoder_step):
                    decoder_step_num_encode = torch.ones(
                        (tmp_batch_size, 1), dtype=tdtype
                    ) * np.log(decoder_step_num + 1)
                    fcst_cur_idx = cur_idx + decoder_step_num * params.fcst_window
                    if params.gmfeature is not None:
                        fcst_features = batch.get_features(
                            fcst_cur_idx, fcst_cur_idx + params.fcst_window
                        )
                        input_fcst = torch.cat(
                            [tmp_encode, fcst_features, decoder_step_num_encode], dim=1
                        )
                    else:
                        input_fcst = torch.cat(
                            [tmp_encode, decoder_step_num_encode], dim=1
                        )

                    fcst = decoder(input_fcst) + torch.log(anchor_level)

                    tmp_cur_training_loss, valid_res, fcst_store = self._calculate(
                        fcst_cur_idx,
                        params,
                        batch,
                        fcst,
                        loss_func,
                        valid_loss_func,
                        is_valid,
                        training_mode,
                        valid_res,
                        fcst_monitor,
                        fcst_store,
                    )

                    cur_training_loss = (
                        cur_training_loss + tmp_cur_training_loss
                        if tmp_cur_training_loss != 0
                        else cur_training_loss
                    )

                if is_training:
                    cur_training_loss /= encoder_step
                    train_loss.append(cur_training_loss)
                    train_res.append(cur_training_loss.detach().numpy())

            if is_valid:
                # once we arrive at validation/forecasting stage, we only need to go through it once.
                break

            prev_idx = cur_idx
            cur_step += step_delta
            if cur_step > first_valid_step:  # adjust the step to first_valid_step
                cur_step = first_valid_step
        return train_loss, train_res, valid_res, fcst_store

    def _calculate(
        self,
        cur_idx: int,
        params: GMParam,
        batch: GMBatch,
        fcst: Tensor,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_func: Optional[Callable],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        valid_loss_func: Optional[Callable],
        is_valid: bool,
        training_mode: bool,
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        valid_res: List,
        fcst_monitor: bool,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        fcst_store: Dict,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
    ) -> Tuple[Union[float, Tensor], List, Dict]:
        """Calculate and store training or validation losses and forecasts."""

        cur_training_loss = 0.0
        if (not is_valid) and training_mode:
            actuals = batch.x[:, cur_idx : cur_idx + params.fcst_window].clone().log()
            # pyre-fixme
            cur_training_loss = torch.sum(loss_func(fcst, actuals))
        elif is_valid:
            # restoring to original scale
            fcst = (torch.exp(fcst).detach() - batch.offset).numpy()
            if training_mode:  # get evaluated performance on validation set
                actuals = (
                    batch.x[:, cur_idx : cur_idx + params.fcst_window].clone()
                    - batch.offset
                ).numpy()
                # pyre-fixme
                valid_res.append(valid_loss_func(fcst, actuals))

                if fcst_monitor:
                    fcst_store["fcst"].append(fcst)
                    fcst_store["actual"].append(actuals)
            else:  # get forecasts
                fcst_store["fcst"].append(fcst)
        return cur_training_loss, valid_res, fcst_store


def load_gmmodel_from_file(file_name: str) -> GMModel:
    """Function for loading global model from a binary file.

    Args:
        file_name: A string representing the file path and file name to be loaded.

    Returns:
        A :class:`GMModel` object representing the loaded model.
    """

    try:
        with open(file_name, "rb") as f:
            info = joblib.load(f)
        gmparam = gmparam_from_string(info["gmparam_string"])
        gmmodel = GMModel(gmparam)
        if gmparam.model_type == "rnn":
            gmmodel.build_rnn()
            gmmodel.rnn.load_state_dict(info["state_dict"])
        elif gmparam.model_type == "s2s":
            gmmodel.build_s2s()
            gmmodel.encoder.load_state_dict(info["encoder_state_dict"])
            gmmodel.decoder.load_state_dict(info["decoder_state_dict"])

        logging.info(f"Successfully load global model from {file_name}.")
    except Exception as e:
        msg = f"Fail to load global model from {file_name} with exception {e}."
        logging.error(msg)
        raise ValueError(msg)
    return gmmodel
