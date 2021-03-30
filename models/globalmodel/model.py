import collections
import logging
from typing import List, Optional, Union, Callable, Any, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.models.globalmodel.data_processor import (
    GMParam,
    GMBatch,
    GMDataLoader,
)

from infrastrategy.kats.models.globalmodel.utils import (
    GMParam,
    DilatedRNNStack,
    PinballLoss,
    AdjustedPinballLoss,
    calc_smape,
    calc_exceed,
    calc_sbias,
)
from torch import Tensor
from torch.optim import Adam

NoneT = torch.FloatTensor([-1e38])


class GMModel:
    """
    The class for global model.

    :Parameters:
    params: GMParam
        A GMParam object for global model.
    load_model: bool
        Whether to load a pretrained model or not.
    :Example:
    >>> from infrastrategy.kats.models.globalmodel.model import GMModel
    >>> # create an object of global model with param
    >>> gmm = GMModel(params)
    >>> # train a model and get training info (e.g., training/validation losses)
    >>> training_info = gmm.train(train_TSs, valid_TSs)
    >>> # make prediction
    >>> gmm.predict(train_TSs)
    >>> # save model
    >>> gmm.save_model("global_model.pickle")
    >>> # load model
    >>> gmm2 = GMModel(params = None, load_model = True)
    >>> gmm2.load_model("global_model.pickle")
    """

    def __init__(self, params: GMParam, load_model: bool = False):

        if not load_model:
            if not isinstance(params, GMParam):
                msg = f"params should be a GMParam object but receives {type(params)}."
                logging.error(msg)
                raise ValueError(msg)

            self.params = params
            self.debug = False

    def build_rnn(self) -> None:
        """
        Helper function for building RNN.

        :Returns:
        None
        """
        params = self.params
        feature_size = params.gmfeature.feature_size if params.gmfeature else 0
        input_size = (
            params.input_window + feature_size + 2
        )  # two additional positions for step_num_encode and step_size_encode
        output_size = (
            params.fcst_window * len(params.quantile) + 1
        )  # one additional position for level smoothing parameter
        if params.seasonality > 1:
            input_size += 2 * params.seasonality
            output_size += (
                1  # one additional position for seasonality smoothing parameter
            )
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

    def build_loss_function(self, mode="train") -> nn.Module:
        """
        Helper function for building loss function.

        :Returns:
        Loss function
        """
        if isinstance(self.params.loss_function, str):
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
            return loss_func
        else:
            return self.params.loss_function

    def build_validation_function(self) -> Callable:
        """
        Helper function for building validation function.

        :Returns:
        Validation function
        """
        valid_metric_names = self.params.validation_metric
        quantile = np.array(self.params.quantile[1:])

        def valid_func(fcst, target):
            ans = {}
            if len(fcst.shape) == 1:
                fcst = fcst.reshape(1, -1)
            if len(target.shape) == 1:
                target = target.reshape(1, -1)
            d = target.shape[1]
            for name in valid_metric_names:
                if name == "smape":
                    ans["smape"] = calc_smape(fcst[:, :d], target)
                elif name == "sbias":
                    ans["sbias"] = calc_sbias(fcst[:, :d], target)
                elif name == "exceed" and len(quantile) > 1:
                    tmp_val = calc_exceed(fcst[:, d:], target, quantile)
                    tmp_dict = {
                        f"exceed_{quantile[i]}": tmp_val[i]
                        for i in range(len(quantile))
                    }
                    ans.update(tmp_dict)
            return ans

        return valid_func

    def build_optimizer(self) -> Tuple[Any, Dict]:
        """
        Helper function for building optimizer

        :Returns:
        optimizer: Any
            A torch optimizer
        optimizer_param: Dict[str, Any]
            A dictionary of optimizer parameters
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

    def train(
        self,
        train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
        valid_TSs: Optional[
            Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
        ] = None,
        fcst_monitor: bool = False,
        debug: bool = False,
    ) -> Dict[str, Optional[List]]:
        """
        Train RNN model.

        :Parameters:
        train_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
            A list or a dictionary of time series for training
        valid_TSs: Optional[Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
            A list or a dictionary of time series for validation
        fcst_monitor: bool = False
            Whether to return the forecasts during training for the validation data.
        debug: bool = False
            Whether to run bebug helper function during training.

        :Returns:
        training_info: Dict[str, Optional[List]]
            A dictionary collects training information. It has three keys: "train_loss_monitor", "valid_loss_monitor" and "valid_fcst_monitor".
        training_info["train_loss_monitor"]: List[Dict]
            A list of dictionaries storing the averaged traning losses of each epoch.
        training_info["valid_loss_monitor"]: List[Dict]
            A list of dictionaries storing the averaged validation losses of each epoch.
        training_info["valid_fcst_monitor"]: Optional[List[Dict]]
            A list of dictionaries storing the forecasts generated by global model for validation time series during training.
        """
        if not hasattr(self, "rnn"):
            self.build_rnn()
        self.debug = debug
        rnn = self.rnn
        rnn.train()
        loss_func = self.build_loss_function()
        valid_loss_func = self.build_validation_function()
        optimizer, optimizer_param = self.build_optimizer()
        dl = GMDataLoader(train_TSs)
        params = self.params
        len_quantile = len(params.quantile)

        single_pass = (
            self._single_pass_seasonal
            if params.seasonality > 1
            else self._single_pass_nonseasonal
        )

        train_loss_monitor = []
        valid_loss_monitor = []

        if fcst_monitor:
            valid_fcst_monitor = []
        else:
            valid_fcst_monitor = None

        for epoch in range(params.epoch_num):

            tmp_train_loss_monitor = []
            tmp_valid_loss_monitor = []

            if epoch in params.batch_size:
                batch_size = params.batch_size[epoch]
            if epoch in params.learning_rate:
                lr = params.learning_rate[epoch]

            trainer = optimizer(params=rnn.parameters(), lr=lr, **optimizer_param)

            logging.info(
                f"Training for epoch {epoch} with batch_size = {batch_size} and learning_rate = {lr}."
            )

            for _ in range(params.epoch_size):
                rnn.reset_state()

                # fetch batch_ids
                batch_ids = dl.get_batch(batch_size)

                # batch TSs
                batch = GMBatch(params, batch_ids, train_TSs, valid_TSs, mode="train")

                train_loss, train_res, valid_res, fcst_store = single_pass(
                    rnn,
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
                avg_train_loss.backward()
                trainer.step()

                # record training_info

                tmp_train_loss_monitor.extend(train_res)
                tmp_valid_loss_monitor.extend(valid_res)
                if fcst_monitor:
                    fcst_store["epoch"] = epoch
                    valid_fcst_monitor.append(fcst_store)

            avg_vals = np.average(tmp_train_loss_monitor, axis=0)
            train_res = {"epoch": epoch}
            train_res.update(
                {str(params.quantile[i]): avg_vals[i] for i in range(len_quantile)}
            )
            train_loss_monitor.append(train_res)

            valid_res = pd.DataFrame(tmp_valid_loss_monitor).mean(skipna=True).to_dict()
            valid_res["epoch"] = epoch
            valid_loss_monitor.append(valid_res)

            logging.info(f"Successfully finished training for epoch {epoch}")

        training_info = {
            "train_loss_monitor": train_loss_monitor,
            "valid_loss_monitor": valid_loss_monitor,
            "valid_fcst_monitor": valid_fcst_monitor,
        }

        rnn.reset_state()
        return training_info

    def _format_fcst(
        self,
        ids: List[Any],
        fcst_store: Dict[str, np.ndarray],
        steps: int,
        last_time: np.ndarray,
    ) -> Dict[Any, pd.DataFrame]:
        """
        Helper function for transforming raw forecast data into pd.DataFrame.

        :Parameters:
        ids: List[Any]
            The list of time series ids that are associated with the forecasts.
        fcst_store: Dict[str, np.ndarray]
            Raw forecasts generated by global model.
        steps: int
            The step of forecasts.
        last_time:
            The list of the last timestamps of the training time series.

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
                last_time[i] + self.params.freq, freq=self.params.freq, periods=steps
            )
            if "actual" in fcst_store:
                df["actual"] = actual[i]
            ans[idx] = df
        return ans

    def predict(
        self,
        test_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
        steps: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        raw: bool = False,
    ) -> Dict[Any, Union[pd.DataFrame, List[np.ndarray]]]:
        """
        Generate forecasts for target time series.

        :Parameters:
        test_TSs: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]]
            A collection of time series to generate forecasts for.
        steps: Optional[int] = None
            Forecast steps. If None, then take the maximum step value, which is params.fcst_window * params.fcst_step_num
        test_batch_size: Optional[int] = None
            Batch size for testing. If None, then take default value as 500.
        raw: bool = False
            Whether to return raw forecasts. If True, then return the raw forecasts for all possible forecast steps.
            If False, then return the pd.DataFrame.

        :Returns:
        fcst_collects: Dict[Any, Union[pd.DataFrame, List[np.ndarray]]]
            A dictionary of forecasts with its keys are ids for time series, and values are the corresponding forecasts.

        """
        params = self.params

        max_step = params.fcst_step_num * params.fcst_window
        if steps is None:
            steps = max_step
        elif not isinstance(steps, int) or steps <= 0 and steps > max_step:
            msg = f"step should be a positive integer less than {max_step} but receives {steps}."
            logging.error(msg)
            raise ValueError(msg)
        if test_batch_size is None:
            test_batch_size = 500
        if not isinstance(test_batch_size, int) or test_batch_size <= 0:
            msg = f"test_batch_size should be a positive integer but receives {test_batch_size}."
            logging.error(msg)
            raise ValueError(msg)

        n = len(test_TSs)
        batch_size = 500  # here we set the maximum batch_size for prediction is 1000.
        m = n // batch_size + (n % batch_size != 0)
        dl = GMDataLoader(test_TSs)
        rnn = self.rnn
        rnn.eval()
        fcst_collects = {}
        single_pass = (
            self._single_pass_seasonal
            if params.seasonality > 1
            else self._single_pass_nonseasonal
        )
        for i in range(m):

            rnn.reset_state()

            ids = dl.get_batch(batch_size)
            batch = GMBatch(params, batch_ids=ids, train_TSs=test_TSs, mode="test")

            _, _, _, fcst_store = single_pass(
                rnn,
                params,
                batch,
                training_mode=False,
                fcst_monitor=False,
                loss_func=None,
                valid_loss_func=None,
            )

            if not raw:
                fcst_store = self._format_fcst(
                    ids, fcst_store, steps, batch.time[:, -1]
                )
                fcst_collects.update(fcst_store)
            else:
                tmp = {
                    ids[i]: [t[i] for t in fcst_store["fcst"]] for i in range(len(ids))
                }
                fcst_collects.update(tmp)

        return fcst_collects

    def save_model(self, file_name: str) -> None:
        """
        Save global model.
        """
        joblib.dump(self.__dict__, file_name)
        logging.info(f"Successfully save the model as {file_name}.")

    def load_model(self, file_name: str) -> None:
        """
        Load global model.
        """
        try:
            self.__dict__ = joblib.load(file_name)
            logging.info(f"Successfully load the model from {file_name}.")
        except Exception as e:
            msg = f"Fail to load model from {file_name} with exception {e}."
            logging.error(msg)
            raise ValueError(msg)

    def _single_pass_seasonal(
        self,
        rnn: nn.Module,
        params: GMParam,
        batch: GMBatch,
        training_mode: bool = True,
        fcst_monitor: bool = False,
        loss_func: Optional[Callable] = None,
        valid_loss_func: Optional[Callable] = None,
    ) -> Tuple[List[Tensor], List[np.ndarray], List[Dict], Dict[str, np.ndarray]]:
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
        season_sm = torch.ones(tmp_batch_size, 1) * params.init_smoothing_params[1]

        period = params.seasonality
        x_lt, levels, seasonality = [], [], []
        seasonality = [
            batch.init_seasonality[:, i].view(tmp_batch_size, 1) for i in range(period)
        ]
        seasonality.append(batch.init_seasonality[:, 0].view(tmp_batch_size, 1))

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

        fcst_col_idx = 2
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

            input_t = torch.cat([input_t, next_season - 1.0, diff_season], dim=1)

            self._valid_tensor(input_t, "input")

            fcst_all = rnn(input_t)

            self._valid_tensor(fcst_all, "fcst")

            level_sm = torch.sigmoid(fcst_all[:, 0]).view(tmp_batch_size, 1)

            season_sm = torch.sigmoid(fcst_all[:, 1]).view(tmp_batch_size, 1)

            if (
                not is_valid
            ) and training_mode:  # training stage: inputs to loss_func are of logrithmic scale

                fcst = (
                    fcst_all[:, fcst_col_idx:]
                    + torch.log(anchor_level)
                    + torch.log(fcst_season).repeat(1, len_quantile)
                )
                actuals = (
                    batch.x[:, cur_idx : cur_idx + params.fcst_window].clone().log()
                )
                tmp_loss = loss_func(fcst, actuals)
                train_loss.append(torch.sum(tmp_loss))
                train_res.append(tmp_loss.detach().numpy())

            elif (
                is_valid
            ):  # validation stage: inputs to validation_func are of original scale
                fcst = (
                    (
                        fcst_all[:, fcst_col_idx:].exp()
                        * fcst_season.repeat(1, len_quantile)
                        - batch.offset
                    )
                    .detach()
                    .numpy()
                )
                if training_mode:
                    actuals = (
                        batch.x[:, cur_idx : cur_idx + params.fcst_window].clone()
                        - batch.offset
                    ).numpy()
                    tmp_loss = valid_loss_func(fcst, actuals)
                    valid_res.append(tmp_loss)
                    if fcst_monitor:

                        fcst_store["fcst"].append(fcst)
                        fcst_store["actual"].append(actuals)
                else:
                    fcst_store["fcst"].append(fcst)

            cur_step += step_delta
            if (
                cur_step >= last_train_step
                and (cur_step - step_delta) < last_train_step
            ):
                cur_step = last_train_step
                step_delta = 1
            prev_idx = cur_idx

        return train_loss, train_res, valid_res, fcst_store

    def _single_pass_nonseasonal(
        self,
        rnn: nn.Module,
        params: GMParam,
        batch: GMBatch,
        training_mode: bool = True,
        fcst_monitor: bool = False,
        loss_func: Optional[Callable] = None,
        valid_loss_func: Optional[Callable] = None,
    ) -> Tuple[List[Tensor], List[np.ndarray], List[Dict], Dict[str, np.ndarray]]:
        """
        Helper function for passing a batch into the non-seasonal RNN.
        """
        tmp_batch_size = batch.batch_size
        tdtype = torch.get_default_dtype()

        # initialize data
        train_loss = []
        train_res = []
        valid_res = []
        fcst_store = collections.defaultdict(list)
        level_sm = torch.ones(tmp_batch_size, 1) * params.init_smoothing_params[0]
        season_sm = torch.ones(tmp_batch_size, 1) * params.init_smoothing_params[1]

        period = params.seasonality
        x_lt, levels, seasonality = [], [], []

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

        fcst_col_idx = 1
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

            fcst_all = rnn(input_t)

            self._valid_tensor(fcst_all, "fcst")

            level_sm = torch.sigmoid(fcst_all[:, 0]).view(tmp_batch_size, 1)

            if (
                not is_valid
            ) and training_mode:  # training stage: inputs to loss_func are of logrithmic scale

                fcst = fcst_all[:, fcst_col_idx:] + torch.log(anchor_level)
                actuals = (
                    batch.x[:, cur_idx : cur_idx + params.fcst_window].clone().log()
                )
                tmp_loss = loss_func(fcst, actuals)
                train_loss.append(torch.sum(tmp_loss))
                train_res.append(tmp_loss.detach().numpy())

            elif (
                is_valid
            ):  # validation stage: inputs to validation_func are of original scale
                fcst = (
                    (fcst_all[:, fcst_col_idx:].exp() * anchor_level - batch.offset)
                    .detach()
                    .numpy()
                )
                if training_mode:
                    actuals = (
                        batch.x[:, cur_idx : cur_idx + params.fcst_window].clone()
                        - batch.offset
                    ).numpy()
                    tmp_loss = valid_loss_func(fcst, actuals)
                    valid_res.append(tmp_loss)
                    if fcst_monitor:

                        fcst_store["fcst"].append(fcst)
                        fcst_store["actual"].append(actuals)
                else:
                    fcst_store["fcst"].append(fcst)

            cur_step += step_delta
            if (
                cur_step >= last_train_step
                and (cur_step - step_delta) < last_train_step
            ):
                cur_step = last_train_step
                step_delta = 1
            prev_idx = cur_idx

        return train_loss, train_res, valid_res, fcst_store
