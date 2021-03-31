#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import logging
from typing import List, Optional, Union, Any, Dict, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


all_validation_metric_name = ["smape", "sbias", "exceed"]
import numpy as np
import pandas as pd

# for jit
NoneT = torch.FloatTensor([-1e38])


class LSTM2Cell(torch.nn.Module):
    """
    A version of LSTM cell where the output (of size=state_size) is split between h state (of size=h_size) and
    the real output that goes to the next layer (of size=state_size-h_size)

    :Parameters:
    input_size: int
        The number of expected features in the input x.
    h_size: int
        h state size.
    state_size: int
        c state size.

    :Inputs:
    input_t: Tensor
        Tensor containing input features of shape (batch_size, input_size)
    has_prev_state : bool
        Whether have previous state.
    has_delayed_state: bool
        Whether have delayed state.
    prev_h_state: Tensor = NoneT
        Tensor containing previsous h_state of shape (batch_size, h_size)
    delayed_h_state: Tensor = NoneT
        Tensor containing delayed h_state of shape (batch_size, h_size)
    c_state: Tensor = NoneT
        Tensor containing c_state of shape (batch_size, state_size)

    :Outputs:
    output_t, (h_state, new_state)

    output_t: Tensor
        Tensor containing outputs of shape (batch_size, state_size-h_size)
    h_state: Tensor
        Tensor containing the next h_state of shape (batch_size, h_size)
    new_state: Tensor
        Tensor containing the next c_state of shape (batch_size, state_size)

    """

    def __init__(self, input_size: int, h_size: int, state_size: int):
        super(LSTM2Cell, self).__init__()
        self.lxh = torch.nn.Linear(input_size + 2 * h_size, 4 * state_size)
        self.h_size = h_size
        self.out_size = state_size - h_size

    # jit does not like Optional, so we have to use bool variables and NoneT
    def forward(
        self,
        input_t: Tensor,
        has_prev_state: bool,
        has_delayed_state: bool,
        prev_h_state: Tensor = NoneT,
        delayed_h_state: Tensor = NoneT,
        c_state: Tensor = NoneT,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        if has_delayed_state:
            xh = torch.cat([input_t, prev_h_state, delayed_h_state], dim=1)
        elif has_prev_state:
            xh = torch.cat([input_t, prev_h_state, prev_h_state], dim=1)
        else:
            empty_h_state = torch.zeros(
                input_t.shape[0], 2 * self.h_size, dtype=torch.float32
            )
            xh = torch.cat([input_t, empty_h_state], dim=1)

        gates = self.lxh(xh)

        chunked_gates = torch.chunk(gates, 4, dim=1)

        forget_gate = (chunked_gates[0] + 1).sigmoid()
        in_gate = chunked_gates[1].sigmoid()
        out_gate = chunked_gates[2].sigmoid()
        new_state = chunked_gates[3].tanh()

        if has_prev_state:
            new_state = (forget_gate * c_state) + (in_gate * new_state)
        whole_output = out_gate * new_state.tanh()

        output_t, h_state = torch.split(
            whole_output, [self.out_size, self.h_size], dim=1
        )
        return output_t, (h_state, new_state)


class S2Cell(torch.nn.Module):
    """
    Slawek's S2 cell: a kind of mix of GRU and LSTM. Also splitting output into h and the "real output".

    :Parameters:
    input_size: int
        The number of expected features in the input x.
    h_size: int
        The number of expected features in the h_state.
    state_size: int
        The number of expected features in the c_state.

    :Inputs:
    input_t: Tensor
        Tensor containing input features of shape (batch_size, input_size).
    has_prev_state : bool
        Whether have previous state.
    has_delayed_state: bool
        Whether have delayed state.
    prev_h_state: Tensor = NoneT
        Tensor containing previsous h_state of shape (batch_size, h_size).
    delayed_h_state: Tensor = NoneT
        Tensor containing delayed h_state of shape (batch_size, h_size).
    prev_c_state: Tensor = NoneT
        Tensor containing previous c_state of shape (batch_size, state_size).
    delayed_c_state: Tensor = NoneT
        Tensor containing delayed c_state of shape (batch_size, state_size).

    :Outputs:
    output_t, (h_state, new_state)

    output_t: Tensor
        Tensor containing outputs of shape (batch_size, state_size-h_size).
    h_state: Tensor
        Tensor containing the next h_state of shape (batch_size, h_size).
    new_state: Tensor
        Tensor containing the next c_state of shape (batch_size, state_size).

    """

    def __init__(self, input_size: int, h_size: int, state_size: int):
        super(S2Cell, self).__init__()
        self.lxh = torch.nn.Linear(input_size + 2 * h_size, 4 * state_size)
        self.h_size = h_size
        self.state_size = state_size
        self.out_size = state_size - h_size

    # jit does not like Optional, so we have to use bool variables and NoneT
    def forward(
        self,
        input_t: Tensor,
        has_prev_state: bool,
        has_delayed_state: bool,
        prev_h_state: Tensor = NoneT,
        delayed_h_state: Tensor = NoneT,
        prev_c_state: Tensor = NoneT,
        delayed_c_state: Tensor = NoneT,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if has_delayed_state:
            xh = torch.cat([input_t, prev_h_state, delayed_h_state], dim=1)
        elif has_prev_state:
            xh = torch.cat([input_t, prev_h_state, prev_h_state], dim=1)
        else:
            empty_h_state = torch.zeros(
                input_t.shape[0], 2 * self.h_size, dtype=torch.float
            )
            xh = torch.cat([input_t, empty_h_state], dim=1)

        gates = self.lxh(xh)
        chunked_gates = torch.chunk(gates, 4, dim=1)

        forget_gate = (chunked_gates[0] + 1).sigmoid()
        new_stat = chunked_gates[1].tanh()
        out_gate = chunked_gates[3].sigmoid()

        if has_prev_state:
            if has_delayed_state:
                alpha = chunked_gates[2].sigmoid()
                weighted_c_state = alpha * prev_c_state + (1 - alpha) * delayed_c_state
            else:
                weighted_c_state = prev_c_state

            new_stat = forget_gate * weighted_c_state + (1 - forget_gate) * new_stat

        whole_output = out_gate * new_stat

        output_t, h_state = torch.split(
            whole_output, [self.out_size, self.h_size], dim=1
        )
        return output_t, (h_state, new_stat)


class DilatedRNNStack(torch.nn.Module):
    """
    The recurrent neural network for global model.

    :Parameters:
    nn_structure: List[List[int]]
        Strucuture of neural network. For example,[[1,3],[6,12]] - this defines 2 blocks of 2 layers each and output adaptor layer, with a resNet-style shortcut between output of the first block (output of the second layer)
        and output of the second block (output of 4th layer). The positive integers are the dilation number.
    cell_name: str
        Name of the cells, currently support LSTM, LSTM2Cell and S2Cell.
    input_size: int
        The number of expected features in the input x.
    state_size: int
        c state size (which is hidden_size for standard LSTM).
    output_size: int
        The number of expected features in the final output.
    h_size: Optional[int] = None
        The number of expected features in h_state (not needed for standard LSTM).
    jit: bool = False
        Whether to jit every cell.

    :Inputs:
    input_t: Tensor
        Tensor containing input features of shape (batch_size, input_size).

    :Outputs:
    output_t: Tensor
         Tensor containing outputs of shape (batch_size, output_size).
    """

    def __init__(
        self,
        nn_structure: List[List[int]],
        cell_name: str,
        input_size: int,
        state_size: int,
        output_size: int,
        h_size=None,
        jit=False,
    ) -> None:
        super(DilatedRNNStack, self).__init__()
        block_num = len(nn_structure)
        self.nn_structure = nn_structure
        self.cell_name = cell_name
        self.input_size = input_size
        self.h_size = h_size
        self.jit = jit

        self.reset_state()

        if cell_name not in ["LSTM2Cell", "S2Cell", "LSTM"]:
            msg = f"Only support cells 'S2Cell', 'LSTM2Cell', 'LSTM' but receive {cell_name}."
            logging.error(msg)
            raise ValueError(msg)

        if cell_name == "LSTM2Cell" or cell_name == "S2Cell":
            if h_size is None:
                msg = "h_size should be a positive integer smaller than state_size for LSTM2Cell or S2Cell."
                logging.error(msg)
                raise ValueError(msg)
            if h_size >= state_size:
                msg = "h_size should be smaller than state_size."
                logging.error(msg)
                raise ValueError(msg)
            out_size = state_size - h_size
        else:
            out_size = state_size

        self.cells = []
        layer = 0
        iblock = 0
        for iblock in range(block_num):
            for lay in range(len(nn_structure[iblock])):
                if lay == 0 and iblock == 0:
                    tmp_input_size = input_size
                else:
                    tmp_input_size = out_size

                if cell_name == "LSTM2Cell":
                    if jit:
                        cell = torch.jit.script(
                            LSTM2Cell(tmp_input_size, h_size, state_size)
                        )
                    else:
                        cell = LSTM2Cell(tmp_input_size, h_size, state_size)
                elif cell_name == "S2Cell":
                    if jit:
                        cell = torch.jit.script(
                            S2Cell(tmp_input_size, h_size, state_size)
                        )
                    else:
                        cell = S2Cell(tmp_input_size, h_size, state_size)
                else:
                    cell = torch.nn.LSTMCell(tmp_input_size, state_size)
                self.add_module("Cell_{}".format(layer), cell)
                self.cells.append(cell)
                layer += 1

        self.adaptor = torch.nn.Linear(out_size, output_size)
        self.block_num = block_num
        self.out_size = out_size

    def _forward_S2Cell(
        self,
        tmp_input: Tensor,
        layer: int,
        has_prev_state: bool,
        has_delayed_state: bool,
        t: int,
        ti_1: int,
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        .forward function for S2Cell (to avoid lint warning).
        """
        if has_delayed_state:
            output_t, (h_state, new_state) = self.cells[layer](
                tmp_input,
                has_prev_state,
                has_delayed_state,
                prev_h_state=self.h_state_store[t - 1][layer],
                delayed_h_state=self.h_state_store[ti_1][layer],
                prev_c_state=self.c_state_store[t - 1][layer],
                delayed_c_state=self.c_state_store[ti_1][layer],
            )
        elif has_prev_state:
            output_t, (h_state, new_state) = self.cells[layer](
                tmp_input,
                has_prev_state,
                has_delayed_state,
                prev_h_state=self.h_state_store[t - 1][layer],
                prev_c_state=self.c_state_store[t - 1][layer],
            )
        else:
            output_t, (h_state, new_state) = self.cells[layer](tmp_input, False, False)
        return output_t, (h_state, new_state)

    def _forward_LSTM2Cell(
        self,
        tmp_input: Tensor,
        layer: int,
        has_prev_state: bool,
        has_delayed_state: bool,
        t: int,
        ti_1: int,
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        .forward function for LSTM2Cell (to avoid lint warning).
        """
        if has_delayed_state:
            output_t, (h_state, new_state) = self.cells[layer](
                tmp_input,
                has_prev_state,
                has_delayed_state,
                prev_h_state=self.h_state_store[t - 1][layer],
                delayed_h_state=self.h_state_store[ti_1][layer],
                c_state=self.c_state_store[ti_1][layer],
            )
        elif has_prev_state:
            output_t, (h_state, new_state) = self.cells[layer](
                tmp_input,
                has_prev_state,
                has_delayed_state,
                prev_h_state=self.h_state_store[t - 1][layer],
                c_state=self.c_state_store[t - 1][layer],
            )
        else:
            output_t, (h_state, new_state) = self.cells[layer](tmp_input, False, False)
        return output_t, (h_state, new_state)

    def forward(self, input_t: Tensor) -> Tensor:
        prev_block_output = torch.zeros(
            input_t.shape[0], self.out_size, dtype=torch.float
        )
        t = len(self.h_state_store)
        self.h_state_store.append([])
        self.c_state_store.append([])
        output_t = NoneT  # just to initialize output_t
        has_prev_state = t > 0

        layer = 0
        for iblock in range(self.block_num):
            for lay in range(len(self.nn_structure[iblock])):
                if lay == 0:
                    if iblock == 0:
                        tmp_input = input_t
                    else:
                        tmp_input = prev_block_output
                else:
                    tmp_input = output_t

                ti_1 = t - self.nn_structure[iblock][lay]
                has_delayed_state = ti_1 >= 0

                if self.cell_name == "S2Cell":
                    output_t, (h_state, new_state) = self._forward_S2Cell(
                        tmp_input, layer, has_prev_state, has_delayed_state, t, ti_1
                    )

                elif self.cell_name == "LSTM2Cell":
                    output_t, (h_state, new_state) = self._forward_LSTM2Cell(
                        tmp_input, layer, has_prev_state, has_delayed_state, t, ti_1
                    )
                else:  # LSTM
                    if has_delayed_state:
                        h_state, new_state = self.cells[layer](
                            tmp_input,
                            (
                                self.h_state_store[ti_1][layer],
                                self.c_state_store[ti_1][layer],
                            ),
                        )
                    elif has_prev_state:
                        h_state, new_state = self.cells[layer](
                            tmp_input,
                            (
                                self.h_state_store[t - 1][layer],
                                self.c_state_store[t - 1][layer],
                            ),
                        )
                    else:
                        h_state, new_state = self.cells[layer](tmp_input)
                    output_t = h_state

                self.h_state_store[t].append(h_state)
                self.c_state_store[t].append(new_state)
                layer += 1
            prev_block_output = output_t + prev_block_output

        output_t = self.adaptor(prev_block_output)
        return output_t

    def reset_state(self) -> None:
        """
        Clear all stored state tensors.
        """
        self.h_state_store = []
        self.c_state_store = []


class PinballLoss(_Loss):
    """
    Pinball Loss function.

    For quantile q (0<q<1), forecast value y_hat and true value y, pinball loss is defined as:
        pinball(y_hat, y)=max((y-y_hat)*q, (y-y_hat)*(q-1)).

    :Parameters:
    quantile: Tensor
        A 1-dimensional tensor containing quantiles to be calculated.
    weight: Optional[Tensor]
        A 1-dimensional tensor containing weights for quantiles. If None, the quantiles will be equally weighted.
    reduction: str = 'mean'
        Reduction method, we currently support 'mean', 'sum' and None.


    :Inputs:
    input: Tensor
        Tensor containing forecasted values of shape (num, n_steps * n_quantiles), where n_quantiles is the length of quantile.
    target: Tensor
        Tensor contianing true values of shape (num, n_steps)

    :Outputs:
    ans: Tensor
        Tensor containing loss values. If reduction == 'mean' or 'sum', output is a 1-dimensional tensor of length n_quantiles.
        If reduction is None, output is tensor of shape (num, n_quantiles).

    """

    def __init__(
        self, quantile: Tensor, weight: Optional[Tensor] = None, reduction: str = "mean"
    ) -> None:

        super(PinballLoss, self).__init__(
            size_average=None, reduce=None, reduction=reduction
        )
        if len(quantile) < 1:
            msg = "quantile should not be empty."
            logging.error(msg)
            raise ValueError(msg)
        if len(quantile.size()) != 1:
            msg = "quantile should be a 1-dimentional tensor."
            logging.error(msg)
            raise ValueError(msg)
        self.register_buffer("quantile", quantile)
        if weight is None:
            d = len(quantile)
            weight = torch.ones(d) / d
        else:
            if weight.size() != quantile.size():
                msg = "weight and quantile should have the same size."
                logging.error(msg)
                raise ValueError(msg)

        self.register_buffer("weight", weight)
        self.reduction = reduction

    def _check(self, input: Tensor, target: Tensor) -> None:
        """
        Check input tensor and target tensor size.
        """
        if target.size()[0] != input.size()[0]:
            msg = "Input batch size is not equal to target batch size."
            logging.error(msg)
            raise ValueError(msg)
        num_feature = target.size()[1] * len(self.quantile)
        if input.size()[1] != num_feature:
            msg = f"Input should contain {num_feature} features but receive {input.size()[1]}."
            logging.error(msg)
            raise ValueError(msg)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self._check(input, target)
        n = len(input)
        m = len(self.quantile)
        horizon = target.size()[1]
        nans = torch.isnan(target).detach()
        # clean up NaNs to avoid NaNs in gradient
        target[nans] = 1.0
        num_not_nan = (~nans).float().sum(dim=1)
        num_not_nan[num_not_nan == 0] += 1

        target = target.repeat(1, m)
        nans = nans.repeat(1, m)

        quants = self.quantile.repeat(horizon, 1).t().flatten()
        weights = self.weight.repeat(horizon, 1).t().flatten()

        diff = target - input
        res = torch.max(diff * quants, diff * (quants - 1.0))
        res[nans] = 0.0
        res = res * weights
        res = (
            res.view(n, -1, horizon).sum(dim=2) / num_not_nan[:, None]
        )  # row_wise operation

        if self.reduction == "mean":
            return res.mean(dim=0)
        if self.reduction == "sum":
            return res.sum(dim=0)

        return res


class AdjustedPinballLoss(_Loss):
    """
    Adjusted Pinball Loss function, which aligns better with sMAPE loss.

    This is an adjusted version of pinball loss function in that when for the first quantile (i.e., should be 0.5 or quantile close to 0.5),
    we normalized the original pinball loss with the average value of target and forecasts. The idea is to optimize for sMAPE for the first quantile.

    For the other quantile q (0<q<1), pinball loss is defined as:
        pinball(y_hat, y)=max((y-y_hat)*q, (y-y_hat)*(q-1)),
    where y_hat is the forecast value and y is the true value.

    Note that the first quantiles loss operates on "real" (i.e. not logrithemic values), and the other quantile losses operate on logrithemic values.

    :Parameters:
    quantil: Tensor
        A 1-dimensional tensor containing quantiles to be calculated.
    weight: Optional[Tensor]
        A 1-dimensional tensor containing weights for quantiles. If None, the quantiles will be equally weighted.
    reduction: str = 'mean'
        Reduction method, we currently support 'mean', 'sum' and None.
    input_log: bool = True
        Whether the target and forecasts are of logarithmic scale.

    :Inputs:
    input: Tensor
        Tensor containing forecasted values of shape (num, n_steps * n_quantiles), where n_quantiles is the length of quantile.
    target: Tensor
        Tensor contianing true values of shape (num, n_steps)

    :Outputs:
    ans: Tensor
        Tensor containing loss values. If reduction == 'mean' or 'sum', output is a 1-dimensional tensor of length n_quantiles.
        If reduction is None, output is tensor of shape (num, n_quantiles).

    """

    def __init__(
        self,
        quantile: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        input_log: bool = True,
    ) -> None:

        super(AdjustedPinballLoss, self).__init__(
            size_average=None, reduce=None, reduction=reduction
        )
        if len(quantile) < 1:
            msg = "quantile should not be empty."
            logging.error(msg)
            raise ValueError(msg)
        if len(quantile.size()) != 1:
            msg = "quantile should be a 1-dimentional tensor."
            logging.error(msg)
            raise ValueError(msg)
        self.register_buffer("quantile", quantile)
        if weight is None:
            d = len(quantile)
            weight = torch.ones(d) / d
        else:
            if weight.size() != quantile.size():
                msg = "weight and quantile should have the same size."
                logging.error(msg)
                raise ValueError(msg)
        self.register_buffer("weight", weight)
        self.reduction = reduction
        self.input_log = input_log

    def _check(self, input: Tensor, target: Tensor) -> None:
        """
        Check input tensor and target tensor size.
        """
        if target.size()[0] != input.size()[0]:
            msg = "Input batch size is not equal to target batch size."
            logging.error(msg)
            raise ValueError(msg)
        num_feature = target.size()[1] * len(self.quantile)
        if input.size()[1] != num_feature:
            msg = f"Input should contain {num_feature} features but receive {input.size()[1]}."
            logging.error(msg)
            raise ValueError(msg)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self._check(input, target)
        n = len(input)
        m = len(self.quantile)
        horizon = target.size()[1]
        nans = torch.isnan(target).detach()
        # avoid nans appear in the loss
        target[nans] = 1.0
        num_not_nan = (~nans).float().sum(dim=1)
        num_not_nan[num_not_nan == 0] += 1

        if self.input_log:
            target_exp = torch.exp(target)
            fcst_exp = torch.exp(input[:, :horizon])
        else:
            target_exp = target
            fcst_exp = input[:, :horizon]
        diff = target_exp - fcst_exp
        res = (
            torch.max(diff * self.quantile[0], diff * (self.quantile[0] - 1.0))
            / (target_exp + fcst_exp)
            * 2
        )
        res[nans] = 0.0

        if m > 1:
            if self.input_log:
                fcst = input[:, horizon:]
            else:
                fcst = torch.log(input[:, horizon:])
            m -= 1

            target = target.repeat(1, m)
            nans = nans.repeat(1, m)

            quants = self.quantile[1:].repeat(horizon, 1).t().flatten()

            diff_q = target - fcst
            res_q = torch.max(diff_q * quants, diff_q * (quants - 1.0))
            res_q[nans] = 0.0

            res = torch.cat([res, res_q], dim=1)

        weights = self.weight.repeat(horizon, 1).t().flatten()
        res = res * weights
        res = res.view(n, -1, horizon).sum(dim=2) / num_not_nan[:, None]

        if self.reduction == "mean":
            return res.mean(dim=0)
        if self.reduction == "sum":
            return res.sum(dim=0)

        return res


def squash(t: Tensor) -> Tensor:
    """
    Squash function for TS.

    """
    return torch.log(t)


def expand(t: Tensor) -> Tensor:
    """
    Expand function for TS (i.e., transform squashed values to its original scale.)
    """
    return torch.exp(t)


class GMParam:
    """
    A class for storing all parameters of a global model.

    :Parameters:
    freq: Union[str,pd.Timedelta]
        frequency of TS.
    input_window: int
        Length of input TS.
    fcst_window: int
        Length of forecast horizon.
    seasonality: int=1
        Seasonality period (seasonality==1 indicates there is non-seasonality).
    uplifting_ratio: float = 3.0
        For TS containing negative values, we add offset to it such that (max(TS)+offset)/(min(TS)+offset)=uplifing_ratio
    gmfeature: Optional[Any]= None
        GMFeature object for computing TS feature.
    nn_structure: Optional[List[List[int]]]= None
        Neural network structure. If None, default value is [[1,3]].
    cell_name: str = 'LSTM'
        Name of NN cell. We currently support 'LSTM2Cell', 'S2Cell' and 'LSTM'.
    state_size: int = 50
        c state size.
    h_size: Optional[int] = None
        h state size. When cell_name == 'LSTM2Cell' or 'S2Cell', h_size should be a positive integer less than state_size.
    optimizer: Optional[Union[str, Dict[str, Any]]]= None
        Optimizer for training NN. If None, default input is {'name':'Adam', params:{'eps': 1e-7}}.
    loss_function: Union[str, torch.nn.Module]='Pinball'
        Loss function for training NN. Currently support 'Pinball' and 'AdjustedPinball'.
    quantile: Optional[List[float]]= None
        Forecast quantiles. If None, default value is [0.5,0.05,0.95,0.99].
    training_quantile: Optional[List[float]]= None
        Quantiles used for pinball loss function during training, if None, then taken the same value as quantile.
    quantile_weight: Optional[List[float]]=None
        Weights for quantiles during training. If None, then all quantiles are equally weighted.
    validation_metric: Optional[List[float]]=None
        Metric names for validation. If None, then default value is ["smape", "sbias", "exceed"]
    batch_size: Union[None, int, Dict[int, int]]=None
        Dictionary for batch_size schedule. Keys are epoch numbers and values are batch sizes. If None, default value is
        {0:2,3:5,4:15,5:50,6:150,7:500}.
    learning_rate: Union[None, float, Dict[int, float]]= None
        Dictionary for learning_rate schedule. Keys are epoch numbers and values are learning rate. If None, default value is
        {0: 1e-3, 2: 1e-3/3.}.
    epoch_num: int= 8
        Totoal number of epoches.
    epoch_size: int= 3000
        Number of batches per epoch.
    init_seasonality: Optional[List[float]]= None
        Lower and upper bounds for initial seasonalities. If None, default value is [0.1, 10.].
    init_smoothing_params: Optional[List[float]]= None
        Initial values for smoothing parameters: level smoothing parameter and seasonality smoothing parameter. If None,
        default value is [0.4, 0.6].
    min_training_step_num: int = 4
        Minimum number of training steps.
    min_training_step_length: Optional[int] = None
        Minimum training step length. If None, then min_training_step_length = min(1, seasonality-1)
    soft_max_training_step_num: int = 10
        Soft maxinum value for training step.
    validation_step_num: int = 3
        Maximum number of validation steps (maximum validation horizon = validation_step_num * fcst_window).
    min_warming_up_step_num: int = 2
        Mininum number of warming-up steps for forecasting.
    fcst_step_num: int = 1
        Maximum number of forecasting steps (maximum forecasting horizon = fcst_step_num * fcst_window).
    jit: bool = False
        Whether to jit every cell of the RNN.
    name: Optional[str]=None
        Name of the GMParam.

    """

    def __init__(
        self,
        freq: Union[str, pd.Timedelta],
        input_window: int,
        fcst_window: int,
        seasonality: int = 1,
        uplifting_ratio: float = 3.0,
        gmfeature: Optional[Any] = None,  # need to be changed once gmfeature is defined
        nn_structure: Optional[List[List[int]]] = None,
        cell_name: str = "LSTM",
        state_size: int = 50,
        h_size: Optional[int] = None,
        optimizer: Optional[Union[str, Dict[str, Any]]] = None,
        loss_function: Union[str, torch.nn.Module] = "Pinball",
        quantile: Optional[List[float]] = None,
        training_quantile: Optional[List[float]] = None,
        quantile_weight: Optional[List[float]] = None,
        validation_metric: Optional[List[float]] = None,
        batch_size: Union[None, int, Dict[int, int]] = None,
        learning_rate: Union[float, Dict[int, float]] = None,
        epoch_num: int = 8,
        epoch_size: int = 3000,
        init_seasonality: Optional[List[float]] = None,
        init_smoothing_params: Optional[List[float]] = None,
        min_training_step_num: int = 4,
        min_training_step_length: Optional[int] = None,
        soft_max_training_step_num: int = 10,
        validation_step_num: int = 3,
        min_warming_up_step_num: int = 2,
        fcst_step_num: int = 1,
        jit: bool = False,
        name: Optional[str] = None,
    ) -> None:

        self._valid_freq(freq)

        # valid uplifiting ratio
        if uplifting_ratio < 0:
            msg = f"uplifting_ratio should be a positive float but receive {uplifting_ratio}."
            logging.error(msg)
            raise ValueError(msg)
        self.uplifting_ratio = uplifting_ratio

        # need validation func later
        self.gmfeature = gmfeature

        self.nn_structure = nn_structure if nn_structure is not None else [[1, 3]]
        self.cell_name = cell_name

        self.state_size = state_size
        self.h_size = h_size

        batch_size = (
            batch_size
            if batch_size is not None
            else {0: 2, 3: 5, 4: 15, 5: 50, 6: 150, 7: 500}
        )
        self._valid_union_dict(batch_size, "batch_size", int, int)
        learning_rate = (
            learning_rate if learning_rate is not None else {0: 1e-3, 2: 1e-3 / 3.0}
        )
        self._valid_union_dict(learning_rate, "learning_rate", int, float)

        self._valid_loss_function(loss_function)
        self._valid_optimizer(optimizer)

        quantile = quantile if quantile is not None else [0.5, 0.05, 0.95, 0.99]
        self._valid_list(quantile, "quantile", float, 0, 1)

        # additional check needed for filling NaNs during training.
        if self.quantile[0] != 0.5:
            msg = f"The first element of quantile should be 0.5 but receives {self.quantile[0]}."
            logging.error(msg)
            raise ValueError(msg)

        if training_quantile is None:
            self.training_quantile = quantile
        else:
            self._valid_list(training_quantile, "training_quantile", float, 0, 1)

        if quantile_weight is None:
            self.quantile_weight = [1.0 / len(quantile)] * len(quantile)
        else:
            if len(quantile_weight) != len(quantile):
                msg = "quantile and quantile_weight should be of the same length."
                logging.error(msg)
                raise ValueError(msg)
            self._valid_list(quantile_weight, "quantile_weight", float, 0, np.inf)

        if validation_metric is None:
            self.validation_metric = all_validation_metric_name
        else:
            if isinstance(validation_metric, list):
                for name in validation_metric:
                    if name not in all_validation_metric_name:
                        msg = f"Invalid metric_name {name}!"
                        logging.error(msg)
                        raise ValueError(msg)
                self.validation_metric = validation_metric
            else:
                msg = f"validation_metric should be a list of str, but receives {type(validation_metric)}."
                logging.error(msg)
                raise ValueError(msg)

        init_seasonality = (
            init_seasonality if init_seasonality is not None else [0.1, 10.0]
        )
        self._valid_list(init_seasonality, "init_seasonality", float, 0, np.inf)
        init_smoothing_params = (
            init_smoothing_params if init_smoothing_params is not None else [0.4, 0.6]
        )
        self._valid_list(
            init_smoothing_params, "init_smoothing_params", float, 0, np.inf
        )

        if min_training_step_length is None:
            min_training_step_length = max(1, seasonality - 1)

        pos_integer_params = {
            "input_window": input_window,
            "fcst_window": fcst_window,
            "validation_step_num": validation_step_num,
            "min_training_step_num": min_training_step_num,
            "min_training_step_length": min_training_step_length,
            "seasonality": seasonality,
            "state_size": state_size,
            "epoch_num": epoch_num,
            "epoch_size": epoch_size,
            "soft_max_training_step_num": soft_max_training_step_num,
            "validation_step_num": validation_step_num,
            "min_warming_up_step_num": min_warming_up_step_num,
            "fcst_step_num": fcst_step_num,
        }
        self._valid_positive_integer_params(pos_integer_params)

        # max_step_delta needed for training/testing
        self.max_step_delta = min(input_window, fcst_window) // min_training_step_length

        self.jit = jit
        self.name = name

    def _valid_optimizer(self, optimizer):
        opt_methods = ["adam"]
        if optimizer is None:
            self.optimizer = {"name": "Adam", "params": {"eps": 1e-7}}
        elif isinstance(optimizer, str):
            if optimizer.lower() in opt_methods:
                self.optimizer = optimizer
            else:
                msg = f"optimizer {optimizer} is not implemented."
        elif isinstance(optimizer, dict):
            if "name" in optimizer and optimizer["name"].lower() in opt_methods:
                self.optimizer = optimizer
            else:
                msg = f"optimizer {optimizer} is invalid."
                logging.error(msg)
                raise ValueError(msg)
        else:
            msg = f"optimizer should be either a str or a dict but receives {type(optimizer)}."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_loss_function(self, loss_function):
        """
        Helper function to verify loss function.
        """
        if isinstance(loss_function, str):
            if loss_function.lower() in ["pinball", "adjustedpinball"]:
                self.loss_function = loss_function.lower()
            else:
                msg = f"loss function {loss_function} is not implemented."
                logging.error(msg)
                raise ValueError(msg)
        elif isinstance(loss_function, _Loss):
            self.loss_function = loss_function
        else:
            msg = f"loss function should be either a str or a _Loss object but receives {type(loss_function)}."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_list(
        self, value: List, name: str, value_type: type, lower: float, upper: float
    ) -> None:
        """
        Helper function to verify list inputs.
        """
        if isinstance(value, list):
            for q in value:
                if isinstance(q, value_type) and q < upper and q > lower:
                    continue
                else:
                    msg = f"Each element in {name} should be a {value_type} in ({lower}, {upper}) but receives {q} of type {type(q)}."
                    logging.error(msg)
                    raise ValueError(msg)
            setattr(self, name, value)
        else:
            msg = f"{name} should be a list."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_union_dict(
        self,
        value: Union[int, float, dict],
        name: str,
        key_type: type,
        value_type: type,
    ) -> None:
        """
        Helper function to verify batch_size and learning_rate.
        """
        if isinstance(value, value_type) and value > 0:
            setattr(self, name, {0: value})
            return
        elif isinstance(value, dict):
            if 0 not in value:
                msg = f"0 should be in {name}."
                logging.error(msg)
                raise ValueError(msg)
            for n in value:
                if (
                    (not isinstance(n, key_type))
                    or (not isinstance(value[n], value_type))
                    or (value[n] <= 0)
                ):
                    msg = f"""
                    The key in {name} should be a non-negative {key_type},
                    and the value in batch_size should be a positive {value_type}.
                    """
                    logging.error(msg)
                    raise ValueError(msg)
            setattr(self, name, value)
            return
        else:
            msg = f"{name} should be either positive {value_type} or a dictionary, but receive {value}."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_freq(self, freq: Union[str, pd.Timedelta]):
        """
        Helper function to verify freq.
        """
        if isinstance(freq, pd.Timedelta):
            self.freq = freq
        elif isinstance(freq, str) and len(freq) > 0:
            try:
                if freq[0].isdigit():
                    freq = pd.to_timedelta(freq)
                else:
                    freq = pd.to_timedelta("1" + freq)
            except Exception as e:
                msg = f"Fail to convert freq to pd.Timedelta with error message {e}."
                logging.error(msg)
                raise ValueError(msg)
        else:
            msg = f"freq should be either pd.Timedelta or str but receive {type(freq)}."
            logging.error(msg)
            raise ValueError(msg)
        self.freq = freq

    def _valid_positive_integer_params(self, params: Dict[str, int]) -> None:
        """
        Verify params are positive integers.
        """
        for elm in params:
            if not isinstance(params[elm], int) or params[elm] < 1:
                msg = f"{elm} should be a positive integer but receive {params[elm]}."
                logging.error(msg)
                raise ValueError(msg)
            setattr(self, elm, params[elm])


def calc_smape(fcst: np.ndarray, actuals: np.ndarray):
    """
    Compute smape between fcst and actuals.
    """
    diff = 2 * np.abs(fcst - actuals) / (np.abs(fcst) + np.abs(actuals))
    return np.nanmean(diff)


def calc_sbias(fcst: np.ndarray, actuals: np.ndarray):
    """
    Compute sbias between fcst and actuals.
    """
    diff = 2 * (fcst - actuals) / (np.abs(fcst) + np.abs(actuals))

    return np.nanmean(diff)


def calc_exceed(fcst: np.ndarray, actuals: np.ndarray, quantile: np.ndarray):
    """
    Compute exceed rate for quantile estimates.
    """
    if len(fcst.shape) == 1:
        fcst = fcst.reshape(-1, 1)
    if len(actuals.shape) == 1:
        actuals = actuals.reshape(-1, 1)
    m = len(quantile)
    n, horizon = actuals.shape
    actuals = np.tile(actuals, m)
    mask = np.repeat((quantile > 0.5) * 2 - 1, horizon)

    diff = (actuals - fcst) * mask > 0
    return np.nanmean(diff.reshape(n, m, -1), axis=2).mean(axis=0)
