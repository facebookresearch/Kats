# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
from numba import jit
from torch import Tensor
from torch.nn.modules.loss import _Loss

all_validation_metric_name = ["smape", "sbias", "exceed"]

import pandas as pd

"""
A module for utility functions of global models, including:
    1) Helper functions for preprocessing and calculating error metrics.
    2) NN cells Classes and RNN Class: :class:`LSTM2Cell`, :class:`S2Cell`, and :class:`DilatedRNNStack`.
    3) Loss function Classes: :class:`PinballLoss` and :class:`AdjustedPinballLoss`.
    4) Basic Classes for global model hyper-parameters and time series features: :class:`GMParam` and :class:`GMFeature`.
"""

# for jit
NoneT = torch.FloatTensor([-1e38])

# Define all possible gmfeatures
all_possible_gmfeatures = [
    "last_date",
    "simple_date",
    "tsfeatures",
    "ts2vec",
    "last_hour",
    "last_hour_minute",
    "last_month",
]


@jit
# pyre-fixme[2]: Parameter must be annotated.
def get_filters(isna_idx, seasonality) -> np.ndarray:
    """Helper function for adding NaNs to time series.

    Args:
        isna_idx: A np.ndarry indicating whether the corresponding element is NaN or not.
        seasonality: An integer indicating the seasonality period.

    Returns:
        A `numpy.ndarray` object representing whether or not to keep the corresponding element.
    """
    n = len(isna_idx)
    i = 0
    flips = []
    while i < n:
        if isna_idx[i]:
            cnt = 1
            j = i + 1
            while j < n and isna_idx[j]:
                cnt += 1
                j += 1
            if cnt >= seasonality:
                diff = cnt % seasonality
                flips.append((i + diff, j))
            i = j
        else:
            i += 1
    filters = np.array([True] * n)
    for (i, j) in flips:
        filters[i:j] = False
    return filters


def fill_missing_value_na(
    ts: TimeSeriesData,
    seasonality: int,
    freq: Optional[Union[str, pd.Timedelta]] = None,
) -> TimeSeriesData:
    """Padding holes in time series with NaNs, such that the timestamp difference between any two consecute timestamps is either zero or a multipler of seasonality.

    Args:
        ts: A :class:`kats.consts.TimeSeriesData` object representing the time series to be padded.
        seasonality: An integer representing the period of seasonality, should be positive integer.
        freq: A string or a `pandas.Timedelta` object representing the frequency of time series data.

    Returns:
        A :class:`kats.consts.TimeSeriesData` object representing the padded time series.
    """

    if freq is None:
        freq = ts.infer_freq_robust()
    elif isinstance(freq, str):
        try:
            if freq[0].isdigit():
                freq = pd.to_timedelta(freq)
            else:
                freq = pd.to_timedelta("1" + freq)
        except Exception as e:
            msg = f"Fail to convert freq to pd.Timedelta with error message {e}."
            logging.error(msg)
            raise ValueError(msg)
    elif not isinstance(freq, pd.Timedelta):
        msg = f"freq should be either str or pd.Timedela but receives {type(freq)}."
        logging.error(msg)
        raise ValueError(msg)
    if len(ts) == ((ts.time.max() - ts.time.min()) / freq + 1) or seasonality == 1:
        return ts
    else:
        df = ts.to_dataframe()
        col_name = [t for t in df.columns.values if t != ts.time_col_name][0]
        time_name = ts.time_col_name

        all_ds = pd.DataFrame(
            pd.date_range(df.time.iloc[0], df.time.iloc[-1], freq=freq),
            columns=[time_name],
            copy=False,
        )
        all_ds = all_ds.merge(df, on=time_name, how="left")
        isna_idx = all_ds[col_name].isna().values

        filters = get_filters(isna_idx, seasonality)
        return TimeSeriesData(all_ds.loc[filters])


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def split(
    splits: int,
    overlap: bool,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    train_TSs: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    valid_TSs: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData], None],
) -> List[Tuple[Dict[Any, TimeSeriesData], Optional[Dict[Any, TimeSeriesData]]]]:
    """Split dataset into sub-datasets.

    Args:
        splits: An integer representing the number of sub-datasets to create.
        overlap: A boolean indicating whether the sub-datasets overlap with each other.
        train_TSs: A dictionary or a list of :class:`kats.consts.TimeSeriesData` objects representing the training time series.
        valid_TSs: A dictionary or a list of :class:`kats.consts.TimeSeriesData` objects representing the validation time series.

    Return:
        A list of tuples of dictionaries of :class:`kats.consts.TimeSeriesData` objects. Each element t is a tuple, t[0] is a dictionary of training time series and t[1] is a dictionary of validation time series.
    """

    n = len(train_TSs)

    keys = (
        np.array(list(train_TSs.keys()))
        if isinstance(train_TSs, dict)
        else np.arange(n)
    )

    if splits == 1:  # no need to split the dataset
        return [
            (
                {t: train_TSs[t] for t in keys},
                {t: valid_TSs[t] for t in keys} if valid_TSs is not None else None,
            )
        ]

    m = n // splits
    if m == 0:
        msg = f"Fail to split {n} time series into {splits} subsets."
        logging.error(msg)
        raise ValueError(msg)

    seps = list(range(0, n, m))
    if len(seps) == splits + 1:
        seps[-1] = n
    else:
        seps.append(n)
    index = []
    for i in range(splits):
        tmp = np.array([False] * n)
        tmp[seps[i] : seps[i + 1]] = True
        index.append(tmp)

    if overlap:
        split_data = [
            (
                {t: train_TSs[t] for t in keys[~index[i]]},
                {t: valid_TSs[t] for t in keys[~index[i]]}
                if valid_TSs is not None
                else None,
            )
            for i in range(splits)
        ]
    else:
        split_data = [
            (
                {t: train_TSs[t] for t in keys[index[i]]},
                {t: valid_TSs[t] for t in keys[index[i]]}
                if valid_TSs is not None
                else None,
            )
            for i in range(splits)
        ]
    return split_data


class LSTM2Cell(torch.nn.Module):
    """A modified version of LSTM cell where the output (of size=state_size) is split between h state (of size=h_size) and
    the real output that goes to the next layer (of size=state_size-h_size)

    Attributes:
        input_size: An integer representing the number of expected features in the input tensor.
        h_size: An integer representing h state size.
        state_size: An integer representing c state size.
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self, input_size: int, h_size: int, state_size: int):
        super(LSTM2Cell, self).__init__()
        self.lxh = torch.nn.Linear(input_size + 2 * h_size, 4 * state_size)
        self.h_size = h_size
        # pyre-fixme[4]: Attribute must be annotated.
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
        """Forward function of LSTM2Cell.

        Args:
            input_t: A `torch.Tensor` object representing input features of shape (batch_size, input_size)
            has_prev_state : A boolean specifying whether or not to have previous state.
            has_delayed_state: A boolean specifying whether or not to have delayed state.
            prev_h_state: Optional; A `torch.Tensor` object representing previsous h_state of shape (batch_size, h_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).
            delayed_h_state: Optional; A `torch.Tensor` object representing delayed h_state of shape (batch_size, h_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).
            c_state: Optional; A `torch.Tensor` object representing c_state of shape (batch_size, state_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).

        Returns:
            output_t, (h_state, new_state), where output_t is `torch.Tensor` object representing outputs of shape (batch_size, state_size-h_size);
            h_state is a `torch.Tensor` object representing the next h_state of shape (batch_size, h_size);
            new_state is a `torch.Tensor` object representing the next c_state of shape (batch_size, state_size).
        """

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
    """Slawek's S2 cell.

    A NN cell which is a mix of GRU and LSTM, which also splits output into h and the "real output".

    Attributes:
        input_size: int
            The number of expected features in the input tensor.
        h_size: int
            The number of expected features in the h_state.
        state_size: int
            The number of expected features in the c_state.
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self, input_size: int, h_size: int, state_size: int):
        super(S2Cell, self).__init__()
        self.lxh = torch.nn.Linear(input_size + 2 * h_size, 4 * state_size)
        self.h_size = h_size
        self.state_size = state_size
        # pyre-fixme[4]: Attribute must be annotated.
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
        """Forward method of S2Cell module.

        Args:
            input_t: A `torch.Tensor` object representing input features of shape (batch_size, input_size).
            has_prev_state : A boolean specifying whether or not to have previous state.
            has_delayed_state: A boolean specifying whether or not to have delayed state.
            prev_h_state: Optional; A `torch.Tensor` object representing previsous h_state of shape (batch_size, h_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).
            delayed_h_state: Optional; A `torch.Tensor` object representing delayed h_state of shape (batch_size, h_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).
            prev_c_state: Optional; A `torch.Tensor` object representing previous c_state of shape (batch_size, state_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).
            delayed_c_state: A `torch.Tensor` object representing delayed c_state of shape (batch_size, state_size). Default is NoneT = torch.FloatTensor([-1e38]) (i.e., will not be used).

        Returns:
            A tuple of `torch.tensor` objects, (i.e., output_t, (h_state, new_stat)), where output_t is `torch.Tensor` object representing outputs of shape (batch_size, state_size-h_size);
            h_state is a `torch.Tensor` object representing the next h_state of shape (batch_size, h_size); new_state is a `torch.Tensor` object representing the next c_state of shape (batch_size, state_size).
        """

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
    """The recurrent neural network module for global model.

    Attributes:
        nn_structure: A list of lists of integers representing the strucuture of neural network. For example, [[1,3],[6,12]] defines 2 blocks of 2 layers each and output adaptor layer, with a resNet-style shortcut between output of the first block (output of the second layer)
            and output of the second block (output of 4th layer). The positive integers are the dilation number.
        cell_name: A string representing the name of the cells, can be 'LSTM', 'LSTM2Cell' or 'S2Cell'.
        input_size: An integer representing the number of expected features in the input tensor.
        state_size: An integer representing the c state size (which is hidden_size for a standard LSTM cell).
        output_size: An integer representing the number of expected features in the final output.
        h_size: Optional; An integer representing the number of expected features in h_state. Default is None (i.e., not specified).
        jit: Optional; A boolean specifying whether or not to jit each cell. Default is False.
    """

    def __init__(
        self,
        nn_structure: List[List[int]],
        cell_name: str,
        input_size: int,
        state_size: int,
        output_size: Optional[int] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        h_size=None,
        # pyre-fixme[2]: Parameter must be annotated.
        jit=False,
    ) -> None:
        super(DilatedRNNStack, self).__init__()
        block_num = len(nn_structure)
        self.nn_structure = nn_structure
        self.cell_name = cell_name
        self.input_size = input_size
        # pyre-fixme[4]: Attribute must be annotated.
        self.h_size = h_size
        # pyre-fixme[4]: Attribute must be annotated.
        self.jit = jit
        # pyre-fixme[4]: Attribute must be annotated.
        self.h_state_store = []
        # pyre-fixme[4]: Attribute must be annotated.
        self.c_state_store = []
        # pyre-fixme[4]: Attribute must be annotated.
        self.max_dilation = np.max([np.max(t) for t in nn_structure])

        self.reset_state()

        out_size = self._validate(cell_name, state_size, h_size)

        # pyre-fixme[4]: Attribute must be annotated.
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
        if isinstance(output_size, int) and output_size > 0:
            # pyre-fixme[4]: Attribute must be annotated.
            self.adaptor = torch.nn.Linear(out_size, output_size)
        elif output_size is None:
            self.adaptor = None
        else:
            msg = f"output_size should be either None (for encoder) or a positive integer, but receives {output_size}."
            logging.error(msg)
            raise ValueError(msg)

        # pyre-fixme[4]: Attribute must be annotated.
        self.block_num = block_num
        # pyre-fixme[4]: Attribute must be annotated.
        self.out_size = out_size

    def _validate(self, cell_name: str, state_size: int, h_size: Optional[int]) -> int:
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

        return out_size

    # pyre-fixme[2]: Parameter must be annotated.
    def prepare_decoder(self, decoder) -> None:
        """Prepare a DilatedRNNStack object used as decoder.

        This function copies the last max_dilation tensors in h_state_store and c_state_store to decoder.

        Args:
            decoder: A :class:`DilatedRNNStack` object representing the decoder.
        """
        decoder.h_state_store = self.h_state_store[-self.max_dilation :]
        decoder.c_state_store = self.c_state_store[-self.max_dilation :]
        return

    def _forward_S2Cell(
        self,
        tmp_input: Tensor,
        layer: int,
        has_prev_state: bool,
        has_delayed_state: bool,
        t: int,
        ti_1: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """forward function for S2Cell (to avoid lint warning)."""

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
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward function for LSTM2Cell (to avoid lint warning)."""

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
        """Forward method of DilatedRNNStack

        Args:
            input_t: A `torch.Tensor` object representing input features of shape (batch_size, input_size).

        Returns:
            A `torch.Tensor` object representing outputs of shape (batch_size, output_size).
        """

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

        if self.adaptor is not None:
            output_t = self.adaptor(prev_block_output)
        else:
            output_t = prev_block_output
        return output_t

    def reset_state(self) -> None:
        """Clear all stored state tensors."""
        self.h_state_store = []
        self.c_state_store = []


class PinballLoss(_Loss):
    """Pinball Loss function module.

    For quantile q (0<q<1), forecast value y_hat and true value y, the pinball loss function is defined as:
        pinball(y_hat, y, q)=max((y-y_hat)*q, (y-y_hat)*(q-1)).
    For quantiles Q = [q_1, q_2, ..., q_n] and weights W = [w_1, w_2, ..., w_n], forecasts Y_hat=[y_hat_1, ..., yhat_n] and true value y, the weighted pinball loss is defined as:
        PinballLoss(Y_hat, Y) = Sum_i^n pinball(y_hat_i, y, q_i)*w_i.
    This module provides functionality for computing weighted pinball loss.

    Attributes:
        quantile: A 1-dimensional  `torch.Tensor` object representing the quantiles to be calculated.
        weight: Optional; A 1-dimensional  `torch.Tensor` object representing the weights for quantiles. Default is torch.Tensor([1/n,..,1/n]) where n the number of quantiles.
        reduction: Optional; A string representing the reduction method. Can be 'mean' or 'sum'. Default is 'mean'.
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

        self.quantile = quantile
        self.quantile.requires_grad = False

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
        """
        Args:
            input: A `torch.Tensor` object representing forecasted values of shape (num, n_steps * n_quantiles), where n_quantiles is the length of quantile.
            target: A `torch.Tensor` object contianing true values of shape (num, n_steps).

        Returns:
            A 1-dimensional `torch.Tensor` object representing the computed pinball loss of length the number of quantiles.
        """

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
    """Adjusted Pinball Loss function.

    This is an adjusted version of pinball loss function in that when for the first quantile (i.e., should be 0.5 or close to 0.5), we normalize the original pinball loss with the average value of target and forecasts.
    The idea is to optimize for sMAPE for the median forecasts (i.e., the forecasts for quantile 50). For the other quantile q (0<q<1), pinball loss is defined as:
        pinball(y_hat, y, q)=max((y-y_hat)*q, (y-y_hat)*(q-1)),
    where y_hat is the forecast value and y is the true value.

    For quantiles Q = [q_1, q_2, ..., q_n] and weights W = [w_1, w_2, ..., w_n], forecasts Y_hat=[y_hat_1, ..., yhat_n] and true value y, the adjusted weighted pinball loss is defined as:
        PinballLoss(Y_hat, Y) = 2*pinball(y_hat_1, y, q_1)/(y_hat_1+q_1) + Sum_{i=2}^n pinball(log(y_hat_i), log(y), q_i)*w_i.

    Attributes:
        quantil: A 1-dimensional `torch.Tensor` object representing quantiles to be calculated.
        weight: Optional; A 1-dimensional `torch.Tensor` object representing the weights for quantiles. Default is torch.Tensor([1/n,..,1/n]) where n the number of quantiles.
        reduction: Optional; A string representing the reduction method. Can be 'mean' or 'sum'. Default is 'mean'.
        input_log: Optional; A boolean specifying whether or not the target and the forecast are of logarithmic scale. Default is True.
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
        self.quantile = quantile
        self.quantile.requires_grad = False

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
        """Forward method of AdjustedPinballLoss module.

        Args:
            input: A `torch.Tensor` object representing the forecasts of shape (num, n_steps * n_quantiles), where n_quantiles is the length of quantile.
            target: A `torch.Tensor` object representing true values of shape (num, n_steps)

        Returns:
            A 1-dimensional `torch.Tensor` object representing the computed pinball loss of length the number of quantiles.
        """

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


class GMFeature:
    """Module for computing time series features for global model

    We currently support the following features:
        1) last date feature: a binary features computed on the last timestamp
        2) simple date feature: such as date of week/month/year, etc
        3) tsfeatures: features defined in Kats tsfeature module
        4) time series embedding: embedding from Kats time2vec model # TODO

    This class provides methods including get_base_features and get_on_the_fly_features.

    Attributes:
        feature_type: A string or a list of strings representing the feature names. Each string should be in ['last_date', 'simple_date', 'tsfeatures', 'ts2vec', 'last_hour'].
    """

    def __init__(self, feature_type: Union[List[str], str]) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.all_possible_gmfeatures = all_possible_gmfeatures
        if isinstance(feature_type, str):
            feature_type = [feature_type]

        if not set(feature_type).issubset(set(self.all_possible_gmfeatures)):
            msg = f"feature_type must from {self.all_possible_gmfeatures}."
            logging.error(msg)
            raise ValueError(msg)
        self.feature_type = feature_type

    def get_feature_size(self, ts_length: int) -> int:
        """Calculate the length of feature matrix (i.e., dim 1 of feature matrix) of a time series of length ts_length.

        Args:
            ts_length: An integer representing the length of the time series.

        Returns:
            An integer presenting the length of the feature.
        """
        fixed_feature_lengths = {
            "tsfeatures": 40,
            "ts2vec": 0,
            "last_date": 7 + 27 + 31,
            "last_hour": 24,
            "last_hour_minute": 2,
            "last_month": 12,
        }
        varied_feature_lengths = {"simple_date": 4}
        ans = 0
        for f in self.feature_type:
            ans += fixed_feature_lengths.get(f, 0)
            ans += varied_feature_lengths.get(f, 0) * ts_length
        return int(ans)

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
            features.append(
                np.log(
                    np.abs(
                        list(
                            # pyre-fixme[16]: `List` has no attribute `values`.
                            TsFeatures()
                            .transform(
                                TimeSeriesData(
                                    pd.DataFrame(
                                        {"time": time[i], "value": x[i]}, copy=False
                                    ).dropna()  # NaNs would mess-up tsfeatures
                                )
                            )
                            .values()
                        )
                    )
                )
            )
        features = torch.tensor(features)
        # filter out NaN and inf
        features[torch.isnan(features)] = 0.0
        features[torch.isinf(features)] = 0.0
        return features

    @staticmethod
    def _get_date_feature(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """Private method to get simple date features

        We leverage the properties from `pandas.DatetimeIndex`, and the feature includes:
            - day
            - month
            - dayofweek
            - dayofyear
        """
        feature = []

        for i in range(len(x)):
            pdt = pd.to_datetime(
                time[i]
            )  # converting data type only once to speed up computation
            feature.append(
                np.concatenate(
                    [
                        pdt.day.values,
                        pdt.month.values,
                        pdt.dayofweek.values,
                        pdt.dayofyear.values,
                    ]
                )
            )
        feature = (torch.tensor(feature) + 1.0).log()
        return feature

    @staticmethod
    def _get_last_date_feature(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """Compute date features for the last timestamp."""
        n = len(time)
        m = 7 + 27 + 31
        offset = np.arange(0, n * m, m)
        ans = np.zeros(n * m)
        pdt = pd.to_datetime(time[:, -1])
        indices = []
        # compute day of week indices
        indices.append(pdt.dayofweek.values + offset)
        # compute bi-week indices
        indices.append((pdt.weekofyear.values - 1) // 2 + 7 + offset)
        # compute day of month indices
        indices.append(pdt.day.values + 6 + 27 + offset)
        indices = np.concatenate(indices)
        ans[indices] = 1.0
        return torch.tensor(ans.reshape(n, -1), dtype=torch.get_default_dtype())

    @staticmethod
    def _get_last_hour_feature(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute hour features for the last timestamp.
        """
        n = len(time)
        ans = np.zeros(n * 24)
        indices = pd.to_datetime(time[:, -1]).hour.values + np.arange(0, n * 24, 24)
        ans[indices] = 1.0

        return torch.tensor(ans.reshape(n, -1), dtype=torch.get_default_dtype())

    @staticmethod
    def _get_last_month_feature(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute month features for the last timestamp.
        """
        n = len(time)
        ans = np.zeros(n * 12)
        indices = pd.to_datetime(time[:, -1]).month.values + np.arange(0, n * 12, 12)
        ans[indices] = 1.0
        return torch.tensor(ans.reshape(n, -1), dtype=torch.get_default_dtype())

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    def _get_ts2vec(
        x: np.ndarray,
        time: np.ndarray,
    ):
        # TODO after ts2vec model lands
        pass

    @staticmethod
    def _get_last_hour_minute_feature(
        x: np.ndarray,
        time: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute minute features for the last timestamp.
        """
        pdt = pd.to_datetime(time[:, -1])

        hr = pdt.hour.values + 1
        minute = pdt.minute.values + 1

        return torch.tensor(np.column_stack([hr, minute])).log()

    def get_base_features(
        self,
        x: np.ndarray,
        time: np.ndarray,
    ) -> Optional[Tensor]:
        """Compute selected base features, i.e., the features to be computed only once for each time series.

        Args:
            x: A `numpy.ndarry` object representing the values of the time series data.
            time: A `numpy.ndarry` object representing the timestamps of the time series data.

        Returns:
            A `torch.Tensor` object representing the features.
        """

        funcs = {
            "tsfeatures": self._get_tsfeatures,
            "ts2vec": self._get_ts2vec,
        }
        # get features by given feature types
        features = []
        for ft in self.feature_type:
            if ft in funcs:
                features.append(funcs[ft](x, time))
        if len(features) > 0:
            return torch.cat(features, 1)
        return None

    def get_on_the_fly_features(
        self,
        x: np.ndarray,
        time: np.ndarray,
    ) -> Optional[Tensor]:
        """Compute selected on-the-fly features, i.e., the features to be computed when stepping through RNN.

        Args:
            x: A `numpy.ndarry` object representing the values of the time series data.
            time: A `numpy.ndarry` object representing the timestamps of the time series data.

        Returns:
            A `torch.Tensor` object representing the features.
        """

        funcs = {
            "last_date": self._get_last_date_feature,
            "simple_date": self._get_date_feature,
            "last_hour": self._get_last_hour_feature,
            "last_hour_minute": self._get_last_hour_minute_feature,
            "last_month": self._get_last_month_feature,
        }

        # get features by given feature types
        features = []

        for ft in self.feature_type:
            if ft in funcs:
                features.append(funcs[ft](x, time))

        if len(features) > 0:
            return torch.cat(features, 1)
        return None

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __eq__(self, gmfeature):
        if isinstance(gmfeature, GMFeature):
            if set(gmfeature.feature_type) == set(self.feature_type):
                return True
        return False


class GMParam:
    """A class for storing all the parameters of a global model.

    This module storing all necessary information for building a global model object.

    Attributes:
        freq: A string or a `pandas.Timedelta` object representing the frequency of the time series.
        input_window: An integer representing the length of the input window, i.e., the length of time series feeded into RNN.
        fcst_window: An integer representing the length of the forecast window, i.e., the length of forecasts generated in one forecast step.
        seasonality: Optional; An integer representing the seasonality period. Default is 1, which represents non-seasonality.
        model_type: Optional; A string representing the type of neural network for global model. Can be either 'rnn' or 's2s'. Default is 'rnn'.
        uplifting_ratio: Optional; A float representing the uplifting ratio, which is used for computing the offset value of a time series with negative values given that (max(TS)+offset)/(min(TS)+offset)=uplifing_ratio.
                        Default is 3.0.
        gmfeature: Optional; A string, a list of strings or a :class:`GMFeature` object representing the time series features. Default is None, which means no time series features.
        nn_structure: Optional; A list of lists of integers representing the NN structure of RNN (or encoder). Default value is [[1,3]].
        decoder_nn_structure: Optional; A list of lists of integers representing the NN structure of decoder. Default is None, which takes the same NN structure as encoder when model_type is 's2s', else takes None.
        cell_name: A string representing the name of NN cells, can be 'LSTM2Cell', 'S2Cell' or 'LSTM'. Default is 'LSTM'.
        state_size: Optional; An integer representing the c state size. Default is 50.
        h_size: Optional; An integer representing the h state size. When cell_name is 'LSTM2Cell' or 'S2Cell', h_size should be a positive integer which is less than state_size. Default is None, i.e., not specified.
        optimizer: Optional; A string or a dictionary representing the name and the parameters of the optimizer for training NN. Default is {'name':'Adam', params:{'eps': 1e-7}}.
        loss_function: Optonal; A string representing the loss function, can be 'Pinball' or 'AdjustedPinball'. Default is 'Pinball'.
        quantile: Optional; A list of floats representing the forecast quantiles. Default value is [0.5,0.05,0.95,0.99].
        training_quantile: Optional; A list of floats representing quantiles used for training. Default is None, which takes training_quantile the same value as quantile.
        quantile_weight: Optional; A list of floats representing weights for quantiles during training. Default is None, which sets weight as torch.Tensor([1/n,...,1/n]), where n is the length of quantil.
        validation_metric: Optional; A list of strings representing the names of the error metrics for validation. Default is None, which sets validation_metric as ['smape', 'sbias', 'exceed'].
        batch_size: Optional; A dictionary representing the batch size schedule, whose keys are the epoch numbers and the corresponding values are the batch sizes. Default is None, which sets batch_size as {0:2,3:5,4:15,5:50,6:150,7:500}.
        learning_rate: Optional; A dictionary representing the learning rate schedule, whose keys are epoch numbers and the corresponding values are the learning rates. Default is None, which sets learning_rate as {0: 1e-3, 2: 1e-3/3.}.
        epoch_num: Optional; An integer representing the totoal number of epoches. Default is 8.
        epoch_size: Optional; An integer representing the batches per epoch. Default is 3000.
        init_seasonality: Optional; A list of two floats representing the lower and upper bounds for the initial seasonalities. Default is None, which sets init_seasonality as [0.1, 10.].
        init_smoothing_params: Optional; A list of two floats representing initial values for smoothing parameters, i.e., level smoothing parameter and seasonality smoothing parameter. Default is None, which sets init_smoothing_params as [0.4, 0.6].
        min_training_step_num: Optional; An integer representing the minimum number of training steps. Default is 4.
            Minimum number of training steps.
        min_training_step_length: Optional; An integer representing the minimum training step length. Default is min(1, seasonality-1).
        soft_max_training_step_num: Optional; An integer representing the soft maxinum value for training step. Default is 10.
        validation_step_num: Optional; An integer representing the maximum number of validation steps (maximum validation horizon = validation_step_num * fcst_window). Default is 3.
        min_warming_up_step_num: Optional; An integer representing the mininum number of warming-up steps for forecasting. Default is 2.
        fcst_step_num: Optional; An integer representing the maximum number of forecasting steps. Default is 1.
        jit: Optional; A boolean specifying whether or not to jit every cell of the RNN. Default is False.
        gmname: Optional; A string representing the name of the `GMParam` object. Default is None.
    """

    def __init__(
        self,
        freq: Union[str, pd.Timedelta],
        input_window: int,
        fcst_window: int,
        seasonality: int = 1,
        model_type: str = "rnn",
        uplifting_ratio: float = 3.0,
        gmfeature: Union[
            None, GMFeature, str, List[str]
        ] = None,  # need to be changed once gmfeature is defined
        nn_structure: Optional[List[List[int]]] = None,
        decoder_nn_structure: Optional[List[List[int]]] = None,
        cell_name: str = "LSTM",
        state_size: int = 50,
        h_size: Optional[int] = None,
        optimizer: Optional[Union[str, Dict[str, Any]]] = None,
        loss_function: str = "Pinball",
        quantile: Optional[List[float]] = None,
        training_quantile: Optional[List[float]] = None,
        quantile_weight: Optional[List[float]] = None,
        validation_metric: Optional[List[float]] = None,
        batch_size: Union[None, int, Dict[int, int]] = None,
        learning_rate: Optional[Union[float, Dict[int, float]]] = None,
        epoch_num: int = 8,
        epoch_size: int = 3000,
        init_seasonality: Optional[List[float]] = None,
        init_smoothing_params: Optional[List[float]] = None,
        min_training_step_num: int = 4,
        min_training_step_length: int = -1,
        soft_max_training_step_num: int = 10,
        validation_step_num: int = 3,
        min_warming_up_step_num: int = 2,
        fcst_step_num: int = 1,
        jit: bool = False,
        gmname: Optional[str] = None,
    ) -> None:

        self._valid_freq(freq)

        self.model_type: str = self._valid_model_type(model_type)

        # valid uplifiting ratio
        if uplifting_ratio < 0:
            msg = f"uplifting_ratio should be a positive float but receive {uplifting_ratio}."
            logging.error(msg)
            raise ValueError(msg)
        self.uplifting_ratio = uplifting_ratio

        nn_structure, decoder_nn_structure = self._valid_nn_structure(
            nn_structure, decoder_nn_structure
        )
        self.nn_structure: List[List[int]] = nn_structure
        self.decoder_nn_structure: List[List[int]] = decoder_nn_structure

        self.cell_name = cell_name
        self.state_size = state_size
        self.h_size = h_size

        batch_size = (
            batch_size
            if batch_size is not None
            else {0: 2, 3: 5, 4: 15, 5: 50, 6: 150, 7: 500}
        )
        self.batch_size: Dict[int, int] = cast(
            Dict[int, int],
            # pyre-fixme: pyre fail to infer correct data type
            self._valid_union_dict(batch_size, "batch_size", int, int),
        )
        learning_rate = (
            learning_rate if learning_rate is not None else {0: 1e-3, 2: 1e-3 / 3.0}
        )
        self.learning_rate: Dict[int, float] = self._valid_union_dict(
            learning_rate, "learning_rate", int, float
        )
        self.loss_function: str = self._valid_loss_function(loss_function)

        self.optimizer: Dict[str, Any] = self._valid_optimizer(optimizer)

        self.quantile: List[float] = (
            quantile if quantile is not None else [0.5, 0.05, 0.95, 0.99]
        )
        self._valid_list(self.quantile, "quantile", 0, 1)

        # additional check needed for filling NaNs during training.
        if self.quantile[0] != 0.5:
            msg = f"The first element of quantile should be 0.5 but receives {self.quantile[0]}."
            logging.error(msg)
            raise ValueError(msg)

        if training_quantile is None:
            self.training_quantile: List[float] = self.quantile
        else:
            self._valid_list(training_quantile, "training_quantile", 0, 1)
            self.training_quantile = training_quantile

        if quantile_weight is None:
            self.quantile_weight: List[float] = [1.0 / len(self.quantile)] * len(
                self.quantile
            )
        else:
            if len(quantile_weight) != len(self.quantile):
                msg = "quantile and quantile_weight should be of the same length."
                logging.error(msg)
                raise ValueError(msg)
            self._valid_list(quantile_weight, "quantile_weight", 0, np.inf)
            self.quantile_weight = quantile_weight

        if validation_metric is None:
            # pyre-fixme[4]: Attribute must be annotated.
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

        self.init_seasonality: List[float] = (
            init_seasonality if init_seasonality is not None else [0.1, 10.0]
        )
        self._valid_list(self.init_seasonality, "init_seasonality", 0, np.inf)

        self.init_smoothing_params: List[float] = (
            init_smoothing_params if init_smoothing_params is not None else [0.4, 0.6]
        )
        self._valid_list(self.init_smoothing_params, "init_smoothing_params", 0, np.inf)

        if min_training_step_length <= 0:
            min_training_step_length = max(1, seasonality - 1)

        self.input_window = input_window

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

        self._valid_gmfeature(gmfeature, input_window)

        self._valid_positive_integer_params(pos_integer_params)

        self.input_window = input_window
        self.fcst_window = fcst_window
        self.validation_step_num = validation_step_num
        self.min_training_step_num = min_training_step_num
        self.min_training_step_length = min_training_step_length
        self.seasonality = seasonality
        self.state_size = state_size
        self.epoch_num = epoch_num
        self.epoch_size = epoch_size
        self.soft_max_training_step_num = soft_max_training_step_num
        self.validation_step_num = validation_step_num
        self.min_warming_up_step_num = min_warming_up_step_num
        self.fcst_step_num = fcst_step_num

        # max_step_delta needed for training/testing
        self.max_step_delta: int = (
            min(input_window, fcst_window) // min_training_step_length
        )

        self.jit = jit
        self.gmname = gmname

    def _valid_model_type(self, model_type: str) -> str:
        if model_type not in ["rnn", "s2s"]:
            msg = (
                f"model_type should be either 'rnn' or 's2s but receives {model_type}."
            )
            logging.error(msg)
            raise ValueError(msg)
        return model_type

    def _valid_nn_structure(
        self,
        nn_structure: Optional[List[List[int]]],
        decoder_nn_structure: Optional[List[List[int]]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        nn_structure = nn_structure if nn_structure is not None else [[1, 3]]
        if self.model_type == "s2s":
            decoder_nn_structure = (
                decoder_nn_structure
                if decoder_nn_structure is not None
                else nn_structure
            )
        else:
            decoder_nn_structure = []
        return nn_structure, decoder_nn_structure

    def _valid_optimizer(
        self, optimizer: Union[str, Dict[str, Any], None]
    ) -> Dict[str, Any]:
        opt_methods = ["adam"]
        if optimizer is None:
            return {"name": "Adam", "params": {"eps": 1e-7}}
        elif isinstance(optimizer, str) and optimizer.lower() == "adam":
            return {"name": "Adam"}
        elif isinstance(optimizer, dict):
            if "name" in optimizer and optimizer["name"].lower() in opt_methods:
                return optimizer
        msg = f"`optimizer`={optimizer} is invalid."
        logging.error(msg)
        raise ValueError(msg)

    def _valid_loss_function(self, loss_function: str) -> str:
        """Helper function to verify loss function."""

        if loss_function.lower() in ["pinball", "adjustedpinball"]:
            return loss_function.lower()
        else:
            msg = f"loss function {loss_function} is not implemented."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_list(
        self,
        value: List[float],
        name: str,
        lower: float,
        upper: float,
    ) -> None:
        """
        Helper function to verify list inputs.
        """
        for q in value:
            if q >= upper or q <= lower:
                msg = f"Each element in `{name}` should be a float in ({lower}, {upper}) but receives {q}."
                logging.error(msg)
                raise ValueError(msg)

    def _valid_union_dict(
        self,
        value: Union[int, float, Dict[int, Union[int, float]]],
        name: str,
        key_type: Type[int],
        value_type: Type[Union[int, float]],
    ) -> Dict[int, Union[int, float]]:
        """
        Helper function to verify batch_size and learning_rate.
        """
        if isinstance(value, value_type) and value > 0:
            return {0: value}
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
            return value
        else:
            msg = f"{name} should be either positive {value_type} or a dictionary, but receive {value}."
            logging.error(msg)
            raise ValueError(msg)

    def _valid_positive_integer_params(self, params: Dict[str, int]) -> None:
        """Verify params are positive integers."""
        for elm in params:
            if not isinstance(params[elm], int) or params[elm] < 1:
                msg = f"{elm} should be a positive integer but receive {params[elm]}."
                logging.error(msg)
                raise ValueError(msg)

    def _valid_freq(self, freq: Union[str, pd.Timedelta]) -> None:
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

    def to_string(self) -> str:
        params = self.__dict__.copy()
        # encode freq
        params["freq"] = params["freq"].value
        # encode gmfeature
        if params["gmfeature"] is not None:
            params["gmfeature"] = params["gmfeature"].feature_type
        return json.dumps(params)

    # pyre-fixme[2]: Parameter must be annotated.
    def __eq__(self, gmparam) -> bool:
        if isinstance(gmparam, GMParam):
            for elm in self.__dict__:
                if self.__dict__[elm] != getattr(gmparam, elm):
                    return False
            return True
        return False

    # pyre-fixme[2]: Parameter must be annotated.
    def _valid_gmfeature(self, gmfeature, input_window) -> None:

        if gmfeature is None:
            self.gmfeature = None
            return

        if isinstance(gmfeature, GMFeature):
            pass
        elif isinstance(gmfeature, str) or isinstance(gmfeature, List):
            gmfeature = GMFeature(gmfeature)
        else:
            msg = f"gmfeature {gmfeature} is invalid."
            logging.error(msg)
            raise ValueError(msg)

        self.gmfeature = gmfeature

    def copy(self) -> object:
        """Generate a copy of the :class:`GMParam` object.

        Returns:
            A :class:`GMParam` object which is the copy of the original :class:`GMParam` object.
        """
        tmp_dict = self.__dict__.copy()
        del tmp_dict["max_step_delta"]
        return GMParam(**tmp_dict)


def gmparam_from_string(gmstring: str) -> GMParam:
    """Convert a json format string to a :class:`GMParam` object.

    Args:
        gmstring: A string reprenting the json format string encoding the :class:`GMParam` object.

    Returns:
        A :class:`GMParam` object.
    """

    gmparam_dict = json.loads(gmstring)
    gmparam_dict["freq"] = pd.to_timedelta(gmparam_dict["freq"], unit="ns")
    del gmparam_dict["max_step_delta"]

    if isinstance(gmparam_dict["batch_size"], dict):
        gmparam_dict["batch_size"] = {
            int(t): gmparam_dict["batch_size"][t] for t in gmparam_dict["batch_size"]
        }
    if isinstance(gmparam_dict["learning_rate"], dict):
        gmparam_dict["learning_rate"] = {
            int(t): gmparam_dict["learning_rate"][t]
            for t in gmparam_dict["learning_rate"]
        }
    gmparam = GMParam(**gmparam_dict)
    return gmparam


# pyre-fixme[3]: Return annotation cannot contain `Any`.
def gmpreprocess(
    gmparam: GMParam,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    data: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
    mode: str,
    valid_set: bool = True,
) -> Tuple[Dict[Any, TimeSeriesData], Optional[Dict[Any, TimeSeriesData]]]:
    """Proprecessing funtion for global models.

    This function preprocesses time series data for global models. It mainly provides two functionalities:
    1) fill in missing values with NaNs if necessary; and 2) discard time series that are too short.

    Args:
        gmparam: A :class:`GMParam` object which will be used for global model.
        data: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects storing the time series data.
        mode: A string specifying the mode. Can be 'train' or 'test'.
        valid_set: Optional; A boolean specifying whether or not to extract a validation set (i.e., spliting a single time series into two parts). Default is True.
    """

    if mode == "train":
        length = (
            gmparam.input_window
            + gmparam.fcst_window * gmparam.fcst_step_num
            + gmparam.min_training_step_num * gmparam.min_training_step_length
        )
        valid_length = (
            gmparam.fcst_window * gmparam.validation_step_num if valid_set else 0
        )
        length += valid_length

    # for test mode
    elif mode == "test":
        valid_length = gmparam.fcst_step_num * gmparam.fcst_window
        length = valid_length + gmparam.seasonality

    else:
        msg = f"mode should be either 'train' or 'test' but receives {mode}."
        logging.error(msg)
        raise ValueError(msg)

    keys = (
        np.array(list(data.keys())) if isinstance(data, dict) else np.arange(len(data))
    )

    train_TSs = {}
    valid_TSs = {}
    for k in keys:
        ts = fill_missing_value_na(data[k], gmparam.seasonality, gmparam.freq)
        if len(ts) >= length:
            if valid_set:
                train_TSs[k] = ts[:-valid_length]
                valid_TSs[k] = ts[-valid_length:]
            else:
                train_TSs[k] = ts
    if not valid_set:
        valid_TSs = None
    logging.info(
        f"Processed {len(data)} time series and returned {len(train_TSs)} valid time series."
    )
    return train_TSs, valid_TSs


def calc_min_input_length(gmparam: GMParam) -> int:
    """Calculate the minimum length of training data given the :class:`GMParam` object.

    Args:
        gmparam: A :class:`GMParam` object.

    Returns:
        An integer representing the minimum length of training data.
    """

    return int(
        gmparam.input_window
        + gmparam.min_warming_up_step_num * gmparam.min_training_step_length
    )


def calc_max_fcst_horizon(gmparam: GMParam) -> int:
    """Calculate the maximum length of forecasting horizon given :class:`GMParam`.

    Args:
        gmparam: A :class:`GMParam` object.

    Returns:
        An integer representing the maximum length of forecasting horizon.
    """

    return int(gmparam.fcst_window * gmparam.fcst_step_num)


def pad_ts(ts: TimeSeriesData, n: int, freq: pd.Timedelta) -> TimeSeriesData:
    """Pad time series data to increase its length by n.

    Args:
        ts: A :class:`kats.consts.TimeSeriesData` object representing the time series data.
        n: An integer representing the increase length.
        freq: A `pandas.Timedelta` object representing the frequency of the time series data.

    Returns:
        A :class:`kats.consts.TimeSeriesData` object after padding.
    """
    if n < 1 or not isinstance(n, int):
        msg = f"Padding length n should be a positive integer, but receive {n}."
        logging.error(msg)
        raise ValueError(msg)

    df = ts.to_dataframe()
    time_col = ts.time_col_name
    val_col = [col for col in df.columns if col != time_col][0]
    pad_val = df[val_col].values[df[val_col].first_valid_index()]
    # add the padding value
    df = df.append(
        {time_col: df[time_col].iloc[0] - n * freq, val_col: pad_val},
        ignore_index=True,
    )
    return TimeSeriesData(df)
