# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The LSTM model stands for Long short-term memory, it is a recurrent neural network model that can be used for sequential data.

More information for the model can be found: https://en.wikipedia.org/wiki/Long_short-term_memory
We directly adopt the PyTorch implementation and apply the model for time series forecast. More details for the PyTorch modules are here: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, List, Dict, Any

import kats.models.model as mm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kats.consts import Params, TimeSeriesData
from sklearn.preprocessing import MinMaxScaler


class LSTMParams(Params):
    """Parameter class for time series LSTM model

    This is the parameter class for time series LSTM model, it currently contains three parameters

    Attributes:
        hidden_size: LSTM hidden unit size
        time_window: Time series sequence length that feeds into the model
        num_epochs: Number of epochs for the training process
    """

    __slots__ = ["hidden_size", "time_window", "num_epochs"]

    def __init__(self, hidden_size: int, time_window: int, num_epochs: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.time_window = time_window
        self.num_epochs = num_epochs

        logging.debug(
            "Initialized LSTMParams instance."
            f"hidden_size:{hidden_size}, time_window:{time_window}, num_epochs:{num_epochs}"
        )

    def validate_params(self):
        logging.info("Method validate_params() is not implemented.")
        pass


class LSTMForecast(nn.Module):
    """Torch forecast class for time series LSTM model

    This is the forecast class for time series LSTM model inherited from the PyTorch module, detailed implementation for the core LSTM and Linear modules can be gound here:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    Attributes:
        params: A LSTMParams instance for parameters
        input_size: Input unit feature size for the LSTM layer
        output_size: Output unit feature size from the output Linear layer
    """

    def __init__(self, params: LSTMParams, input_size: int, output_size: int) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=params.hidden_size)

        self.linear = nn.Linear(
            in_features=params.hidden_size, out_features=output_size
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """The forward method for the LSTM forecast PyTorch module

        Args:
            input_seq: A torch tensor contains the input data sequence for the LSTM layer

        Returns:
            prediction: A torch tensor contains the output prediction from the output Linear layer
        """

        # pyre-fixme[16]: `LSTMForecast` has no attribute `hidden_cell`.
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTMModel(mm.Model):
    """Kats model class for time series LSTM model

    This is the Kats model class for time series forecast using the LSTM model

    Attributes:
        data: :class:`kats.consts.TimeSeriesData`, the input data
        params: A LSTMParams object for the parameters
    """

    def __init__(self, data: TimeSeriesData, params: LSTMParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = (
                f"Only support univariate time series, but get {type(self.data.value)}."
            )
            logging.error(msg)
            raise ValueError(msg)

    def __setup_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare input data for the LSTM model

        This method will perform a min-max normalization on the input data, then output normalized input sequence data and true values for the prediction. More details for the normalizer:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

        Args:
            None

        Returns:
            A list of tuples that include both the input sequence tensor (with normalized values) and ground truth value for prediction
        """

        train_data = self.data.value.values.astype(float)

        # scaling using MinMaxScaler
        # pyre-fixme[16]: `LSTMModel` has no attribute `scaler`.
        # pyre-fixme[16]: Module `sklearn` has no attribute `preprocessing`.
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1))
        # converting to Tensor
        # pyre-fixme[16]: `LSTMModel` has no attribute `train_data_normalized`.
        self.train_data_normalized = torch.FloatTensor(train_data_scaled).view(-1)

        # generating sequence
        inout_seq = []

        for i in range(len(self.train_data_normalized) - self.params.time_window):
            train_seq = self.train_data_normalized[i : i + self.params.time_window]
            train_label = self.train_data_normalized[
                i + self.params.time_window : i + self.params.time_window + 1
            ]
            inout_seq.append((train_seq, train_label))

        return inout_seq

    def fit(self, **kwargs) -> None:
        """Fit the LSTM forecast model

        Args:
            None

        Returns:
            The fitted LSTM model object
        """

        logging.debug("Call fit() with parameters." f"kwargs:{kwargs}")

        # learning rate
        # pyre-fixme[16]: `LSTMModel` has no attribute `lr`.
        self.lr = kwargs.get("lr", 0.001)

        # supports univariate time series, multivariate support in the future
        # pyre-fixme[16]: `LSTMModel` has no attribute `model`.
        # pyre-fixme[16]: `LSTMModel` has no attribute `params`.
        self.model = LSTMForecast(params=self.params, input_size=1, output_size=1)

        # loss function
        loss_function = nn.MSELoss()

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # inout_seq
        train_inout_seq = self.__setup_data()

        for i in range(self.params.num_epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                # pyre-fixme[16]: `LSTMForecast` has no attribute `hidden_cell`.
                self.model.hidden_cell = (
                    torch.zeros(1, 1, self.params.hidden_size),
                    torch.zeros(1, 1, self.params.hidden_size),
                )
                # prediction using input data
                y_pred = self.model(seq)
                # calculating loss
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 25 == 1:
                logging.info(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

        # pyre-fixme[7]: Expected `None` but got `LSTMModel`.
        return self

    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
        """Prediction function for a multi-step forecast

        Args:
            steps: number of steps for the forecast

        Returns:
            A pd.DataFrame that includes the forecast and confidence interval
        """

        logging.debug(
            "Call predict() with parameters. " f"steps:{steps}, kwargs:{kwargs}"
        )
        # pyre-fixme[16]: `LSTMModel` has no attribute `freq`.
        # pyre-fixme[16]: `LSTMModel` has no attribute `data`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))

        # pyre-fixme[16]: `LSTMModel` has no attribute `model`.
        self.model.eval()

        # get last train input sequence
        # pyre-fixme[16]: `LSTMModel` has no attribute `train_data_normalized`.
        # pyre-fixme[16]: `LSTMModel` has no attribute `params`.
        test_inputs = self.train_data_normalized[-self.params.time_window :].tolist()

        for _ in range(steps):
            seq = torch.FloatTensor(test_inputs[-self.params.time_window :])
            with torch.no_grad():
                self.model.hidden = (
                    torch.zeros(1, 1, self.params.hidden_size),
                    torch.zeros(1, 1, self.params.hidden_size),
                )
                test_inputs.append(self.model(seq).item())

        # inverse transform
        # pyre-fixme[16]: `LSTMModel` has no attribute `scaler`.
        fcst_denormalized = self.scaler.inverse_transform(
            np.array(test_inputs[self.params.time_window :]).reshape(-1, 1)
        ).flatten()
        logging.info("Generated forecast data from LSTM model.")
        logging.debug(f"Forecast data: {fcst_denormalized}")

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        # pyre-fixme[16]: `LSTMModel` has no attribute `dates`.
        self.dates = dates[dates != last_date]  # Return correct number of periods
        # pyre-fixme[16]: `LSTMModel` has no attribute `y_fcst`.
        self.y_fcst = fcst_denormalized
        # pyre-fixme[16]: `LSTMModel` has no attribute `y_fcst_lower`.
        self.y_fcst_lower = fcst_denormalized * 0.95
        # pyre-fixme[16]: `LSTMModel` has no attribute `y_fcst_upper`.
        self.y_fcst_upper = fcst_denormalized * 1.05

        # pyre-fixme[16]: `LSTMModel` has no attribute `fcst_df`.
        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            }
        )

        logging.debug(f"Return forecast data: {self.fcst_df}")

        return self.fcst_df

    def plot(self):
        """Plot forecast results from the LSTM model"""

        mm.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "LSTM"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, Any]]:
        """Get default parameter search space for the LSTM model

        Args:
            None

        Returns:
            A dictionary with the default LSTM parameter search space.
        """

        return [
            {
                "name": "hidden_size",
                "type": "choice",
                "values": list(range(1, 500, 10)),
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "time_window",
                "type": "choice",
                "values": list(range(1, 20, 1)),
                "value_type": "int",
                "is_ordered": True,
            },
            {
                "name": "num_epochs",
                "type": "choice",
                "values": list(range(50, 2000, 50)),
                "value_type": "int",
                "is_ordered": True,
            },
        ]
