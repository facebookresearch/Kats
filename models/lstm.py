#!/usr/bin/env python3

'''
Forecasting with LSTM Model
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import infrastrategy.kats.models.model as mm
from infrastrategy.kats.consts import Params, TimeSeriesData


class LSTMParams(Params):
    __slots__ = ["hidden_size", "time_window", "num_epochs"]

    def __init__(self, hidden_size : int, time_window : int, num_epochs: int) -> None:
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

    def __init__(self, params: LSTMParams, input_size, output_size) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=params.hidden_size)

        self.linear = nn.Linear(in_features=params.hidden_size, out_features=output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        '''
        forward pass through lstm, linear layers and prediction is returned to calling function

        Args: input_seq (Tensor) - Preprocessed Input Data

        Returns: prediction(Tensor)
        '''

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTMModel(mm.Model):

    def __init__(self, data: TimeSeriesData, params: LSTMParams) -> None:
        super().__init__(data, params)
        if not isinstance(self.data.value, pd.Series):
            msg = f"Only support univariate time series, but get {type(self.data.value)}."
            logging.error(msg)
            raise ValueError(msg)

    def __setup_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Scales and converts the input data to sequence of length defined in time_window

        Args: None

        Returns: (X,Y)-> List[Tuple[Tensor,Tensor]]]
        '''
        train_data = self.data.value.values.astype(float)

        #scaling using MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1))
        #converting to Tensor
        self.train_data_normalized = torch.FloatTensor(train_data_scaled).view(-1)

        #generating sequence
        inout_seq = []

        for i in range(len(self.train_data_normalized) - self.params.time_window):
            train_seq = self.train_data_normalized[i:i + self.params.time_window]
            train_label = self.train_data_normalized[i + self.params.time_window : i + self.params.time_window + 1]
            inout_seq.append((train_seq, train_label))

        return inout_seq

    def fit(self, **kwargs) -> None:
        '''
        Fit the LSTM model

        Args: None

        Returns: None
        '''
        logging.debug("Call fit() with parameters."
        f"kwargs:{kwargs}")

        # learning rate
        self.lr = kwargs.get("lr", 0.001)

        # supports univariate time series, multivariate support in the future
        self.model = LSTMForecast(params=self.params, input_size=1, output_size=1)

        # loss function
        loss_function = nn.MSELoss()

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #inout_seq
        train_inout_seq = self.__setup_data()

        for i in range(self.params.num_epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.params.hidden_size),
                                    torch.zeros(1, 1, self.params.hidden_size))
                # prediction using input data
                y_pred = self.model(seq)
                # calculating loss
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 25 == 1:
                logging.info(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        return self

    def predict(self, steps: int, **kwargs) -> pd.DataFrame:
        '''
        Perform multi-step forecasting

        Args: steps (int)

        Returns: fcst_df (pd.DataFrame)
        '''
        logging.debug(
            "Call predict() with parameters. "
            f"steps:{steps}, kwargs:{kwargs}"
        )
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))

        self.model.eval()

        # get last train input sequence
        test_inputs = self.train_data_normalized[-self.params.time_window:].tolist()

        for _ in range(steps):
            seq = torch.FloatTensor(test_inputs[-self.params.time_window:])
            with torch.no_grad():
                self.model.hidden = (torch.zeros(1, 1, self.params.hidden_size),
                                torch.zeros(1, 1, self.params.hidden_size))
                test_inputs.append(self.model(seq).item())

        # inverse transform
        fcst_denormalized = self.scaler.inverse_transform(np.array(test_inputs[self.params.time_window:]).reshape(-1, 1)).flatten()
        logging.info("Generated forecast data from LSTM model.")
        logging.debug(f"Forecast data: {fcst_denormalized}")

        last_date = self.data.time.max()
        dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)
        self.dates = dates[dates != last_date]  # Return correct number of periods
        self.y_fcst = fcst_denormalized
        self.y_fcst_lower = fcst_denormalized * 0.95
        self.y_fcst_upper = fcst_denormalized * 1.05

        self.fcst_df = pd.DataFrame(
            {
                "time": self.dates,
                "fcst": self.y_fcst ,
                "fcst_lower": self.y_fcst_lower,
                "fcst_upper": self.y_fcst_upper,
            }
        )

        logging.debug(f"Return forecast data: {self.fcst_df}")

        return self.fcst_df

    def plot(self):
        '''
        Method to plot forecast graph

        Args: None

        Returns:None
        '''
        mm.Model.plot(self.data, self.fcst_df)

    def __str__(self):
        return "LSTM"

    @staticmethod
    def get_parameter_search_space() -> List[Dict[str, object]]:
        '''
        get parameter search space for LSTM model

        Args: None

        Returns: parameter search space for hidden_size, time_window, num_epochs (List[Dict[str,obj]])
        '''
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
