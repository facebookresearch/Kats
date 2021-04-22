#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
import pandas as pd
import torch
from kats.consts import TimeSeriesData
from kats.models.globalmodel.backtester import GMBackTester
from kats.models.globalmodel.data_processor import GMDataLoader, GMBatch
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.utils import (
    LSTM2Cell,
    S2Cell,
    DilatedRNNStack,
    PinballLoss,
    GMParam,
    AdjustedPinballLoss,
    GMFeature,
    gmparam_from_string,
)


def get_ts(n, start_time, freq="D", has_nans=True):
    """
    Helper function for generating TimeSeriesData.
    """
    t = pd.Series(pd.date_range(start_time, freq=freq, periods=n))
    val = np.random.randn(n)
    if has_nans:
        idx = np.random.choice(range(n), int(n * 0.2), replace=False)
        val[idx] = np.nan
    val = pd.Series(val)
    return TimeSeriesData(time=t, value=val)


class TestGMParam(TestCase):
    def test_daily(self):
        GMParam(
            freq="d",
            input_window=45,
            fcst_window=30,
            seasonality=7,
            quantile=[0.5, 0.05, 0.95, 0.99],
            training_quantile=[0.58, 0.05, 0.94, 0.985],
        )


class LSTM2CellTest(TestCase):
    def test_cell(self):
        cell = LSTM2Cell(30, 2, 5)
        input_t = torch.randn(5, 30)
        prev_h = torch.randn(5, 2)
        delayed_h = torch.randn(5, 2)
        c = torch.randn(5, 5)
        cell(input_t, False, False)
        cell(input_t, True, False, prev_h, c)
        cell(input_t, True, True, prev_h, delayed_h, c)


class S2CellTest(TestCase):
    def test_cell(self):
        cell = S2Cell(20, 2, 5)
        input_t = torch.randn(5, 20)
        prev_h = torch.randn(5, 2)
        delayed_h = torch.randn(5, 2)
        prev_c = torch.randn(5, 5)
        delayed_c = torch.randn(5, 5)
        cell(input_t, False, False)
        cell(input_t, True, False, prev_h_state=prev_h, prev_c_state=prev_c)
        cell(input_t, True, True, prev_h, delayed_h, prev_c, delayed_c)


class DilatedRNNStackTest(TestCase):
    def test_rnn(self):
        # test LSTM
        input_t = torch.randn(5, 20)
        rnn = DilatedRNNStack([[1, 2], [1, 2, 3]], "LSTM", 20, 50, 10)
        for _ in range(6):
            rnn(input_t)
        # test LSTM2Cell
        rnn = DilatedRNNStack([[1, 2], [1, 2, 2]], "LSTM2Cell", 20, 50, 10, 2)
        for _ in range(6):
            rnn(input_t)
        # test S2Cell
        rnn = DilatedRNNStack([[1, 1, 2], [2, 2]], "S2Cell", 20, 50, 10, 2)
        for _ in range(6):
            rnn(input_t)

    def test_others(self):
        self.assertRaises(ValueError, DilatedRNNStack, [], "randomcell", 20, 20, 10)
        self.assertRaises(ValueError, DilatedRNNStack, [], "LSTM2Cell", 20, 20, 10)
        self.assertRaises(ValueError, DilatedRNNStack, [], "LSTM2Cell", 20, 20, 10, 20)


class PinballLossTest(TestCase):
    def test_pinballloss(self):
        quantile = torch.tensor([0.5, 0.05, 0.95, 0.85])
        rnn = DilatedRNNStack(
            [[1, 2], [2, 2, 4]],
            "LSTM",
            input_size=20,
            state_size=50,
            output_size=3 * len(quantile),
        )
        fcst = rnn(torch.randn(3, 20))
        actuals = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                ],
                [
                    np.nan,
                    2.0,
                    1.0,
                ],
                [np.nan, np.nan, np.nan],
            ]
        ).log()
        pbl = PinballLoss(quantile)
        loss = pbl(fcst, actuals)
        sum_loss = loss.sum()
        # test auto_grad
        sum_loss.backward()

    def test_other(self):
        self.assertRaises(ValueError, PinballLoss, quantile=torch.tensor([]))
        self.assertRaises(ValueError, PinballLoss, quantile=torch.tensor([[]]))
        self.assertRaises(ValueError, PinballLoss, quantile=torch.tensor([[0.5]]))

        self.assertRaises(
            ValueError,
            PinballLoss,
            quantile=torch.tensor([0.5]),
            weight=torch.tensor([0.5, 0.6]),
        )

        quantile = torch.tensor([0.5, 0.05, 0.95])
        pbl = PinballLoss(quantile)

        self.assertRaises(ValueError, pbl, torch.randn(2, 9), torch.randn(3, 3))
        self.assertRaises(ValueError, pbl, torch.randn(2, 9), torch.randn(2, 5))


dl_dataset = [
    TimeSeriesData(
        pd.DataFrame(
            {"time": pd.date_range("2020-05-06", periods=t), "y": np.random.randn(t)}
        )
    )
    for t in range(10)
]
names = ["ts_" + str(i) for i in range(10)]

dl_dataset_dict = {names[i]: dl_dataset[i] for i in range(10)}


class GMFeatureTest(TestCase):
    def test_gmfeature(self):
        TSs = [get_ts(30, "2020-05-06")]
        x = np.row_stack([np.abs(ts.value.values) for ts in TSs])
        time = np.row_stack([ts.time.values for ts in TSs])

        gmfs = [
            GMFeature(feature_size=40, feature_type="tsfeatures"),
            GMFeature(feature_size=7 + 27 + 31, feature_type="last_date"),
            GMFeature(feature_size=4 * 30, feature_type="simple_date"),
            GMFeature(
                feature_size=4 * 30,
                feature_type=["simple_date", "tsfeatures", "last_date"],
            ),
        ]

        for gmf in gmfs:
            base_features = gmf.get_base_features(x, time)
            on_the_fly_features = gmf.get_on_the_fly_features(x, time)
            if base_features is not None:
                self.assertEqual(torch.isnan(base_features).sum(), 0)
                self.assertEqual(torch.isinf(base_features).sum(), 0)
            if on_the_fly_features is not None:
                self.assertEqual(torch.isnan(on_the_fly_features).sum(), 0)
                self.assertEqual(torch.isinf(on_the_fly_features).sum(), 0)


class GMDataLoaderTest(TestCase):
    def test_dataloader(self):

        gmdl = GMDataLoader(dl_dataset)
        collects = []
        for _ in range(5):
            batch = gmdl.get_batch(2)
            collects.extend(batch)
        # verify that all ids in [0,9] are visited
        self.assertEqual(set(collects) == set(range(10)), True)
        for _ in range(5):
            batch = gmdl.get_batch(4)
        # verify that all ids are returned
        batch = gmdl.get_batch(20)
        self.assertEqual(set(range(10)) == set(batch), True)

        gmdl = GMDataLoader(dl_dataset_dict)
        collects = []
        for _ in range(5):
            batch = gmdl.get_batch(3)
            collects.extend(batch)
        self.assertEqual(set(collects) == set(dl_dataset_dict.keys()), True)

    def test_others(self):
        self.assertRaises(ValueError, GMDataLoader, [])
        gmdl = GMDataLoader(dl_dataset)
        self.assertRaises(ValueError, gmdl.get_batch, 0.5)
        self.assertRaises(ValueError, gmdl.get_batch, -1)


class GMBatchTest(TestCase):
    def test_batch(self):
        train_ts = {
            str(i): get_ts(n * 3, "2020-05-06") for i, n in enumerate([10, 12, 15])
        }
        valid_ts = {
            str(i): get_ts(n * 3, "2021-05-06") for i, n in enumerate([5, 6, 7])
        }
        batch_ids = [str(i) for i in range(3)]

        gmparam_1 = GMParam(
            freq="d", input_window=10, fcst_window=7, seasonality=3, fcst_step_num=2
        )
        batch = GMBatch(
            gmparam_1,
            train_TSs=train_ts,
            valid_TSs=valid_ts,
            batch_ids=batch_ids,
            mode="train",
        )
        self._valid(batch, gmparam_1)

        batch = GMBatch(
            gmparam_1,
            train_TSs=train_ts,
            valid_TSs=valid_ts,
            batch_ids=batch_ids,
            mode="test",
        )
        self._valid(batch, gmparam_1)

        batch = GMBatch(
            gmparam_1,
            train_TSs=train_ts,
            valid_TSs=None,
            batch_ids=batch_ids,
            mode="train",
        )
        self._valid(batch, gmparam_1)

        batch = GMBatch(
            gmparam_1,
            train_TSs=valid_ts,
            valid_TSs=None,
            batch_ids=batch_ids,
            mode="test",
        )
        self._valid(batch, gmparam_1)

        gmparam_2 = GMParam(freq="d", input_window=10, fcst_window=7, seasonality=1)
        batch = GMBatch(
            gmparam_2,
            train_TSs=train_ts,
            valid_TSs=valid_ts,
            batch_ids=batch_ids,
            mode="train",
        )
        self._valid(batch, gmparam_2)

        gmparam_3 = GMParam(
            freq="d",
            input_window=10,
            fcst_window=7,
            seasonality=1,
            gmfeature="simple_date",
        )
        batch = GMBatch(
            gmparam_3,
            train_TSs=train_ts,
            valid_TSs=valid_ts,
            batch_ids=batch_ids,
            mode="train",
        )
        self._valid(batch, gmparam_3)
        features = batch.get_features(0, 10)
        self.assertEqual(
            torch.isnan(features).sum(), 0, "features should not contain NaN."
        )
        self.assertEqual(
            torch.isinf(features).sum(), 0, "features should not contain inf."
        )

    def _valid(self, batch, params):
        seasonality = params.seasonality
        # valid tensors
        for name in ["x", "init_seasonality"]:
            x = getattr(batch, name)
            if x is not None:
                msg = f"All values of {name} should either be positive or NaN."
                self.assertEqual((x[~torch.isnan(x)] <= 0).sum(), 0, msg)

                if "valid" not in name:
                    msg = f"All values within the first seasonality period of {name} should not be NaN."
                    self.assertEqual(torch.isnan(x[:, :seasonality]).sum(), 0, msg)
        # valid indices
        if batch.training:  # for training
            self.assertEqual(
                batch.train_indices[-1] + params.fcst_window, batch.train_length
            )
            self.assertTrue(len(batch.train_indices) >= params.min_training_step_num)
            if batch.valid_length > 0:
                self.assertEqual(batch.valid_indices[0], batch.train_length)
                self.assertTrue(len(batch.valid_indices) <= params.validation_step_num)
        else:  # for testing
            self.assertEqual(
                batch.train_indices[-1] + params.min_training_step_length,
                batch.train_length,
            )
            self.assertTrue(len(batch.train_indices) >= params.min_warming_up_step_num)
            self.assertEqual(batch.valid_indices[0], batch.train_length)
            self.assertEqual(len(batch.valid_indices), params.fcst_step_num)


class AdjustedPinballLossTest(TestCase):
    def test_adjustedpinballloss(self):
        quantile = torch.tensor([0.5, 0.05, 0.95, 0.9])
        rnn = DilatedRNNStack(
            [[1, 2], [2, 2, 4]], "LSTM", input_size=20, state_size=50, output_size=4 * 3
        )
        fcst = rnn(torch.randn(3, 20))
        actuals = torch.tensor(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                ],
                [
                    np.nan,
                    2.0,
                    1.0,
                ],
                [np.nan, np.nan, np.nan],
            ]
        ).log()
        pbl = AdjustedPinballLoss(quantile)
        loss = pbl(fcst, actuals)
        sum_loss = loss.sum()
        # test auto_grad
        sum_loss.backward()

    def test_other(self):
        self.assertRaises(ValueError, AdjustedPinballLoss, quantile=torch.tensor([]))
        self.assertRaises(ValueError, AdjustedPinballLoss, quantile=torch.tensor([[]]))
        self.assertRaises(
            ValueError, AdjustedPinballLoss, quantile=torch.tensor([[0.5]])
        )

        self.assertRaises(
            ValueError,
            AdjustedPinballLoss,
            quantile=torch.tensor([0.5]),
            weight=torch.tensor([0.5, 0.6]),
        )

        quantile = torch.tensor([0.5, 0.05, 0.95])
        pbl = AdjustedPinballLoss(quantile)

        self.assertRaises(ValueError, pbl, torch.randn(2, 9), torch.randn(3, 3))
        self.assertRaises(ValueError, pbl, torch.randn(2, 9), torch.randn(2, 5))


class GMModelTest(TestCase):
    def test_model(self):
        train_ts = {
            str(i): get_ts(n * 3, "2020-05-06") for i, n in enumerate(range(20, 40))
        }
        valid_ts = {
            str(i): get_ts(n * 2, "2021-05-06") for i, n in enumerate(range(10, 30))
        }

        gmparam = GMParam(
            freq="d",
            input_window=20,
            fcst_window=10,
            seasonality=5,
            gmfeature=["last_date", "simple_date"],
            epoch_num=3,
            batch_size={0: 3, 1: 4, 2: 20},
            learning_rate={0: 1e-4, 1: 1e-30},
            nn_structure=[[1, 2, 3], [4, 5]],
            loss_function="adjustedpinball",
            epoch_size=5,
            cell_name="S2Cell",
            h_size=30,
        )
        gmmodel = GMModel(gmparam)
        _ = gmmodel.train(
            train_ts, valid_ts, train_ts, valid_ts, fcst_monitor=True, debug=True
        )
        pred = gmmodel.predict(train_ts)
        _ = gmmodel.evaluate(train_ts, valid_ts)
        # check whether forecasts contain NaN
        msg = "Forecasts contian NaNs."
        num_nan = np.sum([pd.isna(pred[k]).values.sum() for k in pred])
        self.assertEqual(num_nan, 0, msg)


class GMParamConversionTest(TestCase):
    def test_conversion(self):
        origin_gmparam = GMParam(
            freq="d", input_window=10, fcst_window=7, gmfeature=["simple_date"]
        )
        gmparam_str = origin_gmparam.to_string()
        new_gmparam = gmparam_from_string(gmparam_str)
        self.assertEqual(
            origin_gmparam,
            new_gmparam,
            "The new GMParam object is not equal to the origin GMParam object.",
        )


class GMBacktesterTest(TestCase):
    def test_gmbacktester(self):

        TSs = {str(i): get_ts(n * 3, "2019-05-06") for i, n in enumerate(range(20, 40))}

        gmfeature = GMFeature(
            feature_size=7 + 27 + 31 + 7 * 4, feature_type=["last_date", "simple_date"]
        )

        gmparam = GMParam(
            freq="d",
            input_window=7,
            fcst_window=5,
            seasonality=3,
            min_training_step_num=2,
            gmfeature=gmfeature,
            epoch_num=2,
            batch_size={0: 3, 1: 4},
            learning_rate={0: 1e-4, 1: 1e-30},
            nn_structure=[[1, 2, 3], [4, 5]],
            loss_function="adjustedpinball",
            epoch_size=3,
            cell_name="S2Cell",
            h_size=30,
            fcst_step_num=2,
        )

        gmbt = GMBackTester(
            TSs,
            gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            earliest_timestamp="2018-01-01",
            splits=3,
            overlap=False,
            multi=True,
        )

        _ = gmbt.run_backtest()
