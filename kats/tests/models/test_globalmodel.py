# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from typing import Dict, List, Union
from unittest import mock, TestCase

import numpy as np
import pandas as pd
import torch
from kats.consts import TimeSeriesData
from kats.models.globalmodel.backtester import GMBackTester, GMBackTesterExpandingWindow
from kats.models.globalmodel.data_processor import GMBatch, GMDataLoader
from kats.models.globalmodel.ensemble import GMEnsemble, load_gmensemble_from_file
from kats.models.globalmodel.model import GMModel, load_gmmodel_from_file
from kats.models.globalmodel.serialize import (
    global_model_to_json,
    load_global_model_from_json,
)
from kats.models.globalmodel.stdmodel import STDGlobalModel
from kats.models.globalmodel.utils import (
    AdjustedPinballLoss,
    DilatedRNNStack,
    fill_missing_value_na,
    get_filters,
    GMFeature,
    GMParam,
    gmparam_from_string,
    LSTM2Cell,
    pad_ts,
    PinballLoss,
    S2Cell,
    split,
)
from parameterized.parameterized import parameterized


def get_ts(
    n: int, start_time: str, seed: int = 560, freq: str = "D", has_nans: bool = True
) -> TimeSeriesData:
    """
    Helper function for generating TimeSeriesData.
    """
    np.random.seed(seed)
    t = pd.Series(pd.date_range(start_time, freq=freq, periods=n))
    val = np.random.randn(n)
    if has_nans:
        idx = np.random.choice(range(n), int(n * 0.2), replace=False)
        val[idx] = np.nan
    val = pd.Series(val)
    return TimeSeriesData(time=t, value=val)


def _gm_mock_predict_func(
    TSs: TimeSeriesData,
    steps: int,
    fcst_window: int,
    len_quantile: int,
    raw: bool = True,
    test_batch_size: int = 500,
) -> Dict[int, List[np.ndarray]]:
    """
    Helper function for building predict method for mock GMModel.
    """
    m = (steps // fcst_window) + int(steps % fcst_window != 0)
    n = fcst_window * len_quantile
    return {i: [np.random.randn(n)] * m for i in range(len(TSs))}


def _gm_mock_predict_func_2(
    TSs: Dict[int, TimeSeriesData], steps: int
) -> Dict[int, pd.DataFrame]:
    tpd = pd.DataFrame(
        {
            "time": pd.date_range("2021-05-06", periods=steps),
            "fcst_quantile_0.5": np.arange(steps),
        }
    )
    return {k: tpd for k in TSs}


def get_gmmodel_mock(gmparam: GMParam) -> mock.MagicMock:
    """
    Helper function for building mock object for GMModel
    """
    gm_mock = mock.MagicMock()
    gm_mock.predict.side_effect = partial(
        _gm_mock_predict_func,
        fcst_window=gmparam.fcst_window,
        len_quantile=len(gmparam.quantile),
    )
    return gm_mock


def get_gmmodel_mock_2(gmparam: GMParam) -> mock.MagicMock:
    gm_mock = mock.MagicMock()
    gm_mock.predict.side_effect = _gm_mock_predict_func_2
    return gm_mock


# pyre-fixme[5]: Global expression must be annotated.
TSs = [get_ts(i * 5, "2020-05-06", i) for i in range(20, 30)]
# pyre-fixme[5]: Global expression must be annotated.
valid_TSs = [get_ts(i * 2, "2020-05-06", i) for i in range(20, 30)]

ts_missing_val = TimeSeriesData(
    pd.DataFrame(
        {
            "time": [
                "2021-05-06",
                "2021-05-07",
                "2021-05-10",
                "2021-05-11",
                "2021-05-13",
            ],
            "value": np.arange(5),
        }
    )
)


class TestGMParam(TestCase):
    def test_daily(self) -> None:
        GMParam(
            freq="d",
            input_window=45,
            fcst_window=30,
            seasonality=7,
            quantile=[0.5, 0.05, 0.95, 0.99],
            training_quantile=[0.58, 0.05, 0.94, 0.985],
        )

    def test_hourly(self) -> None:
        GMParam(
            freq="H",
            input_window=168,
            fcst_window=168,
            seasonality=24,
            gmfeature=["last_date", "last_hour"],
            quantile=[0.5, 0.05, 0.95, 0.99],
            training_quantile=[0.58, 0.05, 0.94, 0.985],
        )

    def test_valid_freq(self) -> None:
        # test freq type validation with empty string and invalid type
        self.assertRaises(
            ValueError,
            GMParam,
            freq="invalid_arg",
            input_window=168,
            fcst_window=168,
        )

        self.assertRaises(
            ValueError,
            GMParam,
            freq=0,
            input_window=168,
            fcst_window=168,
        )

    def test_valid_optimizer(self) -> None:
        # test optimizer validation with invalid input

        # test with invalid dict key
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            optimizer={"type": "adam"},
        )

        # test with invalid method
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            optimizer="unsupported_method",
        )

        # test with invalid optimizer type
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            optimizer=0,
        )

        # test with valid optimizer
        GMParam(
            freq="H",
            input_window=168,
            fcst_window=168,
            optimizer="adam",
        )

    def test_valid_loss_func(self) -> None:
        # test loss function validation with invalid input

        # test with invalid loss function name
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            loss_function="nonexistent_loss_func",
        )

    def test_valid_union_dict(self) -> None:
        # test union dict validation with invalid dict value
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            batch_size={0: "128"},
        )

        # test union dict validation with empty dict
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            batch_size={},
        )

        # test union dict validation with invalid type
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            batch_size="128",
        )

    def test_equal(self) -> None:
        # test GMParam equal
        param1 = GMParam(
            freq="H",
            input_window=168,
            fcst_window=168,
        )

        param2 = GMParam(
            freq="H",
            input_window=168,
            fcst_window=168,
        )

        param3 = GMParam(
            freq="D",
            input_window=168,
            fcst_window=168,
        )

        param4 = ""

        self.assertEqual(param1 == param2, True)
        self.assertEqual(param1 == param3, False)
        self.assertEqual(param1 == param4, False)

    def test_validation_metric(self) -> None:
        # test validation_metric input

        # test validation_metric with invalid metric
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            validation_metric=["invalid_metric"],
        )

        # test validation_metric with invalid input type
        self.assertRaises(
            ValueError,
            GMParam,
            freq="H",
            input_window=168,
            fcst_window=168,
            validation_metric=0,
        )

    def test_daily_s2s(self) -> None:
        GMParam(
            freq="d", input_window=45, fcst_window=30, seasonality=7, model_type="s2s"
        )


class LSTM2CellTest(TestCase):
    def test_cell(self) -> None:
        cell = LSTM2Cell(30, 2, 5)
        input_t = torch.randn(5, 30)
        prev_h = torch.randn(5, 2)
        delayed_h = torch.randn(5, 2)
        c = torch.randn(5, 5)
        cell(input_t, False, False)
        cell(input_t, True, False, prev_h, c)
        cell(input_t, True, True, prev_h, delayed_h, c)


class S2CellTest(TestCase):
    def test_cell(self) -> None:
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
    def test_rnn(self) -> None:
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

    def test_others(self) -> None:
        self.assertRaises(
            ValueError, DilatedRNNStack, [[1, 2]], "randomcell", 20, 20, 10
        )
        self.assertRaises(ValueError, DilatedRNNStack, [], "LSTM2Cell", 20, 20, 10)
        self.assertRaises(ValueError, DilatedRNNStack, [], "LSTM2Cell", 20, 20, 10, 20)
        self.assertRaises(
            ValueError, DilatedRNNStack, [[1, 2]], "S2Cell", 20, 50, -5, 10
        )

    def test_encoder_decoder(self) -> None:
        x = torch.randn(5, 20)
        encoder = DilatedRNNStack(
            [[1], [3]],
            "S2Cell",
            input_size=20,
            state_size=50,
            output_size=None,
            h_size=2,
        )
        decoder = DilatedRNNStack(
            [[1], [3]], "S2Cell", input_size=20, state_size=50, output_size=30, h_size=2
        )
        for _ in range(2):
            _ = encoder(x)
            encoder.prepare_decoder(decoder)
            _ = decoder(x)


class PinballLossTest(TestCase):
    def test_pinballloss(self) -> None:
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

    def test_other(self) -> None:
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


class GMFeatureTest(TestCase):
    def test_gmfeature(self) -> None:
        x = np.row_stack([np.abs(ts.value.values[:10]) for ts in TSs])
        time = np.row_stack([ts.time.values[:10] for ts in TSs])

        gmfs = [
            GMFeature(feature_type="tsfeatures"),
            GMFeature(feature_type="last_date"),
            GMFeature(feature_type="simple_date"),
            GMFeature(
                feature_type=["simple_date", "tsfeatures", "last_date"],
            ),
            GMFeature(feature_type=["last_date", "last_hour"]),
            GMFeature(feature_type=["last_date", "last_hour_minute"]),
            GMFeature(feature_type=["last_month"]),
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

    def test_others(self) -> None:
        self.assertRaises(ValueError, GMFeature, feature_type="randome_feature")
        self.assertRaises(
            ValueError,
            GMFeature,
            feature_type=["randome_feature_1", "simple_date"],
        )


class GMDataLoaderTest(TestCase):
    def test_dataloader(self) -> None:
        # Input is a list.
        gmdl = GMDataLoader(TSs)
        collects = []
        for _ in range(5):
            batch = gmdl.get_batch(2)
            collects.extend(batch)
        # verify that all ids in [0,9] are visited.
        self.assertEqual(set(collects) == set(range(10)), True)
        for _ in range(5):
            batch = gmdl.get_batch(4)
        # verify that all ids are returned
        batch = gmdl.get_batch(20)
        self.assertEqual(set(range(10)) == set(batch), True)

        # Input is a dictionary.
        gmdl = GMDataLoader({f"ts_{i}": t for i, t in enumerate(TSs)})
        collects = []
        for _ in range(5):
            batch = gmdl.get_batch(3)
            collects.extend(batch)

        self.assertEqual(set(collects) == {f"ts_{i}" for i in range(len(TSs))}, True)

        # When both training set and validation set are provided.
        gmdl = GMDataLoader(TSs, TSs)
        collects = []
        for _ in range(15):
            batch = gmdl.get_batch(3)
            collects.extend(batch)
        self.assertEqual(set(collects) == set(range(len(TSs))), True)

    def test_others(self) -> None:
        self.assertRaises(ValueError, GMDataLoader, [])
        self.assertRaises(ValueError, GMDataLoader, ["test_data"])
        self.assertRaises(ValueError, GMDataLoader, "test_data")

        gmdl = GMDataLoader(TSs)
        self.assertRaises(ValueError, gmdl.get_batch, 0.5)
        self.assertRaises(ValueError, gmdl.get_batch, -1)


class GMBatchTest(TestCase):
    def test_batch(self) -> None:
        train_ts = {str(i): TSs[i] for i in range(len(TSs))}
        valid_ts = {str(i): valid_TSs[i] for i in range(len(TSs))}
        batch_ids = [str(i) for i in range(len(TSs))]

        GMParam_collects = [
            # RNN GM with seasonality
            GMParam(
                freq="d", input_window=10, fcst_window=7, seasonality=3, fcst_step_num=2
            ),
            # RNN GM without seasonality
            GMParam(freq="d", input_window=10, fcst_window=7, seasonality=1),
            # RNN GM with seasonlaity and feature
            GMParam(
                freq="d",
                input_window=10,
                fcst_window=7,
                seasonality=3,
                gmfeature="simple_date",
            ),
            # S2S GM
            GMParam(
                freq="d", input_window=5, fcst_window=3, seasonality=2, model_type="s2s"
            ),
        ]

        GMBatch_params = [
            # training mode with validation set
            {
                "train_TSs": train_ts,
                "valid_TSs": valid_ts,
                "batch_ids": batch_ids,
                "mode": "train",
            },
            # training mode without validation set
            {
                "train_TSs": train_ts,
                "valid_TSs": None,
                "batch_ids": batch_ids,
                "mode": "train",
            },
            # testing mode with validation set
            {
                "train_TSs": train_ts,
                "valid_TSs": valid_ts,
                "batch_ids": batch_ids,
                "mode": "test",
            },
            # testing mode without validation set
            {
                "train_TSs": train_ts,
                "valid_TSs": None,
                "batch_ids": batch_ids,
                "mode": "test",
            },
        ]

        for gmparam in GMParam_collects:
            for batch_param in GMBatch_params:
                # pyre-fixme
                batch_param["params"] = gmparam
                # pyre-fixme
                batch = GMBatch(**batch_param)
                self._valid(batch, gmparam)

    # pyre-fixme[2]: Parameter must be annotated.
    def _valid(self, batch, params) -> None:
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
                batch.train_indices[-1]
                + params.fcst_window * batch.training_encoder_step_num,
                batch.train_length,
            )
            self.assertTrue(len(batch.train_indices) >= params.min_training_step_num)
            if batch.valid_length > 0:
                self.assertEqual(batch.valid_indices[0], batch.train_length)
                if params.model_type == "rnn":
                    self.assertTrue(
                        len(batch.valid_indices) <= params.validation_step_num,
                        f"valid_indices with length = {len(batch.valid_indices)}, {params.validation_step_num}",
                    )
                else:
                    self.assertTrue(
                        len(batch.valid_indices) == batch.training_encoder_step_num
                    )
        else:  # for testing
            self.assertEqual(
                batch.train_indices[-1] + params.min_training_step_length,
                batch.train_length,
            )
            self.assertTrue(len(batch.train_indices) >= params.min_warming_up_step_num)
            self.assertEqual(batch.valid_indices[0], batch.train_length)
            self.assertEqual(len(batch.valid_indices), params.fcst_step_num)

    def test_others(self) -> None:
        # test when minimum length of training time series less than seasonality
        train_ts = {str(i): TSs[i][:i] for i in [2, 6, 7]}
        gmparam = GMParam("D", 10, 10, seasonality=5)
        self.assertRaises(
            ValueError,
            GMBatch,
            params=gmparam,
            batch_ids=list(train_ts.keys()),
            train_TSs=train_ts,
        )


class AdjustedPinballLossTest(TestCase):
    def test_adjustedpinballloss(self) -> None:
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
        sum_loss.backward(retain_graph=True)

        # test sum reduction
        pbl2 = AdjustedPinballLoss(quantile, reduction="sum")
        loss2 = pbl2(fcst, actuals)
        sum_loss2 = loss2.sum()
        # test auto_grad
        sum_loss2.backward()

    def test_other(self) -> None:
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
    def test_model(self) -> None:

        GMParam_collects = [
            # RNN GM
            GMParam(
                freq="d",
                input_window=10,
                fcst_window=5,
                seasonality=3,
                gmfeature=["last_date", "simple_date"],
                epoch_num=3,
                batch_size={0: 3, 1: 4},
                learning_rate={0: 1e-4, 1: 1e-30},
                nn_structure=[[1, 2], [4]],
                loss_function="adjustedpinball",
                epoch_size=2,
                cell_name="S2Cell",
                h_size=5,
                state_size=10,
                jit=True,
            ),
            # S2S GM
            GMParam(
                freq="d",
                model_type="s2s",
                input_window=10,
                fcst_window=5,
                seasonality=2,
                gmfeature=["last_date"],
                epoch_num=2,
                batch_size={0: 3, 1: 4, 2: 20},
                learning_rate={0: 1e-4, 1: 1e-30},
                nn_structure=[[1, 2]],
                epoch_size=2,
                cell_name="S2Cell",
                h_size=5,
                state_size=10,
                jit=True,
            ),
        ]
        for gmparam in GMParam_collects:
            gm = GMModel(gmparam)
            self._test_single_gmmodel(gm, TSs, valid_TSs)

        # check saving/loading trained model
        gm.save_model("test_save_model.p")
        # check loading model
        gm = load_gmmodel_from_file("test_save_model.p")
        # remove temporary file
        try:
            os.remove("test_save_model.p")
        except Exception as e:
            logging.info(f"Fail to remove test_save_model.p with exception {e}.")

    # pyre-fixme[2]: Parameter must be annotated.
    def _test_single_gmmodel(self, gmmodel, train_ts, valid_ts) -> None:
        _ = gmmodel.train(
            train_ts, valid_ts, train_ts, valid_ts, fcst_monitor=True, debug=True
        )
        pred = gmmodel.predict(train_ts, 50)
        _ = gmmodel.predict(train_ts[0], 20)
        _ = gmmodel.evaluate(train_ts, valid_ts)
        _ = gmmodel.evaluate(train_ts[0], valid_ts[0])
        # check whether forecasts starting timestamps are correct
        self.assertTrue(
            pred[0].time.iloc[0] == (train_ts[0].time.iloc[-1] + gmmodel.params.freq),
            "The first timestamp of forecasts is not equal to the last timestamp of training timestamp+frequency.",
        )
        # check whether forecasts contain NaNs.
        msg = "Forecasts contian NaNs."
        num_nan = np.sum([pd.isna(pred[k]).values.sum() for k in pred])
        self.assertEqual(num_nan, 0, msg)


class GMParamConversionTest(TestCase):
    def test_conversion(self) -> None:
        origin_gmparam = GMParam(
            freq="d", input_window=10, fcst_window=7, gmfeature=["simple_date"]
        )
        gmparam_str = origin_gmparam.to_string()
        new_gmparam = gmparam_from_string(gmparam_str)
        self.assertEqual(
            origin_gmparam,
            new_gmparam,
            f"The new GMParam object is not equal to the origin GMParam object with original gmparam string {gmparam_str} and new {new_gmparam.to_string()}.",
        )

    def test_copy(self) -> None:
        origin_gmparam = GMParam(
            freq="d", input_window=10, fcst_window=7, gmfeature=["simple_date"]
        )
        self.assertEqual(
            origin_gmparam,
            origin_gmparam.copy(),
            "The copy of the origin GMParam is not equal to the original GMParam object.",
        )


class GMBacktesterTest(TestCase):
    def test_gmbacktester(self) -> None:
        gmparam = GMParam(
            freq="d",
            input_window=7,
            fcst_window=5,
            seasonality=3,
            min_training_step_num=2,
            gmfeature=["last_date", "simple_date"],
            epoch_num=2,
            batch_size={0: 3, 1: 4},
            learning_rate={0: 1e-4, 1: 1e-30},
            nn_structure=[[1]],
            loss_function="adjustedpinball",
            epoch_size=2,
            cell_name="S2Cell",
            h_size=3,
            state_size=5,
        )

        gmbt = GMBackTester(
            TSs,
            gmparam,
            backtest_timestamp=["2020-06-10"],
            earliest_timestamp="2018-01-01",
            splits=3,
            overlap=False,
            multi=True,
        )

        # Using mock
        # gmbt.gm_collects = {"2020-06-10": [get_gmmodel_mock(gmparam) for _ in range(3)]}

        _ = gmbt.run_backtest()
        gmbt.multi = False
        _ = gmbt.run_backtest()

    def test_other(self) -> None:
        # Check that gmparam is a valid object
        self.assertRaises(
            ValueError, GMBackTester, data=TSs, gmparam=None, backtest_timestamp=[]
        )

        # Check that backtest_timestamp is not empty
        gmparam = GMParam(freq="d", input_window=7, fcst_window=5, epoch_size=2)
        self.assertRaises(
            ValueError, GMBackTester, data=TSs, gmparam=gmparam, backtest_timestamp=[]
        )

        # Check splits is valid
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=TSs,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            splits=0,
        )
        # Check replicate > 0
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=TSs,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            replicate=0,
        )

        # Check valid earliest_timestamp
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=TSs,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            earliest_timestamp=2019,
        )

        # Check valid test_size
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=TSs,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            test_size=None,
        )

        # Check valid max_core
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=TSs,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
            max_core="1",
        )

        # Check data is valid
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=None,
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
        )

        # Check data is valid
        self.assertRaises(
            ValueError,
            GMBackTester,
            data=[1.0, 2.0, 3.0],
            gmparam=gmparam,
            backtest_timestamp=["2019-06-16", "2019-07-05"],
        )


class GMEnsembleTest(TestCase):
    def test_gmensemble(self) -> None:
        # RNN GMEnsemble
        gmparam = GMParam(
            freq="d",
            input_window=7,
            fcst_window=5,
        )
        gme = GMEnsemble(
            gmparam,
            ensemble_type="median",
            splits=2,
            replicate=1,
            overlap=False,
            multi=True,
        )
        self._test_sinle_ensemble(gme, TSs, valid_TSs)
        # S2S GMEnsemble
        gmparam = GMParam(
            freq="d",
            model_type="s2s",
            input_window=5,
            fcst_window=3,
            seasonality=2,
            h_size=5,
            state_size=10,
        )
        gme = GMEnsemble(
            gmparam,
            ensemble_type="mean",
            splits=2,
            replicate=1,
            overlap=True,
            multi=False,
        )
        self._test_sinle_ensemble(gme, TSs, valid_TSs)

    # pyre-fixme[2]: Parameter must be annotated.
    def _test_sinle_ensemble(self, gme, TSs, test_TSs) -> None:
        # mock each single GMModel object
        gme.gm_models = [get_gmmodel_mock(gme.params) for _ in range(gme.model_num)]
        # test train
        gme.train(TSs, test_size=0.1)
        gme.train(TSs, test_size=0)
        # test predict
        _ = gme.predict(TSs, 20)[0]
        # test evaluation
        _ = gme.evaluate(TSs, test_TSs)

    def test_other(self) -> None:
        gmparam = GMParam(freq="d", input_window=7, fcst_window=5)
        gme = GMEnsemble(gmparam)
        # initiate NNs
        [t._initiate_nn() for t in gme.gm_models]
        # test save_model
        gme.save_model("test_gme.p")
        # test load_model
        _ = load_gmensemble_from_file("test_gme.p")
        # remove temporary file
        try:
            os.remove("test_gme.p")
        except Exception as e:
            logging.info(f"Fail to remove test_gme.p with exception {e}.")

        # Check that gmparam is a valid object
        self.assertRaises(
            ValueError,
            GMEnsemble,
            gmparam=None,
        )

        # Check that gmparam ensemble type is invalid
        self.assertRaises(
            ValueError,
            GMEnsemble,
            gmparam=gmparam,
            ensemble_type="random_ensemble_type",
        )

        # Check splits is valid
        self.assertRaises(
            ValueError,
            GMEnsemble,
            gmparam=gmparam,
            splits=0,
        )

        # Check replicate is valid
        self.assertRaises(
            ValueError,
            GMEnsemble,
            gmparam=gmparam,
            replicate=0,
        )

        # Check valid max_core
        self.assertRaises(
            ValueError,
            GMEnsemble,
            gmparam=gmparam,
            max_core=0,
        )

        # Check train test_size is valid
        self.assertRaises(
            ValueError,
            gme.train,
            data=TSs,
            test_size=-1,
        )

        # Check valid steps
        self.assertRaises(
            ValueError,
            gme.predict,
            test_TSs=valid_TSs,
            steps=0,
        )

        # Check valid test_batch_size
        self.assertRaises(
            ValueError,
            gme.predict,
            test_TSs=valid_TSs,
            steps=1,
            test_batch_size=0,
        )

        # Check file_name is valid
        self.assertRaises(
            ValueError,
            gme.save_model,
            file_name=None,
        )

        # Check test_train_TSs type and test_valid_TSs are same
        self.assertRaises(
            ValueError,
            gme.evaluate,
            test_train_TSs=TSs,
            test_valid_TSs=None,
        )


class STDGlobalModelTest(TestCase):
    # pyre-fixme
    @parameterized.expand(
        [
            [{"decomposition_model": "stl"}],
            [{"decomposition_model": "seasonal_decompose"}],
            [
                {
                    "decomposition_model": "prophet",
                    "fit_trend": True,
                    "decomposition": "multiplicative",
                }
            ],
        ]
    )
    def test_stdgm(self, stdparams: Dict[str, Union[str, bool]]) -> None:
        # mock a single global model
        gmparam = GMParam(
            freq="d",
            model_type="s2s",
            input_window=5,
            fcst_window=3,
            seasonality=2,
            h_size=5,
            state_size=10,
        )
        gm = get_gmmodel_mock_2(gmparam)
        # pyre-fixme Incompatible parameter type [6]: In call `STDGlobalModel.__init__`, for 1st positional only parameter expected `Optional[GMParam]` but got `Union[bool, str]`.
        stdgm = STDGlobalModel(**stdparams)
        stdgm.load_global_model(gm)
        _ = stdgm.predict(TSs, steps=5)


class GMBackTesterExpandingWindowTest(TestCase):
    def test_GMBTEW(self) -> None:

        gmparam = GMParam(
            input_window=10, fcst_window=7, seasonality=6, fcst_step_num=2, freq="D"
        )

        gmm1 = GMModel(gmparam)
        gmm1.build_rnn()

        gmm2 = GMModel(gmparam)
        gmm2.build_rnn()

        gme = GMEnsemble(gmparam, splits=2, replicate=1)

        gme.gm_models = [gmm1, gmm2]

        gmbtew1 = GMBackTesterExpandingWindow(
            ["mape", "mse"], TSs[0], gmm1, 60, 80, 20, 5, True
        )

        gmbtew1.run_backtest()

        gmbtew2 = GMBackTesterExpandingWindow(
            ["mape", "mse"], TSs[0], gme, 60, 80, 20, 5, True
        )

        gmbtew2.run_backtest()

        gmbtew3 = GMBackTesterExpandingWindow(
            ["mape", "mse"], TSs[0], gme, 60, 80, 20, 5, False
        )

        gmbtew3.run_backtest()

    def test_other(self) -> None:
        # Check that data and gmparam have same freq
        ts = get_ts(80, "2020-05-06", 560, "W")
        gmparam = GMParam(
            input_window=10, fcst_window=7, seasonality=6, fcst_step_num=2, freq="D"
        )
        gmm1 = GMModel(gmparam)
        gmm1.build_rnn()

        self.assertRaises(
            ValueError,
            GMBackTesterExpandingWindow,
            ["mape", "mse"],
            ts,
            gmm1,
            60,
            80,
            20,
            5,
            True,
        )


class HelperFunctionsTest(TestCase):
    def test_pad_ts(self) -> None:
        # test helper function pad_ts
        ts = TSs[0]
        freq = pd.Timedelta(days=10)
        self.assertRaises(
            ValueError,
            pad_ts,
            ts,
            0,
            freq,
        )

        pad_ts(ts, 1, freq)

    # pyre-fixme
    @parameterized.expand(
        [
            ["non_season_mising", ts_missing_val, 1, "D", 5],
            ["season_2_missing", ts_missing_val, 2, "1D", 6],
            ["season_3_missing", ts_missing_val, 3, "D", 8],
            ["missing_1_day", ts_missing_val[-3:], 4, "D", 4],
        ]
    )
    def test_filling_missing_value(
        self,
        test_name: str,
        test_ts: TimeSeriesData,
        seasonality: int,
        freq: str,
        target_len: int,
    ) -> None:
        new_ts = fill_missing_value_na(test_ts, seasonality, freq)
        self.assertEqual(
            len(new_ts),
            target_len,
            f"Expect filled time series of length {target_len} but receives {len(new_ts)}.",
        )

    def test_get_filters(self) -> None:
        # test helper function get_filters
        isna_idx = np.array([False] + [True] * 2 + [False] + [True] * 3 + [False])
        est_filter = get_filters(isna_idx, 3)
        true_filter = np.array([True] * 4 + [False] * 3 + [True])
        self.assertEqual(np.sum(est_filter == true_filter), len(est_filter))

    def test_split(self) -> None:
        _ = split(1, True, TSs, None)
        _ = split(2, True, TSs, None)
        # test when number of TS is less than splits number
        self.assertRaises(
            ValueError,
            split,
            splits=10,
            overlap=True,
            train_TSs=TSs[:3],
            valid_TSs=None,
        )


class SerializeTest(TestCase):
    def test_serialize(self) -> None:
        models = []
        for model_type in ["rnn", "s2s"]:
            gmparam = GMParam(
                freq="D",
                model_type=model_type,
                input_window=5,
                fcst_window=3,
                nn_structure=[[1]],
            )
            # build and initialize GMModel
            gm = GMModel(gmparam)
            gm._initiate_nn()
            models.append(gm)

            # build and initialize GMEnsemble
            gme = GMEnsemble(gmparam, splits=2, replicate=1)
            [t._initiate_nn() for t in gme.gm_models]
            models.append(gme)

        for m in models:
            model_str = global_model_to_json(m)
            loaded_model = load_global_model_from_json(model_str)
            fcst = m.predict(TSs[0], steps=3)[0]
            loaded_fcst = loaded_model.predict(TSs[0], steps=3)[0]
            self.assertTrue(
                fcst.equals(loaded_fcst),
                "Forecasts generated by the loaded model is different from that generated from the original model.",
            )
