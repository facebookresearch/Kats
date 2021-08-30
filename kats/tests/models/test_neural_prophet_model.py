# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
import pkgutil
import unittest
from unittest import TestCase

import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.neural_prophet import NeuralProphetParams


class NeuralProphetModelTest(TestCase):
    def setUp(self):
        # Expected default parameters
        self.expected_defaults = NeuralProphetParams(
            growth="linear",
            changepoints=None,
            n_changepoints=10,
            changepoints_range=0.9,
            trend_reg=0,
            trend_reg_threshold=False,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality="auto",
            seasonality_mode="additive",
            seasonality_reg=0,
            n_forecasts=1,
            n_lags=0.0,
            num_hidden_layers=0,
            d_hidden=None,
            ar_sparsity=None,
            learning_rate=None,
            epochs=None,
            batch_size=None,
            loss_func="Huber",
            optimizer="AdamW",
            train_speed=None,
            normalize="auto",
            impute_missing=True,
        )

    def test_default_parameters(self) -> None:

        """Check that the default parameters are as expected. The expected values are hard coded."""
        act_params = vars(NeuralProphetParams())

        for param, exp_val in vars(self.expected_defaults).items():
            msg = """param:{param}, exp_val:{exp_val},  val:{val}""".format(
                param=param, exp_val=exp_val, val=act_params[param]
            )
            logging.info(msg)
            self.assertEqual(act_params[param], exp_val)


if __name__ == "__main__":
    unittest.main()
