# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import builtins
import logging
import sys
import unittest
from unittest import TestCase
from unittest.mock import patch

from kats.models.neural_prophet import NeuralProphetParams


class NeuralProphetModelTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        original_import_fn = builtins.__import__

        def mock_neural_prophet_import(module, *args, **kwargs):
            if module == "neuralprophet":
                raise ImportError
            else:
                return original_import_fn(module, *args, **kwargs)

        cls.mock_imports = patch(
            "builtins.__import__", side_effect=mock_neural_prophet_import
        )

    def test_neural_prophet_not_installed(self) -> None:
        # Unload neural_prophet module so its imports can be mocked as necessary
        del sys.modules["kats.models.neural_prophet"]

        with self.mock_imports:
            with self.assertRaises(ImportError):
                from kats.models.neural_prophet import NeuralProphetParams

                NeuralProphetParams()

        from kats.models.neural_prophet import NeuralProphetParams

        # Confirm that the module has been properly reloaded -- should not
        # raise an exception anymore
        NeuralProphetParams()

    def test_default_parameters(self) -> None:
        """
        Check that the default parameters are as expected. The expected values
        are hard coded.
        """
        expected_defaults = NeuralProphetParams(
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
            n_lags=0,
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
        # Expected params should be valid
        expected_defaults.validate_params()

        actual_defaults = vars(NeuralProphetParams())

        for param, exp_val in vars(expected_defaults).items():
            msg = """param:{param}, exp_val:{exp_val},  val:{val}""".format(
                param=param, exp_val=exp_val, val=actual_defaults[param]
            )
            logging.info(msg)
            self.assertEqual(actual_defaults[param], exp_val)


if __name__ == "__main__":
    unittest.main()
