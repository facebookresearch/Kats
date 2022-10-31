# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams
from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.utils.ensemble_predict_interval import ensemble_predict_interval


class testEnsemblePredictInterval(TestCase):
    def setUp(self) -> None:
        # create time series data
        np.random.seed(0)
        val = (
            np.arange(180) / 6
            + np.sin(np.pi * np.arange(180) / 6) * 20
            + +np.cos(np.arange(180)) * 20
            + np.random.randn(180) * 10
        )
        ts = TimeSeriesData(
            pd.DataFrame({"time": pd.date_range("2021-05-06", periods=180), "val": val})
        )
        self.test_ts = ts[120:]
        self.hist_ts = ts[:120]

    def test_EPI_Prophet(self) -> None:
        # test EPI on Prophet model
        epi = ensemble_predict_interval(
            # pyre-fixme[6]: Incompatible parameter type
            model=ProphetModel,
            model_params=ProphetParams(seasonality_mode="additive"),
            ts=self.hist_ts,
            block_size=10,
            n_block=5,
            ensemble_size=4,
        )
        self.assertEqual(len(epi.ts), 60)
        self.assertEqual(epi.error_matrix_flag, False)
        self.assertEqual(epi.projection_flag, False)

        res = epi.get_projection(step=10)
        self.assertEqual(res.shape, (10, 3))
        self.assertEqual(epi.error_matrix_flag, True)
        self.assertEqual(epi.projection_flag, True)

        res_other_conf_level = epi.get_fcst_band_with_level(confidence_level=0.5)
        self.assertEqual(res_other_conf_level.shape, (10, 3))

        epi.pi_comparison_plot(self.test_ts)

    def test_EPI_Sarima(self) -> None:
        # test EPI on SARIMA model
        params = SARIMAParams(p=2, d=1, q=1, trend="ct", seasonal_order=(1, 0, 1, 12))
        epi = ensemble_predict_interval(
            # pyre-fixme[6]: Incompatible parameter type
            model=SARIMAModel,
            model_params=params,
            ts=self.hist_ts,
            n_block=5,
            ensemble_size=4,
            multiprocessing=True,
        )
        res = epi.get_projection(step=20)
        self.assertEqual(res.shape, (20, 3))
        epi.pi_comparison_plot(self.test_ts)

    def test_EPI_HW(self) -> None:
        # test EPI on Holt-Winters model
        epi = ensemble_predict_interval(
            # pyre-fixme[6]: Incompatible parameter type
            model=HoltWintersModel,
            model_params=HoltWintersParams(),
            ts=self.hist_ts,
            n_block=5,
            ensemble_size=10,
            multiprocessing=True,
        )
        res = epi.get_projection(step=40, rolling_based=True)
        self.assertEqual(res.shape, (40, 3))

        res_other_conf_level = epi.get_fcst_band_with_level(confidence_level=0.5)
        self.assertEqual(res_other_conf_level.shape, (40, 3))

        epi.pi_comparison_plot(self.test_ts)
        # test if there is no test_ts provided
        epi.pi_comparison_plot()

        # fcst step is greater than test set length
        epi2 = ensemble_predict_interval(
            # pyre-fixme[6]: Incompatible parameter type
            model=HoltWintersModel,
            model_params=HoltWintersParams(),
            ts=self.hist_ts,
            n_block=5,
            ensemble_size=4,
        )
        res2 = epi2.get_projection(step=80)
        self.assertEqual(res2.shape, (80, 3))

        res_other_conf_level2 = epi2.get_fcst_band_with_level(confidence_level=0.5)
        self.assertEqual(res_other_conf_level2.shape, (80, 3))

        epi2.pi_comparison_plot(self.test_ts)
        epi2.pi_comparison_plot(self.test_ts, test_data_only=True)

    def test_errors(self) -> None:
        # both block_size and n_block are None
        with self.assertRaises(ValueError):
            _ = ensemble_predict_interval(
                # pyre-fixme[6]: Incompatible parameter type
                model=HoltWintersModel,
                model_params=HoltWintersParams(),
                ts=self.hist_ts,
            )

        # not suitable n_block when block_size is None
        for x in [1, -1, 200]:
            with self.assertRaises(ValueError):
                _ = ensemble_predict_interval(
                    # pyre-fixme[6]: Incompatible parameter type
                    model=HoltWintersModel,
                    model_params=HoltWintersParams(),
                    ts=self.hist_ts,
                    n_block=x,
                )

        # not suitable block_size when n_block is None
        for y in [1, -1, 200]:
            with self.assertRaises(ValueError):
                _ = ensemble_predict_interval(
                    # pyre-fixme[6]: Incompatible parameter type
                    model=HoltWintersModel,
                    model_params=HoltWintersParams(),
                    ts=self.hist_ts,
                    block_size=y,
                )

        # not suitable block_size and n_block vals
        for x, y in [(100, 100), (1, 100), (100, 1)]:
            with self.assertRaises(ValueError):
                _ = ensemble_predict_interval(
                    # pyre-fixme[6]: Incompatible parameter type
                    model=HoltWintersModel,
                    model_params=HoltWintersParams(),
                    ts=self.hist_ts,
                    n_block=x,
                    block_size=y,
                )

        # not suitable ensemble size
        for m in [1, -1, 0]:
            with self.assertRaises(ValueError):
                _ = ensemble_predict_interval(
                    # pyre-fixme[6]: Incompatible parameter type
                    model=HoltWintersModel,
                    model_params=HoltWintersParams(),
                    ts=self.hist_ts,
                    n_block=5,
                    ensemble_size=m,
                )

        # error of plot
        epi = ensemble_predict_interval(
            # pyre-fixme[6]: Incompatible parameter type
            model=HoltWintersModel,
            model_params=HoltWintersParams(),
            ts=self.hist_ts,
            n_block=5,
        )
        with self.assertRaises(ValueError):
            # havn't been trained
            epi.pi_comparison_plot(self.test_ts)

        _ = epi.get_projection(step=20)
        with self.assertRaises(ValueError):
            # not test_ts provided
            epi.pi_comparison_plot(test_data_only=True)


if __name__ == "__main__":
    unittest.main()
