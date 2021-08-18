# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models import prophet
from kats.models.reconciliation.base_models import (
    BaseTHModel,
    GetAggregateTS,
    calc_mae,
    calc_mape,
)
from kats.models.reconciliation.thm import TemporalHierarchicalModel


def generate_ts(st="2018-05-06", et="2021-05-06"):
    time = pd.date_range(st, et, freq="D")
    ts = TimeSeriesData(
        pd.DataFrame({"time": time, "y": np.random.uniform(0, 1, len(time))})
    )
    return ts


ts = generate_ts("2020-05-06", "2020-05-15")

bm1 = BaseTHModel(level=1, model_name="prophet", model_params=prophet.ProphetParams())
bm2 = BaseTHModel(level=2, model_name="prophet", model_params=prophet.ProphetParams())
bm5 = BaseTHModel(level=5, fcsts=np.random.randn(5), residuals=np.random.randn(len(ts)))


class testHelperFunctions(TestCase):
    def test_calc_mape(self) -> None:
        pred = np.array([0, 1, 1])
        truth = np.array([0, 1, 2])

        self.assertEqual(0.25, calc_mape(pred, truth))

    def test_calc_mae(self) -> None:
        pred = np.array([0, 1, 1])
        truth = np.array([0, 1, 4])
        self.assertEqual(1.0, calc_mae(pred, truth))


class testBaseTHModel(TestCase):
    def test_initialization(self) -> None:

        BaseTHModel(2, model_name="prophet", model_params=prophet.ProphetParams())
        BaseTHModel(1, residuals=np.random.randn(4), fcsts=np.random.randn(4))

        self.assertRaises(ValueError, BaseTHModel, level=-0.5)
        self.assertRaises(ValueError, BaseTHModel, level=1)
        self.assertRaises(ValueError, BaseTHModel, level=1, model_name="prophet")
        self.assertRaises(
            ValueError, BaseTHModel, level=1, model_params=prophet.ProphetParams()
        )


class testGetAggregateTS(TestCase):
    def test(self) -> None:
        gat1 = GetAggregateTS(ts)
        agg_res1 = gat1.aggregate([1, 5, 10])
        # Aggregated TS for level 1 should be equal to original TS.
        if agg_res1 != ts:
            msg = "Aggregated TS for level 1 should be equal to original TS."
            logging.info(msg)
            raise ValueError(msg)
        # First aggregate TS using level 5 then level 2 should be equal to aggregate TS using level 10.
        gat2 = GetAggregateTS(agg_res1[5])
        agg_res2 = gat2.aggregate([2])
        if agg_res2[2] != agg_res1[10]:
            msg = "First aggregate TS using level 5 then level 2 should be equal to aggregate TS using level 10."
            logging.info(msg)
            raise ValueError(msg)

        self.assertRaises(ValueError, gat1.aggregate, levels=[15])
        self.assertRaises(ValueError, gat1.aggregate, levels=[1.5])
        self.assertRaises(ValueError, gat1.aggregate, levels=3)


class testTemporalHierarchicalModel(TestCase):
    def test_initialization(self) -> None:
        self.assertRaises(
            ValueError,
            TemporalHierarchicalModel,
            ts,
            # pyre-fixme[16]: Module `prophet` has no attribute `Prophet`.
            [prophet.Prophet()],
        )
        self.assertRaises(ValueError, TemporalHierarchicalModel, ts, [bm2])
        self.assertRaises(
            ValueError,
            TemporalHierarchicalModel,
            ts,
            [
                bm1,
                bm2,
                BaseTHModel(
                    level=2,
                    fcsts=np.random.randn(5),
                    residuals=np.random.randn(len(ts)),
                ),
            ],
        )

    def test_model(self) -> None:
        thm = TemporalHierarchicalModel(ts, [bm1, bm2, bm5])
        # should fit model first.
        self.assertRaises(ValueError, thm.predict, steps=30)
        # fit all base models
        thm.fit()
        # generate forecast with different methods
        for method in [
            "bu",
            "median",
            "svar",
            "struc",
            "mint_shrink",
            "mint_sample",
            "hvar",
        ]:
            thm.predict(steps=10, method=method)
            if method == "bu":
                thm.predict(steps=30, method=method)
            else:
                self.assertRaises(ValueError, thm.predict, steps=30, method=method)
        # test median validation
        thm.median_validation(steps=20)
