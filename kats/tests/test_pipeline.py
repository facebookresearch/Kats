#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.utils.cupik import Pipeline
from kats.detectors.trend_mk import MKDetector
from kats.models.theta import ThetaParams, ThetaModel

DATA = pd.read_csv("kats/kats/data/air_passengers.csv")
DATA.columns = ["time", "y"]
TSData = TimeSeriesData(DATA)


class cupikTest(TestCase):
    def test_mkdetector(self) -> None:

        # We will be using 2 different scenarios to test if the results
        # are the same between a directly called MKDetector and one that
        # is called via CuPiK

        # Scene 1: window_size = 7, direction = 'up'
        pipe = Pipeline(
            [
                ("trend_detector", MKDetector(threshold=0.8)),
            ]
        )
        pipe.fit(
            TSData, params={"trend_detector": {"window_size": 7, "direction": "up"}}
        )

        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 50)
        self.assertEqual(
            len(pipe.metadata["trend_detector"][0]),
            len(MKDetector(data=TSData).detector(window_size=7, direction="up")),
        )

        # Scene 2: Default parameters of MKDetector
        pipe = Pipeline(
            [
                ("trend_detector", MKDetector(threshold=0.8)),
            ]
        )
        pipe.fit(TSData)

        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 2)
        self.assertEqual(
            len(pipe.metadata["trend_detector"][0]),
            len(MKDetector(data=TSData).detector()),
        )

    def test_thetamodel(self) -> None:
        pipe = Pipeline([("theta_model", ThetaModel(params = ThetaParams()))])
        fitted = pipe.fit(TSData)
        bools = (
            # pyre-fixme[16]: `None` has no attribute `fitted_values`.
            ThetaModel(TSData, ThetaParams()).fit().fitted_values.values
            == fitted.fitted_values.values
        )
        self.assertEqual(np.sum(bools), 144)
        self.assertEqual(fitted.predict(1).fcst.values[0], 433.328591954023)


        # test if the model can be built on the output from the detector
        pipe = Pipeline(
            [
                ("trend_detector", MKDetector(threshold=0.8)),
                ("theta_model", ThetaModel(params = ThetaParams())),
            ]
        )
        fitted = pipe.fit(TSData)
        self.assertEqual(len(pipe.metadata['trend_detector'][0]), 2)
        self.assertEqual(fitted.predict(1).fcst.values[0], 433.328591954023)
