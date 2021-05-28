#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import TestCase

import re
import statsmodels
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.utils.cupik import Pipeline
from kats.detectors.trend_mk import MKDetector
from kats.models.theta import ThetaParams, ThetaModel

if "kats/tests" in os.getcwd():
    DATA_FILE = os.path.abspath(
        os.path.join(
            os.path.dirname("__file__"),
            "../",
            "data/air_passengers.csv"
            )
            )
elif "/home/runner/work/" in os.getcwd(): # for Githun Action
    DATA_FILE = "kats/data/air_passengers.csv"
else:
    DATA_FILE = "kats/kats/data/air_passengers.csv"

DATA = pd.read_csv(DATA_FILE)
DATA.columns = ["time", "y"]
TSData = TimeSeriesData(DATA)
statsmodels_ver = float(re.findall('([0-9]+\\.[0-9]+)\\..*', statsmodels.__version__)[0])


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
        if statsmodels_ver < 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.328591954023)
        elif statsmodels_ver >= 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.1270492317991)


        # test if the model can be built on the output from the detector
        pipe = Pipeline(
            [
                ("trend_detector", MKDetector(threshold=0.8)),
                ("theta_model", ThetaModel(params = ThetaParams())),
            ]
        )
        fitted = pipe.fit(TSData)
        self.assertEqual(len(pipe.metadata['trend_detector'][0]), 2)
        if statsmodels_ver < 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.328591954023)
        elif statsmodels_ver >= 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.1270492317991)
