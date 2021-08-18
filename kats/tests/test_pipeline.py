# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.detectors.trend_mk import MKDetector
from kats.models.theta import ThetaParams, ThetaModel
from kats.utils.cupik import Pipeline


statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class cupikTest(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

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
            self.TSData,
            params={"trend_detector": {"window_size": 7, "direction": "up"}},
        )

        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 50)
        self.assertEqual(
            len(pipe.metadata["trend_detector"][0]),
            len(MKDetector(data=self.TSData).detector(window_size=7, direction="up")),
        )

        # Scene 2: Default parameters of MKDetector
        pipe = Pipeline(
            [
                ("trend_detector", MKDetector(threshold=0.8)),
            ]
        )
        pipe.fit(self.TSData)

        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 2)
        self.assertEqual(
            len(pipe.metadata["trend_detector"][0]),
            len(MKDetector(data=self.TSData).detector()),
        )

    def test_thetamodel(self) -> None:
        pipe = Pipeline([("theta_model", ThetaModel(params=ThetaParams()))])
        fitted = pipe.fit(self.TSData)
        bools = (
            # pyre-fixme[16]: Optional type has no attribute `values`.
            ThetaModel(self.TSData, ThetaParams()).fit().fitted_values.values
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
                ("theta_model", ThetaModel(params=ThetaParams())),
            ]
        )
        fitted = pipe.fit(self.TSData)
        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 2)
        if statsmodels_ver < 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.328591954023)
        elif statsmodels_ver >= 0.12:
            self.assertEqual(fitted.predict(1).fcst.values[0], 433.1270492317991)
