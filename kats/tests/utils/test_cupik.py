# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
from kats.compat import statsmodels
from kats.data.utils import load_air_passengers
from kats.detectors.trend_mk import MKDetector
from kats.models.theta import ThetaModel, ThetaParams
from kats.utils.cupik import Pipeline


class cupikTest(TestCase):
    def setUp(self) -> None:
        self.TSData = load_air_passengers()

    def test_mkdetector(self) -> None:

        # We will be using 2 different scenarios to test if the results
        # are the same between a directly called MKDetector and one that
        # is called via CuPiK

        # Scene 1: window_size = 7, direction = 'up'
        pipe = Pipeline(
            # pyre-fixme[6]: For 1st param expected `List[Tuple[str, Step]]` but got
            #  `List[Tuple[str, MKDetector]]`.
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
            # pyre-fixme[6]: For 1st param expected `List[Tuple[str, Step]]` but got
            #  `List[Tuple[str, MKDetector]]`.
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
        pipe = Pipeline(
            # pyre-fixme[6]: For 1st param expected `List[Tuple[str, Step]]` but got
            #  `List[Tuple[str, ThetaModel]]`.
            [("theta_model", ThetaModel(data=self.TSData, params=ThetaParams()))]
        )
        fitted = pipe.fit(self.TSData)
        bools = (
            # pyre-fixme[16]: Optional type has no attribute `values`.
            ThetaModel(self.TSData, ThetaParams()).fit().fitted_values.values
            # pyre-fixme[16]: Item `List` of
            #  `Union[List[kats.consts.TimeSeriesData], TimeSeriesData]` has no
            #  attribute `fitted_values`.
            == fitted.fitted_values.values
        )
        self.assertEqual(np.sum(bools), 144)
        old_statsmodels = statsmodels.version < "0.12"
        expected = 433.328591954023 if old_statsmodels else 433.1270492317991
        # pyre-fixme[16]: Item `List` of `Union[List[kats.consts.TimeSeriesData],
        #  TimeSeriesData]` has no attribute `predict`.
        self.assertEqual(expected, fitted.predict(1).fcst.values[0])

        # test if the model can be built on the output from the detector
        pipe = Pipeline(
            # pyre-fixme[6]: For 1st param expected `List[Tuple[str, Step]]` but got
            #  `List[Tuple[str, Union[MKDetector, ThetaModel]]]`.
            [
                ("trend_detector", MKDetector(threshold=0.8)),
                ("theta_model", ThetaModel(data=self.TSData, params=ThetaParams())),
            ]
        )
        fitted = pipe.fit(self.TSData)
        self.assertEqual(len(pipe.metadata["trend_detector"][0]), 2)
        expected = 433.328591954023 if old_statsmodels else 433.1270492317991
        # pyre-fixme[16]: Item `List` of `Union[List[kats.consts.TimeSeriesData],
        #  TimeSeriesData]` has no attribute `predict`.
        self.assertEqual(expected, fitted.predict(1).fcst.values[0])
