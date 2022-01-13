# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from operator import attrgetter
from unittest import TestCase

import pandas as pd
from kats.detectors.trend_mk_model import MKDetectorModel
from kats.tests.detectors.utils import gen_no_trend_data_ndim, gen_trend_data_ndim
from parameterized.parameterized import parameterized


class TestNoTrendMKDetectorModel(TestCase):
    def setUp(self):
        time = pd.Series(pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D"))
        self.no_trend_data = gen_no_trend_data_ndim(time)
        self.response_no_historical = MKDetectorModel().fit_predict(
            data=self.no_trend_data
        )

        test_data_length = 20
        historical_data = self.no_trend_data[:-test_data_length]
        self.test_data = self.no_trend_data[-test_data_length:]

        self.response = MKDetectorModel().fit_predict(
            historical_data=historical_data,
            data=self.test_data,
        )

    # none of the points shoul be stat sig here
    @parameterized.expand(
        [["response_no_historical", "no_trend_data"], ["response", "test_data"]]
    )
    def test_stat_sig(self, response_object, data_object):
        self.assertEqual(
            attrgetter(response_object)(self).stat_sig_ts.value.values.tolist(),
            [0] * len(attrgetter(data_object)(self)),
        )

    # all of the scores should be small here
    @parameterized.expand(
        [["response_no_historical", "no_trend_data"], ["response", "test_data"]]
    )
    def test_scores(self, response_object, data_object):
        self.assertEqual(
            (abs(attrgetter(response_object)(self).scores.value.values) < 0.1).tolist(),
            [True] * len(attrgetter(data_object)(self)),
        )

    def test_serialize(self):
        detector = MKDetectorModel()
        model_dict = {
            "window_size": detector.window_size,
            "training_days": detector.training_days,
            "direction": detector.direction,
            "freq": detector.freq,
            "threshold": detector.threshold,
            "alpha": detector.alpha,
        }
        self.assertEqual(detector.serialize(), json.dumps(model_dict).encode("utf-8"))


class TestDownTrendMKDetectorModel(TestCase):
    def setUp(self):
        time = pd.Series(pd.date_range(start="2020-01-01", end="2020-06-20", freq="1D"))
        trend_data, t_change = gen_trend_data_ndim(time=time)
        self.cp = t_change[0]
        self.threshold = 0.8
        detector = MKDetectorModel(
            window_size=14, direction="down", threshold=self.threshold
        )
        self.response = detector.fit_predict(
            historical_data=trend_data[: self.cp], data=trend_data[self.cp :]
        )
        detector2 = MKDetectorModel(
            window_size=14, direction="down", threshold=self.threshold
        )
        self.response_no_historical = detector2.fit_predict(data=trend_data)

    def test_no_historical(self):
        self.assertTrue(
            self.response_no_historical.stat_sig_ts[: self.cp].value.values.tolist(),
            [0] * self.cp,
        )

    def test_scores(self):
        self.assertEqual(
            (self.response.scores.value.values < 0).tolist(),
            [True] * len(self.response.scores),
        )

    def test_stat_sig(self):
        self.assertEqual(
            (self.response.scores.value.values < -self.threshold).astype(int).tolist(),
            self.response.stat_sig_ts.value.values.tolist(),
        )
