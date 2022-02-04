# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest import TestCase

import numpy as np
import pandas as pd
from kats.detectors.meta_learning.metalearning_detection_model import (
    MetaDetectModelSelect,
)

base_models = ["cusum", "statsig", "bocpd"]


def generate_meta_data(n):
    data = []
    features = np.random.randn(n * 10).reshape(n, -1)
    for i in range(n):
        hpt_res = {}
        for key in base_models:
            hpt_res[key] = ({}, np.random.rand())
        feature_dict = {str(k): features[i, k] for k in range(features.shape[1])}
        best_model = base_models[np.random.randint(0, len(base_models))]
        data.append([hpt_res, feature_dict, best_model])
    return pd.DataFrame(data, columns=["hpt_res", "features", "best_model"])


class MetaDetectModelSelectTest(TestCase):
    def setUp(self) -> None:
        self.df = generate_meta_data(np.random.randint(35, 100))
        self.meta_detector = MetaDetectModelSelect(self.df)

    def test_initialize(self) -> None:
        self.assertRaises(
            ValueError,
            MetaDetectModelSelect,
            np.random.rand(3, 4).tolist(),
        )

        self.assertRaises(
            ValueError,
            MetaDetectModelSelect,
            pd.DataFrame([], columns=["hpt_res", "features", "best_model"]),
        )

        self.assertRaises(ValueError, MetaDetectModelSelect, generate_meta_data(20))

        self.assertRaises(
            ValueError, MetaDetectModelSelect, generate_meta_data(40)[["hpt_res"]]
        )

        self.assertRaises(
            ValueError, MetaDetectModelSelect, generate_meta_data(40)[["features"]]
        )

        self.assertRaises(
            ValueError,
            MetaDetectModelSelect,
            generate_meta_data(40)[["hpt_res", "features"]],
        )

    def test_preprocess(self) -> None:
        preprocessed_data = self.meta_detector.preprocess()
        self.assertEqual(len(preprocessed_data), len(self.df))
        for elem in preprocessed_data:
            self.assertIn("hpt_res", elem)
            self.assertIn("features", elem)
            self.assertIn("best_model", elem)

    def test_train(self) -> None:
        res = self.meta_detector.train()
        for elem in ["fit_error", "pred_error", "clf_accuracy"]:
            self.assertIn(elem, res)

    def test_report_metrics(self) -> None:
        summary = self.meta_detector.report_metrics()
        self.assertIsInstance(summary, pd.DataFrame)
        for model in base_models:
            self.assertIn(model, summary)
        self.assertIn("meta-learn", summary)

    def test_predict(self) -> None:
        self.assertRaises(
            ValueError,
            self.meta_detector.predict,
            np.random.rand(3, 4).tolist(),
        )
