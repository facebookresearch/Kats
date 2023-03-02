#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from kats.detectors.detector import DetectorModel, DetectorModelRegistry
from kats.detectors.prophet_detector import ProphetDetectorModel


class SampleDetectorModel(DetectorModel):
    def __init__(self) -> None:
        print("Test Class")


class TestDetectorRegistry(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_detector_registry(self) -> None:
        sampleModel = DetectorModelRegistry.get_detector_model_by_name(
            "SampleDetectorModel"
        )()
        self.assertTrue(isinstance(sampleModel, SampleDetectorModel))

    def test_detector_import_registry(self) -> None:
        prophetModel = DetectorModelRegistry.get_detector_model_by_name(
            "ProphetDetectorModel"
        )()
        self.assertTrue(isinstance(prophetModel, ProphetDetectorModel))
