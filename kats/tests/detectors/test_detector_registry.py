#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from kats.consts import InternalError
from kats.detectors.detector import DetectorModel, DetectorModelRegistry
from kats.detectors.prophet_detector import ProphetDetectorModel


class TestDetectorRegistry(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_detector_model_registry_on_creation(self) -> None:
        self.assertNotIn("SampleDetectorModel", DetectorModelRegistry.get_registry())
        with self.assertRaises(InternalError):
            DetectorModelRegistry.get_detector_model_by_name("SampleDetectorModel")

        class SampleDetectorModel(DetectorModel):
            def __init__(self) -> None:
                pass

        self.assertIn("SampleDetectorModel", DetectorModelRegistry.get_registry())
        sampleModel = DetectorModelRegistry.get_detector_model_by_name(
            "SampleDetectorModel"
        )()
        self.assertIsInstance(sampleModel, SampleDetectorModel)

    def test_detector_model_registry_on_import(self) -> None:
        prophetModel = DetectorModelRegistry.get_detector_model_by_name(
            "ProphetDetectorModel"
        )()
        self.assertIsInstance(prophetModel, ProphetDetectorModel)

        # Abstract classes shouldn't be registered
        self.assertNotIn("DetectorModel", DetectorModelRegistry.get_registry())
