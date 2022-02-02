# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd
from kats.detectors.meta_learning.exceptions import (
    KatsDetectorHPTIllegalHyperParameter,
    KatsDetectorUnsupportedAlgoName,
    KatsDetectorsUnimplemented,
    KatsDetectorHPTTrainError,
    KatsDetectorHPTModelUsedBeforeTraining,
)
from kats.detectors.meta_learning.hpt_tuning import MetaDetectHptSelect
from kats.detectors.meta_learning.synth_metadata_reader import SynthMetadataReader
from parameterized.parameterized import parameterized


class TestMetaDetectHptSelect(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._synth_data_read = SynthMetadataReader()

    @staticmethod
    def _get_valid_alg_name() -> str:
        supported_algos = list(MetaDetectHptSelect.DETECTION_ALGO.keys())
        return supported_algos[random.randint(0, len(supported_algos) - 1)]

    @classmethod
    def _get_valid_metadata(cls, algorithm_name: str):
        return cls._synth_data_read.get_metadata(algorithm_name)

    @parameterized.expand(MetaDetectHptSelect.DETECTION_ALGO.keys())
    def test_legal_run(self, algorithm_name) -> None:
        MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        ).train()

    @parameterized.expand(["blabla", "mumu"])
    def test_not_supportted_algo(self, algorithm_name) -> None:
        with self.assertRaises(KatsDetectorUnsupportedAlgoName):
            MetaDetectHptSelect(
                **self._get_valid_metadata(self._get_valid_alg_name()),
                algorithm_name=algorithm_name
            )

    @parameterized.expand(MetaDetectHptSelect.DETECTION_ALGO.keys())
    def test_illegal_hyper_parameter(self, algo_name) -> None:
        metadata = self._get_valid_metadata(algo_name)
        data_y = metadata["data_y"]
        corrupted_col = data_y.columns[random.randint(0, len(data_y.columns) - 1)]
        corrupted_data_y = data_y.rename(columns={corrupted_col: "blabla"})

        with self.assertRaises(KatsDetectorHPTIllegalHyperParameter):
            MetaDetectHptSelect(metadata["data_x"], corrupted_data_y, algo_name)

    def test_training_error(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        )
        model._train_model = Mock(side_effect=Exception("unknown sub exception..."))
        with self.assertRaises(KatsDetectorHPTTrainError):
            model.train()

    def test_legal_plot(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        ).train()
        model.plot()

    def test_plot_before_train(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        )
        with self.assertRaises(KatsDetectorHPTModelUsedBeforeTraining):
            model.plot()

    def test_get_hpt(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        ).train()
        with self.assertRaises(KatsDetectorsUnimplemented):
            model.get_hpt(pd.DataFrame())

    def test_save_model(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        ).train()
        with self.assertRaises(KatsDetectorsUnimplemented):
            model.save_model()

    def test_load_model(self) -> None:
        algorithm_name = self._get_valid_alg_name()
        model = MetaDetectHptSelect(
            **self._get_valid_metadata(algorithm_name), algorithm_name=algorithm_name
        ).train()
        with self.assertRaises(KatsDetectorsUnimplemented):
            model.load_model()
