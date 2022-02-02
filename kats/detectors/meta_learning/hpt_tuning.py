# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
MetaDetectHptSelect is a meta learner that predict best hyper parameters of chosen detection algorithm given a time series features
before predicting, user needs to train the model:
    data_x=time_series features
    data_y=best (observed) hyper parameters for those features
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ...models.metalearner.metalearner_hpt import MetaLearnHPT
from ..cusum_model import CUSUMDetectorModel
from ..stat_sig_detector import StatSigDetectorModel
from .exceptions import (
    KatsDetectorHPTTrainError,
    KatsDetectorHPTIllegalHyperParameter,
    KatsDetectorUnsupportedAlgoName,
    KatsDetectorsUnimplemented,
    KatsDetectorHPTModelUsedBeforeTraining,
)


@dataclass
class DetectionAlgoMeta:
    detector_cls: Callable


class MetaDetectHptSelect:
    DETECTION_ALGO = {
        "cusum": DetectionAlgoMeta(detector_cls=CUSUMDetectorModel),
        "statsig": DetectionAlgoMeta(detector_cls=StatSigDetectorModel),
    }
    TRAIN_DEFAULT_VALUES = {
        "lr": 0.001,
    }

    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame, algorithm_name: str):
        self._check_valid_input(data_x, data_y, algorithm_name)
        self._data_x: pd.DataFrame = data_x
        self._data_y: pd.DataFrame = data_y
        self._algorithm_name: str = algorithm_name
        self._meta_hpt_model: Optional[MetaLearnHPT] = None

    @staticmethod
    def _check_valid_input(
        data_x: pd.DataFrame, data_y: pd.DataFrame, algorithm_name: str
    ) -> None:

        # check 1: is given algorithm is supported
        if algorithm_name not in MetaDetectHptSelect.DETECTION_ALGO:
            raise KatsDetectorUnsupportedAlgoName(
                algo_name=algorithm_name,
                valid_algo_names=list(MetaDetectHptSelect.DETECTION_ALGO.keys()),
            )

        # check 2: are all hyper parameters valid for the given algorithm
        for idx, hp in data_y.iterrows():
            try:
                MetaDetectHptSelect._init_detection_model(algorithm_name, hp.to_dict())
            except Exception as e:
                raise KatsDetectorHPTIllegalHyperParameter(
                    algorithm_name, idx, hp.to_dict(), e
                )

    @staticmethod
    def _init_detection_model(algorithm_name, hyper_parameters):
        """
        returns detection algorithm for given algorithm name initialized with given hyper parameters
        """
        return MetaDetectHptSelect.DETECTION_ALGO[algorithm_name].detector_cls(
            **hyper_parameters
        )

    def train(self, **meta_learn_hpt_kwargs) -> MetaDetectHptSelect:
        for k in self.TRAIN_DEFAULT_VALUES:
            if k not in meta_learn_hpt_kwargs:
                meta_learn_hpt_kwargs[k] = self.TRAIN_DEFAULT_VALUES[k]

        model = MetaLearnHPT(
            data_x=self._data_x,
            data_y=self._data_y,
            default_model=self._algorithm_name,
            scale=False,
        )
        model.build_network()
        try:
            self._train_model(model, meta_learn_hpt_kwargs=meta_learn_hpt_kwargs)
        except Exception as e:
            raise KatsDetectorHPTTrainError(e)
        self._meta_hpt_model = model
        return self

    def _train_model(self, model: MetaLearnHPT, meta_learn_hpt_kwargs):
        model.train(**meta_learn_hpt_kwargs)

    def plot(self, **kwargs: Any) -> Sequence[plt.Axes]:
        model = self._meta_hpt_model
        if model is None:
            raise KatsDetectorHPTModelUsedBeforeTraining()
        return model.plot(**kwargs)

    def get_hpt(self, ts: pd.DataFrame):  # -> model_params:
        # get the hpt for a time series
        # ignore for the first bootcamp task
        raise KatsDetectorsUnimplemented()

    def save_model(self):
        raise KatsDetectorsUnimplemented()

    def load_model(self):
        raise KatsDetectorsUnimplemented()
