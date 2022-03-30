# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List


class KatasMetaDetectorException(Exception):
    pass


class KatsDetectorsUnimplemented(KatasMetaDetectorException):
    pass


class KatsDetectorUnsupportedAlgoName(KatasMetaDetectorException):
    def __init__(self, algo_name: str, valid_algo_names: List[str]) -> None:
        valid_algo_str = ", ".join(valid_algo_names)
        super().__init__(
            f"unsupported detection algorithm name: {algo_name}, please choose one of those: {valid_algo_str}"
        )


class KatsDetectorHPTIllegalHyperParameter(KatasMetaDetectorException):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, algorithm_name: str, idx, hyper_params, e: Exception) -> None:
        hp_str = ",".join([f"{k}:{v}" for k, v in hyper_params.items()])
        super().__init__(
            f"detection algorithm {algorithm_name} can not be initialized with {hp_str} (data_y index: {idx}), throw:\n{e}"
        )


class KatsDetectorHPTTrainError(KatasMetaDetectorException):
    pass


class KatsDetectorHPTModelUsedBeforeTraining(KatasMetaDetectorException):
    pass
