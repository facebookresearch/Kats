# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generic, List, Optional, TypeVar, Union

from kats.consts import TimeSeriesData


class EmbeddingParams:
    # Base class for all our embedding parameters
    def __init__(self) -> None:
        pass

    def validate_params(self) -> None:
        pass


EmbedParamType = TypeVar("EmbedParamType", bound=EmbeddingParams)


class EmbeddingModel(Generic[EmbedParamType]):
    # Base class for all our embedding models

    def __init__(
        self,
        data: Optional[List[TimeSeriesData]],
        params: Optional[EmbedParamType],
        load_model: bool = False,
    ) -> None:
        pass

    def train(self) -> None:
        """abstract method for training

        This is a declaration for train method
        """
        pass

    def transform(
        self, new_data: Union[TimeSeriesData, List[TimeSeriesData]]
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """abstract method for transforming one or several time series

        This is a declaration for transform method
        """
        pass

    def batch_transform(
        self, new_data: List[TimeSeriesData]
    ) -> Optional[List[List[float]]]:
        """abstract method for batch transforming

        This is a declaration for batch_transform method
        """
        pass

    def save_model(self, file_path: str) -> None:
        """abstract method for saving a model

        This is a declaration for save_model method
        """
        pass
