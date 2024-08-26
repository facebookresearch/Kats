# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Defines the base class for detectors.
"""

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Generic, Optional, Sequence, Tuple, Type, TypeVar, Union

try:
    import plotly.graph_objs as go

    # pyre-fixme[5]: Global expression must be annotated.
    Figure = go.Figure
except ImportError:
    Figure = object
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import (
    InternalError,
    TimeSeriesChangePoint,
    TimeSeriesData,
    TimeSeriesIterator,
)
from kats.detectors.detector_consts import AnomalyResponse

T = TypeVar("T")
D = TypeVar("D", bound="DetectorModel")


class DetectorModelRegistry(ABCMeta, Generic[D]):
    """
    Registry of instantiable DetectorModel subclasses.

    Can be used to look up and instantiate DetectorModels by their class name
    as follows:
      >> DetectorModelRegistry.get_detector_model_by_name("ConcreteDetectorModel")(...)
    """

    REGISTRY: Dict[str, Type[D]] = {}

    # __new__ is called when a sub class is defined and not instantiated
    def __new__(
        cls, name: str, bases: Tuple[Type[T], ...], attrs: Dict[str, Any]
    ) -> Type[T]:
        # Create the new class object (callable)
        new_cls = type.__new__(cls, name, bases, attrs)
        # Only store instantiable (non-abstract) types
        if not inspect.isabstract(new_cls):
            # Store the class object with the key as the class name.
            # Note that the class name is case sensitive
            cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> Dict[str, Type[D]]:
        return dict(cls.REGISTRY)

    @classmethod
    def get_detector_model_by_name(cls, class_name: str) -> Type[D]:
        try:
            # Return the class object that can be called to instantiate the class
            return cls.REGISTRY[class_name]
        except KeyError as e:
            raise InternalError(
                f"No DetectorModel subclass exists with the name '{class_name}'!"
            ) from e


class Detector(ABC):
    """Base detector class to be inherited by specific detectors

    Attributes:
        data: The input time series data from TimeSeriesData
    """

    iter: Optional[TimeSeriesIterator] = None
    outliers: Optional[Sequence[TimeSeriesChangePoint]] = None

    def __init__(self, data: TimeSeriesData) -> None:
        self.data = data
        self.__type__ = "detector"
        if data is not None:
            self.data.time = pd.to_datetime(self.data.time)

    @abstractmethod
    def detector(self, method: Optional[str] = None) -> Sequence[TimeSeriesChangePoint]:
        raise NotImplementedError()

    def remover(self, interpolate: bool = False) -> TimeSeriesData:
        """Remove the outlier in time series

        Args:
            interpolate: Optional; bool; interpolate the outlier

        Returns:
            A TimeSeriesData with outlier removed.
        """
        outliers = self.outliers
        if outliers is None:
            return self.data

        df = []
        self.detector()
        count = 0
        for count, ts in enumerate(TimeSeriesIterator(self.data)):
            ts.loc[outliers[count], "y"] = np.nan
            df.append(ts)

        # Need to make this a ts object
        df_final = pd.concat(df, axis=1, copy=False)

        if interpolate:
            df_final.interpolate(method="linear", limit_direction="both", inplace=True)
        # may contain multiple time series y
        df_final.columns = [f"y_{i}" for i in range(count + 1)]
        df_final["time"] = df_final.index
        ts_out = TimeSeriesData(df_final)
        return ts_out

    # pyre-fixme[11]: Annotation `Figure` is not defined as a type.
    def plot(self, **kwargs: Any) -> Union[plt.Axes, Sequence[plt.Axes], Figure]:
        raise NotImplementedError()


class DetectorModel(metaclass=DetectorModelRegistry):
    """
    Base Detector model class to be inherited by specific detectors. A DetectorModel
    keeps the state of the Detector, and implements the incremental model training.

    The usage of the DetectorModel is (replace DetectorModel with the proper child class)

    model = DetectorModel(serialized_model)
    model.fit(new_data, ...)

    # the model may be saved through model.serialize() call
    # the model may be loaded again through model = DetectorModel(serialized_model)

    result = model.predict(data, ...)
    """

    @abstractmethod
    def __init__(self, serialized_model: Optional[bytes]) -> None:
        raise NotImplementedError()

    """
    Serialize the model. It's required that the serialized model can be unserialized
    by the next version of the same DetectorModel class. During upgrade of a model class,
    version 1 to 2, the version 2 code will unserialize version 1 model, create the
    new model (version 2) instance, the serialize out the version 2 instance, thus
    completing the upgrade.
    """

    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError()

    """
    Fit the model with the data passes in and update the model's state.
    """

    @abstractmethod
    def fit(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    """
    Given the time series data, returns the anomaly score time series data with
    matching timestamps.
    """

    @abstractmethod
    def predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> AnomalyResponse:
        raise NotImplementedError()

    """
    This method will change the state and return the anomaly scores.
    """

    @abstractmethod
    def fit_predict(
        self,
        data: TimeSeriesData,
        historical_data: Optional[TimeSeriesData],
        **kwargs: Any,
    ) -> AnomalyResponse:
        raise NotImplementedError()
