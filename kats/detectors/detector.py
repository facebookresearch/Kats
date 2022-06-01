# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the base class for detectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

try:
    import plotly.graph_objs as go

    Figure = go.Figure
except ImportError:
    Figure = object
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData, TimeSeriesIterator
from kats.detectors.detector_consts import AnomalyResponse


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

    def plot(self, **kwargs: Any) -> Union[plt.Axes, Sequence[plt.Axes], Figure]:
        raise NotImplementedError()


class DetectorModel(ABC):
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
