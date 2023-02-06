# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector
from scipy.stats import norm, zscore  # @manual


class RobustStatChangePoint(TimeSeriesChangePoint):
    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
        index: int,
        metric: float,
    ) -> None:
        super().__init__(start_time, end_time, confidence)
        self._metric = metric
        self._index = index

    @property
    def metric(self) -> float:
        return self._metric

    @property
    def index(self) -> int:
        return self._index


class RobustStatDetector(Detector):
    def __init__(self, data: TimeSeriesData) -> None:
        super(RobustStatDetector, self).__init__(data=data)
        if not self.data.is_univariate():
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.zscore: Optional[np.ndarray] = None

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(
        self,
        p_value_cutoff: float = 1e-2,
        smoothing_window_size: int = 5,
        comparison_window: int = -2,
    ) -> Sequence[RobustStatChangePoint]:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()
        data_df = data_df.set_index(time_col_name)

        df_ = data_df.loc[:, val_col_name].rolling(window=smoothing_window_size)
        df_ = (
            # Smooth
            df_.mean()
            .fillna(method="bfill")
            # Make spikes standout
            .diff(comparison_window)
            .fillna(0)
        )

        y_zscores = zscore(df_)
        self.zscore = y_zscores
        p_values = norm.sf(np.abs(y_zscores))
        ind = np.where(p_values < p_value_cutoff)[0]

        if len(ind) == 0:
            return []  # empty list for no change points

        change_points = []

        prev_idx = -1
        for idx in ind:
            if prev_idx != -1 and (idx - prev_idx) < smoothing_window_size:
                continue

            prev_idx = idx
            cp = RobustStatChangePoint(
                start_time=data_df.index.values[idx],
                end_time=data_df.index.values[idx],
                # pyre-fixme[16]: `float` has no attribute `__getitem__`.
                confidence=1 - p_values[idx],
                index=idx,
                metric=float(df_.iloc[idx]),
            )

            change_points.append(cp)

        return change_points

    def plot(
        self, change_points: Sequence[RobustStatChangePoint], **kwargs: Any
    ) -> plt.Axes:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        _, ax = plt.subplots()
        ax.plot(data_df[time_col_name].to_numpy(), data_df[val_col_name].to_numpy())

        changepoint_annotated = False
        for change in change_points:
            ax.axvline(x=change.start_time, color="red")
            changepoint_annotated = True
        if not changepoint_annotated:
            logging.warning("No change points detected!")

        return ax
