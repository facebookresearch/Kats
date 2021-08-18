# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from kats.consts import TimeSeriesData, TimeSeriesChangePoint
from kats.detectors.detector import Detector

# pyre-fixme[21]: Could not find name `zscore` in `scipy.stats`.
from scipy.stats import norm, zscore  # @manual


class RobustStatMetadata:
    def __init__(self, index: int, metric: float) -> None:
        self._metric = metric
        self._index = index

    @property
    def metric(self):
        return self._metric

    @property
    def index(self):
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

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(
        self,
        p_value_cutoff: float = 1e-2,
        smoothing_window_size: int = 5,
        comparison_window: int = -2,
    ) -> List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]:
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

        # pyre-fixme[16]: Module `stats` has no attribute `zscore`.
        y_zscores = zscore(df_)
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
            cp = TimeSeriesChangePoint(
                start_time=data_df.index.values[idx],
                end_time=data_df.index.values[idx],
                confidence=1 - p_values[idx],
            )
            metadata = RobustStatMetadata(index=idx, metric=float(df_.iloc[idx]))

            change_points.append((cp, metadata))

        return change_points

    def plot(
        self, change_points: List[Tuple[TimeSeriesChangePoint, RobustStatMetadata]]
    ) -> None:
        time_col_name = self.data.time.name
        val_col_name = self.data.value.name

        data_df = self.data.to_dataframe()

        plt.plot(data_df[time_col_name], data_df[val_col_name])

        if len(change_points) == 0:
            logging.warning("No change points detected!")

        for change in change_points:
            plt.axvline(x=change[0].start_time, color="red")

        plt.show()
