# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements the Dynamic Time Warping (DTW) ChangePoint detector
algorithm as a DetectorModel, to provide a common interface.
"""

from __future__ import annotations

import logging

# Put code into Kats class

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesChangePoint, TimeSeriesData
from kats.detectors.detector import Detector


class DTWCPDChangePoint(TimeSeriesChangePoint):
    """Changepoint detected by the Dynamic Time Warping (DTW) detector model.

    This gives information about the type of detector, and the name of the time
    series.

    Attributes:

        start_time: Start time of the change.
        end_time: End time of the change.
        confidence: The confidence of the change point.
        ts_name: string, name of the time series for which the detector is
            is being run.
    """

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        confidence: float,
        ts_name: Optional[str] = None,
    ) -> None:
        super().__init__(start_time, end_time, confidence)
        self._detector_type = DTWCPDDetector
        self._ts_name = ts_name

    @property
    def detector_type(self) -> Type[Detector]:
        return self._detector_type

    @property
    def ts_name(self) -> Optional[str]:
        return self._ts_name

    def __repr__(self) -> str:
        return (
            f"DTWCPDChangePoint(start_time: {self._start_time}, end_time: "
            f"{self._end_time}, confidence: {self._confidence}, ts_name: "
            f"{self._ts_name}, detector_type={self._detector_type})"
        )


class DTWTimeSeriesTooSmallException(Exception):
    pass


@dataclass
class DTWSubsequenceMatch(object):
    matching_ts_name: str
    matching_ts_index: int
    matched_ts_name: str
    matched_ts_index: int
    distance: float


class DTWCPDDetector(Detector):
    """Dynamic Time Warping (DTW) ChangePoint detector.

    This algorithm detects anomalous time series in a dataset based on the DTW distance
    (see https://en.wikipedia.org/wiki/Dynamic_time_warping for details) between all the
    subsequences of length N in the dataset.

    Given an multivariate time series, this class performs changepoint detection, i.e. it tells
    us when the time series shows a change.

    The basic idea is to raise an alert when we see a new subsequence (via sliding window), which
    is substantially different from those seen in other time series historically. Difference here
    is measured by Dynamic Time Warping distance.

    Attrbutes:
        data: TimeSeriesData, data on which we will run the BOCPD algorithm.
        sliding_window_size: int, the time steps in the sliding window / subsequences to compare.
        mad_threshold:
        min_value:
    """

    def __init__(
        self,
        data: TimeSeriesData,
        sliding_window_size: int,
        skip_size: int,  # Step increment when sliding window moves
        mad_threshold: float = 1,  # Controls sensitivity.
        min_value: float = 1e-9,  # Controls when a value is considered to be zero
        match_against_same_time_series: bool = False,  # Whether to allow historical matches in the same time series.
    ) -> None:

        self.data = data
        self.sliding_window_size = sliding_window_size
        self.MIN_TS_VALUE = min_value
        self.MAD_CUTOFF = mad_threshold
        self.MIN_VALUE = min_value
        self.outliers: "List[DTWCPDChangePoint]" = []
        self.skip_size = skip_size
        self.match_against_same_time_series = match_against_same_time_series

        if self.sliding_window_size > len(self.data.value):
            raise DTWTimeSeriesTooSmallException(
                "The sliding window length is greater than the size of the time series."
            )

        if self.sliding_window_size > 100:
            logging.warning(
                "The model may take a very long time to run with such a large"
                f" sliding_window_size of {self.sliding_window_size}. While the model"
                " does apply fast pruning using lower bounds - it is still designed"
                " for shorter sliding_window_size"
                " (e.g., 15-50 steps), due to the underlying O(sliding_window_size^2)"
                " running time of Dynamic Time Warping. You may wish to"
                " downsample/resample the time series first."
            )

        if str(type(self.data.value)) == "<class 'pandas.core.series.Series'>":
            raise DTWTimeSeriesTooSmallException(
                "The input time series is not multivariate"
            )

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self) -> Sequence[DTWCPDChangePoint]:
        """Overwrite the detector method for DTWCPDDetector.

        Returns:
           A list of time series change points and matching ts information
        """

        logging.info("Starting detection...")
        # We first downsample the time series and only consider non-zero
        # sub-sequences to reduce the computational complexity of the problem.
        all_non_flats_ts = self._get_all_nonzero_subsequences()

        # If all time series are constant zero - then we are done.
        if len(all_non_flats_ts) <= 0:
            return self.outliers

        # If there is only one non-zero time series
        # then return that timeseries with the index of the first non-zero element.
        if len(all_non_flats_ts) == 1:  # There is only one time series
            ts_name = list(all_non_flats_ts.keys())[0]
            min_index = min(all_non_flats_ts[ts_name].keys())
            self.outliers.append(
                DTWCPDChangePoint(
                    start_time=self.data.time[min_index],
                    end_time=self.data.time[min_index + self.sliding_window_size - 1],
                    confidence=1e9,  # maximum confidence since its the only thing
                    ts_name=ts_name,
                )
            )
            return self.outliers  # We are done.

        # Otherwise we naively compare subsequences in each time series via an
        # efficient DTW calc to look for the closest match.
        max_min_dists = self._calculate_distances_for_all_subsequences(all_non_flats_ts)

        logging.info("Calculating anomalies")
        self.outliers = self._find_subsequences_anomalies(max_min_dists)

        return self.outliers

    def _get_nonzero_subsequences(self, ts: List[float]) -> Dict[int, List[float]]:
        win = self.sliding_window_size
        increment = self.skip_size

        return {
            ind: ts[ind : ind + win]
            for ind in range(0, len(ts) - win + 1, increment)
            if (max(ts[ind : ind + win]) > self.MIN_VALUE)
        }

    def _get_all_nonzero_subsequences(self) -> Dict[str, Dict[int, List[float]]]:
        all_non_flats_ts = {}
        df_ts = self.data.value
        c_subs = 0  # Total size of subsequences
        # TODO: vectorize + speed-up the below.
        for ix, ts_name in enumerate(df_ts.columns):
            ts = df_ts[ts_name].to_list()
            subs = self._get_nonzero_subsequences(ts)
            if len(subs) > 0:
                all_non_flats_ts[ts_name] = subs
                c_subs += len(subs)
            preprocess_log_unit = 1000
            if ix % preprocess_log_unit == 0:
                logging.info(f"Preprocessed {ix+1} / {len(df_ts.columns)}")
        logging.info(
            f"non-zero subsequences of length {self.sliding_window_size}"
            f" with minimum value of {self.MIN_TS_VALUE}: {c_subs} subsequences "
            f" on {len(all_non_flats_ts)} time series"
        )
        logging.info("Preprocessed time series.")
        return all_non_flats_ts

    @staticmethod
    def DTWDistance(s1: List[float], s2: List[float], w: int) -> float:
        """Compute DTW distance between the two lists

        Args:
            s1: first list of numbers of interest
            s2: second list of numbers of interest
            w: len(s2) // 3

        returns:
            DTW distance (a float) between the two lists
        """
        DTW = {}
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                DTW[(i, j)] = float("inf")
        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(
                    DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
                )
        return np.sqrt(float(DTW[len(s1) - 1, len(s2) - 1]))

    @staticmethod
    def LB_Keogh(s1: List[float], s2: List[float], w: int) -> float:
        """
        Computes a lower bound on the DTW distance between two time series.

        Method according to Keogh - see # http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html .

        Args:
            s1: first list of numbers of interest
            s2: second list of numbers of interest
            w: a number, the lower bound on the DTW distance

        returns:
            A lower bound (float) for the DTW distance between the two given inputs.
            The lower bound is computed based on <https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm>
        """
        # TODO: vectorize
        LB_sum = 0.0
        for ind, i in enumerate(s1):

            # Rolling min/max
            lower_bound = min(s2[(ind - w if ind - w >= 0 else 0) : (ind + w)])
            upper_bound = max(s2[(ind - w if ind - w >= 0 else 0) : (ind + w)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2

        return np.sqrt(LB_sum)

    def _calculate_distances_for_all_subsequences(
        self, ts2list_of_non_zero_subsequences: Dict[str, Dict[int, List[float]]]
    ) -> Dict[str, DTWSubsequenceMatch]:
        max_min_dists = defaultdict()
        for c, (ts_name_a, subsequences_a) in enumerate(
            ts2list_of_non_zero_subsequences.items()
        ):
            m = []
            for inda, subsequence_a in subsequences_a.items():
                distances = []
                bsf = np.inf  # TODO: rename to upper/lower bound name
                for (
                    ts_name_b,
                    subsequences_b,
                ) in ts2list_of_non_zero_subsequences.items():
                    if (ts_name_a != ts_name_b) or self.match_against_same_time_series:
                        for indb, subsequence_b in subsequences_b.items():
                            # If it's the same time series (and we allow matching - above)
                            # Then they subsequences better not overlap - or we will
                            # obviously match.
                            if (ts_name_a == ts_name_b) and (
                                # They overlap
                                (indb <= inda <= indb + self.sliding_window_size - 1)
                                or (
                                    indb
                                    <= inda + self.sliding_window_size - 1
                                    <= indb + self.sliding_window_size - 1
                                )
                            ):
                                continue

                            if (
                                self.LB_Keogh(
                                    subsequence_a,
                                    subsequence_b,
                                    int(len(subsequence_b) / 3),
                                )
                                < bsf
                            ):
                                dtw_distance = self.DTWDistance(
                                    subsequence_a,
                                    subsequence_b,
                                    int(len(subsequence_b) / 3),
                                )
                                if dtw_distance < bsf:
                                    bsf = dtw_distance
                            else:
                                dtw_distance = np.inf

                            distances.append(
                                DTWSubsequenceMatch(
                                    matching_ts_name=ts_name_a,
                                    matching_ts_index=inda,
                                    matched_ts_name=ts_name_b,
                                    matched_ts_index=indb,
                                    distance=dtw_distance,
                                )
                            )
                if len(distances) > 0:
                    # finding the closest sub to query_1
                    m.append(min(distances, key=lambda x: x.distance))
            # finding the sub with largest distance
            max_min_dists[ts_name_a] = max(m, key=lambda x: x.distance)

            if c % 50 == 0:
                logging.info(
                    "%d out of %d processed ..."
                    % (c, len(ts2list_of_non_zero_subsequences))
                )

        max_min_dists = dict(max_min_dists)
        return max_min_dists

    def _find_subsequences_anomalies(
        self, max_min_dists: Dict[str, DTWSubsequenceMatch]
    ) -> List[DTWCPDChangePoint]:
        """
        Given a set of subsequences with at least one non-zero value,
        detect the ones that are significantly dissimilar from others.
        The dissimilarity is measured by DTW distance.
        """
        # TODO: vectorize
        outliers = []
        highest_dists = [t.distance for _, t in max_min_dists.items()]
        med = np.percentile(highest_dists, 50)
        dev_from_med = [np.abs(med - v) for v in highest_dists]
        mad = np.mean(dev_from_med)
        for ts_name, t in max_min_dists.items():
            score = np.abs(med - t.distance) / mad
            if score > self.MAD_CUTOFF:
                outliers.append(
                    DTWCPDChangePoint(
                        start_time=self.data.time[t.matching_ts_index],
                        end_time=self.data.time[
                            t.matching_ts_index + self.sliding_window_size - 1
                        ],
                        confidence=score,
                        ts_name=ts_name,
                    )
                )
        return outliers
