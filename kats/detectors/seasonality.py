# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module is for seasonality detection.

We provide two seasonality detector: ACFDetector and FFTDetector. ACFDetector uses
autocorrelation function to find seasonality, while FFTDetector uses Fast Fourier
Transform to detect seasonality.

Typical usage example:

>>> timeseries = TimeSeriesData(...)
>>> # initialize detector
>>> detector = ACFDetector(timeseries)
>>> # run detector
>>> detector.detector(diff=1, alpha = 0.01)
>>> # seasonality decomposition, returns trend, seasonal, residual term
>>> detector.remover()
>>> # plot acf and decompsition results
>>> detector.plot()
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objs as go

    Figure = go.Figure
except ImportError:
    Figure = Any
import scipy.fftpack as fp
import statsmodels.api as sm
from kats.consts import TimeSeriesData
from kats.detectors.detector import Detector
from kats.graphics.plots import make_fourier_plot
from kats.utils.decomposition import TimeSeriesDecomposition
from scipy.signal import find_peaks  # @manual
from statsmodels.tsa.stattools import acf

# from numpy.typing import ArrayLike
ArrayLike = Union[np.ndarray, Sequence[float]]


class ACFDetector(Detector):
    """Autocorrelation function seasonality detector.

    Use acf to detect seasonality, and find out the potential cycle lengths

    Attributes:
        data: The input time series data from TimeSeriesData
        decomposed: A bool indicate if we decomposed the time series into trend,
            seasonal and residual.
    """

    ts_diff: Optional[ArrayLike] = None
    lags: Optional[int] = None
    seasonality: Optional[List[int]] = None
    seasonality_detected: bool = False
    decompose: Optional[TimeSeriesDecomposition] = None

    def __init__(self, data: TimeSeriesData):
        super().__init__(data=data)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.decomposed = False

    def _get_seasonality_length(self, d: List[int]) -> List[int]:
        out = []
        while d:
            k = d.pop(0)
            d = [i for i in d if i % k != 0]
            out.append(k)
        return out

    # pyre-fixme[14]: Inconsistent override [14]: `kats.detectors.seasonality.ACFDetector.detector` overrides method defined in `Detector` inconsistently. Could not find parameter `method` in overriding signature.
    # pyre-fixme[14]: `kats.detectors.seasonality.ACFDetector.detector` overrides method defined in `Detector` inconsistently. Returned type `Dict[str, typing.Any]` is not a subtype of the overridden return `None`.
    def detector(
        self, lags: Optional[int] = None, diff: int = 1, alpha: Optional[float] = 0.01
    ) -> Dict[str, Any]:
        """Detect seasonality

        This method runs acf and returns if seasonality detected in the given time series
        and potential cycle lengths

        Args:
            lags: Optional; int; the maximum lags we used in acf.
            diff: Optional; int; times of diff run on timeseries to remove trend before
                apply acf.
            alpha: Optional; float; significant level we use the calcualte
                autocorrelation confidence interval.

        Returns:
            A dict contains
                - seasonality_presence: bool, if seasonality detected
                - seasonalities: List[int], potential seasonlities cycle length(s)
        """

        # Use array to store the data
        ts = self.data.value.values
        ts_diff = ts
        for _ in range(diff):
            ts_diff = np.diff(ts_diff)
        self.ts_diff = ts_diff

        if lags is None:
            lags = int(len(ts) / 3)
        self.lags = lags
        ac, confint, qstat, qval = acf(ts_diff, nlags=lags, qstat=True, alpha=alpha)
        # get seasonality cycle length
        raw_seasonality = []
        for i, _int in enumerate(confint):
            if _int[0] >= 0 and i > 1:
                raw_seasonality.append(i)
        self.seasonality = self._get_seasonality_length(raw_seasonality)
        self.seasonality_detected = True if self.seasonality else False

        return {
            "seasonality_presence": self.seasonality_detected,
            "seasonalities": self.seasonality,
        }

    def plot(self) -> None:
        """Plot detection results.

        Args:
            None

        Returns:
            None
        """
        sm.graphics.tsa.plot_acf(self.ts_diff, lags=self.lags)
        plt.show()
        if self.decomposed:
            decompose = self.decompose
            assert decompose is not None
            decompose.plot()
        plt.show()

    # pyre-fixme[14]: `kats.detectors.seasonality.ACFDetector.remover` overrides method defined in `Detector` inconsistently. Could not find parameter `interpolate` in overriding signature.
    # pyre-fixme[15]: `kats.detectors.seasonality.ACFDetector.remover` overrides method defined in `Detector` inconsistently. Returned type `Optional[Dict[str, typing.Any]]` is not a subtype of the overridden return `TimeSeriesData`.
    def remover(
        self,
        decom: Type = TimeSeriesDecomposition,
        model: str = "additive",
        decompose_any_way: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Remove the seasonality in the time series

        Args:
            decom: Optional; decomposition method.
            model: Optional; model used for decomposition.
            decompose_any_way: Optional; bool; decompose the time series even when
                seasonality is not detected in the time series.

        Returns:
            decomposition results of the decomposition method.
        """

        if decompose_any_way or self.seasonality_detected:
            self.decompose = decompose = decom(self.data, model)
            result = decompose.decomposer()
            self.decomposed = True
            return result
        else:
            logging.info("No seasonality detected, not running decomposition")


class FFTDetector(Detector):
    """Fast Fourier Transform Seasoanlity detector

    Use Fast Fourier Transform to detect seasonality, and find out the
    potential cycle's length.

    Attributes:
        data: The input time series data from TimeSeriesData.
    """

    def __init__(self, data: TimeSeriesData):
        super().__init__(data=data)
        if not self.data.is_univariate():
            msg = "The provided time series data is not univariate."
            logging.error(msg)
            raise ValueError(msg)

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self, sample_spacing: float = 1.0, mad_threshold: float = 6.0) -> Dict:
        """Detect seasonality with FFT

        Args:
            sample_spacing: Optional; float; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0, FFT x axis will be
                1/day.
            mad_threshold: Optional; float; constant for the outlier algorithm for peak
                detector. The larger the value the less sensitive the outlier algorithm
                is.

        Returns:
            FFT Plot with peaks, selected peaks, and outlier boundary line.
        """

        fft = self.get_fft(sample_spacing)
        _, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        # pyre-fixme[6]: Expected `Sized` for 1st param but got
        #  `BoundMethod[typing.Callable(list.index)[[Named(self, List[float]), float,
        #  int, default, int, default], int], List[float]]`.
        seasonality_presence = len(peaks.index) > 0
        selected_seasonalities = []
        if seasonality_presence:
            # pyre-fixme[16]: `float` has no attribute `transform`.
            # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got `str`.
            selected_seasonalities = peaks["freq"].transform(lambda x: 1 / x).tolist()

        return {
            "seasonality_presence": seasonality_presence,
            "seasonalities": selected_seasonalities,
        }

    # pyre-fixme[15]: `plot` overrides method defined in `Detector` inconsistently.
    def plot(
        self,
        time_unit: str,
        sample_spacing: float = 1.0,
        title: str = "FFT",
        mad_threshold: float = 6.0,
    ) -> Figure:
        """Plots an FFT plot as a plotly figure

        Args:
            time_unit: string containing the unit of time (displayed on x axis).
                E.g. 'Hour'.
            sample_spacing: Optional; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0,
                FFT x axis will be 1/day.
            title: Optional; title of the plot.
            mad_threshold: Optional; constant for the outlier algorithm for peak
                detector. The larger the value the less sensitive the outlier algorithm
                is.

        Returns:
            FFT Plot with peaks, selected peaks, and outlier boundary line.
        """
        fft = self.get_fft(sample_spacing)
        thres, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        return make_fourier_plot(
            fft, thres, orig_peaks, peaks, f"1/{time_unit}", title=title
        )

    def get_fft(self, sample_spacing: float = 1.0) -> pd.DataFrame:
        """Computes FFT

        Args:
            sample_spacing: Optional; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0 FFT x axis will be 1/day.

        Returns:
            DataFrame with columns 'freq' and 'ampl'.
        """
        data_fft = fp.fft(self.data.value.values)
        data_psd = np.abs(data_fft) ** 2
        fftfreq = fp.fftfreq(len(data_psd), 1.0 / sample_spacing)
        pos_freq_ix = fftfreq > 0

        freq = (fftfreq[pos_freq_ix],)
        ampl = (10 * np.log10(data_psd[pos_freq_ix]),)

        return pd.DataFrame({"freq": freq[0], "ampl": ampl[0]})

    def get_fft_peaks(
        self, fft: pd.DataFrame, mad_threshold: float = 6.0
    ) -> Tuple[float, List[float], List[float]]:
        """Computes peaks in fft, selects the highest peaks (outliers) and
            removes the harmonics (multiplies of the base harmonics found)

        Args:
            fft: FFT computed by FFTDetector.get_fft
            sample_spacing: Optional; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0 FFT x axis will be 1/day.
            mad_threshold: Optional; constant for the outlier algorithm for peak detector.
                The larger the value the less sensitive the outlier algorithm is.

        Returns:
            outlier threshold, peaks, selected peaks.
        """
        pos_fft = fft.loc[fft["ampl"] > 0]
        median = pos_fft["ampl"].median()
        pos_fft_above_med = pos_fft[pos_fft["ampl"] > median]
        mad = pos_fft_above_med["ampl"].mad()

        threshold = median + mad * mad_threshold

        peak_indices = find_peaks(fft["ampl"], threshold=0.1)
        peaks = fft.loc[peak_indices[0], :]

        orig_peaks = peaks.copy()

        peaks = peaks.loc[peaks["ampl"] > threshold].copy()
        peaks["Remove"] = [False] * len(peaks.index)
        peaks.reset_index(inplace=True)

        # Filter out harmonics
        for idx1 in range(len(peaks)):
            curr = peaks.loc[idx1, "freq"]
            for idx2 in range(idx1 + 1, len(peaks)):
                if peaks.loc[idx2, "Remove"] is True:
                    continue
                fraction = (peaks.loc[idx2, "freq"] / curr) % 1
                if fraction < 0.01 or fraction > 0.99:
                    peaks.loc[idx2, "Remove"] = True
        peaks = peaks.loc[~peaks["Remove"]]
        peaks.drop(inplace=True, columns="Remove")
        return threshold, orig_peaks, peaks

    def plot_fft(
        self,
        time_unit: str,
        sample_spacing: float = 1.0,
        title: str = "FFT",
        mad_threshold: float = 6.0,
    ) -> Figure:
        """Plots an FFT plot as a plotly figure

        Args:
            time_unit: string containing the unit of time (displayed on x axis).
                            E.g. 'Hour'
            sample_spacing: Optional; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0,
                FFT x axis will be 1/day
            title: Optional; title of the plot
            mad_threshold: Optional; constant for the outlier algorithm for peak
                detector. The larger the value the less sensitive the outlier algorithm
                is.

        Returns:
            FFT Plot with peaks, selected peaks, and outlier boundary line
        """
        fft = self.get_fft(sample_spacing)
        thres, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        return make_fourier_plot(
            fft, thres, orig_peaks, peaks, f"1/{time_unit}", title=title
        )
