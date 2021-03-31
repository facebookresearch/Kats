#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from itertools import chain, combinations
from typing import Dict, List, Tuple
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fp
import statsmodels.api as sm
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detector import Detector
from infrastrategy.kats.graphics.plots import make_fourier_plot
from infrastrategy.kats.models.prophet import ProphetModel, ProphetParams
from infrastrategy.kats.utils.backtesters import BackTesterRollingWindow
from infrastrategy.kats.utils.decomposition import TimeSeriesDecomposition
from scipy.signal import find_peaks  # @manual
from statsmodels.tsa.stattools import acf


class ACFDetector(Detector):
    """
    Use acf to detect seasonality, and find out the potential cycles length
    """

    def __init__(self, data: TimeSeriesData):
        super().__init__(data=data)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)
        self.decomposed = False

    def _get_seasonality_length(self, d):
        out = []
        while d:
            k = d.pop(0)
            d = [i for i in d if i % k != 0]
            out.append(k)
        return out

    def detector(self, lags=None, diff=1, alpha=0.01):
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

    def plot(self):
        sm.graphics.tsa.plot_acf(self.ts_diff, lags=self.lags)
        plt.show()
        if self.decomposed:
            self.decompose.plot()
        plt.show()

    def remover(
        self, decom=TimeSeriesDecomposition, model="additive", decompose_any_way=False
    ):

        if decompose_any_way or self.seasonality_detected:
            self.decompose = decom(self.data, model)
            result = self.decompose.decomposer()
            self.decomposed = True
            return result
        else:
            logging.info("No seasonality detected, not running decomposition")


class FFTDetector(Detector):
    """
        Use Fast Fourier Transform to detect seasonality,
        and find out the potential cycle's length
    """

    def __init__(self, data: TimeSeriesData):
        super().__init__(data=data)
        if not self.data.is_univariate():
            msg = "The provided time series data is not univariate."
            logging.error(msg)
            raise ValueError(msg)

    def detector(self, sample_spacing: float = 1.0, mad_threshold: float = 6.0) -> Dict:
        """Detect seasonality with FFT
        Parameters
        ----------
        sample_spacing: Scaling FFT for a different time unit.
            I.e. for hourly time series, sample_spacing=24.0,
            FFT x axis will be 1/day
        mad_threshold: constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.
        Returns
        -------
        FFT Plot with peaks, selected peaks, and outlier boundary line
        """

        fft = self.get_fft(sample_spacing)
        _, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        seasonality_presence = len(peaks.index) > 0
        selected_seasonalities = []
        if seasonality_presence:
            selected_seasonalities = peaks["freq"].transform(lambda x: 1 / x).tolist()

        return {
            "seasonality_presence": seasonality_presence,
            "seasonalities": selected_seasonalities,
        }

    def plot(
        self,
        time_unit: str,
        sample_spacing: float = 1.0,
        title: str = "FFT",
        mad_threshold: float = 6.0,
    ) -> go.Figure:
        """Plots an FFT plot as a plotly figure
        Parameters
        ----------
        time_unit: string containing the unit of time (displayed on x axis).
                        E.g. 'Hour'
        sample_spacing: Scaling FFT for a different time unit.
            I.e. for hourly time series, sample_spacing=24.0,
            FFT x axis will be 1/day
        title: Title of the plot
        mad_threshold: constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.
        Returns
        -------
        FFT Plot with peaks, selected peaks, and outlier boundary line
        """
        fft = self.get_fft(sample_spacing)
        thres, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        return make_fourier_plot(
            fft, thres, orig_peaks, peaks, f"1/{time_unit}", title=title
        )

    def get_fft(self, sample_spacing: float = 1.0) -> pd.DataFrame:
        """Computes FFT
        Parameters
        ----------
        sample_spacing: Scaling FFT for a different time unit.
            I.e. for hourly time series, sample_spacing=24.0 FFT x axis will be 1/day
        Returns
        -------
        DataFrame with columns 'freq' and 'ampl'
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
        Parameters
        ----------
        fft: FFT computed by FFTDetector.get_fft
        sample_spacing: Scaling FFT for a different time unit.
            I.e. for hourly time series, sample_spacing=24.0 FFT x axis will be 1/day
        mad_threshold: constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.
        Returns
        -------
        outlier threshold, peaks, selected peaks
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
    ) -> go.Figure:
        """Plots an FFT plot as a plotly figure
        Parameters
        ----------
        time_unit: string containing the unit of time (displayed on x axis).
                        E.g. 'Hour'
        sample_spacing: Scaling FFT for a different time unit.
            I.e. for hourly time series, sample_spacing=24.0,
            FFT x axis will be 1/day
        title: Title of the plot
        mad_threshold: constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.
        Returns
        -------
        FFT Plot with peaks, selected peaks, and outlier boundary line
        """
        fft = self.get_fft(sample_spacing)
        thres, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)
        return make_fourier_plot(
            fft, thres, orig_peaks, peaks, f"1/{time_unit}", title=title
        )
