# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Utility functions for plotting

from typing import Any, List

import numpy as np
import pandas as pd

try:
    import plotly.graph_objs as go

    _no_plotly = False
    Figure = go.Figure
except ImportError:
    _no_plotly = True
    Figure = Any


def plot_scatter_with_confints(val: List[float], confint: np.ndarray, title) -> Figure:
    """Plots a scatter plot with confidence intervals used to plot ACF and PACF
    Parameters
    ----------
    val: list containing the values
    confint: two dimensional ndarray; confint[:, 0] is the lower confidence
             interval; confint[:, 1] is the upper
    title: Title of the plot
    Returns
    -------
    Plot with confidence intervals
    """
    if _no_plotly:
        raise RuntimeError("requires plotly to be installed")
    prediction_color = "#0072B2"
    error_color = "rgba(0, 114, 178, 0.2)"  # '#0072B2' with 0.2 opacity
    fig = go.Figure(
        {
            "data": [
                go.Scatter(
                    x=list(range(0, len(val))),
                    y=confint[:, 0],
                    name="ConfIntLow",
                    line={"color": error_color, "width": 0},
                    mode="lines",
                    showlegend=False,
                ),
                go.Scatter(
                    x=list(range(0, len(val))),
                    y=val,
                    mode="markers",
                    name=title,
                    fillcolor=error_color,
                    fill="tonexty",
                    line={"color": prediction_color},
                ),
                go.Scatter(
                    x=list(range(0, len(val))),
                    y=confint[:, 1],
                    mode="lines",
                    name="ConfIntHight",
                    line={"color": error_color, "width": 0},
                    fill="tonexty",
                    fillcolor=error_color,
                    showlegend=False,
                ),
            ],
            "layout": go.Layout(
                title=title, yaxis={"title": "Correlation"}, xaxis={"title": "Lag"}
            ),
        }
    )
    return fig


def make_fourier_plot(
    fft: pd.DataFrame,
    threshold: float,
    orig_peaks: List[float],
    peaks: List[float],
    xlabel: str = "",
    ylabel: str = "PSD(dB)",
    title: str = "DFT Plot",
) -> Figure:
    """Plots a scatter plot with fft highlighting the thresholds, peaks,
    and selected peaks
        Parameters
        ----------
        fft: fft information
        threshold: outlier threshold
        orig_peaks: all the peaks
        peaks: selected peaks
        xlabel: label for the X axis
        ylabel: label for the Y axis
        Returns
        -------
        FFT plot
    """
    if _no_plotly:
        raise RuntimeError("requires plotly to be installed")
    return go.Figure(
        {
            "data": [
                go.Scatter(x=fft["freq"], y=fft["ampl"], name="FFT"),
                go.Scatter(
                    x=fft["freq"],
                    y=[threshold] * len(fft.index),
                    name="Outlier Threshold",
                ),
                go.Scatter(
                    # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got
                    #  `str`.
                    x=orig_peaks["freq"],
                    # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got
                    #  `str`.
                    y=orig_peaks["ampl"],
                    mode="markers",
                    name="Original peaks",
                ),
                go.Scatter(
                    # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got
                    #  `str`.
                    x=peaks["freq"],
                    # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got
                    #  `str`.
                    y=peaks["ampl"],
                    mode="markers",
                    name="Selected peaks",
                ),
            ],
            "layout": go.Layout(
                title=title, yaxis={"title": ylabel}, xaxis={"title": xlabel}
            ),
        }
    )


def plot_fitted_harmonics(
    times: pd.Series, original_values: pd.Series, fitted_values: np.ndarray
) -> Figure:
    """Plots a scatter plot of the fitted harmonics
    Parameters
    ----------
    times: date list of the time series
    original_values: values of the original function
    fitted_values: fitted values
    Returns
    -------
    Plot with fitted harmonics
    """
    if _no_plotly:
        raise RuntimeError("requires plotly to be installed")
    return go.Figure(
        {
            "data": [
                go.Scatter(x=times, y=original_values, name="Original signal"),
                go.Scatter(x=times, y=fitted_values, name="Fitted harmonics"),
            ],
            "layout": go.Layout(
                title="Time series v.s Fitted Harmonics",
                yaxis={"title": "Value"},
                xaxis={"title": "Time"},
            ),
        }
    )
