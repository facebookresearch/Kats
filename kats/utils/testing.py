# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for testing and evaluation.
"""

try:
    from plotly.graph_objs import Figure
except ImportError:
    Figure = object


class PlotlyAdapter:
    def __init__(self, fig: Figure) -> None:
        self.fig = fig

    def save_fig(self, path: str) -> None:
        # pyre-ignore[16]: `plotly.graph_objs.graph_objs.Figure` has no attribute `write_image`.
        self.fig.write_image(path)


__all__ = [
    "PlotlyAdapter",
]
