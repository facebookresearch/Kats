import os
from typing import Any
import matplotlib as mpl


def pytest_sessionstart(session: Any) -> None:
    # Set the matplotlib backend to Agg for UI-less testing
    # unless the developer manually overrides by setting
    # MPLBACKEND to something else (such as "TkAgg").
    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "agg"
        # The above should be enough, but I found I needed to use:
        mpl.use("agg")
