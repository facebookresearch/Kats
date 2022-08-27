# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
from kats.compat import compat
from sklearn.metrics import mean_squared_error as mse, mean_squared_log_error as msle


version: compat.Version = compat.Version("sklearn")


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
    squared: bool = True,
) -> float:
    # sklearn >= 1.0 expects args beyond the first two to be passed as named kwargs
    return mse(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


def mean_squared_log_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
    squared: bool = True,
) -> float:
    if version <= "0.24":
        result = msle(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
        if not squared:
            result = np.sqrt(result)
    else:
        # sklearn >= 1.0 expects args beyond the first two to be passed as named kwargs
        # pyre-ignore
        result = msle(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=squared,
        )
    return result
