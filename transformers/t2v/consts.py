#!/usr/bin/env python3

from typing import Any, Optional, NamedTuple

from infrastrategy.kats.transformers.t2v import utils


def T2VParam(
    mode: str = "regression",
    normalizer: Optional[Any] = utils.MinMax
):

    """
    A function for storing all parameters of a t2v model.

    :Parameters:
    mode: str
        Should we treat the training as a classification problem
        or regression problem. Default treats the training as a
        unlabeled regression problem.
    normalizer: Any
        Normalization function imported from utils.
        Currently, only MinMax and Standardize are supported.
    """

    T2VParam = NamedTuple(
        'T2VParam',
        [
            ('mode', str),
            ('normalizer', Optional[Any]),
        ]
    )

    T2VParam.mode = mode
    T2VParam.normalizer = normalizer

    return T2VParam
