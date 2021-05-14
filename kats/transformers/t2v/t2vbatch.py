#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import NamedTuple, List, Tuple

import numpy as np
import torch


class T2VBatch:
    """
    A class for batching sequence-label combinations into tensor batches
    for feeding neural network.

    :Parameters:
    preprocessed: NamedTuple
        The output from t2vpreprocessing, a NamedTuple with preprocessed
        sequences, labels, and output_size.
    param: NamedTuple
        T2V param object containing information of the batch size.
    """

    def __init__(
        self,
        preprocessed: NamedTuple,
        param: NamedTuple,
    ):
        # pyre-fixme[16]: `NamedTuple` has no attribute `seq`.
        self.seq = preprocessed.seq
        # pyre-fixme[16]: `NamedTuple` has no attribute `label`.
        self.label = preprocessed.label
        # pyre-fixme[16]: `NamedTuple` has no attribute `output_size`.
        self.output_size = preprocessed.output_size
        # pyre-fixme[16]: `NamedTuple` has no attribute `window`.
        self.window = preprocessed.window
        logging.info("all attributes inherited from T2VProcessed.")

        # pyre-fixme[16]: `NamedTuple` has no attribute `batch_size`.
        self.batch_size = param.batch_size

    def transform(
        self,
    ) -> NamedTuple:
        tensors = []

        batch_counter = 0
        current_batch = []
        for vec, label in zip(self.seq, self.label):

            if batch_counter == self.batch_size:
                tensors.append(current_batch)
                current_batch = []
                batch_counter = 0

            current_batch.append(
                (
                    torch.from_numpy(vec),  # turning sequence and label into tensors
                    torch.tensor(label),
                )
            )
            batch_counter += 1

        if current_batch != []:
            tensors.append(current_batch)
        logging.info("tensor batching completed.")

        # output a named tuple, with all previous attributes from T2VProcessed
        T2VBatched = NamedTuple(
            "T2VBatched",
            [
                ("seq", List[np.ndarray]),
                ("label", List),
                ("batched_tensors", List[List[Tuple[torch.Tensor, torch.Tensor]]]),
                ("output_size", int),
                ("window", int),
                ("batch_size", int),
                ("batched", bool),
            ],
        )

        # pyre-fixme[41]: Cannot reassign final attribute `seq`.
        # pyre-fixme[41]: Cannot reassign final attribute `label`.
        # pyre-fixme[41]: Cannot reassign final attribute `window`.
        # pyre-fixme[41]: Cannot reassign final attribute `output_size`.
        T2VBatched.seq, T2VBatched.label, T2VBatched.window, T2VBatched.output_size = (
            self.seq,
            self.label,
            self.window,
            self.output_size,
        )
        # pyre-fixme[41]: Cannot reassign final attribute `batched_tensors`.
        T2VBatched.batched_tensors = tensors  # batched tensors
        # pyre-fixme[41]: Cannot reassign final attribute `batch_size`.
        T2VBatched.batch_size = self.batch_size
        # pyre-fixme[41]: Cannot reassign final attribute `batched`.
        T2VBatched.batched = True  # setting batched to True for downstream functions

        # pyre-fixme[7]: Expected `NamedTuple` but got
        #  `Type[T2VBatch.transform.T2VBatched]`.
        return T2VBatched
