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
    batch_size: int
        User defined number of sequence-label tensor combos in each batch.
    """

    def __init__(
        self,
        preprocessed: NamedTuple,
        batch_size: int,
    ):
        self.seq = preprocessed.seq
        self.label = preprocessed.label
        self.output_size = preprocessed.output_size
        self.window = preprocessed.window
        logging.info("all attributes inherited from T2VProcessed.")

        self.batch_size = batch_size

    def _reshaping(
        self,
        sequence: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        This internal function turns each array into [window_size, 1] vector
        for Pytorch. Currently only support univariate time series data.
        """
        sequence = [seq.reshape([self.window, 1]) for seq in sequence]

        logging.info("vector reshaping completed.")
        return sequence

    def transform(
        self,
    ) -> NamedTuple:
        tensors = []

        sequence_reshaped = self._reshaping(self.seq)

        batch_counter = 0
        current_batch = []
        for vec, label in zip(sequence_reshaped, self.label):

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

        T2VBatched.seq, T2VBatched.label, T2VBatched.window, T2VBatched.output_size = (
            self.seq,
            self.label,
            self.window,
            self.output_size,
        )
        T2VBatched.batched_tensors = tensors  # batched tensors
        T2VBatched.batch_size = self.batch_size
        T2VBatched.batched = True  # setting batched to True for downstream functions

        return T2VBatched
