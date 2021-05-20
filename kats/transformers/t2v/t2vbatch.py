#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import NamedTuple, List, Tuple

import numpy as np
import torch
from kats.transformers.t2v.t2vpreprocessing import T2VPreprocessed
from kats.transformers.t2v.consts import T2VParam

@dataclass
class T2VBatched:
    seq: List[np.ndarray]
    label: List
    batched_tensors: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    output_size: int
    window: int
    batch_size: int
    batched: bool


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
        preprocessed: T2VPreprocessed,
        param: T2VParam,
    ):
        self.seq = preprocessed.seq
        self.label = preprocessed.label
        self.output_size = preprocessed.output_size
        self.window = preprocessed.window
        logging.info("all attributes inherited from T2VProcessed.")

        self.batch_size = param.batch_size

    def transform(
        self,
    ) -> T2VBatched:
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

        t2v_batched = T2VBatched(
            seq = self.seq,
            label = self.label,
            window = self.window,
            output_size = self.output_size,
            batched_tensors = tensors,  # batched tensors
            batch_size = self.batch_size,
            batched = True,  # setting batched to True for downstream functions
        )

        return t2v_batched
