import logging
from typing import List, Optional, NamedTuple

import numpy as np
from infrastrategy.kats.consts import TimeSeriesData


class T2VPreprocessing:

    """
    A class for preprocessing time series data. Steps
    include - 1. TO-DO: Segmentation 2. TO-DO: Padding
    3. Normalization 4. Label Separation

    :Parameters:
    param: NamedTuple
        T2VParam object containing necessary user defined
        parameters for data preprocessing.
    data: List[TimeSeriesData]
        A list of processed timeseries data.
    label: Optional[List]
        A list of labels provided by the user. If provided,
        treat as 'supervised' modeling, else, treat as 'unsupervised'
        modeling.
    """

    def __init__(
        self,
        param: NamedTuple,  # TO-DO: Add segmented transformation (21Q2)
        data: List[TimeSeriesData],
        label: Optional[List] = None,
    ):
        self.data = data
        self.param = param

        self.label = label
        self.mode = param.mode
        self.output_size = (
            (np.max(label) + 1)
            if ((not isinstance(label, type(None))) & (self.mode == "classification"))
            else 1
        )  # if label is provided and it's we train it in a classification
        # fashion, then output_size is determined by the max of the labels,
        # else, we train it in a regression fashion

    # TO-DO: function for padding (21Q1)

    def transform(
        self,
    ) -> NamedTuple:

        # sanity check
        if self.mode == "classification":
            if isinstance(self.label, type(None)):
                msg = (
                    "Labels should be provided for training in classification fashion."
                )
                logging.error(msg)
                raise ValueError(msg)
            for label in self.label:
                if (type(label) != int) & (type(label) != np.int64):
                    msg = "Float cannot be used as label for classification training."
                    logging.error(msg)
                    raise ValueError(msg)

        # TO-DO: apply padding first (21Q1)
        ts_sample = self.data[0]
        end = len(ts_sample)
        if isinstance(self.label, type(None)):
            end -= 1  # when data is unlabeled, using last
            # element as label for training embedding

        seq = [
            self.param.normalizer(ts.value.values[:end]) for ts in self.data
        ]  # normalize each time series

        label = (
            self.label
            if not isinstance(self.label, type(None))
            else [ts.value.values[end] for ts in self.data]
        )  # retrieve the label of each time series

        # sanity check: do we have same count for labels and time series data
        if len(label) != len(seq):
            msg = "Number of labels and time series data mismatch."
            logging.error(msg)
            raise ValueError(msg)

        T2VPreprocessed = NamedTuple(
            "T2VPreprocessed",
            [
                ("seq", List[np.ndarray]),
                ("label", List),
                ("output_size", int),
                ("window", int),
                ("batched", bool),
            ],
        )  # A named tuple for storing relevant content of processed timeseries
        # data sequences.

        T2VPreprocessed.seq = seq
        T2VPreprocessed.label = label
        T2VPreprocessed.output_size = self.output_size
        T2VPreprocessed.window = end # currently only supporting feeding
        # the entire time series data, segmentaion will come later.
        T2VPreprocessed.batched = False # for downstream functions

        return T2VPreprocessed
