import logging
import time
from typing import Dict, List, Tuple, NamedTuple

import numpy as np
import torch
from kats.transformers.t2v.consts import LSTM


class T2VNN:

    """
    This is the main function for training models to translate
    time series data into embedding vectors. This major class
    includes three functions, train, validation and translation.

    :Parameters:
    data: NamedTuple
        Either a T2VProcessed NamedTuple, which is processed data
        output from t2vpreprocessing, or a T2VBatched NamedTuple,
        a batched data output from t2vbatch.
    param: NamedTuple
        T2VParam object containing necessary user defined
        parameters for data preprocessing.
    """

    def __init__(
        self,
        data: NamedTuple,
        param: NamedTuple,
    ):
        self.mode = param.mode
        self.epochs = param.epochs
        self.vector_length = param.vector_length
        self.output_size = data.output_size
        self.window = data.window
        self.hidden = param.hidden
        self.dropout = param.dropout
        self.learning_rate = param.learning_rate
        self.loss_function = param.loss_function
        self.optimizer = param.optimizer
        self.batched = data.batched
        self.validator = param.validator

        if self.batched:
            self.data = data.batched_tensors
        elif not self.batched:
            self.data = zip(data.seq, data.label)

    def _translate(
        self,
        sequences,
    ) -> Tuple[List, List]:
        # internal function for performing translations
        self.model.eval()
        start_time = time.time()
        embeddings = []
        labels = []
        for seq in sequences:
            with torch.no_grad():
                if self.mode == "regression":
                    label, embedding = self.model(seq.double())
                    labels.extend(list(label.detach().numpy()))
                elif self.mode == "classification":
                    label, embedding = self.model(seq.float())
                    labels.append(np.argmax(label.detach().numpy()))
                embeddings.append(embedding.view(-1).numpy())
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding translation time: {elapsed_time}")
        return embeddings, labels

    def _train_regression(
        self,
        model,
        loss_function,
        optimizer,
    ) -> None:
        # sub training block for regression mode
        model.double()
        model.train()
        losses = []
        start_time = time.time()
        for i in range(self.epochs):
            training_loss = []
            for b in self.data:
                optimizer.zero_grad()
                if self.batched:  # if data type is t2vbatched
                    seq, label = (
                        torch.tensor([bb[0].numpy() for bb in b]),
                        torch.tensor([bb[1].numpy() for bb in b]),
                    )
                    y_pred, _ = model(seq.double())

                elif not self.batched:  # if data type is t2vprocessed
                    seq, label = torch.from_numpy(b[0]), torch.tensor(b[1])
                    y_pred, vector = model(seq.view([1, seq.shape[0], 1]).double())

                loss = loss_function(
                    y_pred.view(-1),
                    label.view(-1),
                )
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())

            losses.append(np.mean(training_loss))
            logging.info(
                f"""
                epoch: {i:3}, avg train loss: {np.mean(training_loss):10.8f}
            """
            )

        elapsed_time = time.time() - start_time
        logging.info(f"Embedding training time: {elapsed_time}")
        logging.info("Training has concluded")
        self.model = model

    def _train_classification(
        self,
        model,
        loss_function,
        optimizer,
    ) -> None:
        # sub training block for classification mode
        model.train()
        losses = []
        start_time = time.time()
        for i in range(self.epochs):
            training_loss = []
            for b in self.data:
                optimizer.zero_grad()
                if self.batched:  # if data type is t2vbatched
                    seq, label = (
                        torch.tensor([bb[0].numpy() for bb in b]),
                        torch.tensor(np.array([bb[1].numpy() for bb in b])),
                    )
                    y_pred, _ = model(seq.float())
                elif not self.batched:  # if data type is t2vprocessed
                    seq, label = torch.from_numpy(b[0]), torch.tensor(b[1])
                    y_pred, _ = model(seq.view([1, seq.shape[0], 1]).float())

                output_length = len(b) if self.batched else 1
                loss = loss_function(
                    y_pred.view(output_length, self.output_size),
                    label.view(-1),
                )
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())

            losses.append(np.mean(training_loss))
            logging.info(
                f"""
                epoch: {i:3}, avg train loss: {np.mean(training_loss):10.8f}
            """
            )

        elapsed_time = time.time() - start_time
        logging.info(f"Embedding training time: {elapsed_time}")
        logging.info("Training has concluded")
        self.model = model

    def train(self, translate=False) -> Dict[str, List]:
        # function for training deep NN for embedding translation
        model = LSTM(
            vector_length=self.vector_length,
            hidden_layer=self.hidden,
            output_size=self.output_size,
            window=self.window,
            dropout=self.dropout,
        )

        loss_function = self.loss_function()
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)

        if self.mode == "regression":  # training for regression mode
            self._train_regression(
                model,
                loss_function,
                optimizer,
            )
        elif self.mode == "classification":  # training for classification mode
            self._train_classification(
                model,
                loss_function,
                optimizer,
            )

        # performing translation on training data when specified
        if translate:
            seq_2_translate, labels = [], []
            if self.batched:
                for b in self.data:
                    for seq, label in b:
                        seq_2_translate.append(seq.view([1, seq.shape[0], 1]))
                        labels.append(label.item())
            elif not self.batched:
                for seq, label in self.data:
                    seq_2_translate.append(torch.from_numpy(seq).view([1, seq.shape[0], 1]))
                    labels.append(label)

            train_translated, _ = self._translate(seq_2_translate)

            return {"translated_sequences": train_translated, "labels": labels}

    def val(
        self,
        val_data: NamedTuple,
    ) -> Dict:  # for validation only
        # validating the trained T2V module using the original label of the sequences
        # checking on how accurate can T2V module predict the original labels
        seq_2_translate, labels = [], []
        if val_data.batched:  # when data is of type t2vbatched
            val_data = val_data.batched_tensors
            for b in val_data:
                for seq, label in b:
                    seq_2_translate.append(seq.view([1, seq.shape[0], 1]))
                    labels.append(label)
        elif not val_data.batched:  # when data is of type t2vprocessed
            val_data = zip(val_data.seq, val_data.label)
            for seq, label in val_data:
                seq_2_translate.append(torch.from_numpy(seq).view([1, seq.shape[0], 1]))
                labels.append(label)

        _, val_labels = self._translate(seq_2_translate)
        acc = self.validator(labels, val_labels)
        logging.info(f"validated metric: {acc}")

        if self.mode == "regression":  # currently only support mae for regression
            return {"mae": acc}
        elif (
            self.mode == "classification"
        ):  # currently only support accuracy for classification
            return {"accuracy": acc}

    def translate(
        self,
        test_data: NamedTuple,
    ) -> List:  # translate sequences only
        # turning time series sequences into embedding using trained T2V module
        seq_2_translate = []
        if test_data.batched:  # when data type is of t2vbatched
            test_data = test_data.batched_tensors
            for b in test_data:
                for seq, _ in b:
                    seq_2_translate.append(seq.view([1, seq.shape[0], 1]))
        elif not test_data.batched:  # when data type is of t2vprocessed
            for seq in test_data.seq:
                seq_2_translate.append(torch.from_numpy(seq).view([1, seq.shape[0], 1]))

        embeddings, _ = self._translate(seq_2_translate)
        return embeddings
