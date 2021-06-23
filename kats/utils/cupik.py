# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData


class Pipeline:
    """
    CuPiK (Customized Pipeline for Kats) is created with a similar mindset
    of sklearn pipeline. Users can call multiple methods within Kats library
    and run them in sequential to perform a series of useful timeseries processes
    at once.

    Due to the current distribution of Kats library, we provide the function to
    apply detectors, transformers and time series modeling sequentially using CuPiK.
    We also offer api to sklearn, once feature extraction using TsFeatures is performed,
    users can feed the results directly to an sklearn machine learning model.
    """

    remove: bool = False
    useFeatures: bool = False
    extra_fitting_params: Optional[Dict[str, Any]] = None
    y: Optional[Union[np.ndarray, pd.Series]] = None

    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        inputs:
        steps: a list of the initialized Kats methods/sklearn machine learning model, with
               the format of [('user_defined_method_name', initialized method class)]. User
               can use any name for 'user_defined_method_name' for identification purpose

        initialized attributes:
        steps: same as the "steps" in the inputs

        metadata: an dictionary to store outputs that are not passing to the next step, like
                  results from a detector. These metadata stored here with the format of
                  "user_defined_method_name": output

        univariate: is the data fed in a list of multiple univariate time series or just a single
                    univariate time series

        functions: a look up dictionary linking each method in the steps to what processing
                   function inside CuPiK should we apply
        """
        self.steps = steps
        self.metadata = {}
        self.univariate = False
        self.functions = {
            "detector": self.__detect__,
            "transformer": self.__transform__,
            "model": self.__model__,
        }

    def __detect__(
        self, steps: List[Any], data: List[TimeSeriesData], extra_params: Dict[str, Any]
    ) -> Tuple[List[TimeSeriesData], Any]:
        """
        Internal function for processing the detector steps

        inputs:
        steps: a list of the duplicated initialized detector. We will be using each duplicated
               detector to process one time series data within the data list

        data: a list containing time series data to be processed

        extra_params: a dictionary holding extra customized parameters to be fed in the detector

        outputs:
        data: a list of post-processed data for next steps

        metadata: outputs from the detectors, like changepoints, outliers, etc.
        """
        metadata = []
        for i, (s, d) in enumerate(zip(steps, data)):
            s.data = d
            if not s.data.is_univariate():
                msg = "Only support univariate time series, but get {type}.".format(
                    type=type(s.data.value)
                )
                logging.error(msg)
                raise ValueError(msg)
            s.data.time = pd.to_datetime(s.data.time)
            if s.__subtype__ == "outlier":
                extra_params["pipe"] = True
            metadata.append(s.detector(**extra_params))
            if (
                self.remove and s.__subtype__ == "outlier"
            ):  # outlier removal when the step is outlier detector,
                # and user required us to remove outlier
                data[i] = s.remover(interpolate=True)
        return data, metadata

    def __transform__(
        self, steps: List[Any], data: List[TimeSeriesData], extra_params: Dict[str, Any]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Internal function for processing the transformation/transformer steps. We currently only have
        tsfeatures as a transformation/transformer step in Kats.

        inputs:
        steps: a list of the duplicated initialized transformer. We will be using each duplicated
                transformer to process one time series data within the data list

        data: a list containing time series data to be processed

        extra_params: a dictionary holding extra customized parameters to be fed in the transformer

        outputs:
        data: a list of post-processed data for next steps. We user requires to use the outputs of
        the transformer, this would become the output from the transformer turning time series data
        to tabular data; otherwise, do nothing at the current stage of Kats.

        metadata: outputs from the transformer
        """
        metadata = []
        for s, d in zip(steps, data):
            metadata.append(s.transform(d))
        if self.useFeatures:
            return metadata, metadata
        else:
            return data, metadata

    def __model__(
        self, steps: List[Any], data: List[TimeSeriesData], extra_params: Dict[str, Any]
    ):
        """
        Internal function for processing the modeling step

        inputs:
        steps: a list of the duplicated initialized time series model in Kats. We will be using
               each duplicated model to process one time series data within the data list

        data: a list containing time series data to be processed

        extra_params: a dictionary holding extra customized parameters to be fed in the model

        outputs:
        data: a list of fitted time series model

        None as the placeholder of metadata
        """
        for i, (s, d) in enumerate(zip(steps, data)):
            s.data = d
            if not isinstance(d.value, pd.Series):
                msg = "Only support univariate time series, but get {type}.".format(
                    type=type(d.value)
                )
                logging.error(msg)
                raise ValueError(msg)
            s.fit(**extra_params)
            data[i] = s
        return data, None

    def _fit_sklearn_(
        self,
        step: Any,
        data: List[Dict[str, Any]],
        y: Any,
    ) -> Any:
        """
        Internal function for fitting sklearn model on a tabular data with features
        extracted.

        inputs:
        step: an sklearn model class

        data: a list with each item corresponds to an output from the feature extraction
              methods in Kats

        y: label data for fitting sklearn model

        outputs:
        step: a fitted sklearn model
        """
        assert (type(data) == list) and (
            type(data[0]) == dict
        ), "Require data preprocessed by TsFeatures, please set useFeatures = True"
        assert y is not None, "Missing dependent variable"
        df = pd.DataFrame(data).dropna(axis=1)
        X_train, y_train = df.values, self.y
        step.fit(X_train, y_train)
        return step

    def __fit__(
        self,
        n: str,
        s: Any,
        data: Any,
    ) -> List[
        Any
    ]:  # using list output for adaption of current multi-time series scenarios
        """
        Internal function for performing the detailed fitting functions

        inputs:
        n: short for name, "user_defined_method_name"

        s: short for step, a Kats method or sklearn model

        data: either a list of univariate time series data or a list of dictionaries
               including the output acquired using feature extraction methods in Kats

        outputs:
        data: either a list of post processed univariate time series data or a list
              of dictionaries including the output acquired using feature extraction
              methods in Kats
        """
        if (
            str(s.__class__).split()[1][1:8] == "sklearn"
        ):  # if current step is a scikit-learn model
            return self._fit_sklearn_(s, data, self.y)

        _steps_ = [s for _ in range(len(data))]
        Type = s.__type__

        extra_params = (self.extra_fitting_params or {}).get(n, {})
        data, metadata = self.functions[Type](_steps_, data, extra_params)
        if metadata is not None:
            self.metadata[n] = metadata  # saving the metadata of the current step into
            # the dictionary of {"user_defined_method_name": corresponding_metadata}
        return data

    def fit(self, data: Any, params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        This function is the external function for user to fit the pipeline

        inputs:
        data: a single univariate time series data or a list of multiple univariate
              time series data

        params: a dictionary with the extra parameters for each step. The dictionary
                holds the format of {"user_defined_method_name": {"parameter": value}}

        extra key word arguments:
        remove: a boolean for telling the pipeline to remove outlier or not

        useFeatures: a boolean for telling the pipeline whether to use TsFeatures to process
                     the data for sklearn models, or merely getting the features as metadata
                     for other usage

        y: label data for fitting sklearn model, an array or a list

        outputs:
        data: output a single result for univariate data, or a list of results for multiple
              univariate time series data fed originally in the format of a list. Determined by
              the last step, the output could be processed time series data, or fitted kats/sklearn
              model, etc.
        """
        # Initialize a place holder for params
        if params is None:
            params = {}

        # Judging if extra functions needed
        ####
        self.remove = kwargs.get("remove", False)  # remove outliers or not
        self.useFeatures = kwargs.get(
            "useFeatures", False
        )  # do you want to use tsfeatures as transformation or analyzer
        self.y = kwargs.get("y", None)
        ####
        # Extra parameters for specific method of each step
        self.extra_fitting_params = params

        if type(data) != list:  # Since we support multiple timeseries in a list,
            # when data is univariate, we put them in a list
            self.univariate = True
            data = [data]

        for (
            n,
            s,
        ) in self.steps:  # Iterate through each step and perform the internal fitting
            # function
            data = self.__fit__(n, s, data)

        if (
            self.univariate
        ):  # When input data is one univariate time series, we directly
            # present the output (not in a list)
            return data[0]
        else:
            return data
