#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import pickle
from ManagedCompression import ManagedCompressionFactory


# compression version of saving/loading model
# proof of concept in
# https://www.internalfb.com/intern/anp/view/?id=374769


# original from from data.ai.forecasting.forecastservice.common.common_util.SimpleJsonSerializer
# need to adapt for model saving using pickle
class SimplePickleSerializer:
    def _jdefault(self, o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, bool):
            return str(o).lower()
        if isinstance(o, int):
            return str(o)
        if isinstance(o, float):
            return str(o)
        return o.__dict__

    def serialize(self, obj) -> bytes:
        if obj is None:
            return b""
        return pickle.dumps(obj)

    def deserialize(self, serialized_data):
        if serialized_data is None:
            return None
        decoded = serialized_data#.decode("utf-8")
        if not decoded:
            return None
        return pickle.loads(decoded)

def get_compression_factory():
    return ManagedCompressionFactory(b"data_ai_forecasting", b"forecast").getCodec(
        b"default"
    )  # ZSTD


def serialize_for_zippy(input):
    compress_factory = get_compression_factory()
    serializer = SimplePickleSerializer()
    return base64.b64encode(compress_factory.compress(serializer.serialize(input)))


def deserialize_from_zippy(input, use_case_id=None):
    compress_factory = get_compression_factory()
    serializer = SimplePickleSerializer()
    return serializer.deserialize(
        compress_factory.uncompress(base64.decodebytes(input))
    )
