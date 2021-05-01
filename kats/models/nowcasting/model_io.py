#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

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

def serialize_for_zippy(input):
    serializer = SimplePickleSerializer()
    return serializer.serialize(input)

def deserialize_from_zippy(input, use_case_id=None):
    serializer = SimplePickleSerializer()
    return serializer.deserialize(input)
