# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pickle
from typing import Any


class SimplePickleSerializer:
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize(self, obj: Any) -> bytes:
        """Performs model saving.

        Args:
            obj is an object to be saved. Usually it is an sklearn model.

        Returns:
            A bytes object which is the compressed model.
        """

        if obj is None:
            return b""
        return pickle.dumps(obj)

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def deserialize(self, serialized_data: bytes) -> Any:
        """Performs model decoding.

        Args:
            serialized_data is a bytes object to be decoded.

        Returns:
            A decompressed model. Usually a sklearn model.
        """

        if serialized_data is None:
            return None
        decoded = serialized_data  # .decode("utf-8")
        if not decoded:
            return None
        return pickle.loads(decoded)


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def serialize_for_zippy(input: Any) -> bytes:
    """Performs model compression.

    Args:
        Input is an sklearn model.

    Returns:
        A compressed version of the model.
    """

    serializer = SimplePickleSerializer()
    return serializer.serialize(input)


# pyre-fixme[2]: Parameter must be annotated.
def deserialize_from_zippy(input: bytes, use_case_id=None) -> None:
    """Performs model serialization for Zippydb.

    Args:
        Input is an encoded sklearn model.

    Returns:
        A compressed version of the model.
    """

    serializer = SimplePickleSerializer()
    return serializer.deserialize(input)
