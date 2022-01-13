# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest.mock as mock
from datetime import datetime
from unittest import IsolatedAsyncioTestCase

from kats.transformers.t2v.utils_intern import (
    load_prediction_model,
    save_prediction_model,
)


class test_t2vnn(IsolatedAsyncioTestCase):
    async def test_upload_download(self) -> None:
        test_binary_string = b"\xC2\xA9\x20\xF0\x9D\x8C\x86\x20\xE2\x98\x83"

        # upload the model to the database
        with mock.patch(
            "kats.transformers.t2v.utils_intern.pickle.dumps",
            return_value=test_binary_string,
        ) as mocked_dump:
            t2vnn = mock.MagicMock()
            await save_prediction_model("unit_testing", "thomashyde23", t2vnn)
            mocked_dump.assert_called_with(t2vnn)

        # read the model from the database
        with mock.patch(
            "kats.transformers.t2v.utils_intern.pickle.loads",
        ) as mock_loads:
            ds = str(datetime.now())[:10]
            await load_prediction_model("unit_testing", "thomashyde23", ds)
            mock_loads.assert_called_with(test_binary_string)
