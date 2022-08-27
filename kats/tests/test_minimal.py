# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest

import kats
import pytest


class TestMinimal(unittest.TestCase):
    def test_install(self) -> None:
        self.assertIsNotNone(kats.models)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument `"torch" in
    #  sys.modules` to decorator factory `pytest.mark.skipif`.
    @pytest.mark.skipif("torch" in sys.modules, reason="not minimal")
    def test_minimal_install(self) -> None:
        try:
            from kats.detectors import prophet_detector
            from kats.models import lstm, neuralprophet

            self.assertFalse(
                (
                    lstm is not None
                    and neuralprophet is not None
                    and prophet_detector is not None
                )
            )
        except ImportError:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
