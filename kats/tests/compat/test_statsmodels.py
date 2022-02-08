# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from typing import Dict
import unittest

from kats.compat import compat, statsmodels


class TestStatsmodels(unittest.TestCase):
    def test_version(self) -> None:
        self.assertTrue(statsmodels.version == compat.Version("statsmodels"))


if __name__ == "__main__":
    unittest.main()
