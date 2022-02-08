# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock as mock

from kats.compat import compat


class TestCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.v = compat.Version("0.12.2")

    def test_Version_from_Version(self) -> None:
        result = compat.Version(self.v)
        self.assertEqual("0.12.2", result)

    def test_Version_from_packaging(self) -> None:
        result = compat.Version(self.v.version)
        self.assertEqual("0.12.2", result)

    @mock.patch("kats.compat.compat.metadata")
    def test_Version_module(self, metadata: object) -> None:
        metadata.version.return_value = "3.11.2"
        result = compat.Version("kats")
        self.assertEqual((3, 11, 2), result.version._key[1])

    @mock.patch("kats.compat.compat.metadata")
    def test_Version_module_truncated(self, metadata: object) -> None:
        metadata.version.return_value = "2.3.0"
        result = compat.Version("kats")
        self.assertEqual((2, 3), result.version._key[1])

    def test_Version_compare_str(self) -> None:
        self.assertTrue(self.v < "0.12.3")
        self.assertTrue(self.v <= "0.12.3")
        self.assertFalse(self.v > "0.12.2")
        self.assertTrue(self.v >= "0.12.2")
        self.assertTrue(self.v == "0.12.2")
        self.assertFalse(self.v == "0.12")
        self.assertTrue(self.v != "12.2")

    def test_Version_compare_Version(self) -> None:
        v12_3 = compat.Version("0.12.3")
        self.assertTrue(self.v < v12_3)
        self.assertTrue(self.v <= v12_3)
        self.assertFalse(self.v > self.v)
        self.assertTrue(self.v >= self.v)
        self.assertTrue(self.v == self.v)
        self.assertFalse(self.v == compat.Version("0.12"))
        self.assertTrue(self.v != compat.Version("12.2"))

    def test_incompatible_Version(self) -> None:
        def method(_x: compat.V, _y: compat.V) -> bool:
            raise TypeError()

        self.assertEqual(NotImplemented, self.v._compare(self.v, method))


if __name__ == "__main__":
    unittest.main()
