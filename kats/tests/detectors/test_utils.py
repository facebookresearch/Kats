# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from kats.detectors.utils import (
    get_interval_len,
    get_list_intersection,
    merge_interval_list,
)


class TestIntervalIntesection(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.intervalist1 = [(0.0, 10.0), (20.0, 100.0), (60.0, 70.0)]
        self.intervalist2 = [(5.0, 15.0), (30.0, 35.0), (105.0, 110.0)]
        self.intervalist3 = [(0.0, 10.0), (20.0, 65.0), (60.0, 70.0), (65, 90)]
        self.intervalist4 = [(60.0, 70.0), (0.0, 10.0), (68, 90), (20.0, 65.0)]

    def test_list_merge(self) -> None:

        self.assertEqual(
            merge_interval_list(self.intervalist1), [(0.0, 10.0), (20.0, 100.0)]
        )

    def test_list_intersect(self) -> None:

        self.assertEqual(
            get_list_intersection(self.intervalist1, self.intervalist2),
            [(5.0, 10.0), (30.0, 35.0)],
        )

    def test_list_len(self) -> None:

        self.assertEqual(get_interval_len(self.intervalist1), float(10 + 80 + 10))

    def test_list_merge_hard(self) -> None:

        self.assertEqual(
            merge_interval_list(self.intervalist3), [(0.0, 10.0), (20.0, 90.0)]
        )

    def test_list_merge_no_sort(self) -> None:

        self.assertEqual(
            merge_interval_list(self.intervalist4), [(0.0, 10.0), (20.0, 90.0)]
        )
