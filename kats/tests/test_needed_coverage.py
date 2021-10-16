from typing import List
from unittest import TestCase

from kats.internal.code_coverage.needed_code_coverage import parse_rows
from parameterized import parameterized


class testParseRows(TestCase):
    ROOT = "kats/kats/internal/code_coverage/"

    def load_file(self, filename: str) -> List:
        filepath = self.ROOT + filename
        with open(filepath) as f:
            return f.read().split("\n")

    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            [
                "single_line",
                "targets_single_line_test.txt",
                [(99, "kats/models/bayesian_var.py")],
            ],
            [
                "one_row",
                "targets_one_row_test.txt",
                [(99, "kats/models/bayesian_var.py")],
            ],
            [
                "multiple_rows",
                "targets_multiple_rows_test.txt",
                [
                    (100, "kats/models/holtwinters.py"),
                    (67, "kats/models/holtsummers.py"),
                ],
            ],
            [
                "commented_out_test",
                "targets_commented_out_test.txt",
                [],
            ],
            [
                "commented_out_coverage",
                "targets_commented_out_coverage_test.txt",
                [(95, "kats/models/arima.py")],
            ],
            [
                "square_bracket_format",
                "targets_square_bracket_format_test.txt",
                [
                    (65, "kats/models/harmonic_regression.py"),
                    (100, "kats/models/holtspring.py"),
                    (67, "kats/models/holtfall.py"),
                ],
            ],
            [
                "ignore_non_tests",
                "targets_ignore_non_tests.txt",
                [
                    (95, "kats/models/arima.py"),
                    (99, "kats/models/bayesian_var.py"),
                    (65, "kats/models/harmonic_regression.py"),
                    (100, "kats/models/holtwinters.py"),
                    (67, "kats/models/holtsummers.py"),
                    (100, "kats/models/holtspring.py"),
                    (67, "kats/models/holtfall.py"),
                ],
            ],
        ]
    )
    def test_parameterized(self, name: str, test_file: str, expected: List) -> None:
        contents = self.load_file(test_file)
        rows = parse_rows(contents)
        self.assertEqual(rows, expected)
