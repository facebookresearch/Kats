from unittest import TestCase

from kats.internal.code_coverage.needed_code_coverage import parse_rows


class testParseRows(TestCase):
    ROOT = "kats/kats/internal/code_coverage/"

    def test_single_line_row(self) -> None:
        rows = parse_rows(self.ROOT + "targets_single_line_test.txt")
        expected = [(99, "kats/models/bayesian_var.py")]
        self.assertEqual(rows, expected)

    def test_one_row(self) -> None:
        rows = parse_rows(self.ROOT + "targets_one_row_test.txt")
        expected = [(99, "kats/models/bayesian_var.py")]
        self.assertEqual(rows, expected)

    def test_multiple_rows(self) -> None:
        rows = parse_rows(self.ROOT + "targets_multiple_rows_test.txt")
        expected = [
            (100, "kats/models/holtwinters.py"),
            (67, "kats/models/holtsummers.py"),
        ]
        self.assertEqual(rows, expected)

    def test_commented_out_test(self) -> None:
        rows = parse_rows(self.ROOT + "targets_commented_out_test.txt")
        expected = []
        self.assertEqual(rows, expected)

    def test_commented_out_coverage(self) -> None:
        rows = parse_rows(self.ROOT + "targets_commented_out_coverage_test.txt")
        expected = [(95, "kats/models/arima.py")]
        self.assertEqual(rows, expected)

    def test_square_bracket_format(self) -> None:
        rows = parse_rows(self.ROOT + "targets_square_bracket_format_test.txt")
        expected = [
            (65, "kats/models/harmonic_regression.py"),
            (100, "kats/models/holtspring.py"),
            (67, "kats/models/holtfall.py"),
        ]
        self.assertEqual(rows, expected)

    def test_ignore_non_tests(self) -> None:
        rows = parse_rows(self.ROOT + "targets_ignore_non_tests.txt")
        expected = [
            (95, "kats/models/arima.py"),
            (99, "kats/models/bayesian_var.py"),
            (65, "kats/models/harmonic_regression.py"),
            (100, "kats/models/holtwinters.py"),
            (67, "kats/models/holtsummers.py"),
            (100, "kats/models/holtspring.py"),
            (67, "kats/models/holtfall.py"),
        ]
        self.assertEqual(rows, expected)
