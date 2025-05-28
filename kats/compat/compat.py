# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import TypeAlias

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata

from typing import Callable, Union

import sklearn

import statsmodels

from packaging import version as pv

VERSION_TYPE: TypeAlias = pv.Version
V: TypeAlias = Union[str, "Version", pv.Version]


class Version:
    """Extend packaging Version to allow comparing to version strings.

    Wraps instead of extends, because pv.parse can return either a
    pv.Version or a pv.LegacyVersion for version under 21.3
    """

    version: VERSION_TYPE

    def __init__(self, version: V) -> None:
        """Parse a version.

        Args:
            version: the name of a package, a version string, or a packaging
                version object.
        """
        if isinstance(version, str):
            self.version: VERSION_TYPE = self._parse(version)
        elif isinstance(version, Version):
            self.version: VERSION_TYPE = version.version
        else:
            self.version: VERSION_TYPE = version

    def _parse(self, version: str) -> pv.Version:
        if version == "statsmodels":
            return pv.Version(
                statsmodels.__version__
            )  # TODO: importlib.metadata.version is spuriously returning 0.0.0 as statsmodels version, breaking compat

        if version == "sklearn":
            return pv.parse(sklearn.__version__)

        try:
            version = metadata.version(version)
        except metadata.PackageNotFoundError:
            pass

        return pv.parse(version)

    def __lt__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other: V) -> bool:
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other: V, method: Callable[[V, V], bool]) -> bool:
        if isinstance(other, Version):
            other = other.version
        elif isinstance(other, str):
            other = self._parse(other)
        try:
            return method(self.version._key, other._key)  # type: ignore
        except (AttributeError, TypeError):
            return NotImplemented
