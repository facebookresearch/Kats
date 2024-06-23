# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata

import logging
from typing import Any, Callable, Type, Union

import packaging
import statsmodels

from packaging import version as pv

OLD_PACKAGING_VERSION: bool = pv.parse(packaging.__version__) <= pv.parse("21.3")

# type: ignore
VERSION_TYPE: Type[Any] = (
    Union[pv.Version, pv.LegacyVersion] if OLD_PACKAGING_VERSION else pv.Version
)

if OLD_PACKAGING_VERSION:
    V = Union[str, "Version", pv.Version, pv.LegacyVersion]
else:
    V = Union[str, "Version", pv.Version]


class Version:
    """Extend packaging Version to allow comparing to version strings.

    Wraps instead of extends, because pv.parse can return either a
    pv.Version or a pv.LegacyVersion for version under 21.3
    """

    # type: ignore
    version: VERSION_TYPE

    def __init__(self, version: V) -> None:
        """Parse a version.

        Args:
            version: the name of a package, a version string, or a packaging
                version object.
        """
        if isinstance(version, str):
            # type: ignore
            self.version: VERSION_TYPE = self._parse(version)
        elif isinstance(version, Version):
            # type: ignore
            self.version: VERSION_TYPE = version.version
        else:
            # type: ignore
            self.version: VERSION_TYPE = version

    def _parse(self, version: str) -> Union[pv.Version, pv.LegacyVersion]:
        if version == "statsmodels":
            return pv.Version(
                statsmodels.__version__
            )  # TODO: importlib.metadata.version is spuriously returning 0.0.0 as statsmodels version, breaking compat

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
