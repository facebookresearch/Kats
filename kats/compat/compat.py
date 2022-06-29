# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata

from typing import Callable, Union

from packaging import version as pv


V = Union[str, "Version", pv.Version, pv.LegacyVersion]


class Version:
    """Extend packaging Version to allow comparing to version strings.

    Wraps instead of extends, because pv.parse can return either a
    pv.Version or a pv.LegacyVersion.
    """

    version: Union[pv.Version, pv.LegacyVersion]

    def __init__(self, version: V) -> None:
        """Parse a version.

        Args:
            version: the name of a package, a version string, or a packaging
                version object.
        """
        if isinstance(version, str):
            self.version = self._parse(version)
        elif isinstance(version, Version):
            self.version = version.version
        else:
            self.version = version

    def _parse(self, version: str) -> Union[pv.Version, pv.LegacyVersion]:
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
            return method(self.version._key, other._key)
        except (AttributeError, TypeError):
            return NotImplemented
