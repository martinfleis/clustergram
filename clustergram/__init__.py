"""Clustergam initialisation. Imports main class and dunders."""

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .clustergram import Clustergram  # noqa

__author__ = "Martin Fleischmann"
__author_email__ = "martin@martinfleischmann.net"

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("clustergram")
