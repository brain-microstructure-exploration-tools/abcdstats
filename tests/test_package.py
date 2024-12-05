from __future__ import annotations

import importlib.metadata

import abcdstats as m


def test_version():
    assert importlib.metadata.version("abcdstats") == m.__version__
