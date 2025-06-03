"""
Copyright (c) 2024 Ebrahim Ebrahim. All rights reserved.

abcdstats: Statistical analysis tools for use with ABCD neuroimaging and tabular data
"""

from __future__ import annotations

from .workflow import Basic as BasicWorkflow
from ._version import version as __version__

__all__ = ["BasicWorkflow", "__version__"]
