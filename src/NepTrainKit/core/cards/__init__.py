"""Reusable Make Dataset card operations.

This package contains UI-independent card logic. Qt widgets should collect
parameters and delegate computation here instead of reading widget state inside
the processing path.
"""

from .operation import DatasetOperation, GeneratorOperation, StructureOperation

__all__ = [
    "DatasetOperation",
    "GeneratorOperation",
    "StructureOperation",
]
