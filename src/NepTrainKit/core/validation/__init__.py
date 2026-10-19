"""Model validation against sealed DFT benchmark sets.

* :mod:`.library`   – where managed data lives (one relocatable folder)
* :mod:`.protocols` – one class per property (EOS, generic E/F/V, ...)
* :mod:`.service`   – capture models, import sets, evaluate, scorecard
"""

from .library import FileRef, Library, get_library, resolve_library_root
from .service import SetRejectedError, SetTooLargeError, ValidationService, parse_model_header

__all__ = [
    "FileRef",
    "Library",
    "SetRejectedError",
    "SetTooLargeError",
    "ValidationService",
    "get_library",
    "parse_model_header",
    "resolve_library_root",
]
