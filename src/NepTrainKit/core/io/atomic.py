"""Atomic text exports, with open streams left under their caller's control."""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_text_output(destination, *, append=False):
    """Publish a path only after writing succeeds; leave supplied streams open."""
    if hasattr(destination, "write"):
        yield destination
        return
    destination = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            if append and destination.exists():
                with destination.open("rb") as source:
                    shutil.copyfileobj(source, handle.buffer)
                handle.seek(0, os.SEEK_END)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
