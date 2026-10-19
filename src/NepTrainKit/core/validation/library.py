"""The validation library: one relocatable folder that owns all managed data.

Layout::

    <root>/library.json              marker with the layout version
    <root>/library.db                SQLite index (paths inside are relative)
    <root>/objects/sha256/ab/<hash>  immutable, content-addressed files
    <root>/results/<run_id>.json     evaluation details (curves, per-frame data)

Small data that results depend on (model files, benchmark sets) is copied in as
objects. Large user-owned data (training sets, trajectories, raw DFT output) is
only *referenced* through a :class:`FileRef` fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from NepTrainKit.paths import get_user_config_path

LAYOUT_VERSION = 1
ENV_VAR = "NEPTRAINKIT_LIBRARY"
CONFIG_SECTION = "validation"
CONFIG_LIBRARY_PATH = "library_path"
_QUICK_HASH_LIMIT = 64 * 1024 * 1024
_QUICK_HASH_CHUNK = 4 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quick_hash(path: Path, size: int) -> str:
    """Full hash for ordinary files; size + head + tail for very large ones."""
    if size <= _QUICK_HASH_LIMIT:
        return sha256_file(path)
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(_QUICK_HASH_CHUNK))
        handle.seek(-_QUICK_HASH_CHUNK, os.SEEK_END)
        digest.update(handle.read(_QUICK_HASH_CHUNK))
    return "quick:" + digest.hexdigest()


@dataclass(frozen=True)
class FileRef:
    """Fingerprint of a user-owned file that the library does not copy."""

    path: str
    size: int
    mtime_ns: int
    digest: str

    @classmethod
    def of(cls, path: str | Path) -> FileRef:
        path = Path(path).expanduser().resolve()
        info = path.stat()
        return cls(str(path), info.st_size, info.st_mtime_ns, _quick_hash(path, info.st_size))

    @classmethod
    def from_dict(cls, data: dict) -> FileRef | None:
        if not data:
            return None
        return cls(str(data["path"]), int(data["size"]), int(data["mtime_ns"]), str(data["digest"]))

    def to_dict(self) -> dict:
        return asdict(self)

    def status(self) -> str:
        """Return ``ok``, ``changed`` or ``missing`` (cheap stat first, hash on doubt)."""
        path = Path(self.path)
        try:
            info = path.stat()
        except OSError:
            return "missing"
        if info.st_size == self.size and info.st_mtime_ns == self.mtime_ns:
            return "ok"
        if info.st_size != self.size:
            return "changed"
        return "ok" if _quick_hash(path, info.st_size) == self.digest else "changed"

    def relocate(self, search_root: str | Path) -> FileRef | None:
        """Find the same content under ``search_root`` after the user moved it."""
        name = Path(self.path).name
        for candidate in Path(search_root).rglob(name):
            try:
                if candidate.is_file() and candidate.stat().st_size == self.size:
                    if _quick_hash(candidate, self.size) == self.digest:
                        return FileRef.of(candidate)
            except OSError:
                continue
        return None


def default_library_root() -> Path:
    return get_user_config_path() / "library"


def resolve_library_root(explicit: str | Path | None = None) -> Path:
    """Explicit path > ``NEPTRAINKIT_LIBRARY`` > user setting > default."""
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    try:
        from NepTrainKit.config import Config

        if Config._instance is not None:
            configured = Config.get(CONFIG_SECTION, CONFIG_LIBRARY_PATH)
            if configured:
                return Path(str(configured)).expanduser()
    except Exception:
        pass
    return default_library_root()


class Library:
    """Managed storage rooted at one folder; see the module docstring."""

    def __init__(self, root: str | Path | None = None, *, create: bool = True):
        self.root = resolve_library_root(root).resolve()
        if create:
            self._initialize()
        elif not (self.root / "library.json").exists():
            raise FileNotFoundError(f"Not a NepTrainKit library: {self.root}")
        self._database = None

    # -- layout ---------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.root / "library.db"

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects" / "sha256"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def _initialize(self) -> None:
        marker = self.root / "library.json"
        if marker.exists():
            layout = json.loads(marker.read_text(encoding="utf-8")).get("layout", 0)
            if layout > LAYOUT_VERSION:
                raise RuntimeError(
                    f"Library {self.root} was written by a newer NepTrainKit (layout {layout})."
                )
            return
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._adopt_legacy_data()
        _write_atomic(marker, json.dumps({"layout": LAYOUT_VERSION}, indent=2).encode())

    def _adopt_legacy_data(self) -> None:
        """Bring the pre-library database and ``cas/`` folder into a new default library.

        Legacy files are copied, never moved, so downgrading keeps working.
        """
        if self.root != default_library_root().resolve():
            return
        legacy = get_user_config_path()
        legacy_db = legacy / "mlpman.db"
        if legacy_db.exists() and not self.db_path.exists():
            shutil.copy2(legacy_db, self.db_path)
        legacy_cas = legacy / "cas"
        if legacy_cas.is_dir():
            for item in legacy_cas.iterdir():
                if item.is_file() and len(item.name) == 64:
                    self.put_file(item)

    @property
    def database(self):
        if self._database is None:
            from NepTrainKit.core.dataset.database import Database

            self._database = Database(self.db_path)
        return self._database

    # -- objects --------------------------------------------------------------
    def object_path(self, digest: str) -> Path:
        # Ids come from the database or from iter_objects, never from user input.
        if not _is_digest(digest):
            raise ValueError(f"Invalid object id: {digest!r}")
        return self.objects_dir / digest[:2] / digest

    def has_object(self, digest: str) -> bool:
        return self.object_path(digest).is_file()

    def put_bytes(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        target = self.object_path(digest)
        if not target.exists():
            _write_atomic(target, data, read_only=True)
        return digest

    def put_file(self, path: str | Path) -> str:
        path = Path(path)
        digest = sha256_file(path)
        target = self.object_path(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shutil.copyfile(path, temporary)
                if sha256_file(temporary) != digest:
                    raise OSError(f"{path} changed while it was being copied.")
                _publish(temporary, target, read_only=True)
            finally:
                temporary.unlink(missing_ok=True)
        return digest

    def open_object(self, digest: str, *, verify: bool = True) -> Path:
        """Return the stored file, refusing it if the bytes no longer match."""
        path = self.object_path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"Library object {digest[:12]} is missing from {self.root}.")
        if verify and sha256_file(path) != digest:
            raise ValueError(f"Library object {digest[:12]} is corrupt.")
        return path

    def iter_objects(self):
        if self.objects_dir.is_dir():
            for path in self.objects_dir.glob("??/*"):
                # Skip stray files so clean-up never trips over a foreign name.
                if path.is_file() and _is_digest(path.name):
                    yield path.name, path

    def delete_object(self, digest: str) -> int:
        path = self.object_path(digest)
        try:
            size = path.stat().st_size
            path.chmod(stat.S_IWUSR | stat.S_IRUSR)
            path.unlink()
            return size
        except FileNotFoundError:
            return 0

    # -- results --------------------------------------------------------------
    def write_result(self, run_id: str, payload: dict) -> str:
        relative = f"results/{run_id}.json"
        _write_atomic(self.root / relative, json.dumps(payload, separators=(",", ":")).encode())
        return relative

    def read_result(self, relative: str) -> dict:
        if not relative:
            return {}
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    # -- housekeeping ---------------------------------------------------------
    def usage(self) -> dict[str, int]:
        """Bytes used by each part of the library."""
        def total(paths):
            return sum(p.stat().st_size for p in paths if p.is_file())

        db = total(self.root.glob("library.db*"))
        return {
            "objects": total(self.objects_dir.rglob("*")) if self.objects_dir.exists() else 0,
            "results": total(self.results_dir.glob("*.json")) if self.results_dir.exists() else 0,
            "database": db,
        }


def _is_digest(name: str) -> bool:
    return len(name) == 64 and all(c in "0123456789abcdef" for c in name)


def _publish(temporary: Path, target: Path, *, read_only: bool) -> None:
    if read_only:
        temporary.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        temporary.replace(target)
    except PermissionError:
        # Windows refuses to replace a read-only file; identical content is already there.
        if not target.exists():
            raise


def _write_atomic(target: Path, data: bytes, *, read_only: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        _publish(temporary, target, read_only=read_only)
    finally:
        temporary.unlink(missing_ok=True)


_default_library: Library | None = None


def get_library() -> Library:
    """Process-wide library for the GUI; re-resolved when the root changes."""
    global _default_library
    root = resolve_library_root().resolve()
    if _default_library is None or _default_library.root != root:
        _default_library = Library(root)
    return _default_library
