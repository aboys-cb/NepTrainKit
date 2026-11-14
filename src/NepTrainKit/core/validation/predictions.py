"""Reusable fixed-structure predictions, independent of reference analysis recipes."""

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np


class CachedPrediction:
    def __init__(self, predictor, directory, identity, *, refresh=False):
        self.predictor = predictor
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.path = Path(directory) / (hashlib.sha256(self.identity.encode()).hexdigest() + ".npz")
        self.refresh = refresh
        self.source = None

    def predict(self, frames):
        expected = {
            "energy": (len(frames),),
            "forces": (sum(map(len, frames)), 3),
            "structure_virials": (len(frames), 9),
            "mforces": (sum(map(len, frames)), 3),
        }
        if not self.refresh and self.path.is_file():
            try:
                with np.load(self.path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata"]))
                    if metadata["identity"] != self.identity:
                        raise ValueError("Prediction identity mismatch.")
                    values = {key: np.array(archive[key]) for key in expected if key in archive}
                    if not {"energy", "forces", "structure_virials"} <= values.keys():
                        raise ValueError("Incomplete predictions.")
                    for key, value in values.items():
                        if (
                            value.shape != expected[key]
                            or value.dtype != np.float64
                            or not np.isfinite(value).all()
                            or hashlib.sha256(value.tobytes()).hexdigest() != metadata["hashes"][key]
                        ):
                            raise ValueError("Corrupted predictions.")
                self.source = {"kind": "prediction_cache", "key": self.path.stem}
                return SimpleNamespace(**{"mforces": None, **values})
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
                pass
        prediction = self.predictor.predict(frames)
        values = {}
        for key, shape in expected.items():
            raw = getattr(prediction, key, None)
            if raw is None and key == "mforces":
                continue
            value = np.asarray(raw, dtype=np.float64)
            if key == "structure_virials" and value.shape == (len(frames), 3, 3):
                value = value.reshape(shape)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Fixed-structure prediction {key} has an invalid shape or non-finite values.")
            values[key] = value
        self.source = {"kind": "recalculated", "key": self.path.stem}
        metadata = {
            "identity": self.identity,
            "hashes": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in values.items()},
        }
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, metadata=json.dumps(metadata), **values)
            temporary.replace(self.path)
        except OSError:
            # A full/read-only cache must not discard a successful calculation.
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return SimpleNamespace(**{"mforces": None, **values})
