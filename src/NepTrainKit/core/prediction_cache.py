"""Identity checks for official NEP output caches, independent of GUI loaders."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_record_error(record, *, model_hash, dataset_hash, structures, atoms, directory=None):
    """Check available provenance; old outputs without hashes remain viewable only."""
    for section in ("model", "dataset"):
        if not isinstance(record.get(section, {}), dict):
            return f"prediction.meta.json has invalid {section} metadata"
    dataset = record.get("dataset", {})
    if dataset.get("sha256") and dataset["sha256"] != dataset_hash:
        return "prediction.meta.json records a different structure file"
    if dataset.get("structures") is not None and dataset["structures"] != structures:
        return "prediction.meta.json records a different structure count"
    if dataset.get("atoms") is not None and dataset["atoms"] != atoms:
        return "prediction.meta.json records a different atom count"
    if model_hash and record.get("model", {}).get("sha256") and record["model"]["sha256"] != model_hash:
        return "prediction.meta.json records a different NEP model"
    if directory is not None:
        outputs = record.get("outputs", [])
        if not isinstance(outputs, list):
            return "prediction.meta.json has invalid output metadata"
        for output in outputs:
            if not isinstance(output, dict):
                return "prediction.meta.json has invalid output metadata"
            if not output.get("sha256"):
                continue
            name = output.get("name", "")
            if not isinstance(name, str) or not name or Path(name).name != name:
                return "prediction.meta.json has an invalid output filename"
            path = Path(directory) / name
            try:
                matched = path.is_file() and sha256_file(path) == output["sha256"]
            except OSError:
                matched = False
            if not matched:
                return f"prediction.meta.json output hash does not match {name}"
    return None


def load_accuracy_cache(directory, *, model_hash, dataset_hash, labels):
    """Read a fully identified cache and verify the same bytes that are parsed."""
    directory = Path(directory).expanduser()
    manifest_data = (directory / "prediction.meta.json").read_bytes()
    metadata = json.loads(manifest_data)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or not isinstance(metadata.get("predictions"), dict):
        raise ValueError("The cache needs a supported prediction.meta.json manifest.")
    matches = [(name, record) for name, record in metadata["predictions"].items()
               if isinstance(record, dict) and isinstance(record.get("model"), dict)
               and isinstance(record.get("dataset"), dict) and record["model"].get("sha256") == model_hash
               and record.get("dataset", {}).get("sha256") == dataset_hash]
    if len(matches) != 1:
        raise ValueError("The cache must identify exactly one matching model and dataset hash.")
    name, record = matches[0]
    error = prediction_record_error(record, model_hash=model_hash, dataset_hash=dataset_hash,
                                    structures=len(labels.atom_counts), atoms=int(labels.atom_counts.sum()))
    if error:
        raise ValueError(error)
    dataset = record.get("dataset", {})
    if dataset.get("structures") != len(labels.atom_counts) or dataset.get("atoms") != int(labels.atom_counts.sum()):
        raise ValueError("The cache manifest must record matching structure and atom counts.")
    outputs = record.get("outputs", [])
    if not isinstance(outputs, list) or any(not isinstance(item, dict) for item in outputs):
        raise ValueError("The cache manifest has invalid output metadata.")
    predicted, hashes = {}, {}
    for key, count in labels.counts.items():
        if not count["components"]:
            continue
        filename = f"{key}_{Path(name).stem}.out"
        candidates = [item for item in outputs if item.get("name") == filename]
        if len(candidates) != 1 or not candidates[0].get("sha256"):
            raise ValueError(f"The cache has no verified output hash for {filename}.")
        data = (directory / filename).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != candidates[0]["sha256"]:
            raise ValueError(f"The cache output hash does not match {filename}.")
        columns = 1 if key == "energy" else labels.values[key].shape[1]
        expected = (len(labels.values[key]), 2 * columns)
        array = np.loadtxt(io.BytesIO(data), ndmin=2)
        if array.shape != expected:
            raise ValueError(f"{filename} has shape {array.shape}; expected {expected}.")
        prediction = array[:, :columns]
        reference = array[:, columns:]
        if key == "energy":
            prediction, reference = prediction[:, 0], reference[:, 0]
        mask = labels.masks[key]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{filename} contains non-finite predictions.")
        if not np.allclose(reference[mask], labels.values[key][mask], rtol=1e-6, atol=1e-8):
            raise ValueError(f"{filename} reference columns do not match the selected dataset.")
        predicted[key], hashes[filename] = prediction, digest
    return predicted, {"kind": "verified_cache", "outputs": hashes,
                       "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
                       "backend": record.get("backend", {}),
                       "nep_adapters_version": record.get("nep_adapters_version", "unknown")}


def find_output_groups(directory):
    """Find official comparison files directly beside a model, without reading data."""
    import re
    groups = {}
    for path in sorted(Path(directory).iterdir()):
        match = re.fullmatch(r"(energy|force|virial|mforce)_(.+)\.out", path.name, re.IGNORECASE)
        if match and path.is_file():
            groups.setdefault(match[2], {})[match[1].lower()] = path
    return groups


def _output_rmse(path, columns, stopped):
    """Bound memory while hashing and reducing a potentially large output file."""
    filename = Path(path).name
    digest = hashlib.sha256()
    total, components, rows = 0.0, 0, 0

    expected_columns = columns * 2

    def consume(lines):
        nonlocal total, components, rows
        array = np.loadtxt(lines, ndmin=2)
        if not array.size:
            return
        if array.shape[1] != columns * 2:
            raise ValueError(f"{filename} must have {expected_columns} columns.")
        prediction, reference = array[:, :columns], array[:, columns:]
        if not np.isfinite(prediction).all() or np.isinf(reference).any():
            raise ValueError(f"{filename} contains non-finite predictions or reference infinities.")
        finite = np.isfinite(reference)
        if np.any(finite.any(axis=1) != finite.all(axis=1)):
            raise ValueError(f"{filename} contains incomplete reference vectors.")
        selected = finite.all(axis=1)
        delta = prediction[selected] - reference[selected]
        total += float(np.sum(delta * delta))
        components += int(delta.size)
        rows += len(array)
        if not np.isfinite(total):
            raise ValueError(f"{filename} error accumulation is not finite.")

    with Path(path).open('rb') as handle:
        lines = []
        for line in handle:
            digest.update(line)
            if line.strip() and not line.lstrip().startswith(b'#'):
                lines.append(line)
            if len(lines) >= 8192:
                if stopped():
                    return None
                consume(lines)
                lines.clear()
        if stopped():
            return None
        if lines:
            consume(lines)
    return {"rmse": float(np.sqrt(total / components) * 1000) if components else None,
            "components": components, "rows": rows, "sha256": digest.hexdigest()}


def inspect_output_groups(groups, model_hash, stopped=lambda: False):
    """Preview existing output precision only; this never instantiates a model."""
    summaries = {}
    for name, paths in groups.items():
        summary = {"group": name, "model_sha256": model_hash, "status": "unverified",
                   "metrics": {}, "counts": {}, "outputs": {}, "errors": {}, "reason": ""}
        for key, path in paths.items():
            if stopped():
                return {}
            try:
                result = _output_rmse(path, 1 if key == 'energy' else 6 if key == 'virial' else 3, stopped)
                if result is None:
                    return {}
                summary['metrics'][key + '_rmse'] = result.pop('rmse')
                summary['counts'][key] = {field: result[field] for field in ('rows', 'components')}
                summary['outputs'][path.name] = result['sha256']
            except (OSError, ValueError) as error:
                summary['errors'][key] = str(error)
        directory = next(iter(paths.values())).parent
        manifest = directory / 'prediction.meta.json'
        if manifest.is_file():
            try:
                metadata = json.loads(manifest.read_bytes())
                if not isinstance(metadata, dict) or metadata.get('schema_version') != 1:
                    raise ValueError('The cache needs a supported prediction.meta.json manifest.')
                records = [(filename, record) for filename, record in metadata.get('predictions', {}).items()
                           if Path(filename).stem == name and isinstance(record, dict)]
                if len(records) == 1:
                    filename, record = records[0]
                    registered_model = record.get('model', {}).get('sha256')
                    if registered_model and registered_model != model_hash:
                        raise ValueError('prediction.meta.json records a different NEP model')
                    expected = {row.get('name'): row.get('sha256') for row in record.get('outputs', [])}
                    for output_name, digest in summary['outputs'].items():
                        if expected.get(output_name) and digest != expected[output_name]:
                            raise ValueError(f'The cache output hash does not match {output_name}.')
                    dataset = directory / filename
                    if (registered_model and Path(filename).name == filename and dataset.is_file()
                            and all(expected.get(output) for output in summary['outputs'])
                            and summary['outputs'] and not summary['errors']):
                        from .validation.accuracy import accuracy_labels, accuracy_statistics
                        from .validation.protocols import read_frames
                        data = dataset.read_bytes()
                        labels = accuracy_labels(read_frames(data))
                        dataset_hash = hashlib.sha256(data).hexdigest()
                        error = prediction_record_error(record, model_hash=model_hash, dataset_hash=dataset_hash,
                                                        structures=len(labels.atom_counts), atoms=int(labels.atom_counts.sum()))
                        if error:
                            raise ValueError(error)
                        required = {f'{key}_{name}.out' for key, count in labels.counts.items() if count['components']}
                        if not required.issubset(summary['outputs']):
                            summaries[name] = summary
                            continue
                        values, source = load_accuracy_cache(directory, model_hash=model_hash,
                                                             dataset_hash=dataset_hash, labels=labels)
                        metrics, details = accuracy_statistics(labels, values)
                        summary.update(status='verified', metrics=metrics, counts=details['label_counts'],
                                       dataset_sha256=dataset_hash, outputs=source['outputs'], source=source)
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
                summary.update(status='mismatch', reason=str(error))
        summaries[name] = summary
    return summaries
