"""Fixed-structure reference responses, configured without modifying source frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from copy import copy
import json

import numpy as np

from ..grouping import _scalar
from .base import Evaluation, Issue, Metric, ValidationProtocol, _energy, _virial_per_atom, unlabelled_copies

MODES = ("scan", "difference", "combination", "four_state", "cross", "slices")
ROLES = ("unknown", "training", "development", "validation")


@dataclass(frozen=True)
class ResponseSpec:
    mode: str = "scan"
    group_fields: tuple[str, ...] = ()
    x_field: str = "volume_per_atom"
    second_axis: str = ""
    state_fields: tuple[str, ...] = ()
    reference: str = ""
    # Ordered state values and linear coefficients; never executable expressions.
    terms: tuple = ()
    normalization: str = "atom"
    element: str = ""
    energy_unit: str = "meV"
    quantity: str = "energy"
    reference_field: str = ""
    reference_scale: float = 1.0
    relative: bool = True
    skip_incomplete: bool = False
    role: str = "unknown"
    frame_metadata: dict = field(default_factory=dict)
    extract_exchange: bool = False
    pair_fields: tuple[str, str] = ("site_i", "site_j")
    axis_fields: tuple[str, str] = ("axis_i", "axis_j")
    index_base: int = 0
    multiplicity: float = 1.0

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Invalid reference analysis settings.")
        data = dict(data)
        for key in ("group_fields", "state_fields", "terms", "pair_fields", "axis_fields"):
            if key in data:
                if not isinstance(data[key], (list, tuple)):
                    raise ValueError(f"{key} must be a list.")
                data[key] = tuple(data[key])
        spec = cls(**data)
        spec.validate()
        return spec

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    def validate(self):
        if any(
            not isinstance(getattr(self, key), str)
            for key in (
                "mode",
                "x_field",
                "second_axis",
                "reference",
                "normalization",
                "element",
                "energy_unit",
                "quantity",
                "reference_field",
                "role",
            )
        ):
            raise ValueError("Analysis field names and selections must be text.")
        if any(type(getattr(self, key)) is not bool for key in ("relative", "skip_incomplete", "extract_exchange")):
            raise ValueError("Analysis switches must be boolean values.")
        if not isinstance(self.multiplicity, (int, float)) or not np.isfinite(self.multiplicity):
            raise ValueError("Interaction multiplicity must be finite.")
        if self.mode not in MODES or self.role not in ROLES:
            raise ValueError("Choose a supported analysis and dataset role.")
        if not isinstance(self.frame_metadata, dict):
            raise ValueError("Frame metadata must be indexed by zero-based frame numbers.")
        protected = {
            "energy",
            "free_energy",
            "forces",
            "force",
            "virial",
            "virials",
            "stress",
            "mforce",
            "force_mag",
            "spin",
            "spins",
            "initial_magmoms",
            "positions",
            "numbers",
            "Lattice",
            "Properties",
        }
        for index, metadata in self.frame_metadata.items():
            if (
                not isinstance(index, str)
                or not index.isdecimal()
                or not isinstance(metadata, dict)
                or any(
                    not isinstance(key, str) or key in protected or _scalar(value) is None
                    for key, value in metadata.items()
                )
            ):
                raise ValueError(
                    "Metadata overrides must contain finite scalar grouping fields, never reference labels or model inputs."
                )
        if self.normalization not in ("total", "atom", "element") or self.energy_unit not in ("meV", "μeV"):
            raise ValueError("Choose an energy unit and normalization.")
        if self.normalization == "element" and not self.element:
            raise ValueError("Choose the element used for normalization.")
        if self.quantity not in (
            "energy",
            "pressure",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
            "stress_yz",
            "stress_zx",
        ):
            raise ValueError("Choose an available response quantity.")
        if self.mode in ("four_state", "cross") and self.quantity != "energy":
            raise ValueError("Four-state and cross responses require total energies.")
        if (
            not isinstance(self.reference_scale, (int, float))
            or not np.isfinite(self.reference_scale)
            or self.reference_scale == 0
        ):
            raise ValueError("Reference scaling must be finite and nonzero.")
        for names in (self.group_fields, self.state_fields):
            if any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names):
                raise ValueError("Grouping and state fields must be distinct non-empty names.")
        if set(self.group_fields) & set(self.state_fields):
            raise ValueError("State fields must not also be grouping fields.")
        if self.mode == "difference" and (not self.state_fields or not self.reference.strip()):
            raise ValueError("An energy difference needs state fields and a reference state.")
        if self.mode in ("four_state", "cross") and len(self.state_fields) != 2:
            raise ValueError("Four-state and cross responses need two state fields.")
        if self.extract_exchange and (
            self.mode != "four_state"
            or self.normalization != "total"
            or self.index_base not in (0, 1)
            or len(self.pair_fields) != 2
            or len(self.axis_fields) != 2
            or not np.isfinite(self.multiplicity)
            or self.multiplicity <= 0
        ):
            raise ValueError(
                "Exchange extraction needs four states, per-cell energies, an explicit index base and positive interaction multiplicity."
            )
        if self.mode == "combination":
            if not self.state_fields or not self.terms:
                raise ValueError("A combination needs state fields and signed coefficients.")
            states = []
            for term in self.terms:
                if not isinstance(term, (list, tuple)) or len(term) != 2:
                    raise ValueError("Each term must contain state values and one coefficient.")
                state, coefficient = term
                if not isinstance(state, (list, tuple)) or len(state) != len(self.state_fields):
                    raise ValueError("Term state values must match the state fields.")
                if not isinstance(coefficient, (int, float)) or not np.isfinite(coefficient):
                    raise ValueError("Combination coefficients must be finite.")
                states.append(tuple(_text(x) for x in state))
            if len(set(states)) != len(states) or not any(term[1] for term in self.terms):
                raise ValueError("Combination states must be unique and coefficients nonzero.")

    @property
    def unit(self):
        if self.mode == "slices":
            return "meV/atom"
        if self.quantity != "energy":
            return "GPa"
        suffix = {"total": "/cell", "atom": "/atom", "element": "/" + self.element}[self.normalization]
        return self.energy_unit + suffix


def _text(value):
    value = _scalar(value)
    if value is None:
        raise ValueError("A grouping or state value is missing or is not scalar.")
    return format(value, ".12g") if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _field(frame, key):
    if key == "volume_per_atom":
        return frame.get_volume() / len(frame)
    if key == "volume":
        return frame.get_volume()
    if key not in frame.info:
        raise ValueError(f"Missing metadata field: {key}.")
    return frame.info[key]


def _coordinate(frame, key):
    value = _scalar(_field(frame, key))
    if value is None:
        raise ValueError(f"Scan field {key} must contain finite scalar values.")
    return value


def _denominator(frame, spec):
    count = (
        1
        if spec.normalization == "total"
        else len(frame)
        if spec.normalization == "atom"
        else frame.get_chemical_symbols().count(spec.element)
    )
    if not count:
        raise ValueError("The normalization element is absent from a structure.")
    return count


def _reference(frames, spec):
    if spec.reference_field:
        values = [float(_field(a, spec.reference_field)) * spec.reference_scale for a in frames]
    elif spec.quantity == "energy":
        scale = 1000 if spec.energy_unit == "meV" else 1e6
        values = [_energy(a) / _denominator(a, spec) * scale for a in frames]
    else:
        values = []
        for a in frames:
            virial = _virial_per_atom(a)
            if virial is None:
                raise ValueError("Stress response needs virial/stress labels or an explicit reference field.")
            values.append(_stress_value(virial, a, spec.quantity))
    values = np.asarray(values, dtype=float)
    if values.shape != (len(frames),) or not np.isfinite(values).all():
        raise ValueError("Reference response values must be finite scalars.")
    return values


def _stress_value(virial_per_atom, atoms, quantity):
    tensor = -np.asarray(virial_per_atom).reshape(3, 3) * len(atoms) / atoms.get_volume() * 160.21766208
    if quantity == "pressure":
        return float(-np.trace(tensor) / 3)
    index = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "zx": (2, 0)}[
        quantity.removeprefix("stress_")
    ]
    return float(tensor[index])


def _predicted(prediction, frames, spec):
    if spec.quantity == "energy":
        values = np.asarray(prediction.energy, dtype=float)
        if values.shape != (len(frames),):
            raise ValueError("Prediction must contain one finite energy per reference frame.")
        values = values / [_denominator(a, spec) for a in frames] * (1000 if spec.energy_unit == "meV" else 1e6)
    else:
        tensors = np.asarray(prediction.structure_virials, dtype=float)
        if tensors.shape not in ((len(frames), 9), (len(frames), 3, 3)):
            raise ValueError("Predicted virial shapes do not match the reference structures.")
        values = np.asarray([_stress_value(v, a, spec.quantity) for v, a in zip(tensors, frames)])
    if not np.isfinite(values).all():
        raise ValueError("Predicted response values must be finite.")
    return values


def _same_geometry(frames):
    first = frames[0]
    for frame in frames[1:]:
        if (
            not np.array_equal(frame.numbers, first.numbers)
            or not np.array_equal(frame.pbc, first.pbc)
            or not np.allclose(frame.cell.array, first.cell.array, rtol=0, atol=1e-7)
            or not np.allclose(frame.positions, first.positions, rtol=0, atol=1e-7)
        ):
            raise ValueError("States in this group do not share the same atomic geometry.")


def _plan(frames, spec):
    """Resolve signed rows once, using exactly the same rows for reference and model."""
    spec.validate()
    if not frames or any(not len(a) for a in frames):
        raise ValueError("Reference analysis needs non-empty structures.")
    if any(not np.isfinite(a.positions).all() or not np.isfinite(a.cell.array).all() for a in frames):
        raise ValueError("Reference structures need finite positions and cells.")
    groups = {}
    for index, frame in enumerate(frames):
        key = tuple(_text(_field(frame, name)) for name in spec.group_fields)
        groups.setdefault(key, []).append(index)
    rows, series, excluded = [], [], []
    for key, indices in groups.items():
        name = " · ".join(f"{field}={value}" for field, value in zip(spec.group_fields, key)) or "Whole dataset"
        try:
            current = []
            if spec.mode == "slices":
                current = [(index, [(index, 1.0)], str(index)) for index in indices]
            elif spec.mode == "scan":
                if spec.relative and len(indices) < 2:
                    raise ValueError("A relative reference scan needs at least two points.")
                points = [
                    (
                        _coordinate(frames[i], spec.x_field),
                        _coordinate(frames[i], spec.second_axis) if spec.second_axis else None,
                        i,
                    )
                    for i in indices
                ]
                if len({(str(x), str(y)) for x, y, _ in points}) != len(points):
                    raise ValueError("Duplicate scan coordinates: add grouping fields or a second axis.")
                if any(isinstance(x, str) != isinstance(points[0][0], str) for x, _, _ in points):
                    raise ValueError("Scan coordinate types must agree within a group.")
                if spec.second_axis and any(isinstance(y, str) != isinstance(points[0][1], str) for _, y, _ in points):
                    raise ValueError("Scan coordinate types must agree within a group.")
                points.sort(key=lambda point: point[:2])
                reference_index = min(indices, key=lambda i: _reference([frames[i]], spec)[0])
                current = [
                    (x, [(i, 1.0), (reference_index, -1.0)] if spec.relative else [(i, 1.0)], _text(x), y)
                    for x, y, i in points
                ]
            else:
                states = {}
                for i in indices:
                    state = tuple(
                        ""
                        if spec.mode == "difference" and field not in frames[i].info
                        else _text(_field(frames[i], field))
                        for field in spec.state_fields
                    )
                    if not any(state):
                        raise ValueError("Every response state needs at least one state field.")
                    if state in states:
                        raise ValueError("Duplicate states: add grouping fields to distinguish them.")
                    states[state] = i
                if spec.mode in ("difference", "four_state"):
                    _same_geometry([frames[i] for i in indices])
                if spec.mode == "difference":
                    reference = tuple(value.strip() for value in spec.reference.split(","))
                    reference += ("",) * (len(spec.state_fields) - len(reference))
                    if reference not in states:
                        raise ValueError("The reference state is missing from this group.")
                    current = [
                        (
                            " / ".join(value for value in state if value),
                            [(i, 1.0), (states[reference], -1.0)],
                            " / ".join(value for value in state if value),
                        )
                        for state, i in states.items()
                        if state != reference
                    ]
                else:
                    terms = (
                        spec.terms
                        if spec.mode == "combination"
                        else [([1, 1], 1), ([-1, -1], 1), ([1, -1], -1), ([-1, 1], -1)]
                        if spec.mode == "four_state"
                        else [([1, 1], 1), ([0, 0], 1), ([1, 0], -1), ([0, 1], -1)]
                    )
                    weighted = []
                    for state, coefficient in terms:
                        state = tuple(_text(x) for x in state)
                        if state not in states:
                            raise ValueError("A required combination state is missing from this group.")
                        weighted.append((states[state], float(coefficient)))
                    if len(states) != len(terms):
                        raise ValueError("Unexpected extra states in a combination group.")
                    current = [(0, weighted, name)]
            if not current:
                raise ValueError("This group has no comparable response points.")
        except ValueError as error:
            if not spec.skip_incomplete:
                raise ValueError(f"Response group {name}: {error}") from error
            excluded.append(dict(group=name, indices=indices, reason=str(error)))
            continue
        start = len(rows)
        rows.extend(dict(x=p[0], terms=p[1], label=p[2], second_axis=p[3] if len(p) > 3 else None) for p in current)
        series.append(dict(name=name, indices=list(range(start, len(rows))), frame_indices=indices))
    if not rows:
        raise ValueError("No complete response groups remain.")
    return rows, series, excluded


def _apply(values, rows):
    return np.asarray([sum(values[i] * c for i, c in row["terms"]) for row in rows])


def _extremum(x, values):
    if len(x) < 3 or any(not isinstance(v, (int, float)) for v in x) or len(set(x)) != len(x):
        return {}
    i = int(np.argmin(values))
    result = {"minimum_x": x[i], "minimum_bracketed": 0 < i < len(x) - 1}
    if 0 < i < len(x) - 1:
        xs = np.asarray(x[i - 1 : i + 2], dtype=float)
        curvature = 2 * np.polyfit(xs - xs[1], values[i - 1 : i + 2], 2)[0]
        if np.isfinite(curvature):
            result["curvature"] = float(curvature)
    return result


class ReferenceResponseProtocol(ValidationProtocol):
    kind = "response"
    version = 1
    title = "Reference response analysis"
    supports_spin = True

    def __init__(self, settings=None):
        self.spec = ResponseSpec.from_dict(settings or {})
        self.metrics = (
            Metric("response_rmse", "Response RMSE", self.spec.unit, headline=True),
            Metric("max_response_error", "Max response error", self.spec.unit, headline=True),
            Metric("sign_mismatches", "Sign mismatches", "points"),
            Metric("excluded_groups", "Excluded groups", "groups", better="none"),
        )
        if self.spec.mode == "slices":
            from .efv import EnergyForceVirialProtocol

            self.metrics = EnergyForceVirialProtocol.metrics

    def check(self, frames):
        try:
            frames = self._frames(frames)
            if self.spec.mode != "slices":
                _reference(frames, self.spec)
            rows, _, excluded = _plan(frames, self.spec)
            if self.spec.extract_exchange:
                from ..exchange import exchange_rows

                exchange_rows(frames, rows, self.spec)
            if self.spec.mode == "slices":
                from ..accuracy import accuracy_labels

                if not any(value["components"] for value in accuracy_labels(frames).counts.values()):
                    raise ValueError("At least one energy, force, virial or magnetic-force label is required.")
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return [Issue("error", str(error))]
        return [Issue("warning", f"Excluded group {g['group']}: {g['reason']}") for g in excluded]

    def summarize_reference(self, frames):
        frames = self._frames(frames)
        rows, series, excluded = _plan(frames, self.spec)
        if self.spec.mode == "slices":
            from ..accuracy import accuracy_labels

            labels = accuracy_labels(frames)
            for group in series:
                group["label_counts"] = accuracy_labels([frames[i] for i in group["frame_indices"]]).counts
            reference = [
                None
                if not labels.masks["energy"][row["terms"][0][0]]
                else float(labels.values["energy"][row["terms"][0][0]] * 1000)
                for row in rows
            ]
        else:
            reference = _apply(_reference(frames, self.spec), rows).tolist()
        result = {
            "formula": frames[0].get_chemical_formula(),
            "analysis": self.spec.to_dict(),
            "unit": self.spec.unit,
            "x_label": self.spec.x_field,
            "rows": rows,
            "reference": reference,
            "series": series,
            "excluded": excluded,
            "frames": len(frames),
            "role": self.spec.role,
            **({"label_counts": labels.counts} if self.spec.mode == "slices" else {}),
        }
        if self.spec.extract_exchange:
            from ..exchange import exchange_rows, evaluate_exchange

            result["exchange"] = evaluate_exchange(exchange_rows(frames, rows, self.spec), reference)
        if self.spec.mode == "scan" and not self.spec.second_axis:
            for group in series:
                indices = group["indices"]
                group["reference_extremum"] = _extremum([rows[i]["x"] for i in indices], np.asarray(reference)[indices])
        return result

    def evaluate(self, predictor, frames):
        summary = self.summarize_reference(frames)
        frames = self._frames(frames)
        prediction = predictor.predict(unlabelled_copies(frames))
        if self.spec.mode == "slices":
            return self._accuracy_slices(prediction, frames, summary)
        predicted = _apply(_predicted(prediction, frames, self.spec), summary["rows"])
        reference = np.asarray(summary["reference"])
        error = predicted - reference
        metrics = {
            "response_rmse": float(np.sqrt(np.mean(error**2))),
            "max_response_error": float(np.max(np.abs(error))),
            "sign_mismatches": int(np.sum((reference * predicted < 0) & (np.abs(reference) > 1e-10))),
            "excluded_groups": len(summary["excluded"]),
        }
        details = dict(summary, predicted=predicted.tolist(), errors=error.tolist())
        if self.spec.extract_exchange:
            from ..exchange import exchange_rows, evaluate_exchange

            details["exchange"] = evaluate_exchange(
                exchange_rows(frames, summary["rows"], self.spec), reference, predicted
            )
        for series in details["series"]:
            selected = series["indices"]
            series["rmse"] = float(np.sqrt(np.mean(error[selected] ** 2)))
            series["max_error"] = float(np.max(np.abs(error[selected])))
            if self.spec.mode == "scan" and not self.spec.second_axis:
                x = [summary["rows"][i]["x"] for i in selected]
                series["reference_extremum"] = _extremum(x, reference[selected])
                series["model_extremum"] = _extremum(x, predicted[selected])
        return Evaluation(
            metrics, details, [f"Excluded group {g['group']}: {g['reason']}" for g in summary["excluded"]]
        )

    def _frames(self, frames):
        result = list(frames)
        for key, metadata in self.spec.frame_metadata.items():
            index = int(key)
            if not 0 <= index < len(frames):
                raise ValueError("A metadata override refers to a frame outside this dataset.")
            result[index] = copy(frames[index])
            result[index].info = dict(frames[index].info, **metadata)
        return result

    def _accuracy_slices(self, prediction, frames, summary):
        from ..accuracy import accuracy_labels, accuracy_statistics, prediction_values

        labels = accuracy_labels(frames)
        values = prediction_values(prediction, labels)
        metrics, stats = accuracy_statistics(labels, values)
        summary["label_counts"] = stats["label_counts"]
        summary["accuracy_slices"] = []
        counts = np.r_[0, np.cumsum(labels.atom_counts)]
        for series in summary["series"]:
            indices = series["frame_indices"]
            atoms = np.concatenate([np.arange(counts[i], counts[i + 1]) for i in indices])
            lab = accuracy_labels([frames[i] for i in indices])
            selected = {key: value[indices if key in ("energy", "virial") else atoms] for key, value in values.items()}
            if any(lab.counts[key]["components"] for key in selected):
                result, details = accuracy_statistics(lab, selected)
            else:
                result = {key: None for key in metrics}
                details = {
                    "label_counts": {key: dict(value, evaluated_components=0) for key, value in lab.counts.items()}
                }
            summary["accuracy_slices"].append(
                dict(group=series["name"], metrics=result, counts=details["label_counts"])
            )
        summary["predicted"] = [
            float(values["energy"][row["terms"][0][0]] * 1000) if reference is not None else None
            for row, reference in zip(summary["rows"], summary["reference"])
        ]
        summary["errors"] = [None if r is None else p - r for r, p in zip(summary["reference"], summary["predicted"])]
        return Evaluation(metrics, summary)


def suggest_response_settings(frames):
    """Suggest editable metadata mappings; never infer an independent holdout."""
    if not frames:
        return ResponseSpec().to_dict()
    common = set.intersection(*(set(a.info) for a in frames))
    observables = {a.info.get("observable") for a in frames}
    observable = next(iter(observables)) if len(observables) == 1 else None
    group = ("geometry",) if "geometry" in common else ("family",) if "family" in common else ()
    kwargs = dict(group_fields=group)
    if observable == "MAE":
        kwargs.update(
            mode="difference",
            state_fields=("direction", "plane", "angle_deg")
            if any("angle_deg" in a.info for a in frames)
            else ("direction",),
            reference="p" if any(a.info.get("direction") == "p" for a in frames) else "",
            energy_unit="μeV",
        )
        from NepTrainKit.core.magnetism import existing_moment_vectors

        moments = existing_moment_vectors(frames[0], lift_scalar=False)
        elements = (
            {symbol for symbol, m in zip(frames[0].get_chemical_symbols(), moments) if np.dot(m, m) > 1e-14}
            if moments is not None
            else set()
        )
        if len(elements) == 1:
            kwargs.update(normalization="element", element=next(iter(elements)))
    elif observable == "JD":
        kwargs.update(
            mode="four_state",
            group_fields=("geometry", "theta_deg", "axis_i", "axis_j", "site_i", "site_j"),
            state_fields=("sign_i", "sign_j"),
            normalization="total",
        )
    elif "path" in common and "family" in common:
        kwargs.update(
            mode="difference",
            group_fields=("path", "image") + (("gamma",) if "gamma" in common else ()),
            state_fields=("family",),
            reference="Gp" if any(a.info["family"] == "Gp" for a in frames) else "",
        )
    elif "moment_ratio" in common and "volume_ratio" in common:
        volumes = {a.info["volume_ratio"] for a in frames}
        moments = {a.info["moment_ratio"] for a in frames}
        kwargs.update(
            x_field="moment_ratio" if len(volumes) == 1 else "volume_ratio",
            second_axis="moment_ratio" if len(volumes) > 1 and len(moments) > 1 else "",
        )
    elif "state" in common:
        kwargs.update(x_field="state", group_fields=())
    else:
        kwargs.update(mode="slices")
    return ResponseSpec(**kwargs).to_dict()
