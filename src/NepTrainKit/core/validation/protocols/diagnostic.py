"""Unlabelled, reproducible response experiments. No reference accuracy is inferred."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json

import numpy as np
from ase.io import write

from NepTrainKit.core.magnetism import existing_moment_vectors, set_initial_magmoms_safe
from NepTrainKit.core.magnetic_response import MagneticResponseScanOperation, LocalMagneticResponseParams
from .base import Evaluation, Issue, Metric, ValidationProtocol

CONFIG_KEY = "ntk_diagnostic"
KINDS = {
    "volume": ("体积响应", "ΔV/V (%)"),
    "displacement": ("局部位移", "位移 (Å)"),
    "spin_angle": ("局部磁矩转角", "转角 (°)"),
    "spin_magnitude": ("局部磁矩幅值", "磁矩缩放系数"),
}


class UnsupportedDiagnostic(ValueError):
    pass


@dataclass(frozen=True)
class DiagnosticSpec:
    kind: str = "volume"
    low: float = -8.0
    high: float = 8.0
    points: int = 17
    atom: int = 0  # zero-based internally; the editor uses one-based indices
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    derivative_step: float = 0.1  # in the plotted coordinate's unit
    abs_tolerance: float = 0.000001  # total-energy derivative, eV / coordinate unit
    rel_tolerance: float = 0.02
    zero_field: bool = False
    no_soc: bool = False
    variable_moments: bool = False

    @property
    def origin(self):
        return 1.0 if self.kind == "spin_magnitude" else 0.0

    def coordinates(self):
        return np.unique(np.append(np.linspace(self.low, self.high, self.points), self.origin))

    def validate(self, seed):
        if self.kind not in KINDS:
            raise ValueError("请选择支持的诊断类型。")
        numbers = [self.low, self.high, self.derivative_step, self.abs_tolerance, self.rel_tolerance, *self.axis]
        if not np.isfinite(numbers).all() or len(self.axis) != 3 or np.linalg.norm(self.axis) < 1e-10:
            raise ValueError("参数和方向必须是有限数值，方向不能为零。")
        if not self.low < self.high or not self.low <= self.origin <= self.high:
            raise ValueError("扫描范围须包含初始状态，且下限小于上限。")
        if not isinstance(self.points, int) or not 3 <= self.points <= 101:
            raise ValueError("采样点数须为 3–101。")
        if self.derivative_step <= 0 or self.abs_tolerance <= 0 or not 0 <= self.rel_tolerance <= 1:
            raise ValueError("差分步长和绝对容差须为正，相对容差须在 0–1 内。")
        if not len(seed) or not np.isfinite(seed.positions).all() or not np.isfinite(seed.cell).all():
            raise ValueError("输入结构为空或包含非有限坐标。")
        if np.any(seed.pbc) and abs(np.linalg.det(seed.cell)) < 1e-10:
            raise ValueError("周期结构需要非奇异晶胞。")
        if not isinstance(self.atom, int) or not 0 <= self.atom < len(seed):
            raise ValueError("目标原子编号超出范围。")
        if self.kind == "volume" and (abs(np.linalg.det(seed.cell)) < 1e-10 or min(self.low, -self.derivative_step) <= -100):
            raise ValueError("体积扫描需要有效晶胞，体积缩放比例须大于零。")
        if self.kind.startswith("spin"):
            moments = existing_moment_vectors(seed, lift_scalar=False)
            if moments is None or not np.isfinite(moments).all() or np.linalg.norm(moments[self.atom]) < 1e-10:
                raise ValueError("磁性诊断需要明确的三维磁矩；目标原子磁矩不能为零。")
            if self.kind == "spin_angle" and np.linalg.norm(np.cross(moments[self.atom], self.axis)) < 1e-10:
                raise ValueError("旋转轴与目标磁矩平行，请更换旋转轴。")
        if self.kind == "spin_magnitude" and (self.low < 0 or self.derivative_step >= 1):
            raise ValueError("磁矩幅值缩放须非负，差分步长须小于 1。")


def clean_seed(atoms):
    """Keep geometry and explicitly supplied moments; never turn predictions into labels."""
    seed = atoms.copy()
    seed.calc = None
    seed.set_constraint()
    moments = existing_moment_vectors(atoms, lift_scalar=False)
    for key in list(seed.arrays):
        if key not in ("positions", "numbers"):
            del seed.arrays[key]
    seed.info.clear()
    if moments is not None:
        set_initial_magmoms_safe(seed, moments)
    return seed


def encode_experiment(seed, spec):
    spec.validate(seed)
    seed = clean_seed(seed)
    seed.info[CONFIG_KEY] = asdict(spec)
    stream = io.StringIO()
    write(stream, [seed], format="extxyz")
    return stream.getvalue().encode()


def experiment(frames):
    if len(frames) != 1:
        raise ValueError("每项诊断实验须有一个固定初始结构。")
    raw = frames[0].info.get(CONFIG_KEY)
    if isinstance(raw, str):
        raw = json.loads(raw)
    spec = DiagnosticSpec(**raw) if isinstance(raw, dict) else None
    if spec is None:
        raise ValueError("缺少诊断参数。请通过新建无参考诊断创建实验。")
    spec.validate(frames[0])
    return clean_seed(frames[0]), spec


def scan_frames(seed, spec, coordinates):
    """Reuse card magnetic transformations; return exactly the requested coordinates."""
    axis = np.asarray(spec.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    if spec.kind.startswith("spin"):
        # The shared card operation requires a complete group containing its origin.
        coords = np.unique(np.r_[coordinates, spec.origin])
        if len(coords) < 3:
            coords = np.unique(np.r_[coords, spec.origin - spec.derivative_step, spec.origin + spec.derivative_step])
        kwargs = dict(target_mode="Explicit indices", target_indices=str(spec.atom + 1),
                      rotation_axis=tuple(axis), max_outputs=110)
        if spec.kind == "spin_angle":
            params = LocalMagneticResponseParams(response_kind="Single-spin tilt", coordinate_scan_deg=",".join(map(str, coords)), **kwargs)
        else:
            params = LocalMagneticResponseParams(response_kind="Moment magnitude", moment_scale_scan=",".join(map(str, coords)), **kwargs)
        generated = MagneticResponseScanOperation().generate_local(seed, params)
        unit = 180 / np.pi if spec.kind == "spin_angle" else 1
        actual = [float(frame.info["response_coordinate"]) * unit + spec.origin for frame in generated]
        return [clean_seed(generated[int(np.argmin(np.abs(np.asarray(actual) - x)))]) for x in coordinates]
    frames = []
    for x in coordinates:
        frame = clean_seed(seed)
        if spec.kind == "volume":
            frame.set_cell(seed.cell * np.cbrt(1 + x / 100), scale_atoms=True)
        else:
            frame.positions[spec.atom] += x * axis
        frames.append(frame)
    return frames


def _energies(prediction, count):
    values = np.asarray(prediction.energy, dtype=float).reshape(-1)
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError("模型返回的能量数量不符或包含非有限值。")
    return values


def _derivatives(prediction, frames, spec):
    n = len(frames[0])
    axis = np.asarray(spec.axis, dtype=float) / np.linalg.norm(spec.axis)
    if spec.kind == "volume":
        raw = getattr(prediction, "structure_virials", None)
        if raw is None:
            return None
        virial = np.asarray(raw).reshape(len(frames), 3, 3)
        # Adapter structure_virials are mean per atom, while finite differences use total E.
        return -n * np.trace(virial, axis1=1, axis2=2) / (300 * (1 + spec.coordinates() / 100))
    if spec.kind == "displacement":
        raw = getattr(prediction, "forces", None)
        return None if raw is None else -np.asarray(raw).reshape(len(frames), n, 3)[:, spec.atom] @ axis
    raw = getattr(prediction, "mforces", None)
    if raw is None:
        return None
    mforce = np.asarray(raw).reshape(len(frames), n, 3)[:, spec.atom]
    moments = np.asarray([existing_moment_vectors(frame, lift_scalar=False)[spec.atom] for frame in frames])
    if spec.kind == "spin_angle":
        tangent = np.cross(axis, moments) * np.pi / 180
    else:
        origin_index = int(np.argmin(abs(spec.coordinates() - 1)))
        tangent = np.broadcast_to(moments[origin_index], moments.shape)
    return -np.einsum("ij,ij->i", mforce, tangent)


class DiagnosticProtocol(ValidationProtocol):
    kind = "diagnostic"
    title = "无参考诊断"
    supports_spin = True
    metrics = (Metric("energy_span", "响应能量跨度", "meV/atom", better="none", headline=True),
               Metric("check_failures", "一致性检查异常", "", better="none", headline=True))

    def check(self, frames):
        try:
            experiment(frames)
            return []
        except (ValueError, TypeError, KeyError) as exc:
            return [Issue("error", str(exc))]

    def summarize_reference(self, frames):
        seed, spec = experiment(frames)
        return {"formula": seed.get_chemical_formula(), "atoms": len(seed), "kind": spec.kind,
                "title": KINDS[spec.kind][0], "x_label": KINDS[spec.kind][1],
                "coordinates": spec.coordinates().tolist(), "spec": asdict(spec), "reference": None}

    def evaluate(self, predictor, frames):
        seed, spec = experiment(frames)
        if spec.kind.startswith("spin") and not getattr(predictor, "is_spin_model", False):
            raise UnsupportedDiagnostic("该势不支持磁性输入，无法运行磁矩响应。")
        if getattr(predictor, "is_spin_model", False) and existing_moment_vectors(seed, lift_scalar=False) is None:
            raise UnsupportedDiagnostic("磁性势需要明确的三维磁矩；不会自动填入默认磁矩。")
        if spec.kind == "spin_magnitude" and not spec.variable_moments:
            raise UnsupportedDiagnostic("尚未声明模型支持可变磁矩幅值，已跳过幅值实验。")
        xs = spec.coordinates()
        structures = scan_frames(seed, spec, xs)
        prediction = predictor.predict(structures)
        energies = _energies(prediction, len(xs))
        derivatives = _derivatives(prediction, structures, spec)
        if derivatives is not None and not np.isfinite(derivatives).all():
            raise ValueError("模型返回非有限力、维里或磁力。")
        checks = []
        origin = int(np.argmin(abs(xs - spec.origin)))
        h = spec.derivative_step
        probes = [spec.origin + s * h / divisor for divisor in (1, 2, 4) for s in (-1, 1)]
        probe_e = _energies(predictor.predict(scan_frames(seed, spec, probes)), 6)
        numeric = [(probe_e[2*i+1] - probe_e[2*i]) / (2*h/divisor) for i, divisor in enumerate((1, 2, 4))]
        if not np.isfinite(numeric).all():
            raise ValueError("差分导数非有限，请调整步长或缩小扫描范围。")
        drift = abs(numeric[-1] - numeric[-2])
        tolerance = spec.abs_tolerance + spec.rel_tolerance * abs(numeric[-1])
        analytic = None if derivatives is None else float(derivatives[origin])
        if analytic is None:
            status, reason = "not_applicable", "后端未提供该坐标对应的力、维里或磁力。"
        elif drift > tolerance:
            status, reason = "undetermined", "减半步长后导数尚未收敛；请调整步长并重跑。"
        else:
            tolerance = spec.abs_tolerance + spec.rel_tolerance * max(abs(numeric[-1]), abs(analytic))
            status = "pass" if abs(numeric[-1] - analytic) <= tolerance + drift else "fail"
            reason = "初始状态的总能量中心差分与模型解析导数比较；不判断外部准确性。"
        checks.append(dict(name="能量—广义力一致性", status=status, reason=reason, analytic=analytic,
                           numeric=numeric, steps=[h, h/2, h/4], drift=drift, tolerance=tolerance,
                           unit="eV / " + KINDS[spec.kind][1]))
        # Translational invariance is valid for these position-independent NEP models.
        shifted = clean_seed(seed)
        shifted.positions += np.array([0.173, -0.219, 0.137])
        e_shift = _energies(predictor.predict([shifted]), 1)[0]
        delta = float(abs(e_shift - energies[origin]) / len(seed))
        checks.append(dict(name="整体平移不变性", status="pass" if delta <= 1e-5 else "fail",
                           reason="固定晶胞、磁矩不变，平移所有原子；容差 1e-5 eV/atom。", residual=delta, unit="eV/atom"))
        if spec.kind.startswith("spin"):
            moments = existing_moment_vectors(seed, lift_scalar=False)
            for title, enabled, reason, transform in (
                ("时间反演对称性", spec.zero_field, "仅在零外场且无显式时间反演破缺项的假设下检查。", -moments),
                ("整体自旋旋转不变性", spec.zero_field and spec.no_soc,
                 "仅在无 SOC、零外场、无自旋各向异性项的假设下检查。", moments @ np.array([[0,1,0],[0,0,1],[1,0,0]])),
            ):
                if not enabled:
                    checks.append(dict(name=title, status="not_applicable", reason="未声明适用条件。" + reason))
                    continue
                transformed = clean_seed(seed)
                set_initial_magmoms_safe(transformed, transform)
                e = _energies(predictor.predict([transformed]), 1)[0]
                delta = float(abs(e - energies[origin]) / len(seed))
                checks.append(dict(name=title, status="pass" if delta <= 1e-5 else "fail",
                                   reason=reason + " 容差 1e-5 eV/atom。", residual=delta, unit="eV/atom"))
        response = (energies - energies[origin]) * 1000 / len(seed)
        return Evaluation(
            {"energy_span": float(np.ptp(response)), "check_failures": sum(c["status"] == "fail" for c in checks)},
            {"kind": spec.kind, "x": xs.tolist(), "energy": energies.tolist(), "relative_energy": response.tolist(),
             "derivative": None if derivatives is None else derivatives.tolist(), "checks": checks,
             "spec": asdict(spec), "reference": None, "origin_index": origin,
             "note": "仅为模型响应和内部一致性检查；未验证外部准确性。负曲率可能是初始状态不稳定，不自动判错。"},
        )
