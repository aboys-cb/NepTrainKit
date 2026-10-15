"""Reproducible, label-free sampling quality benchmark (not a unit test).

Run as ``python -m tools.benchmark_sampling_toys --output /tmp/sampling-toys
--model path/to/nep.txt [--spin-model path/to/spin-nep.txt]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

# Fix numerical-library concurrency before importing the scientific stack.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from NepTrainKit.core.calculator import NepCalculator  # noqa: E402
from NepTrainKit.core.io.sampling_coverage import _native_sampling, _without_inputs, select_coverage  # noqa: E402
from NepTrainKit.core.io.sampling_features import build_sampling_feature_blocks  # noqa: E402
from NepTrainKit.core.io.sampling_plan import build_physics_sampling_plan  # noqa: E402
from tools.sampling_toy_data import (  # noqa: E402
    evaluate,
    make_cases,
    make_challenges,
    make_mixed_cases,
    minimum_budget,
    write_case,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output, model, spin_model=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep previous runs reviewable rather than silently replacing evidence.
    if any(output.iterdir()):
        raise ValueError("benchmark output directory must be empty")
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("sampling_toy_data.py")]
    paths += [repo / "src/NepTrainKit/core/io" / (name + ".py") for name in
              ("sampling_coverage", "sampling_features", "sampling_plan", "sampling_states")]
    paths.append(repo / "src/native/sampling/module.cpp")
    results = dict(machine=platform.platform(), threads=dict(OMP=1, OPENBLAS=1),
                   sources={str(p.relative_to(repo)): _hash(p) for p in paths},
                   scope="Coverage only; timings are observations, not isolated speedup claims.",
                   cases={}, skipped=[])
    results["native"] = None if _native_sampling is None else dict(
        path=_native_sampling.__file__, sha256=_hash(_native_sampling.__file__),
        api_version=_native_sampling.coverage_api_version)
    for path in paths:
        snapshot = output / "source" / path.relative_to(repo)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    for variant in range(6):
        factory = (make_cases, make_challenges, make_mixed_cases)[variant//2]
        for frames, truth in factory(variant=variant % 2):
            truth = write_case(output, frames, truth)
            if truth["magnetic"] and spin_model is None:
                results["skipped"].append(truth["name"])
                continue
            # Re-read the actual deliverable: truth arrays and source labels
            # are not attached to structures passed to feature construction.
            frames = read(output / f"{truth['name']}.xyz", index=":")
            chosen_model = Path(spin_model if truth["magnetic"] else model).resolve()
            start = time.perf_counter()
            calc = NepCalculator(str(chosen_model), backend="cpu")
            if calc.is_spin_model != truth["magnetic"]:
                raise ValueError("model magnetic kind does not match toy")
            atomic = calc.descriptors(frames, mean=False)
            offsets = np.r_[0, np.cumsum([len(s) for s in frames])]
            mean = np.asarray([atomic[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:])])
            features = build_sampling_feature_blocks(frames, mean, per_atom_descriptors=atomic,
                                                     spin_model=truth["magnetic"])
            plan = build_physics_sampling_plan(frames, spin_model=truth["magnetic"])
            case = dict(model=str(chosen_model), model_sha256=_hash(chosen_model),
                        xyz_sha256=truth["sha256"], preparation_seconds=time.perf_counter()-start,
                        methods={})
            for name, blocks in (("descriptor_fps", _without_inputs(features)),
                                 ("default", features)):
                start = time.perf_counter()
                selection = select_coverage(blocks, plan, budget=truth["budget"])
                seconds = time.perf_counter()-start
                case["methods"][name] = dict(evaluate(truth, selection.indices),
                                              indices=list(selection.indices), seconds=seconds)
            case["methods"]["reference"] = evaluate(truth, truth["oracle_indices"])
            if not case["methods"]["reference"]["passed"]:
                raise RuntimeError("declared reference failed its own independent criteria")
            case["minimum_budget"] = minimum_budget(truth)
            case["random"] = [evaluate(truth, np.random.default_rng(seed).choice(
                len(frames), truth["budget"], replace=False)) for seed in range(20)]
            results["cases"][truth["name"]] = case
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            current = case["methods"]["default"]
            print(truth["name"], "default", current["passed"], current["gap_max"],
                  f"{current['cells_covered']}/{current['cells_total']}",
                  "min budget", case["minimum_budget"], flush=True)
    _report(output, results)
    return results


def _report(output, results):
    lines = ["# 输入结构采样 toy 基准", "",
             "同一案例内所有方法使用相同数据、模型和预算。生成的 XYZ 无 DFT 标签；"
             "真值只在采样完成后评分。transfer 案例预先改变了网格、预算、物理范围和非磁元素，"
             "属于迁移检查，不宣称盲测。另列弯曲路径、缺格、热扰动、相同晶胞局域差异和独立热扰动五类挑战及其迁移案例，"
             "保留预先定义的参考、预算和失败。", "",
             "另加混合路径密度与路径/离散状态四项预算任务。default 是固定数量采样的默认算法："
             "区域见证与物理分层的最小精确覆盖，剩余名额由不加权多视图 FPS 补足；"
             "descriptor_fps 去掉输入视图作对照。", "",
             "格子覆盖、最大缺口和端点分别报告。格子边界是人为预定义的覆盖任务，"
             "不能把格子数直接解释成训练精度；距离更小也不保证每格都有代表。", "",
             "| 案例 | 方法 | 帧数 | 最大缺口 | 格子 | 全标准通过 |",
             "|---|---|---:|---:|---:|---|"]
    for name, case in results["cases"].items():
        for method, score in case["methods"].items():
            gap = "缺失" if score["gap_max"] is None else f"{score['gap_max']:.4f}"
            lines.append(f"| {name} | {method} | {score['count']} | {gap} | "
                         f"{score['cells_covered']}/{score['cells_total']} | {score['passed']} |")
    lines += ["", "## 预算是否可行", "",
              "对有限 toy，用独立的整数覆盖问题求满足既定评估标准的最少帧数。", "",
              "| 案例 | 独立标准最少帧数 |",
              "|---|---:|"]
    for name, case in results["cases"].items():
        a = case["minimum_budget"]
        lines.append(f"| {name} | {a.get('minimum', a['status'])} |")
    lines += ["", "精确求解只用于这些小型 toy 的诊断，不进入生产采样，也不读取真实数据的标签。"
              "缺口按归一化生成坐标计算；稀有分支案例在各分支内单独计算。"
              "分项失败保留在结果中，不改变阈值，不以测试通过替代科学验收。", "",
              "完整选择、20 个随机对照、源码/模型/数据哈希和耗时见 [results.json](results.json)。"
              "耗时是运行记录，未做隔离重复性能对照。"]
    if results["skipped"]:
        lines += ["", "未提供磁模型，跳过：" + ", ".join(results["skipped"])]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spin-model", type=Path)
    args = parser.parse_args()
    run(args.output, args.model, args.spin_model)
