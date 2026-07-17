#!/usr/bin/env python
"""Run ChemicalMotifIdentifier using an official example _frameworks.py."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _missing_modules(names: list[str]) -> list[str]:
    missing = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def _load_frameworks(path: Path):
    spec = importlib.util.spec_from_file_location("cmi_example_frameworks", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_official_inputs(example_dir: Path, crystal_structure: str) -> None:
    inputs_dir = example_dir / "data" / f"inputs_{example_dir.name}"
    missing = [
        path
        for path in [inputs_dir / "net.pt", inputs_dir / f"df_{crystal_structure}.pkl"]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise SystemExit(
            "Missing official ChemicalMotifIdentifier example inputs. "
            "Run the example notebook gdown.download_folder(...) step first.\n"
            f"{missing_text}"
        )


def _patch_old_pyg_transforms() -> None:
    from chemicalmotifidentifier._src import transforms

    for name in ["RemoveCentralNode", "AddEdges", "ModuloConcentration"]:
        cls = getattr(transforms, name, None)
        if cls is None or "forward" in cls.__dict__:
            continue
        cls.forward = cls.__call__
        cls.__abstractmethods__ = frozenset()


def _patch_torch_load_weights_only() -> None:
    import torch

    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load


def _patch_cmi_dataloader_workers() -> None:
    from chemicalmotifidentifier._src import ml
    from torch_geometric.loader import DataLoader

    def create_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    ml.ModelInference._create_data_loader = create_data_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", required=True, type=Path, help="Official examples/.../_frameworks.py")
    parser.add_argument("--dump-file", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path, help="CMI output/cache directory")
    parser.add_argument("--crystal-structure", default="fcc", choices=["fcc", "bcc", "hcp"])
    parser.add_argument("--frame-number", type=int, default=0)
    args = parser.parse_args(argv)

    missing = _missing_modules(["torch_geometric", "torch_scatter"])
    if missing:
        raise SystemExit(f"Missing ChemicalMotifIdentifier runtime modules: {', '.join(missing)}")

    frameworks_path = args.frameworks.resolve()
    dump_file = args.dump_file.resolve()
    root = args.root.resolve()
    cwd = Path.cwd()
    try:
        os.chdir(frameworks_path.parent)
        _check_official_inputs(frameworks_path.parent, args.crystal_structure)
        frameworks = _load_frameworks(frameworks_path)
        _patch_old_pyg_transforms()
        _patch_torch_load_weights_only()
        _patch_cmi_dataloader_workers()
        cls = frameworks.MonteCarloChemicalMotifIdentifier
        eca = cls(crystal_structure=args.crystal_structure)
        df = eca.predict(root=str(root), dump_file=str(dump_file), frame_number=args.frame_number)
        root.mkdir(parents=True, exist_ok=True)
        if hasattr(eca, "central_atoms"):
            import numpy as np

            np.save(root / "central_atoms.npy", eca.central_atoms)
        if hasattr(eca, "concentration_before_permutation"):
            import numpy as np

            np.save(root / "concentration_before_permutation.npy", eca.concentration_before_permutation)
        if "shell_ID" in df.columns:
            import numpy as np

            np.save(root / "shell_ids.npy", df["shell_ID"].to_numpy())
        df.to_pickle(root / "df_microstates.pkl")
    finally:
        os.chdir(cwd)
    print(root / "df_microstates.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
