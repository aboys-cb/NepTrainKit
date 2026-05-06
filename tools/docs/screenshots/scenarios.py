"""Scenario setup functions for documentation screenshots."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from NepTrainKit.main import create_app, create_main_window


@dataclass
class ScenarioContext:
    """Runtime objects shared by screenshot scenarios."""

    app: QApplication
    window: object
    repo_root: Path


def pump_events(app: QApplication, cycles: int = 80, delay_ms: int = 5) -> None:
    """Let Qt finish layout, paint, and pending signal delivery."""
    for _ in range(cycles):
        app.processEvents()
        if delay_ms > 0:
            QThread.msleep(delay_ms)


def wait_until(app: QApplication, predicate, *, cycles: int = 1200, delay_ms: int = 5) -> bool:
    """Process Qt events until predicate returns true or the cycle budget ends."""
    for _ in range(cycles):
        app.processEvents()
        if predicate():
            return True
        if delay_ms > 0:
            QThread.msleep(delay_ms)
    return False


def create_context(repo_root: Path, window_size: tuple[int, int]) -> ScenarioContext:
    """Create a configured main window without entering the Qt event loop."""
    app = create_app(["capture-ui"])
    window = create_main_window(show=True)
    window.resize(*window_size)
    pump_events(app, 80)
    return ScenarioContext(app=app, window=window, repo_root=repo_root)


def prepare_nep_demo_data(ctx: ScenarioContext) -> Path:
    """Copy the tracked NEP fixture to a scratch directory before loading it."""
    source_dir = ctx.repo_root / "tests/data/nep"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing NEP demo fixture: {source_dir}")

    work_dir = ctx.repo_root / ".tmp/docs-screenshots/fixtures/nep"
    resolved_tmp = work_dir.resolve()
    resolved_root = (ctx.repo_root / ".tmp/docs-screenshots").resolve()
    if resolved_root not in resolved_tmp.parents:
        raise RuntimeError(f"Refusing to replace unexpected screenshot fixture path: {work_dir}")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.xyz", "nep.txt", "descriptor.npy", "energy.npy", "forces.npy", "virial.npy"):
        shutil.copy2(source_dir / name, work_dir / name)
    return work_dir / "train.xyz"


def show_nep_overview(ctx: ScenarioContext) -> None:
    """Prepare the default NEP Dataset Display page."""
    ctx.window.switchTo(ctx.window.show_nep_interface)
    pump_events(ctx.app, 100)
    data_path = prepare_nep_demo_data(ctx)
    ctx.window.show_nep_interface.set_work_path(str(data_path))

    def loaded() -> bool:
        data = getattr(ctx.window.show_nep_interface, "nep_result_data", None)
        return bool(data is not None and getattr(data, "load_flag", False))

    if not wait_until(ctx.app, loaded, cycles=1600):
        raise RuntimeError(f"Timed out loading demo dataset: {data_path}")
    pump_events(ctx.app, 220)


def make_data_empty(ctx: ScenarioContext) -> None:
    """Prepare an empty Make Data workspace."""
    ctx.window.switchTo(ctx.window.make_data_interface)
    pump_events(ctx.app, 100)


def make_data_lattice_strain(ctx: ScenarioContext) -> None:
    """Prepare a Make Data quickstart scene with a configured strain card."""
    make_data_empty(ctx)
    card = ctx.window.make_data_interface.add_card("CellStrainCard")
    if card is None:
        raise RuntimeError("CellStrainCard is not registered")
    card.strain_x_frame.set_input_value([-2, 2, 1])
    card.strain_y_frame.set_input_value([-2, 2, 1])
    card.strain_z_frame.set_input_value([-2, 2, 1])
    pump_events(ctx.app, 100)


RUNNERS: dict[str, Callable[[ScenarioContext], None]] = {
    "show_nep_overview": show_nep_overview,
    "make_data_empty": make_data_empty,
    "make_data_lattice_strain": make_data_lattice_strain,
}
