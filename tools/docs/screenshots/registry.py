"""Screenshot scenario registry for documentation assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_WINDOW_SIZE = (1200, 700)
DEFAULT_OUTPUT_DIR = Path("docs/source/_static/image/generated")


@dataclass(frozen=True)
class Annotation:
    """A visual callout drawn on top of a captured UI screenshot."""

    number: str
    label: str
    target: str | tuple[int, int, int, int]
    badge: str | tuple[int, int] | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative metadata for a documentation screenshot."""

    name: str
    title: str
    runner: str
    output: Path
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
    annotations: tuple[Annotation, ...] = ()
    description: str = ""
    options: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, ScenarioSpec] = {
    "show_nep_overview": ScenarioSpec(
        name="show_nep_overview",
        title="NEP Dataset Display overview",
        runner="show_nep_overview",
        output=DEFAULT_OUTPUT_DIR / "show_nep_overview.png",
        description="Main NEP Dataset Display window with top-level regions annotated.",
        annotations=(
            Annotation("1", "Open data", "widget:open_dir_button", "right"),
            Annotation("2", "Error plots", "widget:show_nep_interface.plot_widget", "top-right"),
            Annotation("3", "Structure viewer", "widget:show_nep_interface.struct_widget", "top-left"),
            Annotation("4", "Search and selection", "widget:show_nep_interface.search_lineEdit", "right"),
        ),
    ),
    "make_data_empty": ScenarioSpec(
        name="make_data_empty",
        title="Make Data workspace",
        runner="make_data_empty",
        output=DEFAULT_OUTPUT_DIR / "make_data_empty.png",
        description="Empty Make Data workflow page with the add-card entry point annotated.",
        annotations=(
            Annotation("1", "Add new card", "widget:make_data_interface.setting_group.new_card_button", "right"),
            Annotation("2", "Run selected cards", (292, 72, 26, 26), "right"),
            Annotation("3", "Workflow workspace", "widget:make_data_interface.workspace_card_widget", "top-left"),
        ),
    ),
    "make_data_lattice_strain": ScenarioSpec(
        name="make_data_lattice_strain",
        title="Make Data lattice strain quickstart",
        runner="make_data_lattice_strain",
        output=DEFAULT_OUTPUT_DIR / "make_data_lattice_strain.png",
        description="Quickstart scene with a Lattice Strain card configured for small axial strain.",
        annotations=(
            Annotation("1", "Add Lattice Strain card", "widget:make_data_interface.setting_group.new_card_button", "right"),
            Annotation("2", "Set strain range", "widget:make_data_interface.workspace_card_widget.cards.0", "right"),
            Annotation("3", "Run selected cards", (292, 72, 26, 26), "right"),
            Annotation("4", "Export generated structures", "widget:make_data_interface.workspace_card_widget.cards.0.export_button", "right"),
        ),
    ),
}
