"""Per-property display choices shared by validation navigation, plots and scorecards.

The evaluation protocol owns numbers, units and metric availability. A presentation
chooses how to label an item, identify comparable branches and draw saved results.
"""

from dataclasses import dataclass
from typing import Callable

from NepTrainKit.ui.widgets.diagnostic_plot import redraw_diagnostics
from NepTrainKit.ui.widgets.eos_plot import eos_group_title, preview_eos, redraw_eos
from NepTrainKit.ui.widgets.reference_response_plot import is_magnetic_eos, redraw_reference_responses
from NepTrainKit.core.validation.response_plot import is_mae


@dataclass(frozen=True)
class PropertyPresentation:
    badge: str
    draw_groups: Callable | None = None
    source_group: str | None = None
    group_title: Callable | None = None
    choose_table_metric: bool = False
    import_label: str = ""
    draw_import_preview: Callable | None = None

    def item_badge(self, item):
        if is_magnetic_eos(item):
            return "mEOS"
        if item["protocol"] == "response" and is_mae(item.get("summary") or {}):
            return "MAE"
        return self.badge

    def group(self, item):
        return (item.get("source_ref") or {}).get(self.source_group) or {}

    def family(self, item):
        group = self.group(item)
        if group:
            return item["protocol"], item["object_hash"], group["field"]
        return None

    def item_title(self, item):
        name = item["name"].rsplit(" / ", 1)[-1]
        group = self.group(item)
        if self.group_title and group and name == str(group["value"]):
            name, _ = self.group_title(name, (item.get("summary") or {}).get("formula"))
        return name


_PRESENTATIONS = {
    "eos": PropertyPresentation(
        "EOS", redraw_eos, source_group="eos_group", group_title=eos_group_title,
        choose_table_metric=True, import_label="EOS · Energy–volume curve", draw_import_preview=preview_eos,
    ),
    "efv": PropertyPresentation("EFV", import_label="Energy / force / virial errors"),
    "response": PropertyPresentation(
        "Resp", redraw_reference_responses, import_label="Reference response · scans / differences / combinations",
    ),
    "diagnostic": PropertyPresentation("Diag", redraw_diagnostics),
}


def uses_curve_chart(item):
    return item["protocol"] == "eos" or (item["protocol"] == "response"
                                      and item["summary"]["analysis"]["mode"] != "slices")


def presentation_for(kind):
    return _PRESENTATIONS[kind]


def file_import_choices():
    return [(kind, view.import_label) for kind, view in _PRESENTATIONS.items() if view.import_label]
