"""Configuration-type distribution evidence."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .extract import StructureAuditRecord
from .result import AuditDimension, AuditSlice, AuditStatus


def audit_config_types(
    records: Sequence[StructureAuditRecord],
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    """Expose real Config_type groups without treating rarity as a defect."""
    if not records:
        return (
            AuditDimension(
                "config_types",
                "Configuration types",
                AuditStatus.UNAVAILABLE,
                "No structures are loaded.",
            ),
            (),
            {"group_count": 0, "labeled_count": 0, "missing_count": 0},
        )

    groups: dict[str, list[int]] = defaultdict(list)
    missing_indices: list[int] = []
    for record in records:
        label = str(record.config_type or "").strip()
        if label:
            groups[label].append(record.index)
        else:
            missing_indices.append(record.index)

    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    labels = tuple(label for label, _ in ordered)
    counts = tuple(len(indices) for _, indices in ordered)
    structure_indices = tuple(tuple(indices) for _, indices in ordered)
    plots = ()
    if ordered:
        plots = (
            {
                "kind": "categorical_bars",
                "id": "config_types:distribution",
                "title": "Configuration-type distribution",
                "x_label": "Configuration type",
                "y_label": "Structures",
                "series": (
                    {
                        "id": "config_type",
                        "label": "Config_type",
                        "labels": labels,
                        "counts": counts,
                        "structure_indices": structure_indices,
                    },
                ),
                "missing_count": len(missing_indices),
                "missing_structure_indices": tuple(missing_indices),
            },
        )

    if not ordered:
        status = AuditStatus.UNAVAILABLE
        reason = "Config_type metadata is not available."
    elif missing_indices:
        status = AuditStatus.PARTIAL
        reason = (
            f"Config_type is available for {len(records) - len(missing_indices)} of "
            f"{len(records)} structures."
        )
    else:
        status = AuditStatus.AVAILABLE
        reason = ""

    return (
        AuditDimension(
            "config_types",
            "Configuration types",
            status,
            reason,
            plots=plots,
        ),
        (),
        {
            "group_count": len(groups),
            "labeled_count": len(records) - len(missing_indices),
            "missing_count": len(missing_indices),
            "groups": {label: len(indices) for label, indices in ordered},
        },
    )
