"""Training Set Audit orchestration."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from .composition import audit_composition
from .extract import indexed_structures_from_result_data, records_from_result_data
from .label_ranges import audit_label_ranges
from .local_chemistry import audit_local_chemistry
from .nep_cutoff import parse_nep_cutoff
from .pair_contacts import PairContactCollector
from .result import AuditDimension, AuditResult, AuditStatus


def build_training_set_audit(result_data: Any, *, dataset_id: str = "current") -> AuditResult:
    records = records_from_result_data(result_data)
    dimensions = []
    slices = []
    overview: dict[str, object] = {"structures": len(records)}

    for run_dimension in (audit_composition, audit_label_ranges):
        dimension, dimension_slices, dimension_overview = run_dimension(records)
        dimensions.append(dimension)
        slices.extend(dimension_slices)
        overview[dimension.id] = dimension_overview

    local_overview: dict[str, object] = {
        "available_scopes": (),
        "center_element_count": 0,
        "sparse_bin_count": 0,
    }
    pair_overview: dict[str, object] = {
        "pair_count": 0,
        "co_sampled_pair_count": 0,
        "zero_contact_pair_count": 0,
    }
    model_path = getattr(result_data, "nep_txt_path", None)
    if model_path is None or str(model_path).strip() == "":
        local_dimension = AuditDimension(
            "local_chemistry",
            "Local chemistry",
            AuditStatus.UNAVAILABLE,
            "No active NEP model file is available.",
        )
        local_slices = ()
    else:
        try:
            profile = parse_nep_cutoff(model_path)
            pair_collector = PairContactCollector(profile)
            local_dimension, local_slices, local_overview = audit_local_chemistry(
                indexed_structures_from_result_data(result_data),
                profile,
                pair_contact_collector=pair_collector,
            )
            pair_dimension, pair_slices, pair_overview = pair_collector.finalize()
        except ValueError as exc:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                str(exc) or "The local chemistry input is invalid.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, str(exc))
            pair_slices = ()
        except OSError:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                "The active NEP model file could not be read.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, "The active NEP model file could not be read.")
            pair_slices = ()
        except Exception:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                "Local chemistry could not be audited from the active data.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, "Pair contacts could not be audited from the active data.")
            pair_slices = ()
    if model_path is None or str(model_path).strip() == "":
        pair_dimension = AuditDimension(
            "pair_contacts",
            "Pair contacts",
            AuditStatus.UNAVAILABLE,
            "No active NEP model file is available.",
        )
        pair_slices = ()
    dimensions.append(local_dimension)
    slices.extend(local_slices)
    overview[local_dimension.id] = local_overview
    dimensions.append(pair_dimension)
    slices.extend(pair_slices)
    overview[pair_dimension.id] = pair_overview

    counts = Counter(audit_slice.severity for audit_slice in slices)
    overview.update(
        {
            "finding_count": len(slices),
            "severity_counts": {severity.value: count for severity, count in counts.items()},
            "label_counts": {
                "energy": sum(record.energy_per_atom is not None and isfinite(record.energy_per_atom) for record in records),
                "force": sum(record.max_force is not None and isfinite(record.max_force) for record in records),
                "virial": sum(record.virial_norm is not None and isfinite(record.virial_norm) for record in records),
            },
        }
    )

    return AuditResult(
        dataset_id=dataset_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={"structure_count": len(records)},
        dimensions=tuple(dimensions),
        slices=tuple(slices),
        overview_metrics=overview,
    )
