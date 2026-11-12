"""Headless execution of exported Make Dataset workflows.

This module runs a workflow JSON produced by the Make Dataset workbench without
importing any Qt widget. It mirrors the desktop execution contract:

1. cards run top to bottom; a failure stops the chain;
2. ``CardGroup`` feeds one shared input to every enabled child and merges the
   results, optionally through its post-merge filter card;
3. ``WorkflowFork`` keeps each branch separate and merges only when enabled.

Every stage keeps its own output, so callers can export one card's result
instead of only the final one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .cards.errors import CardOperationError
from .cards.operation import (
    DatasetOperation,
    GeneratorOperation,
    StructureOperation,
)
from .cards.params import UnsupportedCardFormatError, params_from_dict

__all__ = [
    "CardSpec",
    "CardStage",
    "WorkflowError",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowRun",
    "load_workflow",
    "run_workflow",
]

#: Container cards that arrange other cards instead of transforming structures.
CONTAINER_CARDS = frozenset({"CardGroup", "WorkflowFork"})

#: ``card class -> (operation module, operation name, params module, params name, kind)``
#:
#: Kept as data so the headless path imports only the modules a workflow actually
#: uses. Rows mirror ``create_operation``/``get_params`` in the matching card.
CARD_REGISTRY: dict[str, tuple[str, str, str, str, str]] = {
    "BainPathCard": (
        "NepTrainKit.core.cards.lattice", "BainPathOperation",
        "NepTrainKit.core.cards.lattice", "BainPathParams", "structure",
    ),
    "CellScalingCard": (
        "NepTrainKit.core.cards.lattice", "CellScalingOperation",
        "NepTrainKit.core.cards.lattice", "CellScalingParams", "structure",
    ),
    "CellStrainCard": (
        "NepTrainKit.core.cards.lattice", "CellStrainOperation",
        "NepTrainKit.core.cards.lattice", "CellStrainParams", "structure",
    ),
    "CompositionGradientCard": (
        "NepTrainKit.core.cards.alloy", "CompositionGradientOperation",
        "NepTrainKit.core.cards.alloy", "CompositionGradientParams", "structure",
    ),
    "CompositionSweepCard": (
        "NepTrainKit.core.cards.alloy", "CompositionSweepOperation",
        "NepTrainKit.core.cards.alloy", "CompositionSweepParams", "structure",
    ),
    "ConditionalReplaceCard": (
        "NepTrainKit.core.cards.alloy", "ConditionalReplaceOperation",
        "NepTrainKit.core.cards.alloy", "ConditionalReplaceParams", "structure",
    ),
    "CorrelatedRandomSpinCard": (
        "NepTrainKit.core.cards.magnetism", "CorrelatedRandomSpinOperation",
        "NepTrainKit.core.cards.magnetism", "CorrelatedRandomSpinParams", "structure",
    ),
    "CrystalPrototypeBuilderCard": (
        "NepTrainKit.core.cards.structure", "CrystalPrototypeBuilderOperation",
        "NepTrainKit.core.cards.structure", "CrystalPrototypeBuilderParams", "generator",
    ),
    "FPSFilterDataCard": (
        "NepTrainKit.core.cards.filter", "FPSFilterOperation",
        "NepTrainKit.core.cards.filter", "FPSFilterParams", "dataset",
    ),
    "FiniteCellAlloyOccupancyCard": (
        "NepTrainKit.core.cards.alloy", "FiniteCellAlloyOccupancyOperation",
        "NepTrainKit.core.cards.alloy", "FiniteCellAlloyOccupancyParams", "structure",
    ),
    "FoldedHelixCard": (
        "NepTrainKit.core.cards.magnetism", "FoldedHelixOperation",
        "NepTrainKit.core.cards.magnetism", "FoldedHelixParams", "structure",
    ),
    "GeometryFilterCard": (
        "NepTrainKit.core.cards.filter", "GeometryFilterOperation",
        "NepTrainKit.core.cards.filter", "GeometryFilterParams", "dataset",
    ),
    "GroupLabelCard": (
        "NepTrainKit.core.cards.structure", "GroupLabelOperation",
        "NepTrainKit.core.cards.structure", "GroupLabelParams", "structure",
    ),
    "InsertDefectCard": (
        "NepTrainKit.core.cards.defect", "InsertDefectOperation",
        "NepTrainKit.core.cards.defect", "InsertDefectParams", "structure",
    ),
    "InterfaceLayerMixCard": (
        "NepTrainKit.core.cards.alloy", "InterfaceLayerMixOperation",
        "NepTrainKit.core.cards.alloy", "InterfaceLayerMixParams", "structure",
    ),
    "LayerCopyCard": (
        "NepTrainKit.core.cards.structure", "LayerCopyOperation",
        "NepTrainKit.core.cards.structure", "LayerCopyParams", "structure",
    ),
    "LocalMagneticResponseCard": (
        "NepTrainKit.core.magnetic_response", "MagneticResponseScanOperation",
        "NepTrainKit.core.magnetic_response", "LocalMagneticResponseParams", "structure",
    ),
    "LocalSolvationCard": (
        "NepTrainKit.core.cards.solvation", "LocalSolvationOperation",
        "NepTrainKit.core.cards.solvation", "LocalSolvationParams", "structure",
    ),
    "MagneticMomentRotationCard": (
        "NepTrainKit.core.cards.magnetism", "MagneticMomentRotationOperation",
        "NepTrainKit.core.cards.magnetism", "MagneticMomentRotationParams", "structure",
    ),
    "MagneticOrderCard": (
        "NepTrainKit.core.cards.magnetism", "MagneticOrderOperation",
        "NepTrainKit.core.cards.magnetism", "MagneticOrderParams", "structure",
    ),
    "MagnetoelasticResponseCard": (
        "NepTrainKit.core.magnetic_response", "MagneticResponseScanOperation",
        "NepTrainKit.core.magnetic_response", "MagnetoelasticResponseParams", "structure",
    ),
    "OrderedAlloyPrototypeCard": (
        "NepTrainKit.core.cards.alloy", "OrderedAlloyPrototypeOperation",
        "NepTrainKit.core.cards.alloy", "OrderedAlloyPrototypeParams", "generator",
    ),
    "OrganicMolConfigPBCCard": (
        "NepTrainKit.core.cards.structure", "OrganicMolConfigPBCOperation",
        "NepTrainKit.core.cards.structure", "OrganicMolConfigPBCParams", "structure",
    ),
    "PerturbCard": (
        "NepTrainKit.core.cards.lattice", "PerturbOperation",
        "NepTrainKit.core.cards.lattice", "PerturbParams", "structure",
    ),
    "RandomDopingCard": (
        "NepTrainKit.core.cards.alloy", "RandomDopingOperation",
        "NepTrainKit.core.cards.alloy", "RandomDopingParams", "structure",
    ),
    "RandomOccupancyCard": (
        "NepTrainKit.core.cards.alloy", "RandomOccupancyOperation",
        "NepTrainKit.core.cards.alloy", "RandomOccupancyParams", "structure",
    ),
    "RandomPackingCard": (
        "NepTrainKit.core.cards.structure", "RandomPackingOperation",
        "NepTrainKit.core.cards.structure", "RandomPackingParams", "structure",
    ),
    "RandomSlabCard": (
        "NepTrainKit.core.cards.defect", "RandomSlabOperation",
        "NepTrainKit.core.cards.defect", "RandomSlabParams", "structure",
    ),
    "RandomVacancyCard": (
        "NepTrainKit.core.cards.defect", "RandomVacancyOperation",
        "NepTrainKit.core.cards.defect", "RandomVacancyParams", "structure",
    ),
    "SOCTextureResponseCard": (
        "NepTrainKit.core.magnetic_response", "MagneticResponseScanOperation",
        "NepTrainKit.core.magnetic_response", "TextureMagneticResponseParams", "structure",
    ),
    "SetMagneticMomentsCard": (
        "NepTrainKit.core.cards.magnetism", "SetMagneticMomentsOperation",
        "NepTrainKit.core.cards.magnetism", "SetMagneticMomentsParams", "structure",
    ),
    "ShearAngleCard": (
        "NepTrainKit.core.cards.lattice", "ShearAngleOperation",
        "NepTrainKit.core.cards.lattice", "ShearAngleParams", "structure",
    ),
    "ShearMatrixCard": (
        "NepTrainKit.core.cards.lattice", "ShearMatrixOperation",
        "NepTrainKit.core.cards.lattice", "ShearMatrixParams", "structure",
    ),
    "SmallAngleSpinTiltCard": (
        "NepTrainKit.core.cards.magnetism", "SmallAngleSpinTiltOperation",
        "NepTrainKit.core.cards.magnetism", "SmallAngleSpinTiltParams", "structure",
    ),
    "SolventBoxFillCard": (
        "NepTrainKit.core.cards.solvation", "SolventBoxFillOperation",
        "NepTrainKit.core.cards.solvation", "SolventBoxFillParams", "structure",
    ),
    "SpinDisorderCard": (
        "NepTrainKit.core.cards.magnetism", "SpinDisorderOperation",
        "NepTrainKit.core.cards.magnetism", "SpinDisorderParams", "structure",
    ),
    "SpinSpiralCard": (
        "NepTrainKit.core.cards.magnetism", "SpinSpiralOperation",
        "NepTrainKit.core.cards.magnetism", "SpinSpiralParams", "structure",
    ),
    "StackingFaultCard": (
        "NepTrainKit.core.cards.defect", "StackingFaultOperation",
        "NepTrainKit.core.cards.defect", "StackingFaultParams", "structure",
    ),
    "StrictGSFEPathCard": (
        "NepTrainKit.core.cards.defect", "StrictGSFEPathOperation",
        "NepTrainKit.core.cards.defect", "StrictGSFEPathParams", "structure",
    ),
    "SuperCellCard": (
        "NepTrainKit.core.cards.lattice", "SuperCellOperation",
        "NepTrainKit.core.cards.lattice", "SuperCellParams", "structure",
    ),
    "VacancyDefectCard": (
        "NepTrainKit.core.cards.defect", "VacancyDefectOperation",
        "NepTrainKit.core.cards.defect", "VacancyDefectParams", "structure",
    ),
    "VibrationModePerturbCard": (
        "NepTrainKit.core.cards.structure", "VibrationModePerturbOperation",
        "NepTrainKit.core.cards.structure", "VibrationModePerturbParams", "structure",
    ),
}


class WorkflowError(ValueError):
    """Report a workflow that cannot be parsed, planned, or executed."""

    def __init__(self, message: str, *, card: str | None = None) -> None:
        self.card = card
        super().__init__(f"{card}: {message}" if card else message)


@dataclass(frozen=True)
class CardSpec:
    """One card entry from an exported workflow document."""

    card_class: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    children: tuple["CardSpec", ...] = ()
    branches: tuple[tuple["CardSpec", ...], ...] = ()
    filter_card: "CardSpec | None" = None
    merge_enabled: bool = False

    @property
    def is_container(self) -> bool:
        return self.card_class in CONTAINER_CARDS

    @property
    def kind(self) -> str:
        if self.card_class == "CardGroup":
            return "group"
        if self.card_class == "WorkflowFork":
            return "fork"
        try:
            return CARD_REGISTRY[self.card_class][4]
        except KeyError:
            raise WorkflowError(
                "unknown card class; it may come from a newer version",
                card=self.card_class,
            ) from None

    def requires_input(self) -> bool:
        """Report whether the card cannot produce output without an input set."""
        if self.is_container:
            if self.kind == "fork":
                return any(
                    next((child.requires_input() for child in branch if child.enabled), False)
                    for branch in self.branches
                )
            return any(child.requires_input() for child in self.children if child.enabled)
        return self.kind != "generator"


@dataclass(frozen=True)
class CardStage:
    """One planned execution step, flattened for preview and progress output."""

    index: int
    card_class: str
    label: str
    kind: str
    enabled: bool = True


@dataclass(frozen=True)
class WorkflowPlan:
    """Parsed workflow before execution."""

    schema: int
    software_version: str
    cards: tuple[CardSpec, ...]

    def stages(self) -> list[CardStage]:
        """Flatten enabled cards into the order they will run."""
        stages: list[CardStage] = []
        self._collect(self.cards, stages)
        return stages

    def _collect(self, specs: Sequence[CardSpec], stages: list[CardStage]) -> None:
        for spec in specs:
            if not spec.enabled:
                continue
            # A container's own output only exists after its children have run,
            # so list children first to keep preview order identical to run order
            # (and the same numbering as ``WorkflowResult.outputs(index=...)``).
            if spec.is_container:
                self._collect(spec.children, stages)
                for branch in spec.branches:
                    self._collect(branch, stages)
                if spec.filter_card is not None:
                    self._collect((spec.filter_card,), stages)
            stages.append(
                CardStage(
                    index=len(stages) + 1,
                    card_class=spec.card_class,
                    label=spec.card_class,
                    kind=spec.kind,
                )
            )


@dataclass
class CardOutput:
    """Outputs produced by one executed card."""

    index: int
    card_class: str
    structures: list[Any]
    #: How many structures reached this card. Generators start from zero.
    input_count: int = 0

    def __len__(self) -> int:
        return len(self.structures)


class WorkflowResult:
    """Collected outputs with per-card access.

    A workflow usually ends in a single dataset, but callers often want one
    intermediate stage. Every stage is retained here instead of only the last.
    """

    def __init__(self, outputs: Sequence[CardOutput]) -> None:
        self.card_outputs: list[CardOutput] = list(outputs)

    def all(self) -> list[Any]:
        """Return the concatenation of every stage output."""
        merged: list[Any] = []
        for output in self.card_outputs:
            merged.extend(output.structures)
        return merged

    def final(self) -> list[Any]:
        """Return the output of the last executed card."""
        if not self.card_outputs:
            return []
        return list(self.card_outputs[-1].structures)

    def outputs(
        self,
        card: str | None = None,
        *,
        index: int | None = None,
    ) -> list[Any]:
        """Return one card's output by class name or 1-based execution index."""
        if card is None and index is None:
            return self.final()
        matches = self.card_outputs
        if card is not None:
            matches = [item for item in matches if item.card_class == card]
        if index is not None:
            matches = [item for item in matches if item.index == index]
        if not matches:
            target = card if card is not None else f"stage {index}"
            raise WorkflowError(
                f"no output for {target}; available: "
                + ", ".join(f"{item.index}:{item.card_class}" for item in self.card_outputs)
            )
        return list(matches[-1].structures)

    def card_names(self) -> list[str]:
        """Return executed card class names in order."""
        return [item.card_class for item in self.card_outputs]


class WorkflowRun:
    """A loaded workflow ready to execute."""

    def __init__(self, plan: WorkflowPlan, source: Path | None = None) -> None:
        self.plan = plan
        self.source = source

    @property
    def cards(self) -> tuple[CardSpec, ...]:
        return self.plan.cards

    def preview(self) -> list[CardStage]:
        """Return the execution plan without running anything."""
        return self.plan.stages()

    def requires_input(self) -> bool:
        """Report whether the workflow needs an input dataset."""
        return next((spec.requires_input() for spec in self.plan.cards if spec.enabled), False)

    def run(self, structures: Iterable[Any] | None = None) -> WorkflowResult:
        """Execute the workflow and return every stage output.

        Parameters
        ----------
        structures : Iterable, optional
            Input structures. Cards that only generate ignore it.

        Returns
        -------
        WorkflowResult
            Per-card outputs plus merged accessors.
        """
        current = list(structures or [])
        outputs: list[CardOutput] = []
        for spec in self.plan.cards:
            if not spec.enabled:
                continue
            current = _run_spec(spec, current, outputs)
        return WorkflowResult(outputs)


def load_workflow(payload: str | Path | dict[str, Any]) -> WorkflowRun:
    """Load a workflow from a path, JSON string, or mapping.

    Parameters
    ----------
    payload : str | Path | dict
        A path to an exported JSON file, raw JSON text, or a parsed mapping.

    Returns
    -------
    WorkflowRun
        Ready to preview or execute.
    """
    source: Path | None = None
    if isinstance(payload, Path):
        source = payload
        document = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{"):
            document = json.loads(text)
        else:
            source = Path(text)
            document = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(payload, dict):
        document = payload
    else:
        raise WorkflowError(
            f"expected a path, JSON text, or mapping, got {type(payload).__name__}"
        )

    return WorkflowRun(_parse_document(document), source)


def run_workflow(
    payload: str | Path | dict[str, Any],
    structures: Iterable[Any] | None = None,
) -> WorkflowResult:
    """Load and execute a workflow in one call."""
    return load_workflow(payload).run(structures)


def _parse_document(document: Any) -> WorkflowPlan:
    if not isinstance(document, dict):
        raise WorkflowError("workflow JSON must be a JSON object")

    # Accept both a workflow export and a workflow-library record.
    workflow = document.get("workflow", document)
    if not isinstance(workflow, dict):
        raise WorkflowError("workflow JSON does not contain a workflow object")

    cards = workflow.get("cards")
    if not isinstance(cards, list) or not cards:
        raise WorkflowError("workflow JSON does not contain any cards")

    parsed = tuple(_parse_card(entry) for entry in cards)
    enabled = [spec for spec in parsed if spec.enabled]
    if any(spec.kind == "fork" and not spec.merge_enabled for spec in enabled[:-1]):
        raise WorkflowError("A permanent fork without Merge must be the final workflow node. Insert an explicit Merge before adding a shared downstream card.")
    return WorkflowPlan(
        schema=int(workflow.get("workflow_schema", 2)),
        software_version=str(workflow.get("software_version", "")),
        cards=parsed,
    )


def _parse_card(entry: Any) -> CardSpec:
    if not isinstance(entry, dict):
        raise WorkflowError(f"each card entry must be an object, got {type(entry).__name__}")

    card_class = entry.get("class")
    if not isinstance(card_class, str) or not card_class:
        raise WorkflowError("a card entry is missing its class name", card="<unknown>")

    if card_class == "CardGroup":
        children = _parse_children(entry.get("card_list", []), card_class)
        filter_entry = entry.get("filter_card")
        return CardSpec(
            card_class=card_class,
            enabled=bool(entry.get("check_state", True)),
            children=children,
            filter_card=_parse_card(filter_entry) if isinstance(filter_entry, dict) else None,
        )

    if card_class == "WorkflowFork":
        branches = entry.get("branches", [])
        parsed_branches = []
        if isinstance(branches, list):
            for branch in branches:
                if not isinstance(branch, dict):
                    continue
                if not bool(branch.get("enabled", True)):
                    continue
                parsed_branches.append(_parse_children(branch.get("cards", []), card_class))
        return CardSpec(
            card_class=card_class,
            enabled=bool(entry.get("check_state", True)),
            branches=tuple(parsed_branches),
            merge_enabled=bool(entry.get("merge", entry.get("merge_enabled", False))),
        )

    if card_class not in CARD_REGISTRY:
        raise WorkflowError(
            "unknown card class; it may come from a newer version",
            card=card_class,
        )

    params = entry.get("params")
    if params is None:
        raise UnsupportedCardFormatError(
            card_class,
            "no 'params' object was found (legacy flat layout)",
        )
    if not isinstance(params, dict):
        raise WorkflowError("'params' must be an object", card=card_class)

    return CardSpec(
        card_class=card_class,
        enabled=bool(entry.get("check_state", True)),
        params=params,
    )


def _parse_children(entries: Any, parent: str) -> tuple[CardSpec, ...]:
    if not isinstance(entries, list):
        raise WorkflowError("child card list must be an array", card=parent)
    return tuple(_parse_card(entry) for entry in entries)


def _run_spec(
    spec: CardSpec,
    incoming: list[Any],
    outputs: list[CardOutput],
) -> list[Any]:
    """Execute one spec, appending every produced stage to ``outputs``."""
    kind = spec.kind

    if kind == "group":
        return _run_group(spec, incoming, outputs)

    if kind == "fork":
        return _run_fork(spec, incoming, outputs)

    produced = _execute_card(spec, incoming)
    outputs.append(
        CardOutput(
            index=len(outputs) + 1,
            card_class=spec.card_class,
            structures=produced,
            input_count=len(incoming),
        )
    )
    return produced


def _execute_card(spec: CardSpec, incoming: list[Any]) -> list[Any]:
    module_path, operation_name, params_module, params_name, kind = CARD_REGISTRY[spec.card_class]

    try:
        operation_type = getattr(import_module(module_path), operation_name)
        params_type = getattr(import_module(params_module), params_name)
    except (ImportError, AttributeError) as exc:
        raise WorkflowError(f"cannot load its implementation: {exc}", card=spec.card_class) from exc

    params = params_from_dict(params_type, spec.params)
    operation = operation_type()

    if kind == "generator":
        return list(operation.generate(params))

    if kind == "dataset":
        return list(operation.run_dataset(incoming, params))

    if not isinstance(operation, StructureOperation):
        raise WorkflowError(
            f"unsupported operation type {type(operation).__name__}",
            card=spec.card_class,
        )

    produced: list[Any] = []
    for structure in incoming:
        try:
            produced.extend(operation.run_structure(structure, params))
        except CardOperationError as exc:
            raise WorkflowError(str(exc), card=spec.card_class) from exc
        except Exception as exc:  # noqa: BLE001 - report the failing card
            raise WorkflowError(
                f"{type(exc).__name__}: {exc}",
                card=spec.card_class,
            ) from exc
    return produced


def _run_group(
    spec: CardSpec,
    incoming: list[Any],
    outputs: list[CardOutput],
) -> list[Any]:
    """Run every enabled child on the shared input and merge the results."""
    merged: list[Any] = []
    for child in spec.children:
        if not child.enabled:
            continue
        merged.extend(_run_spec(child, list(incoming), outputs))

    if spec.filter_card is not None and spec.filter_card.enabled:
        merged = _run_spec(spec.filter_card, merged, outputs)

    outputs.append(
        CardOutput(
            index=len(outputs) + 1,
            card_class=spec.card_class,
            structures=merged,
            input_count=len(incoming),
        )
    )
    return merged


def _run_fork(
    spec: CardSpec,
    incoming: list[Any],
    outputs: list[CardOutput],
) -> list[Any]:
    """Run each branch independently; merge only when the fork asks for it."""
    branch_outputs: list[list[Any]] = []
    for branch in spec.branches:
        enabled = [child for child in branch if child.enabled]
        if not enabled:
            continue
        terminal = list(incoming)
        for child in enabled:
            terminal = _run_spec(child, terminal, outputs)
        branch_outputs.append(terminal)

    if not spec.merge_enabled:
        # Independent terminal outputs remain accessible in their stage records.
        outputs.append(
            CardOutput(
                index=len(outputs) + 1,
                card_class=spec.card_class,
                structures=[],
                input_count=len(incoming),
            )
        )
        return []

    merged: list[Any] = []
    for produced in branch_outputs:
        merged.extend(produced)
    outputs.append(
        CardOutput(
            index=len(outputs) + 1,
            card_class=spec.card_class,
            structures=merged,
            input_count=len(incoming),
        )
    )
    return merged
