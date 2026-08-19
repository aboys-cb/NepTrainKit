#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UI widgets namespace.

Temporary shim that re-exports symbols from the legacy
`NepTrainKit.custom_widget` package to the new location.
Once migration is complete, actual modules will live here.
"""

from .layout import FlowLayout
from .label import ProcessLabel
from .completer import CompleterModel, JoinDelegate, ConfigCompleter
from .dialog import (
    GetIntMessageBox,
    GetStrMessageBox,
    GetFloatMessageBox,
    ExportFormatMessageBox,
    DistributionExplorerWidget,
    DistributionInspectorMessageBox,
    SparseMessageBox,
    IndexSelectMessageBox,
    RangeSelectMessageBox,
    LatticeRangeSelectMessageBox,
    ArrowMessageBox,
    EditInfoMessageBox,
    ShiftEnergyMessageBox,
    ProgressDialog,
    PeriodicTableDialog,
    DFTD3MessageBox,
    ProjectInfoMessageBox,
    TagManageDialog,
    ModelInfoMessageBox,
    AdvancedModelSearchDialog,
    TrainingOverlayDialog,
)
from .input import SpinBoxUnitInputFrame
from .compact_form import StatusDot, CategoryTag, CompactField, SegmentedControl
from .card_widget import (
    CheckableHeaderCardWidget,
    ShareCheckableHeaderCardWidget,
    MakeDataCardWidget,
    MakeDataCard,
    FilterDataCard,
)
from .doping_rule import DopingRulesWidget
from .vacancy_rule import VacancyRulesWidget
from .docker import MakeWorkflowArea
from .search_widget import ConfigTypeSearchLineEdit
from .filter_bar import TagFilterDialog, ElementsFilterDialog, ExpressionFilterDialog
from .structure_filter_bar import StructureFilterBar, StructureFilterEditorPopup
from .settingscard import MyComboBoxSettingCard, DoubleSpinBoxSettingCard, LineEditSettingCard, ColorSettingCard
from .table import IdNameTableModel
from .tree import TreeModel, TreeItem, TagDelegate
from .audit_chart import AuditChartWidget

__all__ = [
    "FlowLayout",
    "ProcessLabel",
    "CompleterModel",
    "JoinDelegate",
    "ConfigCompleter",
    "GetIntMessageBox",
    "GetStrMessageBox",
    "GetFloatMessageBox",
    "ExportFormatMessageBox",
    "DistributionExplorerWidget",
    "DistributionInspectorMessageBox",
    "SparseMessageBox",
    "IndexSelectMessageBox",
    "RangeSelectMessageBox",
    "LatticeRangeSelectMessageBox",
    "ArrowMessageBox",
    "EditInfoMessageBox",
    "ShiftEnergyMessageBox",
    "ProgressDialog",
    "PeriodicTableDialog",
    "SpinBoxUnitInputFrame",
    "StatusDot",
    "CategoryTag",
    "CompactField",
    "SegmentedControl",
    "ModelInfoMessageBox",
    "AdvancedModelSearchDialog",
    "TrainingOverlayDialog",
    "CheckableHeaderCardWidget",
    "ShareCheckableHeaderCardWidget",
    "MakeDataCardWidget",
    "MakeDataCard",
    "FilterDataCard",
    "MakeWorkflowArea",
    "ConfigTypeSearchLineEdit",
    "TagFilterDialog",
    "ElementsFilterDialog",
    "ExpressionFilterDialog",
    "StructureFilterBar",
    "StructureFilterEditorPopup",
    "MyComboBoxSettingCard",
    "DoubleSpinBoxSettingCard",
    "LineEditSettingCard",
    "ColorSettingCard",
    "DopingRulesWidget",
    "VacancyRulesWidget",
    "DFTD3MessageBox",
    "ProjectInfoMessageBox",
    "TagManageDialog",
    "IdNameTableModel",
    "TreeModel",
    "TreeItem",
    "TagDelegate",
    "AuditChartWidget",
]
