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
    SparseMessageBox,
    IndexSelectMessageBox,
    RangeSelectMessageBox,
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
)
from .input import SpinBoxUnitInputFrame
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
from .settingscard import MyComboBoxSettingCard, DoubleSpinBoxSettingCard, LineEditSettingCard, ColorSettingCard
from .table import IdNameTableModel
from .tree import TreeModel, TreeItem, TagDelegate

__all__ = [
    'FlowLayout',
    'ProcessLabel',
    'CompleterModel',
    'JoinDelegate',
    'ConfigCompleter',
    'GetIntMessageBox',
    'GetStrMessageBox',
    'SparseMessageBox',
    'IndexSelectMessageBox',
    'RangeSelectMessageBox',
    'ArrowMessageBox',
    'EditInfoMessageBox',
    'ShiftEnergyMessageBox',
    'ProgressDialog',
    'PeriodicTableDialog',
    'SpinBoxUnitInputFrame',
    'ModelInfoMessageBox',
    'AdvancedModelSearchDialog',
    'CheckableHeaderCardWidget',
    'ShareCheckableHeaderCardWidget',
    'MakeDataCardWidget',
    'MakeDataCard',
    'FilterDataCard',
    'MakeWorkflowArea',
    'ConfigTypeSearchLineEdit',
    'MyComboBoxSettingCard',
    'DoubleSpinBoxSettingCard',
    'LineEditSettingCard',
    'ColorSettingCard',
    'DopingRulesWidget',
    'VacancyRulesWidget',
    'DFTD3MessageBox',
    'ProjectInfoMessageBox',
    'TagManageDialog',
    'IdNameTableModel',
    'TreeModel',
    'TreeItem',
    'TagDelegate',
]

