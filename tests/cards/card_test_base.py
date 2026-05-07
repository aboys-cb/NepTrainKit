#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from ase import Atoms
from ase.io import read
from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.views._card import (
    SuperCellCard,
    CrystalPrototypeBuilderCard,
    PerturbCard,
    CellScalingCard,
    CellStrainCard,
    ShearMatrixCard,
    ShearAngleCard,
    RandomSlabCard,
    RandomDopingCard,
    CompositionSweepCard,
    CompositionGradientCard,
    RandomOccupancyCard,
    ConditionalReplaceCard,
    MagneticOrderCard,
    SetMagneticMomentsCard,
    SmallAngleSpinTiltCard,
    SpinDisorderCard,
    SpinSpiralCard,
    FoldedHelixCard,
    MagneticMomentRotationCard,
    GroupLabelCard,
    RandomVacancyCard,
    VacancyDefectCard,
    StackingFaultCard,
    InsertDefectCard,
    LayerCopyCard,
    OrganicMolConfigPBCCard,
    VibrationModePerturbCard,
    GeometryFilterCard,
)
from NepTrainKit.core.magnetism import orthonormal_frame
from NepTrainKit.core import CardManager
from NepTrainKit.core.alloy import parse_composition
from NepTrainKit.core.cards.alloy import (
    CompositionSweepOperation,
    CompositionSweepParams,
    CompositionGradientOperation,
    CompositionGradientParams,
    ConditionalReplaceOperation,
    ConditionalReplaceParams,
    RandomDopingOperation,
    RandomDopingParams,
    RandomOccupancyOperation,
    RandomOccupancyParams,
)
from NepTrainKit.core.cards.filter import FPSFilterOperation, FPSFilterParams, GeometryFilterOperation, GeometryFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    CrystalPrototypeBuilderOperation,
    CrystalPrototypeBuilderParams,
    GroupLabelOperation,
    GroupLabelParams,
    LayerCopyOperation,
    LayerCopyParams,
    OrganicMolConfigPBCParams,
    VibrationModePerturbOperation,
    VibrationModePerturbParams,
)
from NepTrainKit.core.cards.defect import (
    InsertDefectOperation,
    InsertDefectParams,
    RandomSlabOperation,
    RandomSlabParams,
    RandomVacancyOperation,
    RandomVacancyParams,
    StackingFaultOperation,
    StackingFaultParams,
    VacancyDefectOperation,
    VacancyDefectParams,
)
from NepTrainKit.core.cards.lattice import (
    CellScalingOperation,
    CellScalingParams,
    CellStrainOperation,
    CellStrainParams,
    PerturbOperation,
    PerturbParams,
    ShearAngleOperation,
    ShearAngleParams,
    ShearMatrixOperation,
    ShearMatrixParams,
    SuperCellOperation,
    SuperCellParams,
)
from NepTrainKit.core.cards.magnetism import (
    FoldedHelixOperation,
    FoldedHelixParams,
    MagneticMomentRotationOperation,
    MagneticMomentRotationParams,
    MagneticOrderOperation,
    MagneticOrderParams,
    SetMagneticMomentsOperation,
    SetMagneticMomentsParams,
    SmallAngleSpinTiltOperation,
    SmallAngleSpinTiltParams,
    SpinDisorderOperation,
    SpinDisorderParams,
    SpinSpiralOperation,
    SpinSpiralParams,
)
from NepTrainKit.ui.widgets import MakeDataCard
from NepTrainKit.ui.widgets.card_metadata import card_tooltip, metadata_html
from NepTrainKit.ui.widgets.doping_rule import DopingRuleItem
from NepTrainKit.ui.widgets.vacancy_rule import VacancyRuleItem
from NepTrainKit.version import DOCS_BASE_URL

BASE_CARD_KEYS = {"class", "check_state", "metadata", "params"}


class _ExternalTestCard(MakeDataCard):
    card_name = "External Test Card"

    def process_structure(self, structure):
        return [structure]


@CardManager.register_card
class _MetadataTestCard(MakeDataCard):
    card_name = "Metadata Test Card"
    description = "Card used to verify contributor metadata."
    card_version = "0.1"
    contributors = [
        {
            "name": "Test Contributor",
            "role": "author",
            "email": "test@example.com",
            "url": "https://example.com/test",
            "affiliation": "Test Lab",
        }
    ]

    def process_structure(self, structure):
        return [structure]




class BaseCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
        cls.test_dir = Path(__file__).resolve().parents[1]
        cls.base_structure = read(cls.test_dir / "data" / "Si2.vasp")
        cls.base_structure.info.setdefault("Config_type", "Si2")

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        self.structure = self.base_structure.copy()
