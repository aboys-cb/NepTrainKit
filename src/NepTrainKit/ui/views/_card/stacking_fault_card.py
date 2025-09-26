"""Card for constructing stacking fault structures."""

from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition
import numpy as np

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class StackingFaultCard(MakeDataCard):
    """Generate stacking fault structures by translating atomic layers along a plane.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """
