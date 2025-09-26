"""Toolbar widgets that expose plotting and structure manipulation actions."""

from PySide6.QtCore import Signal, QSize
from PySide6.QtGui import QAction, QIcon, QActionGroup
from qfluentwidgets import CommandBar, Action, CommandBarView


class KitToolBarBase(CommandBarView):
    """Shared base class providing helpers for QFluent command bars.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the toolbar.
    """

    def __init__(self, parent=None):
        """Initialise the toolbar container and register placeholder actions."""
        super().__init__(parent)
        self._parent = parent
        self._actions: dict[str, Action] = {}
        self.setIconSize(QSize(24, 24))
        self.setSpaing(2)
        self.init_actions()

    def addButton(self, name, icon, callback, checkable: bool = False):
        """Create an action button with an optional checkable state.

        Parameters
        ----------
        name : str
            Display text shown in the tooltip and accessible name.
        icon : QIcon | str
            Icon assigned to the action.
        callback : Callable
            Slot or callable connected to the action.
        checkable : bool, default=False
            Whether the action toggles between checked and unchecked states.

        Returns
        -------
        Action
            The newly created toolbar action.
        """
        action = Action(QIcon(icon), name, self)
        if checkable:
            action.setCheckable(True)
            action.toggled.connect(callback)
        else:
            action.triggered.connect(callback)
        self._actions[name] = action
        self.addAction(action)
        action.setToolTip(name)
        return action

    def init_actions(self):
        """Hook for derived classes to populate toolbar actions."""
        raise NotImplementedError


class NepDisplayGraphicsToolBar(KitToolBarBase):
    """Toolbar that controls NEP result plots and descriptor selections."""

    panSignal = Signal(bool)
    resetSignal = Signal()
    findMaxSignal = Signal()
    sparseSignal = Signal()
    penSignal = Signal(bool)
    undoSignal = Signal()
    discoverySignal = Signal()
    deleteSignal = Signal()
    editInfoSignal = Signal()
    revokeSignal = Signal()
    exportSignal = Signal()
    shiftEnergySignal = Signal()
    inverseSignal = Signal()
    selectIndexSignal = Signal()
    rangeSignal = Signal()
    dftd3Signal = Signal()

    def __init__(self, parent=None):
        """Initialise toolbar actions and keep a reference to the action group."""
        super().__init__(parent)
        self.action_group: QActionGroup

    def init_actions(self):
        """Populate toolbar actions for interacting with NEP plots."""
        self.addButton("Reset View", QIcon(":/images/src/images/init.svg"), self.resetSignal)
        pan_action = self.addButton(
            "Pan View",
            QIcon(":/images/src/images/pan.svg"),
            self.pan,
            True,
        )
        self.addButton(
            "Select by Index",
            QIcon(":/images/src/images/index.svg"),
            self.selectIndexSignal,
        )
        self.addButton(
            "Select by Range",
            QIcon(":/images/src/images/data_range.svg"),
            self.rangeSignal,
        )
        find_max_action = self.addButton(
            "Find Max Error Point",
            QIcon(":/images/src/images/find_max.svg"),
            self.findMaxSignal,
        )
        sparse_action = self.addButton(
            "Sparse samples",
            QIcon(":/images/src/images/sparse.svg"),
            self.sparseSignal,
        )

        pen_action = self.addButton(
            "Mouse Selection",
            QIcon(":/images/src/images/pen.svg"),
            self.pen,
            True,
        )
        self.action_group = QActionGroup(self)
        self.action_group.setExclusive(True)
        self.action_group.addAction(pan_action)
        self.action_group.addAction(pen_action)
        self.action_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)

        discovery_action = self.addButton(
            "Finding non-physical structures",
            QIcon(":/images/src/images/discovery.svg"),
            self.discoverySignal,
        )
        inverse_action = self.addButton(
            "Inverse Selection",
            QIcon(":/images/src/images/inverse.svg"),
            self.inverseSignal,
        )
        revoke_action = self.addButton(
            "Undo",
            QIcon(":/images/src/images/revoke.svg"),
            self.revokeSignal,
        )
        delete_action = self.addButton(
            "Delete Selected Items",
            QIcon(":/images/src/images/delete.svg"),
            self.deleteSignal,
        )

        self.addSeparator()
        self.addButton(
            "Edit Info",
            QIcon(":/images/src/images/edit_info.svg"),
            self.editInfoSignal,
        )
        export_action = self.addButton(
            "Export structure descriptor",
            QIcon(":/images/src/images/export.svg"),
            self.exportSignal,
        )
        self.addSeparator()
        self.addButton(
            "Energy Baseline Shift",
            QIcon(":/images/src/images/alignment.svg"),
            self.shiftEnergySignal,
        )
        self.addButton(
            "DFT D3",
            QIcon(":/images/src/images/dft_d3.png"),
            self.dftd3Signal,
        )

    def reset(self) -> None:
        """Clear any mutually exclusive toggle that is still checked."""
        if self.action_group.checkedAction():
            self.action_group.checkedAction().setChecked(False)

    def pan(self, checked: bool) -> None:
        """Toggle pan mode on the canvas.

        Parameters
        ----------
        checked : bool
            ``True`` enables pan mode; ``False`` disables it.
        """
        self.panSignal.emit(bool(checked))

    def pen(self, checked: bool) -> None:
        """Toggle the lasso selection mode on the canvas.

        Parameters
        ----------
        checked : bool
            ``True`` enables lasso selection; ``False`` disables it.
        """
        self.penSignal.emit(bool(checked))


class StructureToolBar(KitToolBarBase):
    """Toolbar for structure viewing actions inside the 3D viewer."""

    showBondSignal = Signal(bool)
    orthoViewSignal = Signal(bool)
    autoViewSignal = Signal(bool)
    exportSignal = Signal()
    arrowSignal = Signal()

    def init_actions(self):
        """Populate actions for camera control and structure export."""
        view_action = self.addButton(
            "Ortho View",
            QIcon(":/images/src/images/view_change.svg"),
            self.view_changed,
            True,
        )
        auto_action = self.addButton(
            "Automatic View",
            QIcon(":/images/src/images/auto_distance.svg"),
            self.auto_view_changed,
            True,
        )
        show_bond_action = self.addButton(
            "Show Bonds",
            QIcon(":/images/src/images/show_bond.svg"),
            self.show_bond,
            True,
        )

        self.addButton(
            "Show Arrows",
            QIcon(":/images/src/images/xyz.svg"),
            self.arrowSignal,
        )

        export_action = self.addButton(
            "Export current structure",
            QIcon(":/images/src/images/export1.svg"),
            self.exportSignal,
        )

    def view_changed(self, checked: bool) -> None:
        """Emit the orthographic view toggle state."""
        self.orthoViewSignal.emit(bool(checked))

    def auto_view_changed(self, checked: bool) -> None:
        """Emit the automatic view alignment toggle state."""
        self.autoViewSignal.emit(bool(checked))

    def show_bond(self, checked: bool) -> None:
        """Toggle bond visibility and update the corresponding icon."""
        if checked:
            self._actions["Show Bonds"].setIcon(QIcon(":/images/src/images/hide_bond.svg"))
            self.showBondSignal.emit(True)
        else:
            self._actions["Show Bonds"].setIcon(QIcon(":/images/src/images/show_bond.svg"))
            self.showBondSignal.emit(False)
