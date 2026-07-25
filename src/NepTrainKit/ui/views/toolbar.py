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
        self.setSpaing(0)
        self.init_actions()

    def addButton(self, name, icon, callback, checkable: bool = False, action_key: str | None = None):
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
        self._actions[action_key or name] = action
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
    undoSelectionSignal = Signal()
    discoverySignal = Signal()
    deleteSignal = Signal()
    editInfoSignal = Signal()
    revokeSignal = Signal()
    exportSignal = Signal()
    shiftEnergySignal = Signal()
    inverseSignal = Signal()
    selectIndexSignal = Signal()
    rangeSignal = Signal()
    latticeRangeSignal = Signal()
    dftd3Signal = Signal()
    trainingSetCheckSignal = Signal()
    forceBalanceSignal = Signal()

    def __init__(self, parent=None):
        """Initialise toolbar actions and keep a reference to the action group."""
        super().__init__(parent)
        self.action_group: QActionGroup

    def init_actions(self):
        """Populate toolbar actions for interacting with NEP plots."""
        self.addButton(self.tr("Reset View"), QIcon(":/images/src/images/init.svg"), self.resetSignal)
        pan_action = self.addButton(
            self.tr("Pan View"),
            QIcon(":/images/src/images/pan.svg"),
            self.pan,
            True,
        )
        self.addButton(
            self.tr("Select by Index"),
            QIcon(":/images/src/images/index.svg"),
            self.selectIndexSignal,
            action_key="select_by_index",
        )
        self.addButton(
            self.tr("Select by Range"),
            QIcon(":/images/src/images/data_range.svg"),
            self.rangeSignal,
        )
        self.addButton(
            self.tr("Select by Lattice"),
            QIcon(":/images/src/images/supercell.svg"),
            self.latticeRangeSignal,
        )
        find_max_action = self.addButton(
            self.tr("Find Max Error Point"),
            QIcon(":/images/src/images/find_max.svg"),
            self.findMaxSignal,
            action_key="find_max_error",
        )
        sparse_action = self.addButton(
            self.tr("Sparse samples"),
            QIcon(":/images/src/images/sparse.svg"),
            self.sparseSignal,
            action_key="sparse_samples",
        )

        pen_action = self.addButton(
            self.tr("Mouse Selection"),
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
            self.tr("Find non-physical structures"),
            QIcon(":/images/src/images/discovery.svg"),
            self.discoverySignal,
            action_key="find_non_physical",
        )
        self.addButton(
            self.tr("Check net force"),
            QIcon(":/images/src/images/inspect.svg"),
            self.forceBalanceSignal,
            action_key="check_net_force",
        )
        inverse_action = self.addButton(
            self.tr("Invert selection"),
            QIcon(":/images/src/images/inverse.svg"),
            self.inverseSignal,
        )
        undo_selection_action = self.addButton(
            self.tr("Undo selection"),
            QIcon(":/images/src/images/undo_selection.svg"),
            self.undoSelectionSignal,
        )
        revoke_action = self.addButton(
            self.tr("Undo delete"),
            QIcon(":/images/src/images/undo_delete.svg"),
            self.revokeSignal,
        )
        delete_action = self.addButton(
            self.tr("Delete selected items"),
            QIcon(":/images/src/images/delete.svg"),
            self.deleteSignal,
        )

        self.addSeparator()
        self.addButton(
            self.tr("Training Set Audit"),
            QIcon(":/images/src/images/summary.svg"),
            self.trainingSetCheckSignal,
            action_key="training_set_check",
        )
        self.addButton(
            self.tr("Edit info"),
            QIcon(":/images/src/images/edit_info.svg"),
            self.editInfoSignal,
            action_key="edit_info",
        )
        export_action = self.addButton(
            self.tr("Export structure descriptor"),
            QIcon(":/images/src/images/export.svg"),
            self.exportSignal,
        )
        self.addSeparator()
        self.addButton(
            self.tr("Energy baseline shift"),
            QIcon(":/images/src/images/alignment.svg"),
            self.shiftEnergySignal,
            action_key="energy_baseline_shift",
        )
        self.addButton(
            "DFT D3",
            QIcon(":/images/src/images/dft_d3.png"),
            self.dftd3Signal,
            action_key="dft_d3",
        )

    def set_training_set_check_enabled(self, enabled: bool) -> None:
        """Enable the dataset-wide Training Set Audit entry."""
        action = self._actions.get("training_set_check")
        if action is not None:
            action.setEnabled(bool(enabled))

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
    rejectToggledSignal = Signal(bool)
    dropRejectSignal = Signal()

    def init_actions(self):
        """Populate actions for camera control and structure export."""
        self._reject_syncing = False
        view_action = self.addButton(
            self.tr("Orthographic view"),
            QIcon(":/images/src/images/view_change.svg"),
            self.view_changed,
            True,
        )
        auto_action = self.addButton(
            self.tr("Auto view"),
            QIcon(":/images/src/images/auto_distance.svg"),
            self.auto_view_changed,
            True,
        )
        show_bond_action = self.addButton(
            self.tr("Show bonds"),
            QIcon(":/images/src/images/show_bond.svg"),
            self.show_bond,
            True,
            action_key="show_bonds",
        )

        self._arrow_action = self.addButton(
            self.tr("Show arrows"),
            QIcon(":/images/src/images/xyz.svg"),
            self.arrowSignal,
            action_key="show_arrows",
        )

        export_action = self.addButton(
            self.tr("Export current structure"),
            QIcon(":/images/src/images/export1.svg"),
            self.exportSignal,
        )
        self.addSeparator()
        self.addButton(
            self.tr("Mark bad (reject)"),
            QIcon(":/images/src/images/defect.svg"),
            self._reject_changed,
            True,
            action_key="mark_bad",
        )
        self.addButton(
            self.tr("Drop all bad"),
            QIcon(":/images/src/images/delete.svg"),
            self.dropRejectSignal,
        )

    def _reject_changed(self, checked: bool) -> None:
        """Emit the reject toggle state for the current structure."""
        if getattr(self, "_reject_syncing", False):
            return
        self.rejectToggledSignal.emit(bool(checked))

    def set_reject_checked(self, checked: bool) -> None:
        """Update the reject toggle without emitting signals."""
        action = self._actions.get("mark_bad")
        if action is None:
            return
        try:
            self._reject_syncing = True
            action.setChecked(bool(checked))
        finally:
            self._reject_syncing = False

    def set_arrow_enabled(self, enabled: bool, disabled_tooltip: str = "") -> None:
        """Enable or disable the arrow action based on backend capabilities."""
        action = self._actions.get("show_arrows")
        if action is None:
            return
        action.setEnabled(bool(enabled))
        if enabled:
            action.setToolTip(self.tr("Show arrows"))
        elif disabled_tooltip:
            action.setToolTip(disabled_tooltip)

    def view_changed(self, checked: bool) -> None:
        """Emit the orthographic view toggle state."""
        self.orthoViewSignal.emit(bool(checked))

    def auto_view_changed(self, checked: bool) -> None:
        """Emit the automatic view alignment toggle state."""
        self.autoViewSignal.emit(bool(checked))

    def show_bond(self, checked: bool) -> None:
        """Toggle bond visibility and update the corresponding icon."""
        if checked:
            self._actions["show_bonds"].setIcon(QIcon(":/images/src/images/hide_bond.svg"))
            self.showBondSignal.emit(True)
        else:
            self._actions["show_bonds"].setIcon(QIcon(":/images/src/images/show_bond.svg"))
            self.showBondSignal.emit(False)
