"""Filter card that keeps representative points via farthest point sampling."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    FluentIcon,
    LineEdit,
    PushButton,
    ToolTipFilter,
    ToolTipPosition,
    TransparentToolButton,
)

from NepTrainKit import module_path
from NepTrainKit.config import Config
from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.filter import FPSFilterOperation, FPSFilterParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.types import get_configured_nep_backend
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import FilterDataCard


@CardManager.register_card

class FPSFilterDataCard(FilterDataCard):
    """Filter dataset entries via farthest point sampling computed from NEP descriptors.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget managing the card lifecycle.
    """
    separator=True
    group = "Filter"
    card_name= "FPS Filter"
    menu_icon=r":/images/src/images/fps.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self._backend = get_configured_nep_backend().value
        self._chunk_max_atoms = Config.getint("nep", "chunk_max_atoms", 100000)
        self._last_group_report = {}
        self._input_dataset = []
        self.setTitle(self.tr("Representative Sampling (FPS)"))
        self.init_ui()

    def init_ui(self):
        """Build a compact Fluent form with progressive disclosure."""
        self.setObjectName("fps_filter_card_widget")

        self.strategy_label = BodyLabel(self.tr("Sampling strategy"), self.setting_widget)
        self.strategy_combo = ComboBox(self.setting_widget)
        self.strategy_combo.addItem(
            self.tr("All structures together (legacy)"),
            userData="global",
        )
        self.strategy_combo.addItem(
            self.tr("Guarantee every element set"),
            userData="element_set",
        )
        self.strategy_label.setToolTip(
            self.tr(
                "Legacy global FPS starts from the first input unless an existing training set is supplied"
            )
        )
        self.strategy_label.installEventFilter(
            ToolTipFilter(self.strategy_label, 300, ToolTipPosition.TOP)
        )

        self.strategy_hint = CaptionLabel("", self.setting_widget)
        self.strategy_hint.setWordWrap(True)

        self.num_label = BodyLabel(self.tr("Maximum structures to keep"), self.setting_widget)
        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("", 1, "int")
        self.num_condition_frame.setRange(1, 10000000)
        self.num_condition_frame.set_input_value([100])
        self.num_label.setToolTip(
            self.tr("This is an upper bound; the distance cutoff can stop selection earlier")
        )
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.nep_path_label = BodyLabel(self.tr("Descriptor NEP model"), self.setting_widget)

        self.nep_path_lineedit = LineEdit(self.setting_widget)
        self.nep_path_lineedit.setPlaceholderText(self.tr("nep.txt path"))
        self.nep_path_lineedit.setClearButtonEnabled(True)
        self.nep_path_label.setToolTip(
            self.tr("Descriptor distances depend on this model; prefer a model relevant to the candidate chemistry")
        )
        self.nep_path_label.installEventFilter(ToolTipFilter(self.nep_path_label, 300, ToolTipPosition.TOP))

        self.nep89_path = str(module_path/ "Config/nep89.txt" )
        self.nep_path_lineedit.setText(self.nep89_path )
        self.nep_path_widget = QWidget(self.setting_widget)
        self.nep_path_layout = QHBoxLayout(self.nep_path_widget)
        self.nep_path_layout.setContentsMargins(0, 0, 0, 0)
        self.nep_path_layout.setSpacing(4)
        self.nep_path_layout.addWidget(self.nep_path_lineedit, 1)
        self.nep_browse_button = TransparentToolButton(FluentIcon.FOLDER, self.nep_path_widget)
        self.nep_browse_button.setToolTip(self.tr("Browse for a NEP model"))
        self.nep_browse_button.setAccessibleName(self.tr("Browse for a NEP model"))
        self.nep_browse_button.clicked.connect(self._browse_nep_model)
        self.nep_path_layout.addWidget(self.nep_browse_button, 0)

        self.advanced_button = PushButton(
            FluentIcon.SETTING,
            self.tr("Distance cutoff and existing coverage"),
            self.setting_widget,
        )
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolTip(
            self.tr("Optionally stop near duplicates and avoid regions already covered by a training set")
        )
        self.advanced_button.toggled.connect(self._set_advanced_visible)

        self.advanced_frame = QFrame(self.setting_widget)
        self.advanced_layout = QGridLayout(self.advanced_frame)
        self.advanced_layout.setContentsMargins(0, 2, 0, 0)
        self.advanced_layout.setHorizontalSpacing(4)
        self.advanced_layout.setVerticalSpacing(4)
        self.min_distance_condition_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_condition_frame.set_input("", 1,"float")
        self.min_distance_condition_frame.setRange(0, 100)
        self.min_distance_condition_frame.object_list[0].setDecimals(4)   # pyright:ignore
        self.min_distance_condition_frame.set_input_value([0.0])

        self.min_distance_label = BodyLabel(
            self.tr("Descriptor distance cutoff"),
            self.setting_widget,
        )
        self.min_distance_label.setToolTip(
            self.tr("0 disables early stopping; the scale is model-dependent and has no physical unit")
        )

        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))

        self.existing_dataset_label = BodyLabel(self.tr("Existing training set"), self.advanced_frame)
        self.existing_dataset_lineedit = LineEdit(self.advanced_frame)
        self.existing_dataset_lineedit.setPlaceholderText(self.tr("Optional train.xyz for warm start"))
        self.existing_dataset_lineedit.setClearButtonEnabled(True)
        self.existing_dataset_label.setToolTip(
            self.tr(
                "Candidates near this training set are deprioritized; balanced mode compares only matching element sets"
            )
        )
        self.existing_dataset_label.installEventFilter(
            ToolTipFilter(self.existing_dataset_label, 300, ToolTipPosition.TOP)
        )
        self.existing_dataset_widget = QWidget(self.advanced_frame)
        self.existing_dataset_layout = QHBoxLayout(self.existing_dataset_widget)
        self.existing_dataset_layout.setContentsMargins(0, 0, 0, 0)
        self.existing_dataset_layout.setSpacing(4)
        self.existing_dataset_layout.addWidget(self.existing_dataset_lineedit, 1)
        self.existing_browse_button = TransparentToolButton(
            FluentIcon.FOLDER,
            self.existing_dataset_widget,
        )
        self.existing_browse_button.setToolTip(self.tr("Browse for an existing XYZ training set"))
        self.existing_browse_button.setAccessibleName(self.tr("Browse for an existing XYZ training set"))
        self.existing_browse_button.clicked.connect(self._browse_existing_dataset)
        self.existing_dataset_layout.addWidget(self.existing_browse_button, 0)

        self.advanced_layout.addWidget(self.min_distance_label, 0, 0, 1, 1)
        self.advanced_layout.addWidget(self.min_distance_condition_frame, 0, 1, 1, 2)
        self.advanced_layout.addWidget(self.existing_dataset_label, 1, 0, 1, 1)
        self.advanced_layout.addWidget(self.existing_dataset_widget, 1, 1, 1, 2)

        self.settingLayout.addWidget(self.strategy_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.strategy_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.strategy_hint, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.num_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.nep_path_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.nep_path_widget, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.advanced_button, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.advanced_frame, 5, 0, 1, 3)
        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("fpsFilterPreview")
        self.settingLayout.addWidget(self.preview_label, 6, 0, 1, 3)

        self.strategy_combo.currentIndexChanged.connect(self._update_strategy_ui)
        self.nep_path_lineedit.textChanged.connect(self._refresh_preview)
        self.existing_dataset_lineedit.textChanged.connect(self._refresh_preview)
        for control in (
            *self.num_condition_frame.object_list,
            *self.min_distance_condition_frame.object_list,
        ):
            control.valueChanged.connect(self._refresh_preview)
        self._set_advanced_visible(False)
        self._update_strategy_ui()
        self._refresh_preview()

    def _browse_nep_model(self) -> None:
        path = call_path_dialog(
            self,
            self.tr("Select NEP model"),
            "select",
            file_filter=self.tr("NEP model (*.txt);;All files (*.*)"),
        )
        if path:
            self.nep_path_lineedit.setText(str(path))

    def _browse_existing_dataset(self) -> None:
        path = call_path_dialog(
            self,
            self.tr("Select existing training dataset"),
            "select",
            file_filter=self.tr("XYZ files (*.xyz *.extxyz);;All files (*.*)"),
        )
        if path:
            self.existing_dataset_lineedit.setText(str(path))

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_frame.setVisible(bool(visible))

    def _update_strategy_ui(self) -> None:
        balanced = self.strategy_combo.currentData() == "element_set"
        if balanced:
            self.strategy_hint.setText(
                self.tr(
                    "Each element set gets at least one slot; remaining slots follow sqrt(group size)."
                )
            )
        else:
            self.strategy_hint.setText(
                self.tr(
                    "One shared budget; without an existing set, input 1 is always the first selected structure."
                )
            )
        self._refresh_preview()

    def create_operation(self):
        """Return the UI-independent FPS operation."""
        return FPSFilterOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        return self.tr("keep at most {count} · {strategy}").format(
            count=params.n_samples,
            strategy=params.strategy,
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        cutoff = (
            self.tr("distance early-stop disabled")
            if params.min_distance == 0.0
            else self.tr("model-dependent cutoff {value}").format(
                value=f"{params.min_distance:.4g}"
            )
        )
        return self.tr(
            "Use a descriptor model relevant to the candidate chemistry; {cutoff}. "
            "The structure count is an upper bound."
        ).format(cutoff=cutoff)

    def get_params(self) -> FPSFilterParams:
        """Read FPS parameters from UI controls."""
        strategy = str(self.strategy_combo.currentData() or "global")
        return FPSFilterParams(
            nep_path=self.nep_path_lineedit.text(),
            n_samples=int(self.num_condition_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_condition_frame.get_input_value()[0]),
            backend=self._backend,
            chunk_max_atoms=self._chunk_max_atoms,
            strategy=strategy,
            existing_dataset_path=self.existing_dataset_lineedit.text(),
        )

    def set_params(self, params: FPSFilterParams) -> None:
        """Apply FPS parameters to UI controls."""
        self.nep_path_lineedit.setText(params.nep_path)
        self.num_condition_frame.set_input_value([int(params.n_samples)])
        self.min_distance_condition_frame.set_input_value([float(params.min_distance)])
        self._backend = params.backend
        self._chunk_max_atoms = int(params.chunk_max_atoms)
        strategy_index = self.strategy_combo.findData(params.strategy)
        self.strategy_combo.setCurrentIndex(strategy_index if strategy_index >= 0 else 0)
        self.existing_dataset_lineedit.setText(params.existing_dataset_path)
        show_advanced = bool(params.existing_dataset_path) or float(params.min_distance) != 0.0
        self.advanced_button.setChecked(show_advanced)
        self._update_strategy_ui()

    def set_dataset(self, dataset):
        self._last_group_report = {}
        super().set_dataset(dataset)
        self._input_dataset = list(dataset) if dataset is not None else []
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if not self._input_dataset:
            self.preview_label.setText(
                self.tr(
                    "Load upstream structures to preview the output cap and element-set quotas."
                )
            )
            return
        try:
            summary = self.create_operation().selection_summary(
                self._input_dataset,
                self.get_params(),
            )
        except (TypeError, ValueError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return
        if summary["strategy"] == "element_set":
            quota_items = [
                f"{'+'.join(key) or '∅'}:{value}"
                for key, value in sorted(summary["quotas"].items())
            ]
            if len(quota_items) <= 5:
                strategy_text = self.tr("quotas {quotas}").format(
                    quotas=", ".join(quota_items)
                )
            else:
                strategy_text = self.tr(
                    "{count} element sets with guaranteed coverage"
                ).format(count=summary["group_count"])
        else:
            strategy_text = self.tr("one global FPS budget")
        model_text = (
            self.tr("model found: {name}").format(name=summary["model_name"])
            if summary["model_exists"]
            else self.tr("model path is missing")
        )
        cutoff_text = (
            self.tr("no distance early-stop")
            if summary["min_distance"] == 0.0
            else self.tr("distance cutoff {value}").format(
                value=f"{summary['min_distance']:.4g}"
            )
        )
        self.preview_label.setText(
            self.tr(
                "Input {input} structures → keep at most {output} · {strategy} · {cutoff} · {model}"
            ).format(
                input=summary["input_count"],
                output=summary["max_output"],
                strategy=strategy_text,
                cutoff=cutoff_text,
                model=model_text,
            )
        )

    def on_processing_finished(self):
        operation = getattr(getattr(self, "worker_thread", None), "operation", None)
        self._last_group_report = dict(getattr(operation, "last_group_report", {}) or {})
        super().on_processing_finished()

    def _format_dataset_info(self) -> str:
        text = super()._format_dataset_info()
        if self._last_group_report:
            text = self.tr("{summary} | Element groups: {count}").format(
                summary=text,
                count=len(self._last_group_report),
            )
        return text

    def stop(self):
        """Stop background processing and release any worker threads.
        """
        super().stop()
        if hasattr(self, "nep_thread"):
            self.nep_thread.stop()
            del self.nep_thread

    def update_progress(self, progress):
        """Update the visual progress indicators during background execution.
        
        Parameters
        ----------
        progress : float | int
            Latest progress value emitted by the worker thread.
        """
        self.status_label.setText(self.tr("Generating descriptors..."))
        self.status_label.set_progress(progress)

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = FPSFilterParams(
                nep_path=raw_params.get("nep_path", self.nep89_path),
                n_samples=raw_params.get("n_samples", 100),
                min_distance=raw_params.get("min_distance", 0.0),
                backend=raw_params.get("backend", Config.get("nep", "backend", "auto")),
                chunk_max_atoms=raw_params.get("chunk_max_atoms", Config.getint("nep", "chunk_max_atoms", 100000)),
                strategy=raw_params.get("strategy", "global"),
                existing_dataset_path=raw_params.get("existing_dataset_path", ""),
            )
        else:
            params = FPSFilterParams(
                nep_path=data_dict.get("nep_path", self.nep89_path),
                n_samples=data_dict.get("num_condition", [100])[0],
                min_distance=data_dict.get("min_distance_condition", [0.01])[0],
                backend=Config.get("nep", "backend", "auto"),
                chunk_max_atoms=Config.getint("nep", "chunk_max_atoms", 100000),
                strategy="global",
                existing_dataset_path="",
            )
        self.set_params(params)
