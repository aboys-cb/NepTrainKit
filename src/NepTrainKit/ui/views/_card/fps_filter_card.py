"""Filter card that keeps representative points via farthest point sampling."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    FluentIcon,
    LineEdit,
    PushButton,
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
from NepTrainKit.ui.widgets import (
    CompactField,
    FilterDataCard,
    InspectorSection,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


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
        self._last_physics_plan_report = None
        self._input_dataset = []
        self.setTitle(self.tr("FPS Sampling"))
        self.init_ui()

    def init_ui(self):
        """Build a compact inspector around budget, coverage, and preview."""
        self.setObjectName("fps_filter_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.strategy_combo = ComboBox(self.setting_widget)
        self.strategy_combo.addItem(
            self.tr("Global budget"),
            userData="global",
        )
        self.strategy_combo.addItem(
            self.tr("Balance by element set"),
            userData="element_set",
        )
        self.strategy_combo.addItem(
            self.tr("Balance element set, phase, and spin"),
            userData="physics",
        )
        self.strategy_combo.setAccessibleName(self.tr("Sampling plan"))
        self.strategy_field = CompactField(
            self.tr("Sampling plan"),
            self.strategy_combo,
            self.setting_widget,
        )
        self.strategy_label = self.strategy_field.caption

        self.strategy_hint = CaptionLabel("", self.setting_widget)
        self.strategy_hint.setWordWrap(True)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("", 1, "int")
        self.num_condition_frame.setRange(1, 10000000)
        self.num_condition_frame.set_input_value([100])
        self.num_condition_frame.setFixedWidth(144)
        self.num_condition_frame.setAccessibleName(self.tr("Maximum output"))
        self.num_field = CompactField(
            self.tr("Maximum output"),
            self.num_condition_frame,
            self.setting_widget,
            self.tr("This is an upper bound; the distance cutoff can stop selection earlier."),
            inline=True,
            input_max_width=144,
        )
        self.num_label = self.num_field.caption

        self.nep_path_lineedit = LineEdit(self.setting_widget)
        self.nep_path_lineedit.setPlaceholderText(self.tr("nep.txt path"))
        self.nep_path_lineedit.setClearButtonEnabled(True)
        self.nep_path_lineedit.setAccessibleName(self.tr("Descriptor NEP model"))

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

        self.nep_path_field = CompactField(
            self.tr("Descriptor NEP model"),
            self.nep_path_widget,
            self.setting_widget,
            self.tr("Descriptor distances depend on this model; use one that covers the candidate chemistry."),
        )
        self.nep_path_label = self.nep_path_field.caption

        plan_section = InspectorSection(
            self.tr("Sampling plan"),
            self.setting_widget,
            self.tr("FPS keeps candidates that extend coverage in NEP descriptor space."),
        )
        plan_grid = ResponsiveFormGrid(plan_section)
        plan_grid.add_field(self.strategy_field, span=2)
        plan_grid.add_field(self.strategy_hint, span=2)
        plan_grid.add_field(self.num_field, span=2)
        plan_section.addWidget(plan_grid)

        self.advanced_button = PushButton(
            FluentIcon.SETTING,
            self.tr("Supplement existing set"),
            self.setting_widget,
        )
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolTip(
            self.tr(
                "Use an existing training set as the covered baseline, then select new candidates farthest from it."
            )
        )
        self.advanced_button.toggled.connect(self._set_advanced_visible)

        self.advanced_frame = QFrame(self.setting_widget)
        self.advanced_layout = QVBoxLayout(self.advanced_frame)
        self.advanced_layout.setContentsMargins(0, 2, 0, 0)
        self.advanced_layout.setSpacing(4)
        self.min_distance_condition_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_condition_frame.set_input("", 1,"float")
        self.min_distance_condition_frame.setRange(0, 100)
        self.min_distance_condition_frame.object_list[0].setDecimals(4)   # pyright:ignore
        self.min_distance_condition_frame.set_input_value([0.0])
        self.min_distance_condition_frame.setFixedWidth(144)
        self.min_distance_condition_frame.setAccessibleName(self.tr("Distance cutoff"))
        self.min_distance_field = CompactField(
            self.tr("Distance cutoff"),
            self.min_distance_condition_frame,
            self.setting_widget,
            self.tr("Model-dependent and unitless; 0 disables early stopping."),
            inline=True,
            input_max_width=144,
        )
        self.min_distance_label = self.min_distance_field.caption

        self.existing_dataset_lineedit = LineEdit(self.advanced_frame)
        self.existing_dataset_lineedit.setPlaceholderText(self.tr("Optional train.xyz"))
        self.existing_dataset_lineedit.setClearButtonEnabled(True)
        self.existing_dataset_lineedit.setAccessibleName(self.tr("Existing training set"))
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

        self.existing_dataset_field = CompactField(
            self.tr("Existing training set"),
            self.existing_dataset_widget,
            self.advanced_frame,
            self.tr("Used as the selection baseline; only newly selected candidates are output."),
        )
        self.existing_dataset_label = self.existing_dataset_field.caption
        advanced_grid = ResponsiveFormGrid(self.advanced_frame)
        advanced_grid.add_field(self.existing_dataset_field, span=2)
        self.advanced_layout.addWidget(advanced_grid)

        coverage_section = InspectorSection(
            self.tr("Descriptor coverage"),
            self.setting_widget,
        )
        coverage_section.addWidget(self.nep_path_field)
        coverage_section.addWidget(self.min_distance_field)
        coverage_section.addWidget(self.advanced_button)
        coverage_section.addWidget(self.advanced_frame)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("fpsFilterPreview")
        self.preview_section = InspectorSection(self.tr("Selection preview"), self.setting_widget)
        self.preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(plan_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(coverage_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.preview_section, 2, 0, 1, 3)

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
        strategy = self.strategy_combo.currentData()
        if strategy == "element_set":
            self.strategy_hint.setText(
                self.tr(
                    "Plans one slot per element set, then distributes the rest by sqrt(group size). "
                    "Existing coverage or the distance cutoff can reduce actual output."
                )
            )
        elif strategy == "physics":
            self.strategy_hint.setText(
                self.tr(
                    "Partitions by element set first, then structural phase. "
                    "When the selected NEP is a spin model, magnetic order is an additional coverage axis."
                )
            )
        else:
            self.strategy_hint.setText(
                self.tr(
                    "All candidates share one budget; without an existing set, selection starts from input 1."
                )
            )
        self.strategy_hint.updateGeometry()
        self._refresh_preview()

    def create_operation(self):
        """Return the UI-independent FPS operation."""
        return FPSFilterOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        strategy_labels = {
            "global": self.tr("global budget"),
            "element_set": self.tr("balanced by element set"),
            "physics": self.tr("balanced by element set, phase, and spin"),
        }
        strategy = strategy_labels.get(params.strategy, strategy_labels["global"])
        return self.tr("keep at most {count} · {strategy}").format(
            count=params.n_samples,
            strategy=strategy,
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
        if params.strategy == "physics":
            return self.tr(
                "The model type is detected at run time. Spin models require canonical spin:R:3 data; "
                "{cutoff}. The structure count is an upper bound."
            ).format(cutoff=cutoff)
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
        self._last_physics_plan_report = None
        super().set_dataset(dataset)
        self._input_dataset = list(dataset) if dataset is not None else []
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if not self._input_dataset:
            self._set_preview_text(
                self.tr(
                    "Load upstream structures to preview the output cap and sampling plan."
                )
            )
            return
        try:
            summary = self.create_operation().selection_summary(
                self._input_dataset,
                self.get_params(),
            )
        except (TypeError, ValueError) as exc:
            self._set_preview_text(
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
                    "planned quotas for {count} element sets"
                ).format(count=summary["group_count"])
        elif summary["strategy"] == "physics":
            strategy_text = self.tr(
                "{count} element sets; phase and spin strata are detected during the run"
            ).format(count=summary["element_set_count"])
        else:
            strategy_text = self.tr("one global FPS budget")
        model_text = (
            self.tr("model file found: {name}").format(name=summary["model_name"])
            if summary["model_exists"]
            else self.tr("model path is missing")
        )
        if not summary["existing_configured"]:
            existing_text = self.tr("no existing set")
        elif summary["existing_exists"]:
            existing_text = self.tr("supplementing {name}; output contains new selections only").format(
                name=summary["existing_name"]
            )
        else:
            existing_text = self.tr("existing set path is missing")
        cutoff_text = (
            self.tr("no distance early-stop")
            if summary["min_distance"] == 0.0
            else self.tr("distance cutoff {value}").format(
                value=f"{summary['min_distance']:.4g}"
            )
        )
        self._set_preview_text(
            self.tr(
                "Input {input} → keep at most {output} · {strategy} · {cutoff} · {existing} · {model}"
            ).format(
                input=summary["input_count"],
                output=summary["max_output"],
                strategy=strategy_text,
                cutoff=cutoff_text,
                existing=existing_text,
                model=model_text,
            )
        )

    def _set_preview_text(self, text: str) -> None:
        self.preview_label.setText(text)
        self.preview_label.updateGeometry()
        self.preview_section.layout().invalidate()
        self.preview_section.layout().activate()
        self.preview_section.updateGeometry()
        self.settingLayout.invalidate()
        self.settingLayout.activate()
        self.setting_widget.updateGeometry()
        self.updateGeometry()

    def on_processing_finished(self):
        operation = getattr(getattr(self, "worker_thread", None), "operation", None)
        self._last_group_report = dict(getattr(operation, "last_group_report", {}) or {})
        self._last_physics_plan_report = getattr(
            operation,
            "last_physics_plan_report",
            None,
        )
        super().on_processing_finished()

    def _format_dataset_info(self) -> str:
        text = super()._format_dataset_info()
        if self._last_physics_plan_report is not None:
            report = self._last_physics_plan_report
            covered = sum(
                item.selected_count > 0
                for item in self._last_group_report.values()
            )
            phase_labels = ", ".join(label for label, _count in report.phase_counts)
            if report.spin_model:
                text = self.tr(
                    "{summary} | Physical strata: {covered}/{total} | phases: {phases} | spin-aware"
                ).format(
                    summary=text,
                    covered=covered,
                    total=report.stratum_count,
                    phases=phase_labels,
                )
            else:
                text = self.tr(
                    "{summary} | Physical strata: {covered}/{total} | phases: {phases}"
                ).format(
                    summary=text,
                    covered=covered,
                    total=report.stratum_count,
                    phases=phase_labels,
                )
        elif self._last_group_report:
            covered = sum(
                report.selected_count > 0
                for report in self._last_group_report.values()
            )
            text = self.tr("{summary} | Output element-set coverage: {covered}/{total}").format(
                summary=text,
                covered=covered,
                total=len(self._last_group_report),
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
        if self.get_params().strategy == "physics":
            self.status_label.setText(
                self.tr("Analyzing physical coverage and generating descriptors...")
            )
        else:
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
