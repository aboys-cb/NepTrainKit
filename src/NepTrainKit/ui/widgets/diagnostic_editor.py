"""Inline experiment setup for the existing validation workbench."""
import json
import math

from sqlalchemy.exc import SQLAlchemyError
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QWidget, QSizePolicy
from qfluentwidgets import BodyLabel, CaptionLabel, LineEdit, ComboBox, SpinBox, DoubleSpinBox, CheckBox, PushButton, PrimaryPushButton, TransparentToolButton, TransparentPushButton, FluentIcon, SubtitleLabel, ScrollArea

from NepTrainKit.ui.widgets.input import AdaptiveCompactDoubleSpinBox, AdaptiveCompactSpinBox
from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core.validation.protocols.diagnostic import DiagnosticSpec, KINDS, clean_seed, read_seed_structures

PARAMETERS_OPTION = "diagnostic_editor_parameters"


class DiagnosticEditor(QFrame):
    prepared = Signal(str, object, bool)
    readinessChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("diagnosticEditor")
        self.setStyleSheet(f"#diagnosticEditor {{background:{palette.SURFACE}; border:1px solid {palette.BORDER}; border-radius:10px;}}")
        self._seeds = []
        self._restoring_parameters = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        close = TransparentToolButton(FluentIcon.RETURN, self)
        close.setToolTip("返回工作台，保留本次填写")
        close.setAccessibleName("返回工作台")
        close.clicked.connect(self.hide)
        heading.addWidget(close)
        heading.addWidget(SubtitleLabel("新建诊断方案", self))
        heading.addStretch()
        heading.addWidget(CaptionLabel("无需 DFT · 从一个初始结构开始", self))
        layout.addLayout(heading)

        self.body_scroll = ScrollArea(self)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.body_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget {background: transparent;}")
        body = QWidget(self)
        content = QVBoxLayout(body)
        content.setContentsMargins(0, 0, 10, 0)
        content.setSpacing(14)
        self.body_scroll.setWidget(body)
        layout.addWidget(self.body_scroll, 1)

        content.addWidget(BodyLabel("1  初始结构", self))
        source = QGridLayout()
        source.setHorizontalSpacing(12)
        source.addWidget(CaptionLabel("方案名称", self), 0, 0)
        source.addWidget(CaptionLabel("选择一个构型作为扫描起点", self), 0, 1)
        self.name = LineEdit(self)
        self.name.setAccessibleName("方案名称")
        self.name.setPlaceholderText("例如：Fe 晶格与磁性")
        self.name.setText("结构响应诊断")
        self.load_button = PushButton(FluentIcon.FOLDER, "导入结构…", self)
        self.load_button.clicked.connect(self.load_file)
        self.seed_combo = ComboBox(self)
        self.seed_combo.setAccessibleName("初始构型")
        self.seed_combo.setMinimumWidth(180)
        self.seed_combo.currentIndexChanged.connect(self._seed_changed)
        source.addWidget(self.name, 1, 0)
        source.addWidget(self.seed_combo, 1, 1)
        source.addWidget(self.load_button, 1, 2)
        source.setColumnStretch(0, 2)
        source.setColumnStretch(1, 3)
        content.addLayout(source)
        self.seed_hint = CaptionLabel(self)
        self.seed_hint.setWordWrap(True)
        content.addWidget(self.seed_hint)
        parameters_heading = QHBoxLayout()
        parameters_heading.addWidget(BodyLabel("2  选择实验与扫描范围", self))
        parameters_heading.addStretch()
        self.reset_button = TransparentPushButton(FluentIcon.SYNC, "恢复默认", self)
        self.reset_button.setToolTip("恢复默认实验和参数，清除物理条件勾选；保留方案名称和已导入结构。")
        self.reset_button.clicked.connect(self.reset_parameters)
        parameters_heading.addWidget(self.reset_button)
        content.addLayout(parameters_heading)
        self.parameters_hint = CaptionLabel("参数自动记忆；更换结构后，目标原子和物理条件需重新检查。", self)
        self.parameters_hint.setWordWrap(True)
        content.addWidget(self.parameters_hint)

        self.rows = {}
        self.experiment_panels = {}
        self.experiment_cards = {}
        cards = QGridLayout()
        cards.setSpacing(12)
        defaults = (
            ("volume", -8, 8, .2, "均匀缩放晶胞，检查能量与压力响应", "%"),
            ("displacement", -.1, .1, .01, "移动一个原子，检查局部能量与力", " Å"),
            ("spin_angle", -20, 20, 1., "转动一个磁矩，检查角度响应与磁力", "°"),
            ("spin_magnitude", .8, 1.2, .02, "缩放一个磁矩，检查纵向磁性响应", " 倍"),
        )
        for i, (kind, low, high, step, description, unit) in enumerate(defaults):
            card = QFrame(self)
            card.setObjectName("experimentChoice")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(9)
            enabled = CheckBox(KINDS[kind][0], card)
            enabled.setChecked(not kind.startswith("spin"))
            card_layout.addWidget(enabled)
            caption = CaptionLabel(description, card)
            caption.setWordWrap(True)
            card_layout.addWidget(caption)
            panel = QWidget(card)
            params = QGridLayout(panel)
            params.setContentsMargins(0, 2, 0, 0)
            params.setHorizontalSpacing(8)
            lo, hi, h = [AdaptiveCompactDoubleSpinBox(self) for _ in range(3)]
            for box in (lo, hi, h):
                box.setDecimals(4)
                box.setRange(-1000, 1000)
                box.valueChanged.connect(self.update_preview)
            lo.setValue(low)
            hi.setValue(high)
            for box in (lo, hi):
                box.setSingleStep({"volume": 1., "displacement": .01, "spin_angle": 1., "spin_magnitude": .05}[kind])
            lo.setSuffix(unit)
            hi.setSuffix(unit)
            h.setRange(.0001, 100)
            h.setValue(step)
            h.setSingleStep(step / 10)
            count = AdaptiveCompactSpinBox(self)
            count.setRange(3, 101)
            count.setValue(17)
            count.valueChanged.connect(self.update_preview)
            for column, (label, widget) in enumerate((("起点", lo), ("终点", hi), ("采样点数", count))):
                widget.setMinimumWidth(90)
                widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                widget.setAccessibleName(KINDS[kind][0] + label)
                params.addWidget(CaptionLabel(label, card), 0, column)
                params.addWidget(widget, 1, column)
                params.setColumnStretch(column, 1)
            h.setAccessibleName(KINDS[kind][0] + "差分步长")
            card_layout.addWidget(panel)
            card_layout.addStretch()
            enabled.toggled.connect(self.update_preview)
            self.rows[kind] = enabled, lo, hi, count, h
            self.experiment_panels[kind] = panel
            self.experiment_cards[kind] = card
            cards.addWidget(card, i // 2, i % 2)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        content.addLayout(cards)

        self.target_panel = QWidget(self)
        axes = QHBoxLayout(self.target_panel)
        axes.setContentsMargins(0, 0, 0, 0)
        axes.addWidget(CaptionLabel("目标原子（从 1 开始）", self))
        self.atom = SpinBox(self)
        self.atom.setRange(1, 1)
        self.atom.setAccessibleName("目标原子编号")
        axes.addWidget(self.atom)
        self.direction_label = CaptionLabel("位移 / 转动轴", self)
        axes.addWidget(self.direction_label)
        self.axis = ComboBox(self)
        self.axis.setAccessibleName("位移方向或转动轴")
        for axis in ("X", "Y", "Z"):
            self.axis.addItem(axis)
        axes.addWidget(self.axis)
        self.atom_hint = CaptionLabel(self)
        axes.addWidget(self.atom_hint, 1)
        content.addWidget(self.target_panel)

        self.magnetic_panel = QWidget(self)
        magnetic = QVBoxLayout(self.magnetic_panel)
        magnetic.setContentsMargins(0, 0, 0, 0)
        self.variable = CheckBox("模型支持连续可变磁矩幅值", self)
        self.variable_hint = CaptionLabel("需确认模型支持；未声明时，幅值实验运行时会跳过。", self)
        self.variable_hint.setWordWrap(True)
        self.zero_field = CheckBox("零外场且无时间反演破缺项：检查时间反演", self)
        self.no_soc = CheckBox("同时无 SOC 和自旋各向异性：检查整体自旋旋转", self)
        for widget in (self.variable, self.variable_hint, self.zero_field, self.no_soc):
            magnetic.addWidget(widget)
        for box in (self.variable, self.zero_field, self.no_soc):
            box.toggled.connect(self.update_preview)
        content.addWidget(self.magnetic_panel)

        self.advanced_button = CheckBox("高级设置 · 差分步长与容差", self)
        content.addWidget(self.advanced_button)
        self.advanced = QWidget(self)
        advanced = QGridLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.step_labels = {}
        for i, (kind, (_, lo, hi, count, h)) in enumerate(self.rows.items()):
            label = CaptionLabel(KINDS[kind][0] + "差分步长", self)
            self.step_labels[kind] = label
            advanced.addWidget(label, i // 2, (i % 2) * 2)
            advanced.addWidget(h, i // 2, (i % 2) * 2 + 1)
        self.atol = DoubleSpinBox(self)
        self.atol.setDecimals(7)
        self.atol.setRange(.0000001, 10)
        self.atol.setValue(.000001)
        self.rtol = DoubleSpinBox(self)
        self.rtol.setDecimals(3)
        self.rtol.setRange(0, 1)
        self.rtol.setValue(.02)
        for column, (label, widget) in enumerate((("绝对容差（eV / 横轴单位）", self.atol), ("相对容差", self.rtol))):
            widget.setAccessibleName(label)
            advanced.addWidget(CaptionLabel(label, self), 2, column * 2)
            advanced.addWidget(widget, 2, column * 2 + 1)
        content.addWidget(self.advanced)
        self.advanced.hide()
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        content.addStretch(1)

        divider = QFrame(self)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {palette.BORDER};")
        layout.addWidget(divider)
        bottom = QHBoxLayout()
        summary = QVBoxLayout()
        self.preview = CaptionLabel(self)
        self.preview.setWordWrap(True)
        summary.addWidget(self.preview)
        persistence_hint = CaptionLabel("创建后保存结构、参数和运行结果；仅临时试用的方案会在重启后清除。", self)
        persistence_hint.setWordWrap(True)
        summary.addWidget(persistence_hint)
        bottom.addLayout(summary, 1)
        self.temporary_button = PushButton("仅临时试用", self)
        self.temporary_button.setToolTip("仅本次会话有效，重启后方案和结果会清除；可稍后从顶部菜单保存。")
        self.temporary_button.setEnabled(False)
        self.temporary_button.clicked.connect(lambda: self.prepare(temporary=True))
        bottom.addWidget(self.temporary_button)
        self.use_button = PrimaryPushButton(FluentIcon.SAVE, "创建并保存", self)
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(lambda: self.prepare())
        bottom.addWidget(self.use_button)
        layout.addLayout(bottom)
        for widget in (self.atom, self.atol, self.rtol):
            widget.valueChanged.connect(self.update_preview)
        self.axis.currentIndexChanged.connect(self.update_preview)
        self.name.textChanged.connect(self.update_preview)
        tab_order = [close, self.name, self.seed_combo, self.load_button, self.reset_button]
        for enabled, lo, hi, count, _h in self.rows.values():
            tab_order.extend((enabled, lo, hi, count))
        tab_order.extend((self.atom, self.axis, self.variable, self.zero_field, self.no_soc, self.advanced_button))
        tab_order.extend(row[4] for row in self.rows.values())
        tab_order.extend((self.atol, self.rtol, self.temporary_button, self.use_button))
        for before, after in zip(tab_order, tab_order[1:]):
            QWidget.setTabOrder(before, after)
        self._numeric_parameters = {"abs_tolerance": self.atol, "rel_tolerance": self.rtol}
        for kind, (_, lo, hi, count, h) in self.rows.items():
            self._numeric_parameters.update(zip(
                (f"{kind}.{key}" for key in ("low", "high", "points", "step")), (lo, hi, count, h)
            ))
        self._default_parameters = self._parameter_values()
        try:
            stored = json.loads(Config.get("validation", PARAMETERS_OPTION) or "{}")
        except (ValueError, TypeError):
            stored = {}
        self._apply_parameters(stored)
        for widget in self._numeric_parameters.values():
            widget.valueChanged.connect(self._remember_parameters)
        for enabled, *_ in self.rows.values():
            enabled.toggled.connect(self._remember_parameters)
        self.axis.currentIndexChanged.connect(self._remember_parameters)

    def _parameter_values(self):
        # Structure/model-specific declarations and atom indices are never preferences.
        return dict(values={key: widget.value() for key, widget in self._numeric_parameters.items()},
                    experiments={kind: row[0].isChecked() for kind, row in self.rows.items()},
                    axis=self.axis.currentIndex())

    def _apply_parameters(self, parameters):
        self._restoring_parameters = True
        try:
            if isinstance(parameters, dict):
                values = parameters.get("values", {})
                for key, widget in self._numeric_parameters.items():
                    value = values.get(key) if isinstance(values, dict) else None
                    if type(value) not in (int, float) or not math.isfinite(value):
                        continue
                    if type(widget.value()) is int and type(value) is not int:
                        continue
                    if widget.minimum() <= value <= widget.maximum():
                        widget.setValue(value)
                enabled = parameters.get("experiments", {})
                for kind, row in self.rows.items():
                    value = enabled.get(kind) if isinstance(enabled, dict) else None
                    if type(value) is bool:
                        row[0].setChecked(value)
                axis = parameters.get("axis")
                if type(axis) is int and 0 <= axis < self.axis.count():
                    self.axis.setCurrentIndex(axis)
        finally:
            self._restoring_parameters = False
        self.update_preview()

    def _remember_parameters(self, *_args):
        if not self._restoring_parameters:
            try:
                Config.set("validation", PARAMETERS_OPTION, json.dumps(self._parameter_values()))
                self.parameters_hint.setText("参数自动记忆；更换结构后，目标原子和物理条件需重新检查。")
            except (SQLAlchemyError, OSError):
                self.parameters_hint.setText("参数记忆保存失败，当前填写仍可使用，但重启后可能无法恢复。")

    def reset_parameters(self):
        self._apply_parameters(self._default_parameters)
        self.advanced_button.setChecked(False)
        self._seed_changed()
        self._remember_parameters()

    def _seed_changed(self, *_args):
        self.atom.setValue(1)
        for box in (self.variable, self.zero_field, self.no_soc):
            box.setChecked(False)
        self.update_preview()

    def load_experiments(self, name, experiments):
        """Edit a copy of compatible selected experiments, never mutate a saved plan."""
        from dataclasses import asdict
        from NepTrainKit.core.validation.protocols.diagnostic import encode_experiment
        seed, spec = experiments[0]
        shared = lambda item: {k: v for k, v in asdict(item).items() if k not in ("kind", "low", "high", "points", "derivative_step")}
        snapshot = encode_experiment(seed, spec)
        if len({item.kind for _, item in experiments}) != len(experiments) or any(
            shared(item) != shared(spec) or encode_experiment(atoms, spec) != snapshot
            for atoms, item in experiments
        ):
            raise ValueError("所选实验的初始结构或共享条件不同，请先放大一项实验，再编辑其副本。")
        self.set_seeds([seed])
        self.name.setText(name.removesuffix(" · 临时") + "（副本）")
        self.atom.setValue(spec.atom + 1)
        self.axis.setCurrentIndex(max(range(3), key=lambda i: abs(spec.axis[i])))
        axis = tuple(float(i == self.axis.currentIndex()) for i in range(3))
        if tuple(spec.axis) != axis:
            raise ValueError("此实验使用自定义方向；当前编辑器支持 X、Y、Z 正方向。")
        self.atol.setValue(spec.abs_tolerance)
        self.rtol.setValue(spec.rel_tolerance)
        self.variable.setChecked(spec.variable_moments)
        self.zero_field.setChecked(spec.zero_field)
        self.no_soc.setChecked(spec.no_soc)
        selected = {item.kind: item for _, item in experiments}
        for kind, (enabled, lo, hi, count, h) in self.rows.items():
            enabled.setChecked(kind in selected)
            if kind in selected:
                item = selected[kind]
                lo.setValue(item.low)
                hi.setValue(item.high)
                count.setValue(item.points)
                h.setValue(item.derivative_step)
        self.update_preview()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择初始结构", "", "结构文件 (*.xyz *.extxyz *.vasp *.cif POSCAR CONTCAR);;所有文件 (*)")
        if not path:
            return
        # A failed replacement cannot leave a stale preview available for submission.
        self.set_seeds([])
        try:
            self.set_seeds(read_seed_structures(path))
        except Exception as exc:
            self.preview.setText("结构读取失败：" + str(exc))

    def set_seeds(self, seeds):
        self._seeds = [clean_seed(seed) for seed in seeds]
        self.seed_combo.blockSignals(True)
        self.seed_combo.clear()
        for index, seed in enumerate(self._seeds):
            self.seed_combo.addItem(f"{index + 1} · {seed.get_chemical_formula()} · {len(seed)} 原子")
        self.seed_combo.blockSignals(False)
        self._seed_changed()

    def experiments(self):
        if not self.name.text().strip() or not self._seeds:
            raise ValueError("填写方案名并选择初始结构。")
        seed = self._seeds[self.seed_combo.currentIndex()]
        axis = tuple(float(i == self.axis.currentIndex()) for i in range(3))
        experiments = []
        for kind, (enabled, lo, hi, count, h) in self.rows.items():
            if not enabled.isChecked():
                continue
            spec = DiagnosticSpec(kind=kind, low=lo.value(), high=hi.value(), points=count.value(),
                derivative_step=h.value(), atom=self.atom.value()-1, axis=axis,
                abs_tolerance=self.atol.value(), rel_tolerance=self.rtol.value(),
                zero_field=self.zero_field.isChecked(), no_soc=self.no_soc.isChecked(),
                variable_moments=self.variable.isChecked())
            spec.validate(seed)
            experiments.append((seed.copy(), spec))
        if not experiments:
            raise ValueError("请至少选择一项实验。")
        return experiments

    def update_preview(self, *_args):
        if not hasattr(self, "use_button"):
            return
        selected = {kind for kind, row in self.rows.items() if row[0].isChecked()}
        for kind, panel in self.experiment_panels.items():
            active = kind in selected
            panel.setVisible(active)
            self.rows[kind][4].setVisible(active)
            self.step_labels[kind].setVisible(active)
            border = palette.ACCENT if active else palette.BORDER
            self.experiment_cards[kind].setStyleSheet(
                f"#experimentChoice {{background: {palette.SURFACE}; border: 1px solid {border}; border-radius: 8px;}}")
        self.target_panel.setVisible(bool(selected - {"volume"}))
        direction = bool(selected & {"displacement", "spin_angle"})
        self.axis.setVisible(direction)
        self.direction_label.setVisible(direction)
        magnetic = bool(selected & {"spin_angle", "spin_magnitude"})
        self.magnetic_panel.setVisible(magnetic)
        self.variable.setVisible("spin_magnitude" in selected)
        self.variable_hint.setVisible("spin_magnitude" in selected and not self.variable.isChecked())
        self.no_soc.setEnabled(self.zero_field.isChecked())
        if self._seeds:
            from NepTrainKit.core.magnetism import existing_moment_vectors
            seed = self._seeds[self.seed_combo.currentIndex()]
            self.atom.setMaximum(max(1, len(seed)))
            moments = existing_moment_vectors(seed, lift_scalar=False)
            moment_note = "含矢量磁矩，可配置磁性实验" if moments is not None else "未提供矢量磁矩；磁性实验需要带 spin / magmom 的结构"
            self.seed_hint.setText(f"已载入 {len(self._seeds)} 个构型 · {moment_note} · 不使用已有能量 / 力标签")
            index = self.atom.value() - 1
            self.atom_hint.setText((seed[index].symbol + (f" · 磁矩 ({moments[index][0]:.2f}, {moments[index][1]:.2f}, {moments[index][2]:.2f})" if moments is not None else "")) if len(seed) else "空结构")
        else:
            self.seed_hint.setText("可直接使用 show_nep 带入的结构，或导入 XYZ / POSCAR / CIF；无需能量和力标签。")
            self.atom_hint.clear()
        try:
            experiments = self.experiments()
            count = sum(len(spec.coordinates()) for _, spec in experiments)
            note = f"{len(experiments)} 项实验 · {count} 个扫描结构（含初始状态）· 运行时另做差分 / 对称性检查"
            if any(spec.kind == "spin_magnitude" and not spec.variable_moments for _, spec in experiments):
                note += "；幅值实验尚未声明适用条件，运行时将跳过"
            self.preview.setText(note)
            self.use_button.setEnabled(True)
            self.temporary_button.setEnabled(True)
        except ValueError as exc:
            self.preview.setText(str(exc))
            self.use_button.setEnabled(False)
            self.temporary_button.setEnabled(False)

        self.readinessChanged.emit()

    def prepare(self, *, temporary=False):
        try:
            self.prepared.emit(self.name.text().strip(), self.experiments(), temporary)
        except ValueError as exc:
            self.preview.setText(str(exc))
