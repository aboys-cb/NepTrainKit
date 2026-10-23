"""Inline experiment setup for the existing validation workbench."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, LineEdit, ComboBox, SpinBox, DoubleSpinBox, CheckBox, PushButton, PrimaryPushButton, TransparentToolButton, FluentIcon
from ase.io import read

from NepTrainKit import palette
from NepTrainKit.core.validation.protocols.diagnostic import DiagnosticSpec, KINDS, clean_seed


class DiagnosticEditor(QFrame):
    prepared = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("diagnosticEditor")
        self.setStyleSheet(f"#diagnosticEditor {{background:{palette.SURFACE}; border:1px solid {palette.BORDER}; border-radius:10px;}}")
        self._seeds = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        heading = QHBoxLayout()
        heading.addWidget(BodyLabel("新建无参考诊断", self))
        heading.addStretch()
        close = TransparentToolButton(FluentIcon.CLOSE, self)
        close.setToolTip("收起实验设置")
        close.clicked.connect(self.hide)
        heading.addWidget(close)
        layout.addLayout(heading)
        row = QHBoxLayout()
        self.name = LineEdit(self)
        self.name.setPlaceholderText("方案名称")
        self.name.setText("结构响应诊断")
        self.load_button = PushButton(FluentIcon.FOLDER, "选择初始结构…", self)
        self.load_button.clicked.connect(self.load_file)
        self.seed_combo = ComboBox(self)
        self.seed_combo.setMinimumWidth(180)
        self.seed_combo.currentIndexChanged.connect(self.update_preview)
        row.addWidget(self.name, 2)
        row.addWidget(self.load_button)
        row.addWidget(self.seed_combo, 3)
        layout.addLayout(row)
        axes = QHBoxLayout()
        axes.addWidget(CaptionLabel("目标原子（从 1 开始）", self))
        self.atom = SpinBox(self)
        self.atom.setRange(1, 1)
        axes.addWidget(self.atom)
        axes.addWidget(CaptionLabel("位移 / 转动方向", self))
        self.axis = ComboBox(self)
        for axis in ("X", "Y", "Z"):
            self.axis.addItem(axis)
        axes.addWidget(self.axis)
        axes.addStretch()
        layout.addLayout(axes)
        grid = QGridLayout()
        for column, text in enumerate(("实验", "下限", "上限", "点数", "差分步长")):
            grid.addWidget(CaptionLabel(text, self), 0, column)
        self.rows = {}
        defaults = (("volume", -8, 8, .2), ("displacement", -.1, .1, .01),
                    ("spin_angle", -20, 20, 1.), ("spin_magnitude", .8, 1.2, .02))
        for i, (kind, low, high, step) in enumerate(defaults, 1):
            enabled = CheckBox(KINDS[kind][0] + " · " + KINDS[kind][1], self)
            enabled.setChecked(not kind.startswith("spin"))
            lo, hi, h = [DoubleSpinBox(self) for _ in range(3)]
            for box in (lo, hi, h):
                box.setDecimals(4)
                box.setRange(-1000, 1000)
                box.valueChanged.connect(self.update_preview)
            lo.setValue(low)
            hi.setValue(high)
            h.setRange(.0001, 100)
            h.setValue(step)
            count = SpinBox(self)
            count.setRange(3, 101)
            count.setValue(17)
            count.valueChanged.connect(self.update_preview)
            enabled.toggled.connect(self.update_preview)
            for j, widget in enumerate((enabled, lo, hi, count, h)):
                grid.addWidget(widget, i, j)
            self.rows[kind] = enabled, lo, hi, count, h
        layout.addLayout(grid)
        self.advanced_button = CheckBox("适用条件与检查容差", self)
        layout.addWidget(self.advanced_button)
        self.advanced = QWidget(self)
        advanced = QVBoxLayout(self.advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.variable = CheckBox("模型支持连续可变磁矩幅值（仅有 spin 输入不代表支持）", self)
        self.zero_field = CheckBox("零外场、无显式时间反演破缺项：启用时间反演检查", self)
        self.no_soc = CheckBox("无 SOC、无自旋各向异性项：同时满足上一项时检查整体自旋旋转", self)
        for box in (self.variable, self.zero_field, self.no_soc):
            advanced.addWidget(box)
            box.toggled.connect(self.update_preview)
        tolerance = QHBoxLayout()
        tolerance.addWidget(CaptionLabel("导数绝对容差（eV / 横轴单位）", self))
        self.atol = DoubleSpinBox(self)
        self.atol.setDecimals(7)
        self.atol.setRange(.0000001, 10)
        self.atol.setValue(.000001)
        tolerance.addWidget(self.atol)
        tolerance.addWidget(CaptionLabel("相对容差", self))
        self.rtol = DoubleSpinBox(self)
        self.rtol.setDecimals(3)
        self.rtol.setRange(0, 1)
        self.rtol.setValue(.02)
        tolerance.addWidget(self.rtol)
        advanced.addLayout(tolerance)
        layout.addWidget(self.advanced)
        self.advanced.hide()
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        self.preview = CaptionLabel("选择未标注的初始结构；已有能量和力标签不会用于本次诊断。", self)
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        bottom = QHBoxLayout()
        self.use_button = PrimaryPushButton(FluentIcon.ACCEPT, "使用临时方案", self)
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self.prepare)
        bottom.addWidget(CaptionLabel("不自动入库 · 保存方案在顶部保存菜单中", self), 1)
        bottom.addWidget(self.use_button)
        layout.addLayout(bottom)
        layout.addStretch(1)
        for widget in (self.atom, self.atol, self.rtol):
            widget.valueChanged.connect(self.update_preview)
        self.axis.currentIndexChanged.connect(self.update_preview)
        self.name.textChanged.connect(self.update_preview)

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
            self.set_seeds(read(path, index=":"))
        except Exception as exc:
            self.preview.setText("结构读取失败：" + str(exc))

    def set_seeds(self, seeds):
        self._seeds = [clean_seed(seed) for seed in seeds]
        self.seed_combo.blockSignals(True)
        self.seed_combo.clear()
        for index, seed in enumerate(self._seeds):
            self.seed_combo.addItem(f"{index + 1} · {seed.get_chemical_formula()} · {len(seed)} 原子")
        self.seed_combo.blockSignals(False)
        self.update_preview()

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
        if self._seeds:
            self.atom.setMaximum(len(self._seeds[self.seed_combo.currentIndex()]))
        try:
            experiments = self.experiments()
            count = sum(len(spec.coordinates()) for _, spec in experiments)
            note = f"{len(experiments)} 项实验 · {count} 个扫描结构（自动包含初始状态）· 另有差分和对称性探针"
            if any(spec.kind == "spin_magnitude" and not spec.variable_moments for _, spec in experiments):
                note += "；幅值实验尚未声明适用条件，运行时将跳过"
            self.preview.setText(note)
            self.use_button.setEnabled(True)
        except ValueError as exc:
            self.preview.setText(str(exc))
            self.use_button.setEnabled(False)

    def prepare(self):
        try:
            self.prepared.emit(self.name.text().strip(), self.experiments())
        except ValueError as exc:
            self.preview.setText(str(exc))
