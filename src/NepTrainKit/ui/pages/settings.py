#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 19:44
# @email    : 1747193328@qq.com


from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QWidget
 
from qfluentwidgets import SettingCardGroup, HyperlinkCard, PrimaryPushSettingCard, ExpandLayout, OptionsConfigItem, \
    OptionsValidator, EnumSerializer, SwitchSettingCard, FluentIcon, ScrollArea, SettingCard, ComboBox

from NepTrainKit.config import Config
from NepTrainKit.i18n import LANGUAGE_LABELS, SUPPORTED_LANGUAGES, normalize_language
from NepTrainKit.logging_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVELS,
    normalize_log_level,
    set_log_level,
)
from NepTrainKit.ui.messages import MessageManager
from NepTrainKit.ui.widgets import MyComboBoxSettingCard, DoubleSpinBoxSettingCard, LineEditSettingCard
from NepTrainKit.ui.widgets import ColorSettingCard
from NepTrainKit.core.types import (
    ForcesMode,
    CanvasMode,
    NepBackend,
    DataPrecision,
    parse_forces_mode,
    parse_data_precision,
)
from NepTrainKit.ui.update import UpdateWoker, UpdateNEP89Woker, get_pending_update_version
from NepTrainKit.version import HELP_URL, FEEDBACK_URL, __version__, YEAR, AUTHOR


class SettingsWidget(ScrollArea):
    """Provide the scrollable settings page for the NEP application.

    Parameters
    ----------
    parent : QWidget
        Parent container that hosts the settings page.
    """

    def __init__(self,parent):
        """Initialise setting groups, cards, and default values.

        Parameters
        ----------
        parent : QWidget
            Parent container that hosts the settings page.
        """

        super().__init__(parent)
        self.setObjectName('SettingsWidget')
        self.scrollWidget = QWidget()

        self.expand_layout = ExpandLayout(self.scrollWidget)

        self.personal_group = SettingCardGroup(
            self.tr('Personalization'), self.scrollWidget)
        self.nep_group = SettingCardGroup(
            self.tr('NEP Settings'), self.scrollWidget)
        self.plot_group = SettingCardGroup(
            self.tr('Plot Settings'), self.scrollWidget)

        force_mode = parse_forces_mode(Config.get("widget", "forces_data", ForcesMode.Raw))
        default_forces = force_mode.value

        self.optimization_forces_card = MyComboBoxSettingCard(
            OptionsConfigItem("forces", "forces", force_mode, OptionsValidator(ForcesMode), EnumSerializer(ForcesMode)),
            FluentIcon.BRUSH,
            self.tr('Force data format'),
            self.tr("Streamline data and speed up drawing"),
            texts=[
             mode.value for mode in    ForcesMode
            ],
            default=default_forces,
            parent=self.personal_group
        )
        canvas_type = Config.get("widget","canvas_type",str(CanvasMode.AUTO.value))

        self.canvas_card = MyComboBoxSettingCard(
            OptionsConfigItem("canvas","canvas",CanvasMode(canvas_type),OptionsValidator(CanvasMode), EnumSerializer(CanvasMode)),
            FluentIcon.BRUSH,
            self.tr('Canvas Engine'),
            self.tr("Choose GPU with vispy"),
            texts=[
             mode.value for mode in    CanvasMode

            ],
            default=canvas_type,
            parent=self.personal_group
        )

        language_config = normalize_language(Config.get("ui", "language", "auto"))
        self.language_card = SettingCard(
            FluentIcon.SETTING,
            self.tr("Language"),
            self.tr("Restart NepTrainKit to apply language changes"),
            self.personal_group,
        )
        self.language_combo = ComboBox(self.language_card)
        for value in SUPPORTED_LANGUAGES:
            self.language_combo.addItem(self.tr(LANGUAGE_LABELS[value]), userData=value)
        self.language_combo.setCurrentIndex(SUPPORTED_LANGUAGES.index(language_config))
        self.language_card.hBoxLayout.addWidget(self.language_combo, 0, Qt.AlignmentFlag.AlignRight)
        self.language_card.hBoxLayout.addSpacing(16)

        log_level = normalize_log_level(
            Config.get("logging", "level", DEFAULT_LOG_LEVEL)
        )
        self.log_level_card = SettingCard(
            FluentIcon.INFO,
            self.tr("Log level"),
            self.tr(
                "Minimum level written to the console and log file; applies immediately"
            ),
            self.personal_group,
        )
        self.log_level_combo = ComboBox(self.log_level_card)
        for level in LOG_LEVELS:
            self.log_level_combo.addItem(level, userData=level)
        self.log_level_combo.setCurrentIndex(LOG_LEVELS.index(log_level))
        self.log_level_card.hBoxLayout.addWidget(
            self.log_level_combo, 0, Qt.AlignmentFlag.AlignRight
        )
        self.log_level_card.hBoxLayout.addSpacing(16)


        auto_load_config = Config.getboolean("widget","auto_load",False)

        sort_atoms_config = Config.getboolean("widget", "sort_atoms", False)

        use_group_menu_config = Config.getboolean("widget", "use_group_menu", False)

        self.auto_load_card = SwitchSettingCard(
            QIcon(":/images/src/images/auto_load.svg"),
            self.tr('Auto loading'),
            self.tr('Detect startup path data and load'),

            parent=self.personal_group
        )
        self.auto_load_card.setValue(auto_load_config)

        self.sort_atoms_card = SwitchSettingCard(
            QIcon(":/images/src/images/sort.svg"),
            self.tr('Sort atoms'),
            self.tr('Sort atoms in structures when processing cards'),
            parent=self.personal_group
        )
        self.sort_atoms_card.setValue(sort_atoms_config)

        self.use_group_menu_card = SwitchSettingCard(
            QIcon(":/images/src/images/group.svg"),
            self.tr('Use card group menu'),
            self.tr('Group cards by "group" in console menu'),
            parent=self.personal_group
        )
        self.use_group_menu_card.setValue(use_group_menu_config)
        preserve_deepmd = Config.getboolean("widget", "deepmd_preserve_subfolders", True)
        self.deepmd_preserve_card = SwitchSettingCard(
            FluentIcon.FOLDER,
            self.tr('Keep DeepMD subfolders'),
            self.tr('Preserve imported folder hierarchy when exporting deepmd/npy'),
            parent=self.personal_group
        )
        self.deepmd_preserve_card.setValue(preserve_deepmd)

        cache_outputs = Config.getboolean("io", "cache_outputs", True)
        self.cache_outputs_card = SwitchSettingCard(
            FluentIcon.SAVE,
            self.tr('Cache output files'),
            self.tr('Cache *.out and descriptor.out for faster reload (NEP & DeepMD)'),
            parent=self.personal_group
        )
        self.cache_outputs_card.setValue(cache_outputs)

        auto_structure_evidence = Config.getboolean(
            "training_set_audit", "auto_structure_evidence", True
        )
        self.auto_structure_evidence_card = SwitchSettingCard(
            FluentIcon.SYNC,
            self.tr("Automatically analyze structure evidence"),
            self.tr(
                "After the basic dataset audit appears, analyze phases and magnetic order in the background"
            ),
            parent=self.personal_group,
        )
        self.auto_structure_evidence_card.setValue(auto_structure_evidence)

        export_digits = Config.getint("io", "export_significant_digits", 10) or 10
        self.export_digits_card = DoubleSpinBoxSettingCard(
            FluentIcon.SAVE,
            self.tr('Export significant digits'),
            self.tr('Significant digits for per-atom XYZ/extxyz values'),
            self.personal_group
        )
        self.export_digits_card.setRange(6, 17)
        self.export_digits_card.doubleSpinBox.setDecimals(0)
        self.export_digits_card.doubleSpinBox.setSingleStep(1)
        self.export_digits_card.setValue(float(export_digits))
        # Default Config_type text
        default_cfg_type = Config.get("widget", "default_config_type", "neptrainkit")
        self.default_cfg_type_card = LineEditSettingCard(
            FluentIcon.TAG,
            self.tr('Default Config_type'),
            self.tr('Tag assigned when source has no Config_type'),
            self.personal_group
        )
        self.default_cfg_type_card.setValue(default_cfg_type)
        radius_coefficient_config=Config.getfloat("widget","radius_coefficient",0.7)

        self.radius_coefficient_Card = DoubleSpinBoxSettingCard(

            FluentIcon.ALBUM,
            self.tr('Covalent radius coefficient'),
            self.tr('Coefficient used to detect bond length'),
            self.personal_group
        )
        self.radius_coefficient_Card.setValue(radius_coefficient_config)
        self.radius_coefficient_Card.setRange(0.0, 1.5)

        # NEP backend selection
        nep_backend_default = Config.get("nep", "backend","auto")
        self.nep_backend_card = MyComboBoxSettingCard(
            OptionsConfigItem("nep", "backend", NepBackend(nep_backend_default), OptionsValidator(NepBackend),
                              EnumSerializer(NepBackend)),
            QIcon(":/images/src/images/gpu.svg"),
            self.tr('NEP Backend'),
            self.tr('Select CPU/CUDA or let Auto use CUDA when available'),
            texts=[mode.value for mode in NepBackend],
            default=nep_backend_default,
            parent=self.nep_group
        )

        data_precision_default = parse_data_precision(Config.get("nep", "data_precision", DataPrecision.FLOAT32)).value
        self.data_precision_card = MyComboBoxSettingCard(
            OptionsConfigItem(
                "nep",
                "data_precision",
                DataPrecision(data_precision_default),
                OptionsValidator(DataPrecision),
                EnumSerializer(DataPrecision),
            ),
            FluentIcon.TAG,
            self.tr('Data Precision'),
            self.tr('Choose storage precision for imported DFT/structure data'),
            texts=[mode.value for mode in DataPrecision],
            default=data_precision_default,
            parent=self.nep_group
        )

        chunk_max_atoms = Config.getint("nep", "chunk_max_atoms", 100000) or 100000
        self.chunk_max_atoms_card = DoubleSpinBoxSettingCard(
            FluentIcon.SPEED_HIGH,
            self.tr('NEP Chunk Max Atoms'),
            self.tr('Maximum total atoms per prediction chunk on CPU or CUDA'),
            self.nep_group
        )
        self.chunk_max_atoms_card.setRange(1, 100000000)
        self.chunk_max_atoms_card.setValue(float(chunk_max_atoms))

        # Plot settings defaults
        edge_color = Config.get("plot", "marker_edge_color", "#07519C")
        face_color = Config.get("plot", "marker_face_color", "#FFFFFF")
        face_alpha = Config.getint("plot", "marker_face_alpha", 0) or 0
        selected_color = Config.get("plot", "selected_color", "#FF0000")
        show_color = Config.get("plot", "show_color", "#00FF00")
        current_color = Config.get("plot", "current_color", "#FF0000")
        structure_bg_color = Config.get("widget", "structure_bg_color", "#FFFFFF")
        structure_lattice_color = Config.get("widget", "structure_lattice_color", "#000000")
        pg_size = Config.getint("widget", "pg_marker_size", 7) or 7
        vispy_size = Config.getint("widget", "vispy_marker_size", 6) or 6
        vispy_aa = Config.getfloat("widget", "vispy_marker_antialias", 0.5) or 0.5
        current_size = Config.getint("plot", "current_marker_size", 20) or 20

        self.edge_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Scatter edge color'),
            self.tr('Default edge color for points'),
            self.plot_group
        )
        self.edge_color_card.setValue(edge_color)

        self.face_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Scatter face color'),
            self.tr('Default fill color for points'),
            self.plot_group
        )
        self.face_color_card.setValue(face_color)

        self.face_alpha_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            self.tr('Face alpha (0-255)'),
            self.tr('Alpha channel for fill color'),
            self.plot_group
        )
        self.face_alpha_card.setRange(0, 255)
        self.face_alpha_card.setValue(float(face_alpha))

        self.pg_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            self.tr('PyQtGraph scatter size'),
            self.tr('Marker size for PyQtGraph canvas'),
            self.plot_group
        )
        self.pg_size_card.setRange(1, 100)
        self.pg_size_card.setValue(float(pg_size))

        self.vispy_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            self.tr('VisPy scatter size'),
            self.tr('Marker size for VisPy canvas'),
            self.plot_group
        )
        self.vispy_size_card.setRange(1, 100)
        self.vispy_size_card.setValue(float(vispy_size))

        self.vispy_aa_card = DoubleSpinBoxSettingCard(
            FluentIcon.BRUSH,
            self.tr('VisPy antialias'),
            self.tr('Marker antialias value for VisPy (0-2)'),
            self.plot_group
        )
        self.vispy_aa_card.setRange(0.0, 2.0)
        self.vispy_aa_card.setValue(float(vispy_aa))

        self.structure_bg_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Structure background'),
            self.tr('Background color for lattice/structure viewer'),
            self.plot_group
        )
        self.structure_bg_color_card.setValue(structure_bg_color)

        self.structure_lattice_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Lattice line color'),
            self.tr('Line color for lattice edges in structure viewer'),
            self.plot_group
        )
        self.structure_lattice_color_card.setValue(structure_lattice_color)

        self.selected_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Selected color'),
            self.tr('Color for selected points'),
            self.plot_group
        )
        self.selected_color_card.setValue(selected_color)

        self.show_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Show color'),
            self.tr('Color for highlighted "show" points'),
            self.plot_group
        )
        self.show_color_card.setValue(show_color)

        self.current_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            self.tr('Current marker color'),
            self.tr('Color for current star marker'),
            self.plot_group
        )
        self.current_color_card.setValue(current_color)

        self.current_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            self.tr('Current marker size'),
            self.tr('Size of current star marker'),
            self.plot_group
        )
        self.current_size_card.setRange(5, 100)
        self.current_size_card.setValue(float(current_size))


        self.about_group = SettingCardGroup(self.tr("About"), self.scrollWidget)
        self.help_card = HyperlinkCard(
            HELP_URL,
            self.tr('Open Help Page'),
            FluentIcon.HELP,
            self.tr('Help'),
            self.tr('Discover new features and learn useful tips about NepTrainKit.'),
            self.about_group
        )
        self.feedback_card = PrimaryPushSettingCard(
            self.tr("Submit Feedback"),
            FluentIcon.FEEDBACK,
            self.tr("Submit Feedback"),

            self.tr('Help us improve NepTrainKit by providing feedback.'),
            self.about_group
        )
        self.about_card = PrimaryPushSettingCard(
            self.tr('Check for Updates'),
            FluentIcon.INFO,
            self.tr("About"),
            self._base_about_description(),
            self.about_group
        )
        self.about_nep89_card = PrimaryPushSettingCard(
            self.tr('Check and update'),
            FluentIcon.INFO,
            self.tr("About NEP89"),
            self.tr("NEP official NEP89 large model"),
            self.about_group
        )
        self._refresh_about_update_content()
        self.init_layout()
        self.init_signal()

    def init_layout(self):
        """Construct the scrollable layout and register setting cards.

        Returns
        -------
        None
            All setting groups are added to the expand layout.
        """
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.scrollWidget.setLayout(self.expand_layout)



        self.personal_group.addSettingCard(self.optimization_forces_card)
        self.personal_group.addSettingCard(self.canvas_card)
        self.personal_group.addSettingCard(self.language_card)
        self.personal_group.addSettingCard(self.log_level_card)
        self.personal_group.addSettingCard(self.auto_load_card)
        self.personal_group.addSettingCard(self.radius_coefficient_Card)
        self.personal_group.addSettingCard(self.sort_atoms_card)
        self.personal_group.addSettingCard(self.use_group_menu_card)
        self.personal_group.addSettingCard(self.deepmd_preserve_card)
        self.personal_group.addSettingCard(self.cache_outputs_card)
        self.personal_group.addSettingCard(self.auto_structure_evidence_card)
        self.personal_group.addSettingCard(self.export_digits_card)
        self.personal_group.addSettingCard(self.default_cfg_type_card)

        self.nep_group.addSettingCard(self.nep_backend_card)
        self.nep_group.addSettingCard(self.data_precision_card)
        self.nep_group.addSettingCard(self.chunk_max_atoms_card)

 

        self.about_group.addSettingCard(self.about_nep89_card)
        self.about_group.addSettingCard(self.help_card)
        self.about_group.addSettingCard(self.feedback_card)
        self.about_group.addSettingCard(self.about_card)



        self.expand_layout.addWidget(self.personal_group)
        # add plot setting cards into group before adding to layout
        self.plot_group.addSettingCard(self.edge_color_card)
        self.plot_group.addSettingCard(self.face_color_card)
        self.plot_group.addSettingCard(self.face_alpha_card)
        self.plot_group.addSettingCard(self.pg_size_card)
        self.plot_group.addSettingCard(self.vispy_size_card)
        self.plot_group.addSettingCard(self.vispy_aa_card)
        self.plot_group.addSettingCard(self.structure_bg_color_card)
        self.plot_group.addSettingCard(self.structure_lattice_color_card)
        self.plot_group.addSettingCard(self.selected_color_card)
        self.plot_group.addSettingCard(self.show_color_card)
        self.plot_group.addSettingCard(self.current_color_card)
        self.plot_group.addSettingCard(self.current_size_card)
        self.expand_layout.addWidget(self.plot_group)
        self.expand_layout.addWidget(self.nep_group)

        self.expand_layout.addWidget(self.about_group)

    def init_signal(self):
        """Connect widget signals to configuration update callbacks.

        Returns
        -------
        None
            Hooks persist user choices into the ``Config`` store.
        """
        self.canvas_card.optionChanged.connect(lambda option:Config.set("widget","canvas_type",option ))
        self.radius_coefficient_Card.valueChanged.connect(lambda value:Config.set("widget","radius_coefficient",value))
        self.optimization_forces_card.optionChanged.connect(lambda option:Config.set("widget","forces_data",option ))
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.log_level_combo.currentIndexChanged.connect(self._on_log_level_changed)
        self.about_card.clicked.connect(self.check_update)
        self.about_nep89_card.clicked.connect(self.check_update_nep89)

        self.nep_backend_card.optionChanged.connect(lambda option: Config.set("nep", "backend", option))
        self.data_precision_card.optionChanged.connect(lambda option: Config.set("nep", "data_precision", option))
        self.chunk_max_atoms_card.valueChanged.connect(lambda value: Config.set("nep", "chunk_max_atoms", int(value)))

        self.auto_load_card.checkedChanged.connect(lambda state:Config.set("widget","auto_load",state))
        self.sort_atoms_card.checkedChanged.connect(lambda state:Config.set("widget","sort_atoms",state))
        self.use_group_menu_card.checkedChanged.connect(lambda state:Config.set("widget","use_group_menu",state))
        self.deepmd_preserve_card.checkedChanged.connect(lambda state: Config.set("widget", "deepmd_preserve_subfolders", state))
        self.cache_outputs_card.checkedChanged.connect(lambda state: Config.set("io", "cache_outputs", state))
        self.auto_structure_evidence_card.checkedChanged.connect(
            lambda state: Config.set(
                "training_set_audit", "auto_structure_evidence", state
            )
        )
        self.export_digits_card.valueChanged.connect(lambda value: Config.set("io", "export_significant_digits", int(value)))
        self.default_cfg_type_card.textChanged.connect(lambda v: Config.set("widget", "default_config_type", v))
        # plot settings
        from NepTrainKit.core.types import Pens, Brushes
        def refresh_styles():
            Pens.update_from_config()
            Brushes.update_from_config()

        self.edge_color_card.colorChanged.connect(lambda v: (Config.set("plot", "marker_edge_color", v), refresh_styles()))
        self.face_color_card.colorChanged.connect(lambda v: (Config.set("plot", "marker_face_color", v), refresh_styles()))
        self.face_alpha_card.valueChanged.connect(lambda v: (Config.set("plot", "marker_face_alpha", int(v)), refresh_styles()))

        self.pg_size_card.valueChanged.connect(lambda v: Config.set("widget", "pg_marker_size", int(v)))
        self.vispy_size_card.valueChanged.connect(lambda v: Config.set("widget", "vispy_marker_size", int(v)))
        self.vispy_aa_card.valueChanged.connect(lambda v: Config.set("widget", "vispy_marker_antialias", float(v)))
        self.structure_bg_color_card.colorChanged.connect(lambda v: Config.set("widget", "structure_bg_color", v))
        self.structure_lattice_color_card.colorChanged.connect(lambda v: Config.set("widget", "structure_lattice_color", v))

        self.selected_color_card.colorChanged.connect(lambda v: (Config.set("plot", "selected_color", v), refresh_styles()))
        self.show_color_card.colorChanged.connect(lambda v: (Config.set("plot", "show_color", v), refresh_styles()))
        self.current_color_card.colorChanged.connect(lambda v: (Config.set("plot", "current_color", v), refresh_styles()))
        self.current_size_card.valueChanged.connect(lambda v: Config.set("plot", "current_marker_size", int(v)))
        # self.about_card.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))
        self.feedback_card.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL)))

    def _on_language_changed(self, index: int) -> None:
        """Persist the selected UI language."""
        value = self.language_combo.itemData(index)
        value = normalize_language(value)
        Config.set("ui", "language", value)
        MessageManager.send_info_message(
            self.tr("Language saved. Restart NepTrainKit to apply it."),
            title=self.tr("Tip"),
        )

    def _on_log_level_changed(self, index: int) -> None:
        """Persist and immediately apply the selected minimum log level."""
        level = normalize_log_level(self.log_level_combo.itemData(index))
        Config.set("logging", "level", level)
        set_log_level(level)

    def check_update(self):
        """Trigger the application update workflow for NepTrainKit.

        Returns
        -------
        None
            Starts an asynchronous update check.
        """
        self._update_worker = UpdateWoker(self)
        self._update_worker.check_update(manual=True, on_finished=self._on_update_check_finished)

    def check_update_nep89(self):
        """Check for updates of the bundled NEP89 model assets.

        Returns
        -------
        None
            Starts an asynchronous NEP89 update check.
        """
        self._update_nep89_worker = UpdateNEP89Woker(self)
        self._update_nep89_worker.check_update()

    def _base_about_description(self) -> str:
        """Return the base About card description text."""
        return 'Copyright @' + f" {YEAR}, {AUTHOR}. " + self.tr("Version") + f" {__version__}"

    def _refresh_about_update_content(self) -> None:
        """Refresh the About card text based on cached update status."""
        description = self._base_about_description()
        pending_version = get_pending_update_version()
        if pending_version:
            prefix = self.tr("New version available: v")
            description += f"\n{prefix}{pending_version}"
        if hasattr(self.about_card, "setContent"):
            self.about_card.setContent(description)

    def refresh_update_hint(self) -> None:
        """Public hook used by main window to refresh update hint text."""
        self._refresh_about_update_content()

    def _on_update_check_finished(self, _result) -> None:
        """Refresh local/global update hints after manual check."""
        self._refresh_about_update_content()
        win = self.window()
        if hasattr(win, "refresh_update_indicators"):
            win.refresh_update_indicators()
