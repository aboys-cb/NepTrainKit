#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 19:44
# @email    : 1747193328@qq.com


from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QWidget
 
from qfluentwidgets import SettingCardGroup, HyperlinkCard, PrimaryPushSettingCard, ExpandLayout, OptionsConfigItem, \
    OptionsValidator, EnumSerializer, SwitchSettingCard,FluentIcon  , ScrollArea

from NepTrainKit.config import Config
from NepTrainKit.ui.widgets import MyComboBoxSettingCard, DoubleSpinBoxSettingCard, LineEditSettingCard
from NepTrainKit.ui.widgets import ColorSettingCard
from NepTrainKit.core.types import ForcesMode, CanvasMode, NepBackend
from NepTrainKit.ui.update import UpdateWoker,UpdateNEP89Woker
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
             'Personalization' , self.scrollWidget)
        self.nep_group = SettingCardGroup(
             'NEP Settings' , self.scrollWidget)
        self.plot_group = SettingCardGroup(
             'Plot Settings' , self.scrollWidget)

        default_forces = Config.get("widget","forces_data",str(ForcesMode.Raw.value))
        if default_forces=="Row":

            default_forces="Raw"

        self.optimization_forces_card = MyComboBoxSettingCard(
            OptionsConfigItem("forces","forces",ForcesMode(default_forces),OptionsValidator(ForcesMode), EnumSerializer(ForcesMode)),
            FluentIcon.BRUSH,
            'Force data format',
            "Streamline data and speed up drawing",
            texts=[
             mode.value for mode in    ForcesMode
            ],
            default=default_forces,
            parent=self.personal_group
        )
        canvas_type = Config.get("widget","canvas_type",str(CanvasMode.PYQTGRAPH.value))

        self.canvas_card = MyComboBoxSettingCard(
            OptionsConfigItem("canvas","canvas",CanvasMode(canvas_type),OptionsValidator(CanvasMode), EnumSerializer(CanvasMode)),
            FluentIcon.BRUSH,
            'Canvas Engine',
            "Choose GPU with vispy",
            texts=[
             mode.value for mode in    CanvasMode

            ],
            default=canvas_type,
            parent=self.personal_group
        )


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
            'Sort atoms',
            'Sort atoms in structures when processing cards',
            parent=self.personal_group
        )
        self.sort_atoms_card.setValue(sort_atoms_config)

        self.use_group_menu_card = SwitchSettingCard(
            QIcon(":/images/src/images/group.svg"),
            'Use card group menu',
            'Group cards by "group" in console menu',
            parent=self.personal_group
        )
        self.use_group_menu_card.setValue(use_group_menu_config)
        # Default Config_type text
        default_cfg_type = Config.get("widget", "default_config_type", "neptrainkit")
        self.default_cfg_type_card = LineEditSettingCard(
            FluentIcon.TAG,
            'Default Config_type',
            'Tag assigned when source has no Config_type',
            self.personal_group
        )
        self.default_cfg_type_card.setValue(default_cfg_type)
        radius_coefficient_config=Config.getfloat("widget","radius_coefficient",0.7)

        self.radius_coefficient_Card = DoubleSpinBoxSettingCard(

            FluentIcon.ALBUM,
            'Covalent radius coefficient',
            'Coefficient used to detect bond length',
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
            'NEP Backend',
            'Select CPU/GPU or Auto detection',
            texts=[mode.value for mode in NepBackend],
            default=nep_backend_default,
            parent=self.nep_group
        )

        # GPU batch size
        gpu_bs_default = Config.getint("nep", "gpu_batch_size", 1000) or 1000
        self.gpu_bs_card = DoubleSpinBoxSettingCard(
            FluentIcon.SPEED_HIGH,
            'GPU Batch Size',
            'Batch of frames processed GPU slice',
            self.nep_group
        )
        self.gpu_bs_card.setRange(0, 10000000)
        self.gpu_bs_card.setValue(float(gpu_bs_default))

        # Plot settings defaults
        edge_color = Config.get("plot", "marker_edge_color", "#07519C")
        face_color = Config.get("plot", "marker_face_color", "#FFFFFF")
        face_alpha = Config.getint("plot", "marker_face_alpha", 0) or 0
        selected_color = Config.get("plot", "selected_color", "#FF0000")
        show_color = Config.get("plot", "show_color", "#00FF00")
        current_color = Config.get("plot", "current_color", "#FF0000")
        pg_size = Config.getint("widget", "pg_marker_size", 7) or 7
        vispy_size = Config.getint("widget", "vispy_marker_size", 6) or 6
        vispy_aa = Config.getfloat("widget", "vispy_marker_antialias", 0.5) or 0.5
        current_size = Config.getint("plot", "current_marker_size", 20) or 20

        self.edge_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            'Scatter edge color',
            'Default edge color for points',
            self.plot_group
        )
        self.edge_color_card.setValue(edge_color)

        self.face_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            'Scatter face color',
            'Default fill color for points',
            self.plot_group
        )
        self.face_color_card.setValue(face_color)

        self.face_alpha_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            'Face alpha (0-255)',
            'Alpha channel for fill color',
            self.plot_group
        )
        self.face_alpha_card.setRange(0, 255)
        self.face_alpha_card.setValue(float(face_alpha))

        self.pg_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            'PyQtGraph scatter size',
            'Marker size for PyQtGraph canvas',
            self.plot_group
        )
        self.pg_size_card.setRange(1, 100)
        self.pg_size_card.setValue(float(pg_size))

        self.vispy_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            'VisPy scatter size',
            'Marker size for VisPy canvas',
            self.plot_group
        )
        self.vispy_size_card.setRange(1, 100)
        self.vispy_size_card.setValue(float(vispy_size))

        self.vispy_aa_card = DoubleSpinBoxSettingCard(
            FluentIcon.BRUSH,
            'VisPy antialias',
            'Marker antialias value for VisPy (0-2)',
            self.plot_group
        )
        self.vispy_aa_card.setRange(0.0, 2.0)
        self.vispy_aa_card.setValue(float(vispy_aa))

        self.selected_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            'Selected color',
            'Color for selected points',
            self.plot_group
        )
        self.selected_color_card.setValue(selected_color)

        self.show_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            'Show color',
            'Color for highlighted "show" points',
            self.plot_group
        )
        self.show_color_card.setValue(show_color)

        self.current_color_card = ColorSettingCard(
            FluentIcon.BRUSH,
            'Current marker color',
            'Color for current star marker',
            self.plot_group
        )
        self.current_color_card.setValue(current_color)

        self.current_size_card = DoubleSpinBoxSettingCard(
            FluentIcon.ALBUM,
            'Current marker size',
            'Size of current star marker',
            self.plot_group
        )
        self.current_size_card.setRange(5, 100)
        self.current_size_card.setValue(float(current_size))


        self.about_group = SettingCardGroup("About", self.scrollWidget)
        self.help_card = HyperlinkCard(
            HELP_URL,
             'Open Help Page' ,
            FluentIcon.HELP,
             'Help' ,
             'Discover new features and learn useful tips about NepTrainKit.' ,
            self.about_group
        )
        self.feedback_card = PrimaryPushSettingCard(
            "Submit Feedback",
            FluentIcon.FEEDBACK,
            "Submit Feedback",

            'Help us improve NepTrainKit by providing feedback.',
            self.about_group
        )
        self.about_card = PrimaryPushSettingCard(
            'Check for Updates',
            FluentIcon.INFO,
            "About",
            'Copyright 漏' + f" {YEAR}, {AUTHOR}. " +
            "Version" + f" {__version__}",
            self.about_group
        )
        self.about_nep89_card = PrimaryPushSettingCard(
            'Check and update',
            FluentIcon.INFO,
            "About NEP89",
            "NEP official NEP89 large model",
            self.about_group
        )
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
        self.personal_group.addSettingCard(self.auto_load_card)
        self.personal_group.addSettingCard(self.radius_coefficient_Card)
        self.personal_group.addSettingCard(self.sort_atoms_card)
        self.personal_group.addSettingCard(self.use_group_menu_card)
        self.personal_group.addSettingCard(self.default_cfg_type_card)

        self.nep_group.addSettingCard(self.nep_backend_card)
        self.nep_group.addSettingCard(self.gpu_bs_card)

 

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
        self.about_card.clicked.connect(self.check_update)
        self.about_nep89_card.clicked.connect(self.check_update_nep89)

        self.nep_backend_card.optionChanged.connect(lambda option: Config.set("nep", "backend", option))
        self.gpu_bs_card.valueChanged.connect(lambda value: Config.set("nep", "gpu_batch_size", int(value)))

        self.auto_load_card.checkedChanged.connect(lambda state:Config.set("widget","auto_load",state))
        self.sort_atoms_card.checkedChanged.connect(lambda state:Config.set("widget","sort_atoms",state))
        self.use_group_menu_card.checkedChanged.connect(lambda state:Config.set("widget","use_group_menu",state))
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

        self.selected_color_card.colorChanged.connect(lambda v: (Config.set("plot", "selected_color", v), refresh_styles()))
        self.show_color_card.colorChanged.connect(lambda v: (Config.set("plot", "show_color", v), refresh_styles()))
        self.current_color_card.colorChanged.connect(lambda v: (Config.set("plot", "current_color", v), refresh_styles()))
        self.current_size_card.valueChanged.connect(lambda v: Config.set("plot", "current_marker_size", int(v)))
        # self.about_card.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))
        self.feedback_card.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL)))

    def check_update(self):
        """Trigger the application update workflow for NepTrainKit.

        Returns
        -------
        None
            Starts an asynchronous update check.
        """
        UpdateWoker(self).check_update()

    def check_update_nep89(self):
        """Check for updates of the bundled NEP89 model assets.

        Returns
        -------
        None
            Starts an asynchronous NEP89 update check.
        """
        UpdateNEP89Woker(self).check_update()


