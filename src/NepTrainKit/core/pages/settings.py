#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 19:44
# @Author  : 兵
# @email    : 1747193328@qq.com


from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QWidget
from qfluentwidgets import FluentIcon as FIF, ScrollArea
from qfluentwidgets import SettingCardGroup, HyperlinkCard, PrimaryPushSettingCard, ExpandLayout, OptionsConfigItem, \
    OptionsValidator, EnumSerializer, SwitchSettingCard

from NepTrainKit.core import Config
from NepTrainKit.core.custom_widget import MyComboBoxSettingCard, DoubleSpinBoxSettingCard
from NepTrainKit.core.types import ForcesMode, CanvasMode
from NepTrainKit.core.update import UpdateWoker,UpdateNEP89Woker
from NepTrainKit.version import HELP_URL, FEEDBACK_URL, __version__, YEAR, AUTHOR


class SettingsWidget(ScrollArea):
    def __init__(self,parent):

        super().__init__(parent)
        self.setObjectName('SettingsWidget')
        self.scrollWidget = QWidget()

        self.expand_layout = ExpandLayout(self.scrollWidget)

        self.personal_group = SettingCardGroup(
             'Personalization' , self.scrollWidget)


        default_forces = Config.get("widget","forces_data","Raw")
        if default_forces=="Row":
            #没什么用 替换以前的坑 之前写错单词了
            default_forces="Raw"
        self.optimization_forces_card = MyComboBoxSettingCard(
            OptionsConfigItem("forces","forces",ForcesMode(default_forces),OptionsValidator(ForcesMode), EnumSerializer(ForcesMode)),
            FIF.BRUSH,
            'Force data format',
            "Streamline data and speed up drawing",
            texts=[
                "Raw","Norm"
            ],
            default=default_forces,
            parent=self.personal_group
        )
        canvas_type = Config.get("widget","canvas_type","pyqtgraph")

        self.canvas_card = MyComboBoxSettingCard(
            OptionsConfigItem("canvas","canvas",CanvasMode(canvas_type),OptionsValidator(CanvasMode), EnumSerializer(CanvasMode)),
            FIF.BRUSH,
            'Canvas Engine',
            "Choose GPU with vispy",
            texts=[
                "pyqtgraph","vispy"
            ],
            default=canvas_type,
            parent=self.personal_group
        )


        auto_load_config = Config.getboolean("widget","auto_load",False)

        self.auto_load_card = SwitchSettingCard(
            QIcon(":/images/src/images/auto_load.svg"),
            self.tr('Auto loading'),
            self.tr('Detect startup path data and load'),

            parent=self.personal_group
        )
        self.auto_load_card.setValue(auto_load_config)
        radius_coefficient_config=Config.getfloat("widget","radius_coefficient",0.7)

        self.radius_coefficient_Card = DoubleSpinBoxSettingCard(

            FIF.ALBUM,
            'Covalent radius coefficient',
            'Coefficient used to detect bond length',
            self.personal_group
        )
        self.radius_coefficient_Card.setValue(radius_coefficient_config)
        self.radius_coefficient_Card.setRange(0.0, 1.5)

        self.about_group = SettingCardGroup("About", self.scrollWidget)
        self.help_card = HyperlinkCard(
            HELP_URL,
             'Open Help Page' ,
            FIF.HELP,
             'Help' ,
             'Discover new features and learn useful tips about NepTrainKit.' ,
            self.about_group
        )
        self.feedback_card = PrimaryPushSettingCard(
            "Submit Feedback",
            FIF.FEEDBACK,
            "Submit Feedback",

            'Help us improve NepTrainKit by providing feedback.',
            self.about_group
        )
        self.about_card = PrimaryPushSettingCard(
            'Check for Updates',
            FIF.INFO,
            "About",
            'Copyright ©' + f" {YEAR}, {AUTHOR}. " +
            "Version" + f" {__version__}",
            self.about_group
        )
        self.about_nep89_card = PrimaryPushSettingCard(
            'Check and update',
            FIF.INFO,
            "About NEP89",
            "NEP official NEP89 large model",
            self.about_group
        )
        self.init_layout()
        self.init_signal()
    def init_layout(self):
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.scrollWidget.setLayout(self.expand_layout)



        self.personal_group.addSettingCard(self.optimization_forces_card)
        self.personal_group.addSettingCard(self.canvas_card)
        self.personal_group.addSettingCard(self.auto_load_card)
        self.personal_group.addSettingCard(self.radius_coefficient_Card)

        self.about_group.addSettingCard(self.about_nep89_card)
        self.about_group.addSettingCard(self.help_card)
        self.about_group.addSettingCard(self.feedback_card)
        self.about_group.addSettingCard(self.about_card)


        self.expand_layout.addWidget(self.personal_group)
        self.expand_layout.addWidget(self.about_group)

    def init_signal(self):
        self.canvas_card.optionChanged.connect(lambda option:Config.set("widget","canvas_type",option ))
        self.radius_coefficient_Card.valueChanged.connect(lambda value:Config.set("widget","radius_coefficient",value))
        self.optimization_forces_card.optionChanged.connect(lambda option:Config.set("widget","forces_data",option ))
        self.about_card.clicked.connect(self.check_update)
        self.about_nep89_card.clicked.connect(self.check_update_nep89)

        self.auto_load_card.checkedChanged.connect(lambda state:Config.set("widget","auto_load",state))
        # self.about_card.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))
        self.feedback_card.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL)))

    def check_update(self):
        UpdateWoker(self).check_update()
    def check_update_nep89(self):
        UpdateNEP89Woker(self).check_update()