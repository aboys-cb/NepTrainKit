#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import BodyLabel


class TrainNepWidget(QWidget):
    """Simple placeholder widget for NEP training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainNepWidget")
        layout = QVBoxLayout(self)
        layout.addWidget(BodyLabel("Train NEP Placeholder", self))

