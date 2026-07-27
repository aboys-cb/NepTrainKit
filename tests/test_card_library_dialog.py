#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QObject, QPoint, Qt, QTranslator, Signal
from PySide6.QtWidgets import QApplication
from qfluentwidgets import MenuAnimationType, RoundMenu

from NepTrainKit.ui.views.cards import ConsoleWidget
from NepTrainKit.ui.views._card.group_label_card import GroupLabelCard
from NepTrainKit.ui.views._card.magnetic_order_card import MagneticOrderCard
from NepTrainKit.ui.widgets.card_metadata import (
    CardLibraryDialog,
    localized_card_description,
    localized_card_name,
)
from NepTrainKit.core import CardManager


class TestCardLibraryDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_library_search_filters_card_metadata(self):
        dialog = CardLibraryDialog()
        self.assertGreater(dialog.card_list.count(), 1)
        class_name, _metadata = next(iter(dialog._metadata_by_class.items()))

        dialog.search_edit.setText(class_name)

        visible = [
            dialog.card_list.item(row)
            for row in range(dialog.card_list.count())
            if not dialog.card_list.item(row).isHidden()
        ]
        self.assertGreaterEqual(len(visible), 1)
        self.assertTrue(
            any(
                item.data(Qt.ItemDataRole.UserRole) == class_name
                for item in visible
            )
        )
        self.assertIn(str(len(visible)), dialog.result_count_label.text())

    def test_add_button_emits_selected_card_class(self):
        dialog = CardLibraryDialog()
        requested = []
        dialog.cardRequested.connect(requested.append)
        selected = dialog.card_list.currentItem()
        class_name = selected.data(Qt.ItemDataRole.UserRole)

        dialog.add_button.click()

        self.assertEqual(requested, [class_name])

    def test_empty_search_result_disables_add_action(self):
        dialog = CardLibraryDialog()

        dialog.search_edit.setText("card-name-that-does-not-exist")

        self.assertFalse(dialog.add_button.isEnabled())
        self.assertTrue(dialog.result_count_label.text().startswith("0"))

    def test_chinese_catalog_localizes_dynamic_card_metadata(self):
        translator = QTranslator(self._app)
        qm_path = (
            Path(__file__).parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        self.assertTrue(translator.load(str(qm_path)))
        self._app.installTranslator(translator)
        try:
            dialog = CardLibraryDialog()
            for class_name, metadata in dialog._metadata_by_class.items():
                if Path(metadata.source_path).parent.name != "_card":
                    continue
                with self.subTest(card=class_name):
                    self.assertNotEqual(
                        localized_card_name(metadata),
                        metadata.card_name,
                    )
                    self.assertNotEqual(
                        localized_card_description(metadata),
                        metadata.description,
                    )

            item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "CompositionGradientCard"
            )

            self.assertIn("[合金与组分]", item.text())
            self.assertIn("成分梯度", item.text())
            dialog.card_list.setCurrentItem(item)
            self.assertEqual(dialog.detail_title_label.text(), "成分梯度")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "沿晶格 a、b 或 c 方向构造一维成分过渡，不移动原子。",
            )
            self.assertIn("作者", dialog.detail_contributors_label.text())

            dialog.search_edit.setText("成分梯度")
            self.assertFalse(item.isHidden())

            sweep_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "CompositionSweepCard"
            )
            self.assertIn("成分空间采样", sweep_item.text())
            dialog.card_list.setCurrentItem(sweep_item)
            self.assertEqual(
                dialog.detail_title_label.text(),
                "成分空间采样",
            )
            self.assertEqual(
                dialog.detail_description_label.text(),
                "在二至五元成分空间中采样目标配比；仅写入 Comp(...) 标签，需连接“随机占位”生成真实合金结构。",
            )

            replace_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "ConditionalReplaceCard"
            )
            dialog.card_list.setCurrentItem(replace_item)
            self.assertEqual(dialog.detail_title_label.text(), "条件替换")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "按笛卡尔坐标筛选指定元素，并将所有命中位点按给定混合比例替换。",
            )

            finite_cell_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "FiniteCellAlloyOccupancyCard"
            )
            dialog.card_list.setCurrentItem(finite_cell_item)
            self.assertEqual(dialog.detail_title_label.text(), "有限晶胞合金占位")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "按有限位点可实现的整数组成生成真实合金占位，并可分别约束各子晶格。",
            )

            group_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "GroupLabelCard"
            )
            dialog.card_list.setCurrentItem(group_item)
            self.assertEqual(dialog.detail_title_label.text(), "分组标记")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "按坐标规则将原子分成两组，供磁序、掺杂或空位操作使用；不改变坐标和元素。",
            )

            group_card = GroupLabelCard()
            self.assertEqual(
                group_card.mode_combo.itemText(0),
                "分数坐标交替分层",
            )
            self.assertEqual(
                group_card.mode_combo.itemText(1),
                "当前晶胞半网格奇偶",
            )
            self.assertEqual(
                group_card.kvec_combo.itemText(4),
                "111（沿晶格 a+b+c）",
            )

            magnetic_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "MagneticOrderCard"
            )
            dialog.card_list.setCurrentItem(magnetic_item)
            self.assertEqual(dialog.detail_title_label.text(), "磁序")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "根据元素磁矩生成 FM、AFM 和随机 PM 初始自旋构型；不改变坐标和元素。",
            )

            magnetic_card = MagneticOrderCard()
            self.assertEqual(
                magnetic_card.format_combo.itemText(0),
                "共线（沿参考轴）",
            )
            self.assertEqual(
                magnetic_card.afm_mode_combo.itemText(1),
                "已有分组标签",
            )
            self.assertEqual(
                magnetic_card.pm_direction_combo.itemText(0),
                "完整球面",
            )

            vacancy_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "RandomVacancyCard"
            )
            dialog.card_list.setCurrentItem(vacancy_item)
            self.assertEqual(dialog.detail_title_label.text(), "随机空位")
            self.assertEqual(
                dialog.detail_description_label.text(),
                "按元素、已有 group 和数量规则随机选择位点并删除；其余原子坐标保持不变。",
            )

            global_vacancy_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "VacancyDefectCard"
            )
            dialog.card_list.setCurrentItem(global_vacancy_item)
            self.assertEqual(
                dialog.detail_title_label.text(),
                "全局随机空位",
            )
            self.assertEqual(
                dialog.detail_description_label.text(),
                "不区分元素，按整体数量或比例随机删除位点；其余原子坐标保持不变。",
            )

            insert_item = next(
                dialog.card_list.item(row)
                for row in range(dialog.card_list.count())
                if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "InsertDefectCard"
            )
            dialog.card_list.setCurrentItem(insert_item)
            self.assertEqual(
                dialog.detail_title_label.text(),
                "插隙与表面吸附",
            )
            self.assertEqual(
                dialog.detail_description_label.text(),
                "在晶胞内随机生成插隙候选，或在指定上表面生成随机吸附候选；仅保证最小原子间距，不识别晶体学位点。",
            )
        finally:
            self._app.removeTranslator(translator)

    def test_chinese_add_card_dropdown_localizes_card_names(self):
        translator = QTranslator(self._app)
        qm_path = (
            Path(__file__).parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        self.assertTrue(translator.load(str(qm_path)))
        self._app.installTranslator(translator)
        try:
            console = ConsoleWidget()
            action = next(
                action
                for action in console.menu.actions()
                if action.objectName() == "CompositionGradientCard"
            )
            self.assertEqual(action.text(), "成分梯度")
            global_vacancy_action = next(
                action
                for action in console.menu.actions()
                if action.objectName() == "VacancyDefectCard"
            )
            self.assertEqual(
                global_vacancy_action.text(),
                "全局随机空位",
            )
            insert_action = next(
                action
                for action in console.menu.actions()
                if action.objectName() == "InsertDefectCard"
            )
            self.assertEqual(
                insert_action.text(),
                "插隙与表面吸附",
            )
            strict_gsfe_action = next(
                action
                for action in console.menu.actions()
                if action.objectName() == "StrictGSFEPathCard"
            )
            self.assertEqual(
                strict_gsfe_action.text(),
                "层错 / GSFE 路径",
            )
            self.assertNotIn(
                "StackingFaultCard",
                {action.objectName() for action in console.menu.actions()},
            )

            with patch(
                "NepTrainKit.ui.views.cards.Config.getboolean",
                return_value=True,
            ):
                grouped_console = ConsoleWidget()
            alloy_menu = next(
                menu
                for menu in grouped_console.menu._subMenus
                if menu.title() == "合金与组分"
            )
            grouped_action = next(
                action
                for action in alloy_menu.menuActions()
                if action.objectName() == "CompositionGradientCard"
            )
            self.assertEqual(grouped_action.text(), "成分梯度")
        finally:
            self._app.removeTranslator(translator)

    def test_card_details_prioritize_user_facing_metadata(self):
        dialog = CardLibraryDialog()
        item = next(
            dialog.card_list.item(row)
            for row in range(dialog.card_list.count())
            if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "CompositionGradientCard"
        )

        dialog.card_list.setCurrentItem(item)

        self.assertEqual(dialog.detail_title_label.text(), "Composition Gradient")
        self.assertEqual(dialog.detail_group_label.text(), "Alloy")
        self.assertTrue(dialog.detail_description_label.text())
        self.assertTrue(dialog.detail_technical_panel.isHidden())
        self.assertNotIn(
            "/Users/", dialog.detail_source_label.text()
        )
        dialog.detail_technical_button.click()

        self.assertFalse(dialog.detail_technical_panel.isHidden())
        self.assertEqual(
            dialog.detail_class_value.text(), "CompositionGradientCard"
        )
        self.assertEqual(
            dialog.detail_path_value.text(), "composition_gradient_card.py"
        )

    def test_builtin_cards_expose_their_documentation_links(self):
        dialog = CardLibraryDialog()
        builtin_metadata = {
            class_name: metadata
            for class_name, metadata in dialog._metadata_by_class.items()
            if Path(metadata.source_path).parent.name == "_card"
        }

        self.assertEqual(len(builtin_metadata), 38)
        self.assertNotIn("StackingFaultCard", builtin_metadata)
        self.assertIn("StrictGSFEPathCard", builtin_metadata)
        self.assertIn("StackingFaultCard", CardManager.card_info_dict)
        self.assertLessEqual(
            set(builtin_metadata),
            set(CardManager.card_info_dict),
        )
        for class_name, metadata in builtin_metadata.items():
            with self.subTest(card=class_name):
                self.assertTrue(metadata.description)
                self.assertRegex(
                    metadata.docs_url,
                    (
                        r"^https://neptrainkit\.readthedocs\.io/en/latest/"
                        r"module/make-dataset-cards/cards/"
                        r"[a-z0-9-]+\.html$"
                    ),
                )

        item = next(
            dialog.card_list.item(row)
            for row in range(dialog.card_list.count())
            if dialog.card_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "LayerCopyCard"
        )
        dialog.card_list.setCurrentItem(item)
        self.assertFalse(dialog.detail_docs_label.isHidden())
        self.assertIn("layer-copy-card.html", dialog.detail_docs_label.text())

    def test_console_forwards_library_add_request(self):
        class FakeLibraryDialog(QObject):
            cardRequested = Signal(str)

            def __init__(self, parent=None):
                super().__init__(parent)

            def exec(self):
                self.cardRequested.emit("CrystalPrototypeBuilderCard")

        console = ConsoleWidget()
        requested = []
        console.newCardSignal.connect(requested.append)
        with patch(
            "NepTrainKit.ui.views.cards.CardLibraryDialog", FakeLibraryDialog
        ):
            console.show_card_library()

        self.assertEqual(requested, ["CrystalPrototypeBuilderCard"])

    def test_console_exposes_labeled_card_search_button(self):
        console = ConsoleWidget()

        self.assertEqual(console.find_card_button.text(), "Find card")
        self.assertTrue(console.find_card_button.isEnabled())

    def test_console_card_menu_scrolls_instead_of_exceeding_screen_height(self):
        console = ConsoleWidget()

        self.assertEqual(console.menu.view.maxVisibleItems(), 16)
        self.assertGreater(len(console.menu.actions()), 16)

    def test_console_card_menu_skips_off_screen_start_animation(self):
        console = ConsoleWidget()
        with patch.object(RoundMenu, "exec") as exec_mock:
            console.menu.exec(
                QPoint(10, 20),
                aniType=MenuAnimationType.DROP_DOWN,
            )

        exec_mock.assert_called_once_with(
            QPoint(10, 20),
            ani=False,
            aniType=MenuAnimationType.NONE,
        )

    def test_console_exposes_selected_output_handoff_action(self):
        console = ConsoleWidget()
        requests = []
        console.viewOutputSignal.connect(lambda: requests.append(True))

        self.assertFalse(console.view_output_button.isEnabled())
        self.assertEqual(console.view_output_button.text(), "View selected outputs")
        console.set_output_available(True)
        console.view_output_button.click()

        self.assertEqual(requests, [True])


if __name__ == "__main__":
    unittest.main()
