from unittest.mock import Mock, patch

from NepTrainKit.ui.pages.makedata import (
    MAKE_DATA_STRUCTURE_FILE_FILTER,
    MakeDataWidget,
    is_make_data_structure_path,
)
from NepTrainKit.ui.pages.show_nep import RESULT_DATA_FILE_FILTER, ShowNepWidget


class _PageProbe:
    def __init__(self):
        self.load_base_structure = Mock()
        self.set_work_path = Mock()

    @staticmethod
    def tr(text):
        return text


def test_make_data_accepts_documented_structure_names():
    accepted = (
        "seed.xyz",
        "seed.extxyz",
        "seed.vasp",
        "seed.cif",
        "POSCAR",
        "CONTCAR",
    )

    assert all(is_make_data_structure_path(path) for path in accepted)
    assert not is_make_data_structure_path("OUTCAR")


def test_make_data_file_dialog_exposes_documented_formats():
    page = _PageProbe()
    with patch(
        "NepTrainKit.ui.pages.makedata.call_path_dialog",
        return_value=["seed.extxyz"],
    ) as dialog:
        MakeDataWidget.open_file(page)

    assert dialog.call_args.kwargs["file_filter"] == MAKE_DATA_STRUCTURE_FILE_FILTER
    page.load_base_structure.assert_called_once_with(["seed.extxyz"])


def test_dataset_display_file_dialog_exposes_registered_importers():
    page = _PageProbe()
    with patch(
        "NepTrainKit.ui.pages.show_nep.call_path_dialog",
        return_value="OUTCAR (1)",
    ) as dialog:
        ShowNepWidget.open_file(page)

    assert dialog.call_args.kwargs["file_filter"] == RESULT_DATA_FILE_FILTER
    page.set_work_path.assert_called_once_with("OUTCAR (1)")
