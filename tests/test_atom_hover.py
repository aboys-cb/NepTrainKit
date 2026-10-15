from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.atom_inspection import inspect_atom, pick_front_atom
from NepTrainKit.core.structure import Structure
from NepTrainKit.ui.canvas import canvas_factory
from NepTrainKit.ui.canvas.atom_hover import AtomHoverController, format_atom_info


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _structure(positions, **arrays) -> Structure:
    positions = np.asarray(positions, dtype=float)
    count = len(positions)
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    atomic = {"species": np.asarray(["Fe"] * count), "pos": positions}
    for name, values in arrays.items():
        values = np.asarray(values)
        atomic[name] = values
        if not name.startswith("__ntk_"):
            properties.append(
                {"name": name, "type": "R", "count": int(values.reshape(count, -1).shape[1])}
            )
    return Structure(
        lattice=np.diag([10.0, 10.0, 10.0]),
        atomic_properties=atomic,
        properties=properties,
        additional_fields={"pbc": "T T T"},
    )


def test_inspect_atom_reports_forces_error_spin_and_extra_fields():
    structure = _structure(
        [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        force=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        spin=[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
        charge=[0.3, -0.3],
        __ntk_ml_force=np.asarray([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
    )

    info = inspect_atom(structure, 0)

    np.testing.assert_allclose(info.dft_force, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(info.ml_force, [1.0, 0.0, 0.5])
    assert info.force_error == pytest.approx(0.5)
    np.testing.assert_allclose(info.spin, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(info.fractional, [0.1, 0.1, 0.1])
    assert dict(info.extra) == {"charge": (pytest.approx(0.3),)}


def test_inspect_atom_uses_periodic_images_for_nearest_neighbor():
    structure = _structure([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0], [5.0, 5.0, 5.0]])

    info = inspect_atom(structure, 0)

    assert info.nearest_index == 1
    assert info.nearest_distance == pytest.approx(1.0)


def test_nearest_neighbor_handles_unwrapped_coordinates():
    info = inspect_atom(_structure([[0, 0, 0], [30.1, 0, 0]]), 0)
    assert info.nearest_index == 1
    assert info.nearest_distance == pytest.approx(.1, abs=2e-6)


@pytest.mark.parametrize("pbc", ["T T T", "T T F"])
def test_nearest_periodic_self_image_uses_a_reduced_skew_cell(pbc):
    structure = _structure([[0, 0, 0]])
    structure.lattice = np.array([[10., 0, 0], [31., 1, 0], [0, 0, 20.]])
    structure.additional_fields['pbc'] = pbc
    info = inspect_atom(structure, 0)
    # The shortest lattice vector is b - 3*a, absent from the original 27 images.
    assert info.nearest_index == 0
    assert info.nearest_distance == pytest.approx(np.sqrt(2))


def test_partial_periodicity_does_not_wrap_nonperiodic_displacements():
    structure = _structure([[0, 0, 0], [30.1, 0, 7]])
    structure.additional_fields['pbc'] = 'T F F'
    info = inspect_atom(structure, 0)
    assert info.nearest_index == 1
    assert info.nearest_distance == pytest.approx(np.hypot(.1, 7), abs=2e-6)


def test_nonperiodic_single_atom_has_no_neighbor_and_overlap_is_zero_distance():
    structure = _structure([[0, 0, 0]])
    structure.additional_fields['pbc'] = 'F F F'
    assert inspect_atom(structure, 0).nearest_distance is None
    overlap = inspect_atom(_structure([[0, 0, 0], [0, 0, 0]]), 0)
    assert overlap.nearest_index == 1
    assert overlap.nearest_distance == 0


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False), (False, False, False)])
def test_skew_unwrapped_neighbors_match_independent_image_enumeration(pbc):
    rng = np.random.default_rng(71)
    cell = np.array([[4., 0, 0], [12.4, 2.1, 0], [.3, .4, 6.]])
    positions = (rng.random((4, 3)) + rng.integers(-2, 3, size=(4, 3))) @ cell
    structure = _structure(positions)
    structure.lattice = cell
    structure.additional_fields['pbc'] = pbc
    # Deliberately enumerate a generous finite image cube in the original
    # basis, independently of the production minimum-image/reduction helpers.
    images = np.array(list(product(*[range(-12, 13) if v else (0,) for v in pbc])))
    positions = np.asarray(structure.positions, dtype=float)
    for atom in range(4):
        distances = np.linalg.norm(positions[None, :, :] - positions[atom] +
                                   (images @ cell)[:, None, :], axis=2)
        distances[np.all(images == 0, axis=1), atom] = np.inf
        expected = float(distances.min())
        info = inspect_atom(structure, atom)
        assert info.nearest_distance == pytest.approx(expected, abs=1e-6)
        assert distances[:, info.nearest_index].min() == pytest.approx(expected, abs=1e-6)


def test_format_atom_info_only_lists_available_values():
    plain = format_atom_info(inspect_atom(_structure([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]), 0))
    magnetic = format_atom_info(
        inspect_atom(
            _structure([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]], spin=[[0, 0, 2.2], [0, 0, 2.2]]),
            0,
        )
    )

    assert "Spin" not in plain and "Force" not in plain
    assert "Spin" in magnetic and "|m| 2.200" in magnetic


def test_force_mag_is_the_dft_magnetic_force_not_an_extra_field():
    structure = _structure(
        [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        force_mag=[[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
        mforce=[[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
    )

    info = inspect_atom(structure, 0)

    np.testing.assert_allclose(info.dft_mforce, [0.1, 0.2, 0.3])
    assert info.extra == ()


@pytest.mark.parametrize("vector", [[0., 0., 0.], [1., 2., 3.]])
def test_equal_values_do_not_hide_unrelated_or_unknown_fields(vector):
    structure = _structure([[0, 0, 0]], force=[vector], velocity=[vector], renamed_force=[vector])
    info = inspect_atom(structure, 0)
    assert dict(info.extra) == {'velocity': tuple(vector), 'renamed_force': tuple(vector)}


def test_known_forces_alias_is_displayed_in_the_force_row_only():
    structure = _structure([[0, 0, 0]], forces=[[1., 2., 3.]])
    info = inspect_atom(structure, 0)
    np.testing.assert_allclose(info.dft_force, [1., 2., 3.])
    assert info.extra == ()


def test_pick_front_atom_prefers_the_atom_closest_to_the_camera():
    screen = np.asarray([[100.0, 100.0], [102.0, 100.0], [300.0, 300.0]])
    depth = np.asarray([0.9, 0.2, 0.1])
    radius = np.asarray([10.0, 10.0, 10.0])

    assert pick_front_atom(screen, depth, radius, (101.0, 100.0)) == 1
    assert pick_front_atom(screen, depth, radius, (200.0, 200.0)) is None


def test_hover_controller_shows_card_and_clears_on_drag(app):
    host = QWidget()
    host.resize(200, 200)
    host.show()
    calls = []
    view = SimpleNamespace(
        structure=_structure([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
        atom_index_at=lambda pos: 1,
        set_hover_atom=calls.append,
    )
    controller = AtomHoverController(host, view)

    def move(buttons):
        point = QPointF(50, 50)
        event = QMouseEvent(
            QEvent.Type.MouseMove, point, host.mapToGlobal(point),
            Qt.MouseButton.NoButton, buttons, Qt.KeyboardModifier.NoModifier,
        )
        app.sendEvent(host, event)

    move(Qt.MouseButton.NoButton)
    controller._timer.timeout.emit()
    assert controller.card.isVisible()
    assert calls == [1]

    move(Qt.MouseButton.LeftButton)
    assert not controller.card.isVisible()
    assert calls == [1, None]
    host.close()


@pytest.mark.parametrize("backend", ["vispy", "pyqtgraph"])
def test_structure_views_pick_the_front_atom(app, backend):
    try:
        view, _fallback = canvas_factory.create_structure_plot(backend, None)
    except Exception as error:  # pragma: no cover - GL unavailable
        pytest.skip(f"{backend} structure view unavailable: {error}")
    host = canvas_factory.resolve_canvas_host_widget(view)
    host.resize(400, 400)
    host.show()
    # Three atoms on one line; looking along it they share a screen position.
    positions = np.asarray([[5.0, 5.0, 5.0], [5.0, 2.0, 5.0], [5.0, 8.0, 5.0]])
    view.show_structure(_structure(positions))
    if backend == "vispy":
        view.view.camera.set_state({"center": (5, 5, 5), "elevation": 0, "azimuth": 0})
        view.view.camera.distance = 40
        app.processEvents()
        mapped = np.asarray(view.view.scene.node_transform(view.scene).map(positions))
        cursor = tuple(mapped[0, :2] / mapped[0, 3])
        nearest = int(np.argmin(mapped[:, 2] / mapped[:, 3]))
    else:
        from pyqtgraph import Vector

        view.opts["center"] = Vector(5, 5, 5)
        view.setCameraPosition(distance=40, elevation=0, azimuth=90)
        app.processEvents()
        camera = view.cameraPosition()
        camera = np.asarray([camera.x(), camera.y(), camera.z()])
        nearest = int(np.argmin(np.linalg.norm(positions - camera, axis=1)))
        cursor = (host.width() / 2, host.height() / 2)

    assert view.atom_index_at(cursor) == nearest
    host.close()
