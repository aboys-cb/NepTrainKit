import numpy as np
import pytest

from NepTrainKit.ui.canvas.vispy.structure import _arrow_rotation_from_z


@pytest.mark.parametrize(
    "direction",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (-1.0, -2.0, 0.5),
    ],
)
def test_arrow_rotation_maps_mesh_tip_to_vector_direction(direction):
    expected = np.asarray(direction, dtype=float)
    expected /= np.linalg.norm(expected)

    rotation = _arrow_rotation_from_z(expected)
    rendered_direction = (rotation @ np.array([0.0, 0.0, 1.0, 0.0]))[:3]

    np.testing.assert_allclose(rendered_direction, expected, atol=1e-7)
