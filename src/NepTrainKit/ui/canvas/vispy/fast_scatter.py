"""Lightweight VisPy scatter visual optimized for large static point clouds."""

from __future__ import annotations

import numpy as np
from vispy import gloo
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual


VERT_SHADER = """
attribute vec2 a_position;
uniform float u_size;
uniform float u_pixel_scale;

void main(void) {
    gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    gl_PointSize = max(1.0, u_size * u_pixel_scale);
}
"""


FRAG_SHADER = """
uniform vec4 u_face_color;
uniform vec4 u_edge_color;
uniform float u_size;
uniform float u_pixel_scale;
uniform float u_edge_width;

void main(void) {
    vec2 p = gl_PointCoord.xy - vec2(0.5, 0.5);
    float r = length(p);
    if (r > 0.5) {
        discard;
    }
    float point_size = max(1.0, u_size * u_pixel_scale);
    float edge_start = max(0.0, 0.5 - u_edge_width / point_size);
    if (r >= edge_start) {
        gl_FragColor = u_edge_color;
    } else {
        if (u_face_color.a <= 0.0) {
            discard;
        }
        gl_FragColor = u_face_color;
    }
}
"""


class FastScatterVisual(Visual):
    """Draw same-style scatter points using only position VBO plus uniforms."""

    def __init__(self, **_kwargs):
        self._pos_vbo = gloo.VertexBuffer()
        self._index_buffer_obj = None
        self._pos = np.empty((0, 2), dtype=np.float32)
        self._face_color = (1.0, 1.0, 1.0, 1.0)
        self._edge_color = (1.0, 1.0, 1.0, 1.0)
        self._edge_width = 1.0
        self._size = 6.0
        self._pos_changed = True
        self._style_changed = True
        super().__init__(vcode=VERT_SHADER, fcode=FRAG_SHADER)
        self._draw_mode = "points"
        self.set_gl_state(
            depth_test=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self.freeze()

    @property
    def positions(self):
        return self._pos

    def _rgba(self, color, fallback=(1.0, 1.0, 1.0, 1.0)):
        if color is None:
            return tuple(fallback)
        values = np.asarray(color, dtype=np.float32).reshape(-1)
        if values.size >= 4:
            return tuple(float(v) for v in values[:4])
        if values.size == 3:
            return float(values[0]), float(values[1]), float(values[2]), 1.0
        return tuple(fallback)

    def set_data(
        self,
        pos=None,
        size=6,
        face_color=(1.0, 1.0, 1.0, 1.0),
        edge_color=None,
        edge_width=1.0,
        symbol=None,
        **_kwargs,
    ):
        if pos is None:
            pos = np.empty((0, 2), dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] not in (2, 3):
            raise ValueError("FastScatterVisual pos must have shape (N, 2) or (N, 3)")
        self._pos = np.ascontiguousarray(pos[:, :2], dtype=np.float32)
        self._size = float(size or 1)
        self._face_color = self._rgba(face_color)
        self._edge_color = self._rgba(edge_color, fallback=self._face_color)
        self._edge_width = max(0.0, float(edge_width or 0.0))
        self._pos_changed = True
        self._style_changed = True
        self.update()

    def set_indices(self, indices=None):
        if indices is None:
            self._index_buffer_obj = None
        else:
            indices = np.asarray(indices, dtype=np.uint32)
            self._index_buffer_obj = gloo.IndexBuffer(np.ascontiguousarray(indices))
        self._index_buffer = self._index_buffer_obj
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view=None):
        if self._pos.size == 0:
            return False
        if self._pos_changed:
            self._pos_vbo.set_data(self._pos)
            self.shared_program["a_position"] = self._pos_vbo
            self._pos_changed = False
        if self._style_changed:
            self.shared_program["u_face_color"] = self._face_color
            self.shared_program["u_edge_color"] = self._edge_color
            self.shared_program["u_size"] = self._size
            self.shared_program["u_edge_width"] = self._edge_width
            self._style_changed = False
        self.shared_program["u_pixel_scale"] = self.transforms.pixel_scale
        self._index_buffer = self._index_buffer_obj
        return True

    def _compute_bounds(self, axis, view):
        if self._pos.size == 0 or axis > 1:
            return None
        data = self._pos[:, axis]
        return float(np.min(data)), float(np.max(data))


FastScatter = create_visual_node(FastScatterVisual)
