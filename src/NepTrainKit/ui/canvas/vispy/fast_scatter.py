"""Lightweight VisPy scatter visual optimized for large static point clouds."""

from __future__ import annotations

import os
import time

import numpy as np
from loguru import logger
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


def _perf_enabled():
    return os.environ.get("NEPKIT_VISPY_PERF", "").strip().lower() in {"1", "true", "yes", "on"}


def _perf_log(message, **values):
    if not _perf_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    logger.debug(f"[vispy-perf] {message} {payload}".rstrip())


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
        t0 = time.perf_counter()
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
        _perf_log(
            "fastscatter.set_data",
            points=self._pos.shape[0],
            dtype=str(self._pos.dtype),
            ms=f"{(time.perf_counter() - t0) * 1000:.3f}",
        )

    def set_indices(self, indices=None):
        t0 = time.perf_counter()
        if indices is None:
            self._index_buffer_obj = None
            count = 0
        else:
            indices = np.asarray(indices, dtype=np.uint32)
            self._index_buffer_obj = gloo.IndexBuffer(np.ascontiguousarray(indices))
            count = int(indices.size)
        self._index_buffer = self._index_buffer_obj
        self.update()
        _perf_log(
            "fastscatter.set_indices",
            indices=count,
            ms=f"{(time.perf_counter() - t0) * 1000:.3f}",
        )

    def _prepare_transforms(self, view):
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view=None):
        if self._pos.size == 0:
            return False
        total_t0 = time.perf_counter()
        upload_ms = 0.0
        style_ms = 0.0
        if self._pos_changed:
            t0 = time.perf_counter()
            self._pos_vbo.set_data(self._pos)
            self.shared_program["a_position"] = self._pos_vbo
            self._pos_changed = False
            upload_ms = (time.perf_counter() - t0) * 1000
        if self._style_changed:
            t0 = time.perf_counter()
            self.shared_program["u_face_color"] = self._face_color
            self.shared_program["u_edge_color"] = self._edge_color
            self.shared_program["u_size"] = self._size
            self.shared_program["u_edge_width"] = self._edge_width
            self._style_changed = False
            style_ms = (time.perf_counter() - t0) * 1000
        self.shared_program["u_pixel_scale"] = self.transforms.pixel_scale
        self._index_buffer = self._index_buffer_obj
        if upload_ms > 0.0 or style_ms > 0.0:
            _perf_log(
                "fastscatter.prepare_draw",
                points=self._pos.shape[0],
                indices=0 if self._index_buffer_obj is None else "active",
                upload_ms=f"{upload_ms:.3f}",
                style_ms=f"{style_ms:.3f}",
                total_ms=f"{(time.perf_counter() - total_t0) * 1000:.3f}",
            )
        return True

    def _compute_bounds(self, axis, view):
        if self._pos.size == 0 or axis > 1:
            return None
        data = self._pos[:, axis]
        return float(np.min(data)), float(np.max(data))


FastScatter = create_visual_node(FastScatterVisual)
