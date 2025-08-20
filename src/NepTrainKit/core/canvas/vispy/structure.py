#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/5/19 20:45
# @Author  : 兵
# @email    : 1747193328@qq.com
from vispy.visuals.filters import ShadingFilter
from vispy.visuals.transforms import MatrixTransform, STTransform

from NepTrainKit.core import Config
from NepTrainKit.core.structure import table_info
import numpy as np
from vispy.util.transforms import rotate

from vispy import app, scene, visuals
from vispy.geometry import MeshData, create_cylinder, create_cone, create_sphere
from vispy.scene.visuals import Mesh, Line,Text
from vispy.color import Color, get_colormap


def create_arrow_mesh():
    """Return MeshData representing an arrow aligned to +Z axis."""
    cyl = create_cylinder(20, 32, radius=[0.05, 0.05], length=0.8)
    cone = create_cone(32, radius=0.1, length=0.2)
    verts = np.vstack((cyl.get_vertices(), cone.get_vertices() + [0, 0, 0.8]))
    faces = np.vstack((cyl.get_faces(),
                       cone.get_faces() + len(cyl.get_vertices())))
    return MeshData(vertices=verts, faces=faces)


class ArrowAxes:
    """A class to create and manage 3D arrow-based coordinate axes."""
    def __init__(
        self,
        canvas,
        parent,
        directions=None,
        colors=None,
        labels=None,
        label_positions=None,
        scale=0.5,
        font_size=12
    ):
        """
        Initialize the arrow axes.

        Parameters:
        - canvas: SceneCanvas, the Vispy canvas to render on.
        - parent: Node, the parent node for the axes.
        - directions: List of 3D vectors for X, Y, Z axes directions.
        - colors: List of colors for X, Y, Z axes.
        - labels: List of label texts for X, Y, Z axes.
        - label_positions: List of 3D positions for X, Y, Z labels.
        - scale: Float, scale factor for arrows.
        - font_size: Float, font size for labels.
        """
        self.canvas = canvas
        self.axis_root = scene.Node(parent=parent)
        self.arrow_mesh = create_arrow_mesh()
        self.transform = MatrixTransform()
        self.axis_root.transform = self.transform

        # Default values
        self.directions = directions or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.colors = colors or ['red', 'green', 'blue']
        self.labels = labels or ['X', 'Y', 'Z']
        self.label_positions = label_positions or [[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]]
        self.scale = scale
        self.font_size = font_size

        # Create arrows and labels
        self._create_arrows()
        self._create_labels()

        # Connect events
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.resize.connect(self._on_resize)

        # Initial update
        self._update_axis()

    def _create_arrows(self):
        """Create arrow visuals for each axis with precise direction alignment."""
        for i, (direction, color) in enumerate(zip(self.directions, self.colors)):
            arrow = Mesh(meshdata=self.arrow_mesh, color=color, parent=self.axis_root)
            # arrow.set_gl_state(depth_test=False, depth_func='always', cull_face=False)
            transform = MatrixTransform()

            # Normalize direction
            direction = np.array(direction, dtype=float)
            if np.linalg.norm(direction) == 0:
                raise ValueError(f"Direction vector {i} is zero, cannot normalize.")
            direction /= np.linalg.norm(direction)

            # Align arrow (Z-axis to direction) using quaternion rotation
            z_axis = np.array([0, 0, 1], dtype=float)
            if not np.allclose(direction, z_axis) and not np.allclose(direction, -z_axis):

                # Compute rotation axis and angle
                axis = np.cross(z_axis, direction)
                axis_norm = np.linalg.norm(axis)

                if axis_norm > 1e-6:  # Avoid division by zero
                    axis /= axis_norm
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)) * 180 / np.pi
                    transform.rotate(angle, axis)

                elif np.allclose(direction, -z_axis):

                    transform.rotate(180, [1, 0, 0])

            transform.scale([self.scale, self.scale, self.scale])
            arrow.transform = transform

            # Debug direction
            # print(f"Axis {self.labels[i]} direction: {direction}")

    def _create_labels(self):
        """Create text labels for each axis."""
        for label, color, pos in zip(self.labels, self.colors, self.label_positions):
            text = Text(
                label,
                color=color,
                bold=False,
                font_size=self.font_size,
                anchor_x='center',
                anchor_y='center',
                pos=pos,
                parent=self.axis_root
            )
            # text.set_gl_state(depth_test=False)

    def _update_axis(self, event=None):
        """Update axis orientation based on camera angles."""

        cam = self.canvas.view.camera
        self.axis_root.transform.reset()
        self.axis_root.transform.rotate(cam.roll, (0, 0, 1))
        self.axis_root.transform.rotate(cam.elevation+90, (1, 0, 0))
        self.axis_root.transform.rotate(cam.azimuth, (0, 1, 0))
        self.axis_root.transform.scale((100, 100, 0.001))
        self.axis_root.transform.translate((80, self.canvas.size[1] - 80))

        self.axis_root.update()

    def _on_mouse_move(self, event):
        """Handle mouse move events for dragging."""
        if event.button == 1 and event.is_dragging:
            self._update_axis()

    def _on_resize(self, event):
        """Handle canvas resize events."""
        self._update_axis()

class StructurePlotWidget(scene.SceneCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.unfreeze()

        self.bgcolor = 'white'  # Set background to white

        self.grid = self.central_widget.add_grid(margin=5
                                                 )

        self.grid.spacing = 3

        self.view = self.grid.add_view(row=0,col=0,row_span=10,col_span=8)

        self.view.camera = 'turntable'  # Interactive camera
        self.auto_view=False
        self.ortho = False
        self.atom_items = []  # Store atom meshes and metadata
        self.bond_items = []  # Store bond meshes
        self.arrow_items = []
        self.arrow_colorbar = None
        self.arrow_config = None
        self.lattice_item = None  # Store lattice lines
        self.structure = None
        self.show_bond_flag = False
        self.axes=None
        self.scale_factor = 1
        initial_camera_dir = (0, -1, 0)  # for a default initialised camera

        self.initial_light_dir = self.view.camera.transform.imap(initial_camera_dir)[:3]

        # Precompute sphere template (reduced resolution)

        self.sphere_meshdata =create_sphere(15,15,depth=10,radius= 1 ,offset=False)


        # Precompute cylinder template
        self.cylinder_meshdata = create_cylinder(10,10, radius=[0.1,0.1],offset=False)





        self.shading_filter = ShadingFilter(shading="smooth",
                                            ambient_light = (1, 1, 1, .5),


                                            )
        self.set_projection(False)
        self.events.mouse_move.connect(self._on_mouse_move)
        self.events.resize.connect(self._on_resize)
    def _on_mouse_move(self,ev):
        """Handle mouse move events."""
        pass
    def _on_resize(self,ev):
        """Handle canvas resize events."""
        pass
        if self.arrow_colorbar is not None:


            self.arrow_colorbar.pos=[self.size[0]-50,self.size[1]-150]
            self.arrow_colorbar.update()
            # self.update( self.arrow_colorbar)
    def set_auto_view(self,auto_view):
        self.auto_view=auto_view
        if self.structure is not None:

            self.show_structure(self.structure)


    def set_projection(self, ortho=True):
        """Toggle between orthographic and perspective projection."""
        self.ortho = ortho
        current_state = {
            'center': self.view.camera.center,
            'elevation': self.view.camera.elevation,
            'azimuth': self.view.camera.azimuth,

        }
        if self.ortho:
            self.view.camera = scene.cameras.TurntableCamera(
                fov=0,  # Orthographic
                **current_state
            )
            self.view.camera.distance=350

        else:
            self.view.camera = scene.cameras.TurntableCamera(
                fov=60,  # Perspective
                **current_state
            )
            self.view.camera.distance=50

        self.update()

    def set_show_bonds(self, show_bonds=True):
        """Toggle bond visibility and adjust atom scaling."""
        self.show_bond_flag = show_bonds
        if self.structure is not None:
            self.scale_factor = 0.6 if show_bonds else 1
            self.show_structure(self.structure)

    def update_lighting(self):
        """Update light direction to follow camera."""
        # return
        transform = self.view.camera.transform
        dir = np.concatenate((self.initial_light_dir, [0]))
        light_dir = transform.map(dir)[:3]
        # Update shading filter for atoms, bonds, and halos
        self.shading_filter.light_dir  = tuple(light_dir)
        self.update()

        return


    def show_lattice(self, structure):
        """Draw the crystal lattice as 12 distinct edges."""
        if self.lattice_item:
            self.lattice_item.parent = None
        origin = np.array([0.0, 0.0, 0.0])
        a1, a2, a3 = structure.cell
        vertices = np.array([
            origin, a1, a2, a1 + a2, a3, a1 + a3, a2 + a3, a1 + a2 + a3
        ])
        edges = [
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ]
        lines = np.array([vertices[edge] for edge in edges]).reshape(-1, 3)
        self.lattice_item = Line(
            pos=lines,
            color=(0, 0, 0, 1),
            width=1.5,
            connect='segments',
            method='gl',
            parent=self.view.scene,antialias=True
        )


        #
        # def update_axis_visual():
        #     """Sync XYZAxis visual with camera angles"""
        #     axis.transform.reset()
        #
        #     axis.transform.rotate(self.view.camera.roll, (0, 0, 1))
        #     axis.transform.rotate(self.view.camera.elevation+90, (1, 0, 0))
        #     axis.transform.rotate(self.view.camera.azimuth, (0, 1, 0))
        #     axis.transform.scale((50, 50, 0.001))
        #     axis.transform.translate((50., 50.))
        #
        #     axis.update()
        #
        # update_axis_visual()
        #
        # @self.events.mouse_move.connect
        # def on_mouse_move(event):
        #     if event.button == 1 and event.is_dragging:
        #         update_axis_visual()
        #
        #
        self.axes = ArrowAxes(
            canvas=self,
            parent=self.scene,
            directions=[a1 / np.linalg.norm(a1),a2/ np.linalg.norm(a2), a3 / np.linalg.norm(a3)],
            colors=['red', 'green', 'blue'],
            labels=['X', 'Y', 'Z'],
            label_positions=[[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]],
            scale=0.5,
            font_size=12
        )

    def show_bond(self, structure):
        """Draw bonds as cylinders between atom pairs."""
        for item in self.bond_items:
            item.parent = None
        self.bond_items = []
        if not self.show_bond_flag:
            return
        bond_pairs = structure.get_bond_pairs()

        # Use precomputed cylinder template
        z_axis = np.array([0, 0, 1], dtype=float)

        all_vertices = []
        all_faces = []
        all_colors = []
        offset = 0
        base_faces=self.cylinder_meshdata.get_faces()
        base_vertices=self.cylinder_meshdata.get_vertices()
        for pair in bond_pairs:
            elem0 = str(structure.numbers[pair[0]])
            elem1 = str(structure.numbers[pair[1]])
            pos1 = structure.positions[pair[0]]
            pos2 = structure.positions[pair[1]]
            color1 = Color(table_info.get(elem0, {'color': '#808080'})['color']).rgba
            color2 = Color(table_info.get(elem1, {'color': '#808080'})['color']).rgba
            radius1 = table_info.get(elem0, {'color': '#808080'})['radii'] / 150 * self.scale_factor
            radius2 = table_info.get(elem1, {'radii': 70})['radii'] / 150 * self.scale_factor
            bond_radius = 0.12

            vector = pos2 - pos1
            full_length = np.linalg.norm(vector)
            if full_length == 0:
                continue
            direction = vector / full_length

            bond_length = full_length - radius1 - radius2

            start1 = pos1 + direction * radius1
            mid = start1 + direction * (bond_length / 2)
            length1 = bond_length / 2
            length2 = bond_length / 2
            # Compute orthogonal vectors (as in show_bond)
            if abs(direction[2]) < 0.999:
                v1 = np.cross(direction, [0, 0, 1])
            else:
                v1 = np.cross(direction, [0, 1, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(direction, v1)
            v2 = v2 / np.linalg.norm(v2)

            # Construct rotation matrix to align Z-axis with direction
            rot = np.eye(4)
            rot[:3, 0] = v1  # X-axis maps to v1
            rot[:3, 1] = v2  # Y-axis maps to v2
            rot[:3, 2] = direction  # Z-axis maps to bond direction

            for start, length, color in [(start1, length1, color1), (mid, length2, color2)]:
                scale = np.diag([1.0, 1.0, length, 1.0])
                transform = rot @ scale
                transform[:3, 3] = start
                verts = np.c_[base_vertices, np.ones(len(base_vertices))]
                verts = (transform @ verts.T).T[:, :3]
                faces = base_faces + offset
                offset += len(base_vertices)

                color_array = np.tile(color, (len(base_vertices), 1))
                all_vertices.append(verts)
                all_faces.append(faces)
                all_colors.append(color_array)

        # Merge all cylinders
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            colors = np.vstack(all_colors)
            mesh_data = MeshData(vertices=vertices, faces=faces, vertex_colors=colors)
            mesh = Mesh(
                meshdata=mesh_data,
                # shading='smooth',
                parent=self.view.scene
            )
            # mesh.attach(self.shading_filter)
            self.bond_items.append(mesh)

    def show_elem(self, structure):
        """Draw atoms as glossy spheres with merged geometry."""
        for item in self.atom_items:
            if item['mesh']:
                item['mesh'].parent = None
            if item['halo']:
                item['halo'].parent = None
        self.atom_items = []

        # Merge all atoms
        all_vertices = []
        all_faces = []
        all_colors = []
        face_offset = 0
        sphere_vertices=self.sphere_meshdata.get_vertices()
        sphere_faces=self.sphere_meshdata.get_faces()
        for idx, (n, p) in enumerate(zip(structure.numbers, structure.positions)):
            elem = str(n)
            color = Color(table_info.get(elem, {'color': '#808080'})['color']).rgba
            size = table_info.get(elem, {'radii': 70})['radii'] / 150 * self.scale_factor
            scaled_vertices = sphere_vertices * size + p
            all_vertices.append(scaled_vertices)
            all_faces.append(sphere_faces + face_offset)
            all_colors.append(np.repeat([color], len(sphere_vertices), axis=0))
            face_offset += len(sphere_vertices)
            self.atom_items.append({
                'mesh': None,
                'position': p,
                'original_color': color,
                'size': size,
                'halo': None,
                'vertex_range': (len(all_vertices) - 1) * len(sphere_vertices)
            })

        # Create single mesh for atoms
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            colors = np.vstack(all_colors)
            mesh_data = MeshData(vertices=vertices, faces=faces, vertex_colors=colors)
            mesh = Mesh(
                meshdata=mesh_data,
                # shading='smooth',
                parent=self.view.scene
            )
            mesh.attach(self.shading_filter)

            for item in self.atom_items:
                item['mesh'] = mesh

        # Highlight bad bonds

        radius_coefficient = Config.getfloat("widget", "radius_coefficient", 0.7)

        bond_pairs = structure.get_bad_bond_pairs(radius_coefficient)
        for pair in bond_pairs:
            self.highlight_atom(pair[0])
            self.highlight_atom(pair[1])

    def _clear_arrow_visuals(self):
        """Remove existing arrow meshes and colorbar."""
        for item in self.arrow_items:
            item.parent = None
        self.arrow_items = []
        if self.arrow_colorbar:
            self.arrow_colorbar.parent = None
            # self.arrow_colorbar._colorbar.parent = None
            # self.grid.remove_widget(self.arrow_colorbar)
            self.arrow_colorbar = None

    def show_arrow(self, prop_name="spin", scale=1.0, cmap="viridis"):
        """Display vector arrows for a given property."""
        self._clear_arrow_visuals()
        self.arrow_config = {"prop_name": prop_name, "scale": scale, "cmap": cmap}
        if self.structure is None:
            return
        if prop_name not in self.structure.structure_info:
            return
        vectors = self.structure.structure_info[prop_name]
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            return
        vectors = vectors * scale

        arrow_meshdata = create_arrow_mesh()
        z_axis = np.array([0, 0, 1], dtype=float)
        base_vertices = arrow_meshdata.get_vertices()
        base_faces = arrow_meshdata.get_faces()

        mags = np.linalg.norm(vectors, axis=1)
        max_mag = mags.max() if np.any(mags) else 1.0
        min_mag = mags.min() if np.any(mags) else 0

        cmap_obj = get_colormap(cmap)
        color_values = cmap_obj.map(mags / max_mag if max_mag > 0 else mags)

        all_vertices = []
        all_faces = []
        all_colors = []
        offset = 0

        for index, (pos, vec, color) in enumerate(zip(self.structure.positions, vectors, color_values)):
            length = np.linalg.norm(vec)
            radius = self.atom_items[index]['size']
            if length == 0:
                continue
            length += radius
            direction = vec / np.linalg.norm(vec)

            axis = np.cross(z_axis, direction)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
                rot = rotate(angle, axis)
            elif direction[2] < 0:
                rot = rotate(180, (1, 0, 0))
            else:
                rot = np.eye(4)

            scale_mat = np.diag([length, length, length, 1.0])
            transform = rot @ scale_mat
            transform[:3, 3] = pos

            verts = np.c_[base_vertices, np.ones(len(base_vertices))]
            verts = (transform @ verts.T).T[:, :3]

            faces = base_faces + offset
            offset += len(base_vertices)

            all_vertices.append(verts)
            all_faces.append(faces)
            all_colors.append(np.repeat([color], len(base_vertices), axis=0))

        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            colors = np.vstack(all_colors)
            arrow_meshdata = MeshData(vertices=vertices, faces=faces, vertex_colors=colors)
            arrow_mesh = Mesh(meshdata=arrow_meshdata, parent=self.view.scene)
            self.arrow_items.append(arrow_mesh)

            from vispy.scene.widgets import ColorBarWidget
            self.arrow_colorbar = scene.ColorBar(cmap=cmap_obj, orientation='right',
                                                 pos=(self.size[0]-50,self.size[1]-150),
                                                 size=(200,10),
                                                 label=prop_name,clim=(round(min_mag,2),round(max_mag)),parent=self.scene)
            # self.arrow_colorbar.transform = STTransform(translate=(-25, -25, 0))
            # self.arrow_colorbar.update()

            # self.arrow_colorbar.height_max = 100
            # self.arrow_colorbar.width_max = 5
            # self.grid.add_widget(self.arrow_colorbar,row=0, col=9,row_span=10,col_span=1 )

    def clear_arrow(self):
        """Remove arrow visuals and reset configuration."""
        self._clear_arrow_visuals()
        self.arrow_config = None

    def highlight_atom(self, atom_index):
        """Highlight an atom with a translucent halo."""
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            if atom['halo']:
                atom['halo'].parent = None
            halo_size = atom['size'] * 1.2
            halo_color = [1, 1, 0, 0.6]
            vertices = self.sphere_meshdata.get_vertices() * halo_size + atom['position']
            mesh_data = MeshData(vertices=vertices, faces=self.sphere_meshdata.get_faces())
            halo = Mesh(
                meshdata=mesh_data,
                color=halo_color,
                shading='smooth',
                parent=self.view.scene
            )

            self.atom_items[atom_index]['halo'] = halo
            self.update()

    def reset_atom(self, atom_index):
        """Remove halo from an atom."""
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            if atom['halo']:
                atom['halo'].parent = None
                self.atom_items[atom_index]['halo'] = None
            self.update()


    def show_structure(self, structure):
        """Display the entire crystal structure."""
        self.structure = structure
        if self.axes is not None:
            self.axes.axis_root.parent = None

        if self.lattice_item:
            self.lattice_item.parent = None

        self._clear_arrow_visuals()


        if self.auto_view:
            coords = structure.positions
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            max_dimension = np.max(size)
            fov = 60
            distance = max_dimension / (2 * np.tan(np.radians(fov / 2))) * 2.8
            aspect_ratio = size / np.max(size)
            flat_threshold = 0.5
            if aspect_ratio[0] < flat_threshold and aspect_ratio[1] >= flat_threshold and aspect_ratio[2] >= flat_threshold:
                elevation, azimuth = 0, 0
            elif aspect_ratio[1] < flat_threshold and aspect_ratio[0] >= flat_threshold and aspect_ratio[
                2] >= flat_threshold:
                elevation, azimuth = 0, 0
            elif aspect_ratio[2] < flat_threshold and aspect_ratio[0] >= flat_threshold and aspect_ratio[
                1] >= flat_threshold:
                elevation, azimuth = 90, 0
            else:
                elevation, azimuth = 30, 45
            self.view.camera.set_state({
                'center': tuple(center),
                'elevation': elevation,
                'azimuth': azimuth,

            })
            self.view.camera.distance=distance

        self.show_lattice(structure)
        self.show_elem(structure)
        self.show_bond(structure)
        cfg = self.arrow_config
        if cfg and cfg.get("prop_name") in structure.structure_info:
            self.show_arrow(cfg["prop_name"], cfg["scale"], cfg["cmap"])
        self.update_lighting()


    def on_mouse_move(self, event):
        """Update lighting during rotation."""
        if event.is_dragging:
            self.update_lighting()
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    from NepTrainKit.core.structure import Structure
    app = QApplication([])
    view = StructurePlotWidget()
    view.set_show_bonds(True)
    view.set_projection(True)
    view.show()
    import time
    start = time.time()
    atoms = Structure.read_xyz("good.xyz")
    view.show_structure(atoms)  # 修改为show_structure，与代码一致
    print(time.time() - start)
    QApplication.instance().exec_()