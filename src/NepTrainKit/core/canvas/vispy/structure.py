#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/5/19 20:45
# @Author  : 兵
# @email    : 1747193328@qq.com


from NepTrainKit.core import Config
from NepTrainKit.core.structure import table_info
import numpy as np

from vispy import app, scene, visuals
from vispy.geometry import MeshData
from vispy.scene.visuals import Mesh, Line
from vispy.color import Color



class StructurePlotWidget(scene.SceneCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.unfreeze()

        self.bgcolor = 'white'  # Set background to white
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'  # Interactive camera
        self.ortho = False
        self.atom_items = []  # Store atom meshes and metadata
        self.bond_items = []  # Store bond meshes
        self.lattice_item = None  # Store lattice lines
        self.structure = None
        self.show_bond_flag = False
        self.scale_factor = 1
        initial_camera_dir = (0, -1, 0)  # for a default initialised camera

        self.initial_light_dir = self.view.camera.transform.imap(initial_camera_dir)[:3]

        # Precompute sphere template (reduced resolution)
        phi, theta = np.mgrid[0:np.pi:15j, 0:2 * np.pi:15j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        self.sphere_vertices = np.c_[x.ravel(), y.ravel(), z.ravel()]
        self.sphere_faces = []
        n_theta = 15
        for i in range(14):
            for j in range(14):
                v0 = i * n_theta + j
                v1 = v0 + 1
                v2 = (i + 1) * n_theta + j
                v3 = v2 + 1
                self.sphere_faces.append([v0, v1, v2])
                self.sphere_faces.append([v1, v3, v2])
        self.sphere_faces = np.array(self.sphere_faces)

        # Precompute cylinder template
        self.n_segments = 8  # Optimized for performance
        theta = np.linspace(0, 2 * np.pi, self.n_segments, endpoint=False)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        self.cylinder_template = {'cos_theta': cos_theta, 'sin_theta': sin_theta}
        self.cylinder_faces = []
        for i in range(self.n_segments):
            i1, i2 = i, (i + 1) % self.n_segments
            self.cylinder_faces.append([i1, i2, i2 + self.n_segments])
            self.cylinder_faces.append([i1, i2 + self.n_segments, i1 +self.n_segments])
            self.cylinder_faces.append([2 * self.n_segments, i2, i1])  # Bottom cap
            self.cylinder_faces.append([2 * self.n_segments + 1, i1 + self.n_segments, i2 + self.n_segments])  # Top cap
        self.cylinder_faces = np.array(self.cylinder_faces)

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
        else:
            self.view.camera = scene.cameras.TurntableCamera(
                fov=60,  # Perspective
                **current_state
            )
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
        for item in self.atom_items:
            if item['mesh'] and hasattr(item['mesh'], 'shading_filter'):
                item['mesh'].shading_filter.light_dir = tuple(light_dir)
                # item['mesh']._program.draw()  # Force shader update
            if item['halo'] and hasattr(item['halo'], 'shading_filter'):
                item['halo'].shading_filter.light_dir = tuple(light_dir)
                # item['halo']._program.draw()  # Force shader update
        for mesh in self.bond_items:
            if hasattr(mesh, 'shading_filter'):
                mesh.shading_filter.light_dir = tuple(light_dir)
                # mesh._program.draw()  # Force shader update
        self.update()

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
            parent=self.view.scene
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
        cos_theta = self.cylinder_template['cos_theta']
        sin_theta = self.cylinder_template['sin_theta']
        cylinder_vertices = []
        cylinder_faces = []
        cylinder_colors = []
        vertex_offset = 0

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

            # Bond geometry
            bond_vector = pos2 - pos1
            full_length = np.linalg.norm(bond_vector)
            bond_dir = bond_vector / full_length
            start_point = pos1
            mid_point = pos1 + bond_dir * (radius1 + (full_length - radius1 - radius2) / 2)
            bond1_length = (full_length - radius1 - radius2) / 2 + radius1
            bond2_length = (full_length - radius1 - radius2) / 2 + radius2

            # Orthogonal vectors
            if abs(bond_dir[2]) < 0.999:
                v1 = np.cross(bond_dir, [0, 0, 1])
            else:
                v1 = np.cross(bond_dir, [0, 1, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(bond_dir, v1)
            v2 = v2 / np.linalg.norm(v2)

            # Cylinder 1
            vertices = []
            colors = []
            for z in [0, bond1_length]:
                for i in range(self.n_segments):
                    x = cos_theta[i] * bond_radius
                    y = sin_theta[i] * bond_radius
                    pos = start_point + bond_dir * z + v1 * x + v2 * y
                    vertices.append(pos)
                    colors.append(color1)
            vertices.append(start_point)  # Bottom center
            colors.append(color1)
            vertices.append(start_point + bond_dir * bond1_length)  # Top center
            colors.append(color1)
            cylinder_vertices.append(vertices)
            cylinder_faces.append(self.cylinder_faces + vertex_offset)
            cylinder_colors.append(colors)
            vertex_offset += len(vertices)

            # Cylinder 2
            vertices = []
            colors = []
            for z in [0, bond2_length]:
                for i in range(self.n_segments):
                    x = cos_theta[i] * bond_radius
                    y = sin_theta[i] * bond_radius
                    pos = mid_point + bond_dir * z + v1 * x + v2 * y
                    vertices.append(pos)
                    colors.append(color2)
            vertices.append(mid_point)  # Bottom center
            colors.append(color2)
            vertices.append(mid_point + bond_dir * bond2_length)  # Top center
            colors.append(color2)
            cylinder_vertices.append(vertices)
            cylinder_faces.append(self.cylinder_faces + vertex_offset)
            cylinder_colors.append(colors)
            vertex_offset += len(vertices)

        # Merge all cylinders
        if cylinder_vertices:
            vertices = np.vstack(cylinder_vertices)
            faces = np.vstack(cylinder_faces)
            colors = np.vstack(cylinder_colors)
            mesh_data = MeshData(vertices=vertices, faces=faces, vertex_colors=colors)
            mesh = Mesh(
                meshdata=mesh_data,
                shading='smooth',
                parent=self.view.scene
            )

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
        for idx, (n, p) in enumerate(zip(structure.numbers, structure.positions)):
            elem = str(n)
            color = Color(table_info.get(elem, {'color': '#808080'})['color']).rgba
            size = table_info.get(elem, {'radii': 70})['radii'] / 150 * self.scale_factor
            scaled_vertices = self.sphere_vertices * size + p
            all_vertices.append(scaled_vertices)
            all_faces.append(self.sphere_faces + face_offset)
            all_colors.append(np.repeat([color], len(self.sphere_vertices), axis=0))
            face_offset += len(self.sphere_vertices)
            self.atom_items.append({
                'mesh': None,
                'position': p,
                'original_color': color,
                'size': size,
                'halo': None,
                'vertex_range': (len(all_vertices) - 1) * len(self.sphere_vertices)
            })

        # Create single mesh for atoms
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            colors = np.vstack(all_colors)
            mesh_data = MeshData(vertices=vertices, faces=faces, vertex_colors=colors)
            mesh = Mesh(
                meshdata=mesh_data,
                shading='smooth',
                parent=self.view.scene
            )

            for item in self.atom_items:
                item['mesh'] = mesh

        # Highlight bad bonds

        radius_coefficient = Config.getfloat("widget", "radius_coefficient", 0.7)

        bond_pairs = structure.get_bad_bond_pairs(radius_coefficient)
        for pair in bond_pairs:
            self.highlight_atom(pair[0])
            self.highlight_atom(pair[1])

    def highlight_atom(self, atom_index):
        """Highlight an atom with a translucent halo."""
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            if atom['halo']:
                atom['halo'].parent = None
            halo_size = atom['size'] * 1.2
            halo_color = [1, 1, 0, 0.6]
            vertices = self.sphere_vertices * halo_size + atom['position']
            mesh_data = MeshData(vertices=vertices, faces=self.sphere_faces)
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


        if self.lattice_item:
            self.lattice_item.parent = None
        self.show_lattice(structure)
        self.show_elem(structure)
        self.show_bond(structure)
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