"""OpenGL-based structure plotting widgets using PyQtGraph.
"""

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import numpy as np
from OpenGL.GL import GL_PROJECTION, glLoadMatrixf, glMatrixMode
from NepTrainKit.config import Config


from PySide6.QtCore import Qt
from PySide6.QtGui import QColor,QMatrix4x4

from NepTrainKit.core.structure import table_info


class StructurePlotWidget(gl.GLViewWidget):
    """Interactive OpenGL widget for rendering atomic structures with configurable projections and overlays.
    """
    def __init__(self, *args, **kwargs):
        """Initialise the view widget and default rendering state.
        
        Parameters
        ----------
        *args : tuple
            Forwarded to :class:`pyqtgraph.opengl.GLViewWidget`.
        **kwargs : dict
            Forwarded to :class:`pyqtgraph.opengl.GLViewWidget`.
        """
        self.ortho=False
        super().__init__(*args, **kwargs)
        self.setBackgroundColor('w')
        self.setCameraPosition(distance=80, elevation=30, azimuth=30)
        self.atom_items = []
        self.auto_view=False

        self.structure = None
        self.show_bond_flag = None
        self.scale_factor = 1

    def set_auto_view(self, auto_view):
        """Toggle automatic camera framing when a structure is displayed.
        
        Parameters
        ----------
        auto_view : bool
            Whether to re-frame the camera on each call to :meth:`show_structure`.
        """
        self.auto_view = auto_view
        if self.structure is not None:
            self.show_structure(self.structure)

    def set_projection(self,ortho=True):
        """Switch between orthographic and perspective projections.
        
        Parameters
        ----------
        ortho : bool, optional
            Enable orthographic projection when ``True``.
        """
        self.ortho=ortho
        self.setProjection()
        self.update()

    def set_show_bonds(self,show_bonds=True):
        """Toggle rendering of bond cylinders.
        
        Parameters
        ----------
        show_bonds : bool, optional
            Whether bond geometry should be drawn when a structure is shown.
        """
        self.show_bond_flag = show_bonds
        if self.structure is not None:
            if show_bonds:
                self.scale_factor=0.6
                self.show_structure(self.structure)
            else:
                self.scale_factor=1

                self.show_structure(self.structure)

    def setProjection(self, region=None ):
        """Apply the current projection matrix to the OpenGL context.
        
        Parameters
        ----------
        region : tuple[int, int, int, int], optional
            Viewport rectangle used to recompute the projection.
        """
        m = self.projectionMatrix(region)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.array(m.data(), dtype=np.float32))

    def projectionMatrix(self, region=None ):
        """Compute the projection matrix with optional orthographic override.
        
        Parameters
        ----------
        region : tuple[int, int, int, int], optional
            Viewport rectangle passed by the underlying widget.
        
        Returns
        -------
        QMatrix4x4
            Projection matrix used for subsequent draws.
        """
        if self.ortho:
            x0, y0, w, h = self.getViewport()
            aspect = w / h if h != 0 else 1.0
            dist = max(self.opts['distance'], 1e-6)
            fov = self.opts['fov']
            nearClip = dist * 0.001
            farClip = dist * 1000.0
            r = dist * np.tan(np.radians(fov / 2))
            t = r * h / w
            region = region or (x0, y0, w, h)
            left = r * ((region[0] - x0) * (2.0 / w) - 1)
            right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
            bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
            top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)
            mat = QMatrix4x4()
            mat.setToIdentity()

            mat.ortho(left, right, bottom, top, nearClip, farClip)

            return mat
        else:
            return super().projectionMatrix(region)

    def mousePressEvent(self, event):
        """Defer mouse press handling to the base implementation.
        
        Parameters
        ----------
        event : QMouseEvent
            Mouse press event forwarded by Qt.
        """
        super().mousePressEvent(event)
        return
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            x, y = pos.x(), pos.y()

            proj = self.projectionMatrix()
            view = self.viewMatrix()
            viewport = self.getViewport()
            width, height = viewport[2], viewport[3]

            ndc_x = (2.0 * x) / width - 1.0
            ndc_y = 1.0 - (2.0 * y) / height
            ndc_z = 0.0

            proj_data = np.array(proj.data(), dtype=np.float32).reshape(4, 4, order='F')  # Column-major
            view_data = np.array(view.data(), dtype=np.float32).reshape(4, 4, order='F')  # Column-major
            proj_view = proj_data @ view_data
            inv_proj_view = np.linalg.inv(proj_view)

            ndc = np.array([ndc_x, ndc_y, ndc_z, 1.0])
            world_pos = inv_proj_view @ ndc
            world_pos /= world_pos[3]

            camera_pos = self.cameraPosition()
            ray_origin = np.array([camera_pos.x(), camera_pos.y(), camera_pos.z()])
            ray_dir = world_pos[:3] - ray_origin
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            for item in self.items:
                if not isinstance(item, gl.GLMeshItem):
                    continue
                meshdata=item.opts['meshdata']
                vertices = meshdata.vertexes()
                min_bound = vertices.min(axis=0)
                max_bound = vertices.max(axis=0)

                transform = item.transform()
                transform_data = np.array(transform.data(), dtype=np.float32).reshape(4, 4, order='F')
                inv_transform = np.linalg.inv(transform_data)
                local_origin = inv_transform @ np.append(ray_origin, 1.0)
                local_dir = inv_transform @ np.append(ray_dir, 0.0)
                local_origin = local_origin[:3] / local_origin[3]
                local_dir = local_dir[:3]

                t_min = (min_bound - local_origin) / local_dir
                t_max = (max_bound - local_origin) / local_dir
                t1 = np.minimum(t_min, t_max)
                t2 = np.maximum(t_min, t_max)
                t_near = np.max(t1)
                t_far = np.min(t2)

                if t_near <= t_far and t_far >= 0:
                    print(f"Clicked on item: {item}")
                    return

    # def mouseReleaseEvent(self, ev):
    #     super().mouseReleaseEvent(ev)
    #     if ev.button() == Qt.LeftButton:
    #         self.makeCurrent()
    #         pos = ev.pos()
    #         x, y = pos.x(), pos.y()
    #         region = (x,y, 5, 5)
    #         print(self.itemsAt(region))
            # region = (x, self.height() - y - 1, 1, 1)
            # clicked_items = self.itemsAt(region)
            # print(clicked_items)
            # if clicked_items:
            #     item = clicked_items[0]



    def addItems(self, items):
        """Add GL items to the scene and initialise them.
        
        Parameters
        ----------
        items : Iterable[gl.GLGraphicsItem]
            Items to add to the scene.
        """
        for item in items:

            self.items.append(item)

            if self.isValid():
                item.initialize()

            item._setView(self)
        self.update()

    def show_lattice(self, structure):
        """Render lattice edges for the provided structure.
        
        Parameters
        ----------
        structure : Any
            Object exposing ``cell`` coordinates.
        """
        origin = np.array([0.0, 0.0, 0.0])
        a1 = structure.cell[0]
        a2 = structure.cell[1]
        a3 = structure.cell[2]
        vertices = np.array([origin, a1, a2, a3, a1 + a2, a1 + a3, a2 + a3, a1 + a2 + a3])
        edges = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]
        lines = np.array([vertices[edge] for edge in edges]).reshape(-1, 3)
        lattice_lines = gl.GLLinePlotItem(
            pos=lines,
            color=(0, 0, 0, 1),
            width=1.5,
            mode='lines',
            glOptions="translucent",
            antialias=True
        )
        # center = structure.cell.sum(axis=0) / 2
        # self.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.addItem(lattice_lines)

    def show_bond(self,structure):
        """Create bond cylinders for the supplied structure when enabled.
        
        Parameters
        ----------
        structure : Any
            Structure containing atom positions and metadata.
        """
        if not self.show_bond_flag:
            return
        bond_pairs = structure.get_bond_pairs()
        bond_items = []
        for pair in bond_pairs:

            elem0_info = table_info[str(structure.numbers[pair[0]])]
            elem1_info = table_info[str(structure.numbers[pair[1]])]
            pos1 = structure.positions[pair[0]]
            pos2 = structure.positions[pair[1]]
            # bond_length = np.linalg.norm(pos1 - pos2)
            # if (elem0_info["radii"] + elem1_info["radii"]) * radius_coefficient_config > bond_length * 100:
            #     color1 = (1.0, 0.0, 0.0, 0.7)
            #     color2 = (1.0, 0.0, 0.0, 0.7)
            #     bond_radius = 0.3
            # else:
            color1 = QColor(elem0_info["color"]).getRgbF()
            color2 = QColor(elem1_info["color"]).getRgbF()
            bond_radius = 0.1
            radius1 = table_info[str(structure.numbers[pair[0]])]["radii"] / 150*self.scale_factor
            radius2 = table_info[str(structure.numbers[pair[1]])]["radii"] / 150*self.scale_factor
            bond1, bond2 = self.add_bond(pos1, pos2, color1, color2, radius1, radius2, bond_radius=bond_radius)
            bond_items.append(bond1)
            bond_items.append(bond2)

        self.addItems(bond_items)

    def add_bond(self, pos1, pos2, color1, color2, radius1, radius2, bond_radius=0.12):
        """Create two half-cylinders to visualise a bond between two atoms.
        
        Parameters
        ----------
        pos1 : ndarray
            Start point of the bond.
        pos2 : ndarray
            End point of the bond.
        color1 : tuple[float, float, float, float]
            RGBA colour applied to the first half.
        color2 : tuple[float, float, float, float]
            RGBA colour applied to the second half.
        radius1 : float
            Radius of the first atom.
        radius2 : float
            Radius of the second atom.
        bond_radius : float, optional
            Cylinder radius used to render the bond.
        
        Returns
        -------
        tuple of gl.GLMeshItem
            Mesh items representing both halves of the bond cylinder.
        """
        bond_vector = pos2 - pos1
        full_length = np.linalg.norm(bond_vector)
        bond_dir = bond_vector / full_length
        start_point = pos1
        end_point = pos2
        mid_point = (start_point + end_point) / 2
        bond = full_length - radius1 - radius2
        bond1_length = radius1 + bond / 2
        bond2_length = radius2 + bond / 2
        mid_point = start_point + bond_dir * bond1_length

        cylinder1 = gl.MeshData.cylinder(rows=6, cols=12, radius=[bond_radius, bond_radius], length=bond1_length)
        bond1 = gl.GLMeshItem(meshdata=cylinder1, smooth=True, shader='shaded', color=color1)
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, bond_dir)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, bond_dir)) * 180 / np.pi
            bond1.rotate(angle, axis[0], axis[1], axis[2])
        bond1.translate(start_point[0], start_point[1], start_point[2])
        # self.addItem(bond1)

        cylinder2 = gl.MeshData.cylinder(rows=6, cols=12, radius=[bond_radius, bond_radius], length=bond2_length)
        bond2 = gl.GLMeshItem(meshdata=cylinder2, smooth=True, shader='shaded', color=color2)
        if np.linalg.norm(axis) > 0:
            bond2.rotate(angle, axis[0], axis[1], axis[2])
        bond2.translate(mid_point[0], mid_point[1], mid_point[2])
        # self.addItem(bond2)
        return bond1, bond2

    def show_elem(self, structure):
        """Render atom spheres for the current structure and highlight problematic bonds.
        
        Parameters
        ----------
        structure : Any
            Structure instance providing ``numbers`` and ``positions`` arrays.
        """
        self.atom_items = []
        atom_items = []
        for idx, (n, p) in enumerate(zip(structure.numbers, structure.positions)):
            color = QColor(table_info[str(n)]["color"]).getRgbF()
            size = table_info[str(n)]["radii"] / 150*self.scale_factor
            sphere = gl.MeshData.sphere(rows=10, cols=10, radius=size)
            m = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=color)
            m.translate(p[0], p[1], p[2])

            atom_items.append(m)

            self.atom_items.append({"mesh": m, "position": p, "original_color": color, "size": size, "halo": None})
        self.addItems(atom_items)
        radius_coefficient_config = Config.getfloat("widget", "radius_coefficient", 0.7)
        bond_pairs = structure.get_bad_bond_pairs( radius_coefficient_config)
        for pair in bond_pairs:

            self.highlight_atom(pair[0])
            self.highlight_atom(pair[1])

    def highlight_atom(self, atom_index):
        """Emphasise an atom by adding a halo mesh.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom to highlight.
        """
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            highlight_size = atom["size"]
            # highlight_color =atom["original_color"]
            # self.removeItem(atom["mesh"])

            # sphere = gl.MeshData.sphere(rows=10, cols=10, radius=highlight_size)
            # new_mesh = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=highlight_color)
            # new_mesh.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            # self.addItem(new_mesh)

            halo_size = highlight_size * 1.2
            halo_color = ( 1, 1, 0, 0.6)
            halo_sphere = gl.MeshData.sphere(rows=10, cols=10, radius=halo_size)
            halo = gl.GLMeshItem(meshdata=halo_sphere, smooth=True, shader='shaded', color=halo_color, glOptions='translucent')
            halo.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            self.addItem(halo)

            # self.atom_items[atom_index]["mesh"] = new_mesh
            self.atom_items[atom_index]["halo"] = halo
            self.update()

    def reset_atom(self, atom_index):
        """Remove the highlight halo for the specified atom.
        
        Parameters
        ----------
        atom_index : int
            Index of the atom whose highlight should be cleared.
        """
        if 0 <= atom_index < len(self.atom_items):
            atom = self.atom_items[atom_index]
            # self.removeItem(atom["mesh"])
            if atom["halo"] is not None:
                self.atom_items[atom_index]["halo"] = None

                self.removeItem(atom["halo"])
            return
            sphere = gl.MeshData.sphere(rows=10, cols=10, radius=atom["size"])
            new_mesh = gl.GLMeshItem(meshdata=sphere, smooth=True, shader='shaded', color=atom["original_color"])
            new_mesh.translate(atom["position"][0], atom["position"][1], atom["position"][2])
            self.addItem(new_mesh)

            self.atom_items[atom_index]["mesh"] = new_mesh
            # self.atom_items[atom_index]["halo"] = None
            self.update()

    def show_structure(self, structure):
        """Render lattice, atoms, and bonds for the provided structure.
        
        Parameters
        ----------
        structure : Any
            Structure object with ``cell`` and ``positions`` attributes.
        """
        self.atom_items.clear()
        self.clear()
        self.structure = structure
        self.show_lattice(structure)
        self.show_elem(structure)
        self.show_bond(structure )

        if self.auto_view:
            coords = structure.positions
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            max_dimension = np.max(size)
            fov = 60
            distance = max_dimension / (2 * np.tan(np.radians(fov / 2))) * 2.8
            self.opts['center'] = pg.Vector(center[0], center[1], center[2])
            self.opts['distance'] = distance
            aspect_ratio = size / np.max(size)
            # print("aspect_ratio", aspect_ratio)
            flat_threshold=0.5
            if (aspect_ratio[0] < flat_threshold
                    and aspect_ratio[1] >= flat_threshold
                    and aspect_ratio[2] >= flat_threshold):
                self.opts['elevation'] = 0
                self.opts['azimuth'] = 0
            elif (aspect_ratio[1] < flat_threshold
                  and aspect_ratio[0] >= flat_threshold
                  and aspect_ratio[2] >= flat_threshold):
                self.opts['elevation'] = 0
                self.opts['azimuth'] = 0
            elif (aspect_ratio[2] < flat_threshold
                  and aspect_ratio[0] >= flat_threshold
                  and aspect_ratio[1] >= flat_threshold):
                self.opts['elevation'] = 90
                self.opts['azimuth'] = 0
            else:
                self.opts['elevation'] = 30
                self.opts['azimuth'] = 45

            self.setCameraPosition( )


        # self.highlight_atom(0)
