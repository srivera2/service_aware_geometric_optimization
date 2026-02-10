import torch
import numpy as np
import drjit as dr
from drjit.auto import Float, Array3f, UInt
import mitsuba as mi
from shapely.geometry import Point, Polygon, LineString
from scene_parser import extract_building_info

class TxPlacement:
    """
    Handles placement of the transmitter.
    """
    def __init__(self, scene, tx_name, scene_xml_path, building_id, offset=0.0, create_if_missing=True):
        self.scene = scene
        self.tx_name = tx_name
        self.offset = offset
        self.scene_xml_path = scene_xml_path
        self.building_info = extract_building_info(scene_xml_path)
        self.building_id = building_id
        self.building = self.building_info[self.building_id]

        # Get or create the transmitter
        self.tx = scene.get(tx_name)
        if self.tx is None and create_if_missing:
            # Import here to avoid circular dependencies
            from sionna.rt import Transmitter
            # Create transmitter at origin initially
            self.tx = Transmitter(name=tx_name, position=[0.0, 0.0, 0.0])
            scene.add(self.tx)

    def project_to_polygon(self, x, y):
        """
        Project point (x, y) to nearest valid point inside/on polygon.

        Parameters:
        -----------
        x : float
            X coordinate
        y : float
            Y coordinate

        Returns:
        --------
        proj_x : float
            Projected X coordinate
        proj_y : float
            Projected Y coordinate
        """
        # Extract 2D coordinates from vertices
        vertices_2d = self.building['vertices'][:, :2]  # Take only x, y columns

        point = Point(float(x), float(y))
        poly = Polygon(vertices_2d)

        # If inside, return as-is
        if poly.contains(point) or poly.touches(point):
            return float(x), float(y)

        # If outside, project to nearest point on boundary
        boundary = poly.boundary
        nearest_point = boundary.interpolate(boundary.project(point))
        return float(nearest_point.x), float(nearest_point.y)

    def project_to_polygon_edge(self, x, y):
        """
        Project point (x, y) to the nearest point on the polygon boundary (edge).
        Unlike project_to_polygon, this ALWAYS projects to the edge, even if the
        point is inside the polygon.

        Parameters:
        -----------
        x : float
            X coordinate
        y : float
            Y coordinate

        Returns:
        --------
        proj_x : float
            Projected X coordinate on the edge
        proj_y : float
            Projected Y coordinate on the edge
        """
        # Extract 2D coordinates from vertices
        vertices_2d = self.building['vertices'][:, :2]  # Take only x, y columns

        point = Point(float(x), float(y))
        poly = Polygon(vertices_2d)

        # Always project to the boundary (edge), regardless of whether inside or outside
        boundary = poly.boundary
        nearest_point = boundary.interpolate(boundary.project(point))
        return float(nearest_point.x), float(nearest_point.y)

    def set_rooftop_center(self):
        """
        Places the transmitter at the center of the building's roof with optional offset.
        The offset is specified in __init__ and stored in self.offset.
        """
        x_pos = self.building["center"][0]
        y_pos = self.building["center"][1]
        z_pos = self.building["z_height"] + self.offset
        self.tx.position = mi.Point3f(float(x_pos), float(y_pos), float(z_pos))

    def get_line_manifold(self, p_start, p_end):
        """
        Returns a function that maps a scalar `alpha` to a point on the line segment.
        P_tx = p_start + alpha * (p_end - p_start)
        """
        p_start_tensor = torch.tensor(p_start, dtype=torch.float32)
        p_end_tensor = torch.tensor(p_end, dtype=torch.float32)

        def place_on_line(alpha):
            """
            alpha: a tensor with a single value between 0 and 1.
            """
            # Ensure alpha is a tensor
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)

            # Sigmoid activation to keep alpha between 0 and 1
            alpha_sigmoid = torch.sigmoid(alpha)

            p_tx = p_start_tensor + alpha_sigmoid * (p_end_tensor - p_start_tensor)
            return p_tx

        return place_on_line
