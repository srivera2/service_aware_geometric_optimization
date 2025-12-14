import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def parse_scene_buildings(scene_xml_path):
    """
    Parse scene.xml to extract building information

    Parameters:
    -----------
    scene_xml_path : str
        Path to the scene.xml file

    Returns:
    --------
    buildings : dict
        Dictionary mapping building_id to rooftop mesh path
    """
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    buildings = {}

    # Find all rooftop mesh shapes
    for shape in root.findall(".//shape[@type='ply']"):
        shape_id = shape.get('id', '')
        if 'rooftop' in shape_id:
            # Extract building number from id like "mesh-building_0_rooftop"
            parts = shape_id.split('_')
            if len(parts) >= 3 and parts[0] == 'mesh-building':
                building_id = int(parts[1])
                filename = shape.find("string[@name='filename']").get('value')
                buildings[building_id] = filename

    return buildings

def load_rooftop_mesh(mesh_path):
    """
    Load rooftop PLY mesh and extract vertices

    Parameters:
    -----------
    mesh_path : str
        Path to the rooftop PLY file

    Returns:
    --------
    vertices : numpy array
        Nx3 array of vertex coordinates
    z_height : float
        Z-coordinate (height) of the rooftop
    """
    from plyfile import PlyData

    plydata = PlyData.read(mesh_path)
    vertices = plydata['vertex']

    # Extract x, y, z coordinates
    x = np.array(vertices['x'])
    y = np.array(vertices['y'])
    z = np.array(vertices['z'])

    # Stack into Nx3 array
    coords = np.column_stack([x, y, z])

    # Get the rooftop height (max z value)
    z_height = np.max(z)

    return coords, z_height

def extract_building_info(scene_xml_path, verbose=False):
    """
    List all available buildings in the scene

    Parameters:
    -----------
    scene_xml_path : str
        Path to the scene.xml file
    verbose : bool
        If True, print building info to console
    """
    buildings = parse_scene_buildings(scene_xml_path)
    if verbose:
        print(f"Available buildings in scene: {len(buildings)} total")
        print("=" * 80)

    scene_dir = Path(scene_xml_path).parent
    building_info = {}

    for building_id in sorted(buildings.keys()):
        mesh_file = scene_dir / buildings[building_id]

        if mesh_file.exists():
            try:
                coords, z_height = load_rooftop_mesh(str(mesh_file))

                # Calculate bounding box
                x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
                x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                building_info[building_id] = {
                    'z_height': z_height,
                    'x_range': (x_min, x_max),
                    'y_range': (y_min, y_max),
                    'center': (x_center, y_center),
                    'vertices': coords
                }
                if verbose:
                    print(f"\nBuilding {building_id}:")
                    print(f"  Rooftop height: {z_height:.2f} m")
                    print(f"  X range: [{x_min:.2f}, {x_max:.2f}] m (center: {x_center:.2f})")
                    print(f"  Y range: [{y_min:.2f}, {y_max:.2f}] m (center: {y_center:.2f})")
                    print(f"  Vertices: {len(coords)}")

            except Exception as e:
                if verbose:
                    print(f"\nBuilding {building_id}: Error loading mesh - {e}")
        else:
            if verbose:
                print(f"\nBuilding {building_id}: Mesh file not found")
    if verbose:
        print("=" * 80)
    return building_info
