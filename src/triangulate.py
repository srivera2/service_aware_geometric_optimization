import matplotlib.pyplot as plt
import numpy as np
import drjit as dr
from shapely import contains_xy, constrained_delaunay_triangles
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import triangulate, unary_union
import triangle as tr
import scipy.stats.qmc as qmc


def get_zone_polygon_with_exclusions(
    zone_type="box",
    zone_params=None,
    scene_xml_path=None,
    exclude_buildings=True
):
    """
    Extract the target zone polygon and building exclusions for triangulation.

    This function prepares the data needed for constrained Delaunay triangulation
    by returning the outer boundary and building footprints as holes.

    Parameters:
    -----------
    zone_type : str
        Type of zone: "box" or "polygon"
    zone_params : dict
        Zone configuration:
        - For "box": {'center': (x, y), 'width': w, 'height': h}
        - For "polygon": {'vertices': [(x1, y1), (x2, y2), ...]}
    scene_xml_path : str or None
        Path to scene XML for extracting building footprints
    exclude_buildings : bool
        Whether to exclude building footprints as holes

    Returns:
    --------
    target_zone : list of tuples
        Vertices of the outer boundary [(x1, y1), (x2, y2), ...]
    building_exclusions : list of lists
        Each element is a list of (x, y) tuples for a building footprint
    zone_polygon : shapely.geometry.Polygon
        The complete polygon with holes (for visualization/verification)
    """
    if zone_params is None:
        raise ValueError("zone_params must be provided")

    # 1. Get the outer boundary vertices based on zone type
    if zone_type == "box":
        # Create box vertices from center, width, height
        box_center = zone_params.get("center", (0, 0))
        bx, by = box_center
        bw = zone_params.get("width", 20)
        bh = zone_params.get("height", 20)

        if bw <= 0 or bh <= 0:
            raise ValueError(
                f"Box width and height must be positive, got width={bw}, height={bh}"
            )

        # Create clockwise box vertices
        target_zone = [
            (bx - bw/2, by - bh/2),  # Bottom-left
            (bx + bw/2, by - bh/2),  # Bottom-right
            (bx + bw/2, by + bh/2),  # Top-right
            (bx - bw/2, by + bh/2),  # Top-left
        ]

    elif zone_type == "polygon":
        # Use provided vertices
        target_zone = zone_params.get("vertices", [(0, 0), (10, 0), (10, -10), (0, -10)])

        if len(target_zone) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(target_zone)}")

    else:
        raise ValueError(f"Unknown zone_type: {zone_type}. Must be 'box' or 'polygon'")

    # 2. Extract building footprints as exclusions (holes)
    building_exclusions = []

    if exclude_buildings and scene_xml_path is not None:
        from scene_parser import extract_building_info

        # Get building information
        building_info = extract_building_info(scene_xml_path, verbose=False)

        # Create the outer zone polygon to check for intersection
        zone_polygon_outer = Polygon(target_zone)

        # Extract 2D footprints for each building
        for _, info in building_info.items():
            # Get building polygon vertices (only X, Y coordinates)
            vertices_3d = info["vertices"]
            vertices_2d = [(v[0], v[1]) for v in vertices_3d]

            try:
                building_polygon = Polygon(vertices_2d)

                # Only include buildings that intersect with the target zone
                if building_polygon.is_valid and zone_polygon_outer.intersects(building_polygon):
                    # Get the exterior coordinates (shapely returns closed ring, remove duplicate last point)
                    coords = list(building_polygon.exterior.coords)[:-1]
                    building_exclusions.append(coords)

            except Exception:
                # Skip buildings with invalid geometry
                pass

    # 3. Create the complete polygon with holes (for return/verification)
    zone_polygon = Polygon(shell=target_zone, holes=building_exclusions)

    return target_zone, building_exclusions, zone_polygon


def simplify_building_polygons(building_exclusions, tolerance=0.5, verbose=False):
    """
    Simplify building polygons to reduce vertices and improve triangle quality.

    This uses the Douglas-Peucker algorithm to reduce complexity while preserving
    the general shape. Fewer vertices = fewer constraint edges = better triangles.

    Parameters:
    -----------
    building_exclusions : list of lists
        Building footprint vertices
    tolerance : float
        Simplification tolerance in meters. Higher = more simplification.
        - 0.1-0.5m: Minimal simplification (preserves details)
        - 0.5-1.0m: Moderate simplification (recommended)
        - 1.0-2.0m: Aggressive simplification (for distant/small buildings)
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    simplified_buildings : list of lists
        Simplified building footprint vertices
    """
    simplified_buildings = []
    orig_vertices = 0
    new_vertices = 0

    for i, building_coords in enumerate(building_exclusions):
        orig_vertices += len(building_coords)

        try:
            building_poly = Polygon(building_coords)

            if not building_poly.is_valid:
                building_poly = building_poly.buffer(0)

            # Simplify using Douglas-Peucker algorithm
            simplified_poly = building_poly.simplify(tolerance, preserve_topology=True)

            if simplified_poly.is_valid and not simplified_poly.is_empty:
                coords = list(simplified_poly.exterior.coords)[:-1]
                if len(coords) >= 3:
                    simplified_buildings.append(coords)
                    new_vertices += len(coords)
                    if verbose:
                        print(f"  Building {i}: {len(building_coords)} → {len(coords)} vertices")
                else:
                    # Too simplified, keep original
                    if verbose:
                        print(f"  Building {i}: Too simple after simplification, keeping original")
                    simplified_buildings.append(building_coords)
                    new_vertices += len(building_coords)
            else:
                # Simplification failed, keep original
                if verbose:
                    print(f"  Building {i}: Simplification failed, keeping original")
                simplified_buildings.append(building_coords)
                new_vertices += len(building_coords)

        except Exception as e:
            # Error during simplification, keep original
            if verbose:
                print(f"  Building {i}: Error during simplification ({e}), keeping original")
            simplified_buildings.append(building_coords)
            new_vertices += len(building_coords)

    if verbose:
        print(f"\nSimplification complete:")
        print(f"  Original vertices: {orig_vertices}")
        print(f"  Simplified vertices: {new_vertices}")
        if orig_vertices > 0:
            print(f"  Reduction: {100*(1 - new_vertices/orig_vertices):.1f}%")
        else:
            print(f"  No buildings to simplify")

    return simplified_buildings


def triangulate_zone(target_zone, building_exclusions, buffer_distance=-0.01, verbose=False):
    """
    1. Boolean Difference: Calculates (Zone - Buildings) to handle edge intersections.
    2. Triangulation: Meshes the resulting clean geometry.
    """
    
    # --- 1. Geometry Prep (Shapely) ---
    zone_poly = Polygon(target_zone).buffer(0) # Fix self-intersections
    
    # Prepare buildings (clean and union them to handle overlaps)
    valid_buildings = []
    for coords in building_exclusions:
        p = Polygon(coords).buffer(0)
        if buffer_distance != 0:
            p = p.buffer(buffer_distance)
        if not p.is_empty:
            valid_buildings.append(p)
            
    # Create one giant exclusion polygon
    if valid_buildings:
        all_buildings = unary_union(valid_buildings)
        # BOOLEAN SUBTRACTION: This calculates the exact cutting lines
        clean_zone = zone_poly.difference(all_buildings)
    else:
        clean_zone = zone_poly

    # Handle empty result (e.g., building covers entire zone)
    if clean_zone.is_empty:
        if verbose: print("Zone is completely covered by buildings.")
        return np.array([], dtype=np.float32), None

    # --- 2. Convert Shapely -> Triangle Format ---
    # clean_zone could be a Polygon or MultiPolygon (if a building splits the zone)
    
    vertices = []
    segments = []
    holes = []
    
    def parse_polygon(poly):
        """Helper to extract rings and holes from a single Polygon"""
        # A. Exterior Ring
        start_idx = len(vertices)
        ext_coords = list(poly.exterior.coords)[:-1] # Drop duplicate last point
        
        for i, (x, y) in enumerate(ext_coords):
            vertices.append([x, y])
            # Connect i to i+1 (wrap around)
            segments.append([start_idx + i, start_idx + (i + 1) % len(ext_coords)])
            
        # B. Interior Rings (Holes)
        for interior in poly.interiors:
            hole_start_idx = len(vertices)
            int_coords = list(interior.coords)[:-1]
            
            for i, (x, y) in enumerate(int_coords):
                vertices.append([x, y])
                segments.append([hole_start_idx + i, hole_start_idx + (i + 1) % len(int_coords)])
            
            # Add Hole Marker (The Virus Seed)
            # Find a point strictly inside this hole
            p = Polygon(interior).representative_point()
            holes.append([p.x, p.y])

    # Iterate over geometry parts
    if clean_zone.geom_type == 'Polygon':
        parse_polygon(clean_zone)
    elif clean_zone.geom_type == 'MultiPolygon':
        for p in clean_zone.geoms:
            parse_polygon(p)

    # --- 3. Run Triangulation ---
    data = {
        'vertices': np.array(vertices),
        'segments': np.array(segments)
    }
    if holes:
        data['holes'] = np.array(holes)
    
    # 'p': PSLG
    # 'q30': Quality mesh (min angle 30 deg) - prevents slivers!
    try:
        result = tr.triangulate(data, 'pq30')
    except Exception as e:
        if verbose: print(f"Triangulation Error: {e}")
        return np.array([], dtype=np.float32), None

    if 'triangles' not in result:
        return np.array([], dtype=np.float32), None
        
    tri_indices = result['triangles']
    tri_verts = result['vertices'][tri_indices]
    
    return tri_verts.astype(np.float32), clean_zone


def diagnose_polygon_issues(target_zone, building_exclusions, verbose=True):
    """
    Diagnose topology issues with a polygon and its holes.

    Parameters:
    -----------
    target_zone : list of tuples
        Vertices of the outer boundary
    building_exclusions : list of lists
        Building footprint vertices
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    dict with diagnostic information
    """
    diagnostics = {
        'zone_valid': False,
        'zone_area': 0,
        'num_buildings': len(building_exclusions),
        'valid_buildings': 0,
        'invalid_buildings': 0,
        'touching_boundary': 0,
        'outside_zone': 0,
        'overlapping_buildings': 0,
        'issues': []
    }

    # Check outer zone
    zone_poly = Polygon(target_zone)
    diagnostics['zone_valid'] = zone_poly.is_valid
    diagnostics['zone_area'] = zone_poly.area

    if not zone_poly.is_valid:
        diagnostics['issues'].append("Outer zone polygon is invalid")
        if verbose:
            print("ERROR: Outer zone polygon is invalid!")
            print(f"  Reason: {zone_poly.is_valid_reason if hasattr(zone_poly, 'is_valid_reason') else 'Unknown'}")

    # Check each building
    building_polys = []
    for i, building_coords in enumerate(building_exclusions):
        try:
            building_poly = Polygon(building_coords)

            if not building_poly.is_valid:
                diagnostics['invalid_buildings'] += 1
                diagnostics['issues'].append(f"Building {i} is invalid")
                if verbose:
                    print(f"WARNING: Building {i} is invalid")
                continue

            diagnostics['valid_buildings'] += 1

            # Check if building touches boundary
            if zone_poly.boundary.intersects(building_poly):
                diagnostics['touching_boundary'] += 1
                diagnostics['issues'].append(f"Building {i} touches zone boundary")
                if verbose:
                    print(f"WARNING: Building {i} touches the zone boundary")

            # Check if building is outside zone
            if not zone_poly.contains(building_poly) and not zone_poly.intersects(building_poly):
                diagnostics['outside_zone'] += 1
                diagnostics['issues'].append(f"Building {i} is outside zone")
                if verbose:
                    print(f"WARNING: Building {i} is outside the zone")

            building_polys.append((i, building_poly))

        except Exception as e:
            diagnostics['invalid_buildings'] += 1
            diagnostics['issues'].append(f"Building {i} error: {e}")
            if verbose:
                print(f"ERROR: Building {i} - {e}")

    # Check for overlapping buildings
    for i, (idx1, poly1) in enumerate(building_polys):
        for idx2, poly2 in building_polys[i+1:]:
            if poly1.intersects(poly2):
                diagnostics['overlapping_buildings'] += 1
                diagnostics['issues'].append(f"Buildings {idx1} and {idx2} overlap")
                if verbose:
                    print(f"WARNING: Buildings {idx1} and {idx2} overlap")

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("POLYGON DIAGNOSTICS SUMMARY")
        print("="*70)
        print(f"Zone valid: {diagnostics['zone_valid']}")
        print(f"Zone area: {diagnostics['zone_area']:.2f} m²")
        print(f"Buildings: {diagnostics['num_buildings']} total")
        print(f"  Valid: {diagnostics['valid_buildings']}")
        print(f"  Invalid: {diagnostics['invalid_buildings']}")
        print(f"  Touching boundary: {diagnostics['touching_boundary']}")
        print(f"  Outside zone: {diagnostics['outside_zone']}")
        print(f"  Overlapping: {diagnostics['overlapping_buildings']}")
        print(f"\nTotal issues: {len(diagnostics['issues'])}")
        if diagnostics['issues']:
            print("\nRECOMMENDATION:")
            if diagnostics['touching_boundary'] > 0 or diagnostics['overlapping_buildings'] > 0:
                print("  Use buffer_distance=-0.1 in triangulate_zone() to shrink buildings")
            if diagnostics['invalid_buildings'] > 0:
                print("  Some buildings have invalid geometry - they will be skipped")
        else:
            print("\nNo issues detected!")
        print("="*70)

    return diagnostics


def visualize_triangulation(
    triangles,
    target_zone=None,
    building_exclusions=None,
    zone_polygon=None,
    map_config=None,
    tx_position=None,
    title="Zone Triangulation",
    figsize=(14, 10),
    show_triangles=True,
    show_edges=True,
):
    """
    Visualize the triangulation of a zone with buildings excluded.

    Parameters:
    -----------
    triangles : np.ndarray
        Array of shape (N, 3, 2) containing N triangles
    target_zone : list of tuples or None
        Outer boundary vertices
    building_exclusions : list of lists or None
        Building footprint vertices
    zone_polygon : shapely.geometry.Polygon or None
        Complete polygon with holes (alternative to target_zone + building_exclusions)
    map_config : dict or None
        Map configuration with 'center', 'size' for setting plot limits
    tx_position : tuple or None
        (x, y, z) position of transmitter to mark on plot
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    show_triangles : bool
        Whether to show individual triangles with different colors
    show_edges : bool
        Whether to show triangle edges

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare triangle patches for visualization
    if show_triangles:
        # Create a colorful visualization of triangles
        tri_patches = []
        for tri in triangles:
            tri_patches.append(tri)

        # Use PolyCollection for efficient rendering
        tri_collection = PolyCollection(
            tri_patches,
            facecolors=plt.cm.Set3(np.linspace(0, 1, len(triangles))),
            edgecolors='black' if show_edges else 'none',
            linewidths=0.5 if show_edges else 0,
            alpha=0.7
        )
        ax.add_collection(tri_collection)

    # Plot triangle edges only (without fill)
    if show_edges and not show_triangles:
        for tri in triangles:
            tri_closed = np.vstack([tri, tri[0]])  # Close the triangle
            ax.plot(tri_closed[:, 0], tri_closed[:, 1], 'k-', linewidth=0.5, alpha=0.5)

    # Plot the outer boundary
    if target_zone is not None:
        zone_array = np.array(target_zone + [target_zone[0]])  # Close the polygon
        ax.plot(zone_array[:, 0], zone_array[:, 1], 'b-', linewidth=2, label='Target Zone')

    # Plot building exclusions (holes)
    if building_exclusions is not None:
        for building_verts in building_exclusions:
            building_array = np.array(building_verts + [building_verts[0]])  # Close
            ax.fill(building_array[:, 0], building_array[:, 1],
                   facecolor='gray', edgecolor='red',
                   linewidth=1.5, alpha=0.6)

    # Alternative: Plot from zone_polygon if provided
    if zone_polygon is not None and target_zone is None:
        # Plot exterior
        x, y = zone_polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Target Zone')

        # Plot holes (buildings)
        for interior in zone_polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor='gray', edgecolor='red',
                   linewidth=1.5, alpha=0.6)

    # Plot transmitter if provided
    if tx_position is not None:
        ax.plot(tx_position[0], tx_position[1], 'r*',
               markersize=20, label='Transmitter',
               markeredgecolor='darkred', markeredgewidth=1.5)

    # Set axis limits based on map_config or data
    if map_config is not None:
        center_x, center_y, _ = map_config["center"]
        width_m, height_m = map_config["size"]
        ax.set_xlim(center_x - width_m/2, center_x + width_m/2)
        ax.set_ylim(center_y - height_m/2, center_y + height_m/2)
    else:
        # Auto-scale based on triangles
        all_x = triangles[:, :, 0].flatten()
        all_y = triangles[:, :, 1].flatten()
        margin = 0.1 * max(all_x.ptp(), all_y.ptp())
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Add legend
    building_patch = mpatches.Patch(
        facecolor='gray', edgecolor='red', alpha=0.6, label='Buildings'
    )
    handles, _ = ax.get_legend_handles_labels()
    handles.append(building_patch)
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{title}\n{len(triangles)} triangles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')

    # Add text box with triangulation statistics
    stats_text = (
        f"Triangles: {len(triangles)}\n"
        f"Vertices: {len(triangles) * 3}\n"
        f"Buildings excluded: {len(building_exclusions) if building_exclusions else 0}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()

    return fig

def sample_triangulated_zone(tri_verts, num_samples, qrand, ground_z=0.0):
    """
    Snake-Sorted Stratified Sampling:
    Sorts triangles along a Z-order curve (Morton Code) before allocating samples.
    This ensures that the 1D LDS sequence maps to spatially adjacent triangles.
    """
    
    # --- Step 0: The "Snake" Sort (Spatially Reorder Triangles) ---
    # Calculate Centroids
    centroids = np.mean(tri_verts, axis=1) # Shape (N, 2)
    
    # Normalize coordinates to [0, 1] for code generation
    # (We need integers for bitwise operations, so we map 0..1 to 0..2^16)
    min_b, max_b = np.min(centroids, axis=0), np.max(centroids, axis=0)
    norm_centroids = (centroids - min_b) / (max_b - min_b + 1e-8)
    
    # Convert to 16-bit integers (resolution of 65536x65536 grid is plenty)
    # This acts as our "grid binning"
    coords = (norm_centroids * 65535).astype(np.uint32)
    
    # Interleave bits to create Morton Code (Z-Order)
    # This is a fast "bit-shuffling" trick to create the 1D sort key
    x = coords[:, 0]
    y = coords[:, 1]
    
    # "Spread" the bits (e.g. 1111 -> 01010101) to make room for interleaving
    def spread_bits(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    
    morton_codes = spread_bits(x) | (spread_bits(y) << 1)
    
    # SORT the geometry based on this code
    sort_order = np.argsort(morton_codes)
    
    # Apply sort to input vertices
    A = tri_verts[sort_order, 0, :]
    B = tri_verts[sort_order, 1, :]
    C = tri_verts[sort_order, 2, :]
    
    # --- Step 1: Calculate Triangle Areas (Same as before) ---
    # Now calculating areas on the SORTED arrays
    areas = 0.5 * np.abs(
        A[:,0]*(B[:,1] - C[:,1]) + 
        B[:,0]*(C[:,1] - A[:,1]) + 
        C[:,0]*(A[:,1] - B[:,1])
    )
    total_area = np.sum(areas)

    # --- Step 2: Deterministic Allocation (Same as before) ---
    target_counts = (areas / total_area) * num_samples
    int_counts = np.floor(target_counts).astype(int)
    
    # Stochastic Rounding (Fixes the "Blackout" issue for small triangles)
    frac_part = target_counts - int_counts
    # Use independent random (not qrand) for the rounding decision to avoid aliasing
    bonus_points = (np.random.random(len(areas)) < frac_part).astype(int)
    final_counts = int_counts + bonus_points

    # --- Step 3: Expand Indices ---
    # The 'triangle_indices' are now pointing to our SORTED list.
    # Because the list is sorted spatially, index i and index i+1 are neighbors.
    triangle_indices = np.repeat(np.arange(len(areas)), final_counts)
    
    # --- Step 4: Generate LDS Sequence ---
    actual_total = len(triangle_indices)
    raw_samples = np.array(qrand.random(actual_total))
    
    if raw_samples.shape[1] < 2:
        extra = np.array(qrand.random(actual_total))
        raw_samples = np.hstack([raw_samples, extra])

    u_bary_1 = raw_samples[:, 0]
    u_bary_2 = raw_samples[:, 1]

    # --- Step 5: Map to Position ---
    A_selected = A[triangle_indices]
    B_selected = B[triangle_indices]
    C_selected = C[triangle_indices]

    sqrt_r1 = np.sqrt(u_bary_1)
    w_A = (1 - sqrt_r1)[:, None]
    w_B = (sqrt_r1 * (1 - u_bary_2))[:, None]
    w_C = (sqrt_r1 * u_bary_2)[:, None]

    points_2d = w_A * A_selected + w_B * B_selected + w_C * C_selected
    
    # --- Step 6: Z-Dimension ---
    z_col = np.full((actual_total, 1), ground_z)
    sampled_points_3d = np.hstack([points_2d, z_col])

    return sampled_points_3d.astype(np.float32)


def calculate_discrepancy_score(points, zone_polygon, num_probes=1000):
    """
    Estimates discrepancy by placing random test circles inside the zone
    and comparing Actual Count vs. Expected Count.
    """
    total_area = zone_polygon.area
    total_points = len(points)
    max_deviation = 0.0
    
    # Pre-calculate bounds for faster probe generation
    minx, miny, maxx, maxy = zone_polygon.bounds
    
    # Convert points to Shapely for containment checks (slow but accurate)
    # Optimized: Use simple boolean masking for speed if possible, 
    # but Shapely is easiest for "Arbitrary Polygon" proof.
    pts_as_shapely = [Point(p[0], p[1]) for p in points]
    
    for _ in range(num_probes):
        # 1. Generate a random test circle strictly inside the polygon
        #    (Rejection sample the probe itself!)
        valid_probe = False
        while not valid_probe:
            cx = np.random.uniform(minx, maxx)
            cy = np.random.uniform(miny, maxy)
            # Random radius between 1% and 10% of zone width
            r = np.random.uniform((maxx-minx)*0.01, (maxx-minx)*0.1)
            probe = Point(cx, cy).buffer(r)
            
            if zone_polygon.contains(probe):
                valid_probe = True
        
        # 2. Expected Ratio (Area)
        expected_ratio = probe.area / total_area
        
        # 3. Actual Ratio (Count)
        #    Count how many points fall inside this probe
        count = sum(1 for p in pts_as_shapely if probe.contains(p))
        actual_ratio = count / total_points
        
        # 4. Deviation
        diff = abs(actual_ratio - expected_ratio)
        if diff > max_deviation:
            max_deviation = diff
            
    return max_deviation

import time 

# --- Comparison Runner ---
def run_comparison(zone_poly, mesh, num_samples_list=[256, 1024, 4096, 8192]):
    results_rejection = []
    results_cdt = []
    
    print(f"{'N Samples':<10} | {'Rejection D':<12} | {'CDT D':<12} | {'Improvement'}")
    print("-" * 50)
    
    for N in num_samples_list:
        # --- 1. Rejection Sampler (Method A) ---
        # Generate slightly more, filter, then truncate to N
        # (Simulating the 'broken sequence')
        eng_A = qmc.LatinHypercube(d=2, scramble=True)
        # Bounding Box Logic
        minx, miny, maxx, maxy = zone_poly.bounds
        w, h = maxx-minx, maxy-miny
        
        # --- 1. Measure Rejection Sampler ---
        t0 = time.perf_counter()

        # Generate pool (Cost #1: Generating the numbers)
        raw_A = eng_A.random(N * 5) 
        #print(raw_A)

        pts_A = []
        # Filter pool (Cost #2: Geometric checks)
        for p in raw_A:
            px, py = minx + p[0]*w, miny + p[1]*h
            if zone_poly.contains(Point(px, py)):
                pts_A.append([px, py])
                if len(pts_A) == N: break

        t_rejection = (time.perf_counter() - t0) * 1000 # Record time immediately

        # --- 2. Measure CDT Sampler ---
        t0 = time.perf_counter()

        # Initialize Engine (Cost #1: Setup)
        eng_B = qmc.LatinHypercube(d=3, scramble=True)

        # Generate & Map (Cost #2: The Unified Function)
        # This returns a fully computed Numpy array (Eager execution)
        pts_B_3d = sample_triangulated_zone(mesh, N, eng_B, ground_z=0)

        t_cdt = (time.perf_counter() - t0) * 1000 # Record time immediately

        # --- Post-Processing (Outside the Timer) ---
        # Slice to 2D for the discrepancy check
        pts_B = pts_B_3d[:, :2]

        print(f"N={N:<5} | Rej: {t_rejection:6.2f} ms | CDT: {t_cdt:6.2f} ms | Speedup: {t_rejection/t_cdt:.1f}x")
        
        # --- 3. Measure Discrepancy ---
        d_A = calculate_discrepancy_score(pts_A, zone_poly)
        d_B = calculate_discrepancy_score(pts_B, zone_poly) # Will be same as A in this mock
        
        print(f"{N:<10} | {d_A:.5f}      | {d_B:.5f}      | {(d_A/d_B - 1)*100:.1f}%")
        
        results_rejection.append(d_A)
        results_cdt.append(d_B)

    return num_samples_list, results_rejection, results_cdt

# Usage:
# Ns, res_A, res_B = run_comparison(my_shapely_polygon, my_mesh)