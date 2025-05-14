import numpy as np
import pyvista as pv

# ---------------------------
# Geometry Creation Functions
# ---------------------------
def create_icosahedron():
    """
    Returns vertices and faces for a unit icosahedron.
    """
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
         [-1,  phi, 0],
         [ 1,  phi, 0],
         [-1, -phi, 0],
         [ 1, -phi, 0],
         [0, -1,  phi],
         [0,  1,  phi],
         [0, -1, -phi],
         [0,  1, -phi],
         [ phi, 0, -1],
         [ phi, 0,  1],
         [-phi, 0, -1],
         [-phi, 0,  1]
    ])
    # Normalize the vertices to lie on a unit sphere.
    vertices = vertices / np.linalg.norm(vertices[0])
    faces = np.array([
       [0, 11, 5],
       [0, 5, 1],
       [0, 1, 7],
       [0, 7, 10],
       [0, 10, 11],
       [1, 5, 9],
       [5, 11, 4],
       [11, 10, 2],
       [10, 7, 6],
       [7, 1, 8],
       [3, 9, 4],
       [3, 4, 2],
       [3, 2, 6],
       [3, 6, 8],
       [3, 8, 9],
       [4, 9, 5],
       [2, 4, 11],
       [6, 2, 10],
       [8, 6, 7],
       [9, 8, 1]
    ])
    return vertices, faces

def subdivide_triangle(v1, v2, v3, frequency):
    """
    Subdivide a triangle (v1, v2, v3) into smaller triangles
    using barycentric coordinates.

    Parameters
    ----------
    v1, v2, v3 : array-like
         Vertices of the triangle.
    frequency : int
         The subdivision frequency (controls the number of subtriangles).
    
    Returns
    -------
    points : np.ndarray
         Array of subdivided triangle vertices.
    triangles : np.ndarray
         Connectivity (indices into points) for the smaller triangles.
    """
    points = []
    # Create points in barycentric coordinates
    for i in range(frequency + 1):
        for j in range(frequency + 1 - i):
            k = frequency - i - j
            point = (i * v1 + j * v2 + k * v3) / frequency
            points.append(point)
    points = np.array(points)
    
    # Build triangles based on the grid indices
    triangles = []
    index = lambda i, j: int(i * (frequency + 1) - (i*(i-1))//2 + j)
    for i in range(frequency):
        for j in range(frequency - i):
            a = index(i, j)
            b = index(i + 1, j)
            c = index(i, j + 1)
            triangles.append([a, b, c])
            if j < frequency - i - 1:
                d = index(i + 1, j + 1)
                triangles.append([b, d, c])
    return points, np.array(triangles)

def project_to_sphere(points, radius=1.0):
    """
    Project each point onto the surface of a sphere with the provided radius.
    """
    projected = []
    for p in points:
        p_norm = np.linalg.norm(p)
        projected.append(p / p_norm * radius)
    return np.array(projected)

def create_geodesic_sphere(frequency, radius=1.0):
    """
    Create a geodesic sphere by subdividing each face of an icosahedron.

    Parameters
    ----------
    frequency : int
         The level of subdivision.
    radius : float
         The radius of the sphere.

    Returns
    -------
    vertices : np.ndarray
         All vertices of the subdivided sphere.
    triangles : np.ndarray
         Connectivity as an array of triangles (using vertex indices).
    """
    base_vertices, base_faces = create_icosahedron()
    all_vertices = []
    all_triangles = []
    vertex_offset = 0
    for face in base_faces:
        v1, v2, v3 = base_vertices[face[0]], base_vertices[face[1]], base_vertices[face[2]]
        verts, tris = subdivide_triangle(v1, v2, v3, frequency)
        verts = project_to_sphere(verts, radius)
        all_vertices.append(verts)
        all_triangles.append(tris + vertex_offset)
        vertex_offset += verts.shape[0]
    all_vertices = np.vstack(all_vertices)
    all_triangles = np.vstack(all_triangles)
    return all_vertices, all_triangles

def filter_dome(vertices, triangles, upward_axis=1):
    """
    Filter triangles to form a dome (upper hemisphere).
    
    Parameters
    ----------
    upward_axis : int
         Which coordinate axis is "up" (default is y: index 1).
    
    Returns
    -------
    vertices : np.ndarray
         Unchanged array of vertices.
    triangles : np.ndarray
         Only the triangles with all vertices having coordinate >= 0 in the upward_axis.
    """
    keep = []
    for t in triangles:
        if np.all(vertices[t, upward_axis] >= 0):
            keep.append(t)
    return vertices, np.array(keep)

# ---------------------------
# Minecraft Block Conversion Functions
# ---------------------------
def line_to_blocks(p0, p1, steps_mult=2):
    """
    Convert a continuous line segment from p0 to p1 into a set of block coordinates.
    
    Parameters
    ----------
    p0, p1: array-like
         Endpoints of the line.
    steps_mult: int
         Multiplier for the number of interpolation steps.
    
    Returns
    -------
    set of (x, y, z) tuples representing block positions.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    distance = np.linalg.norm(p1 - p0)
    steps = int(np.ceil(distance)) * steps_mult
    blocks = set()
    for t in np.linspace(0, 1, steps):
        point = p0 + t * (p1 - p0)
        block = tuple(np.round(point).astype(int))
        blocks.add(block)
    return blocks

def fill_triangle(p0, p1, p2):
    """
    Approximate filling a triangle with blocks.
    
    The triangle is projected onto a local 2D coordinate system,
    and each block within the bounding box is checked for coverage by the triangle.

    Returns
    -------
    set of (x, y, z) tuples within the triangle.
    """
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    
    # Build a local coordinate system for the triangle.
    e1 = p1 - p0
    e1 = e1 / np.linalg.norm(e1)
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normal / np.linalg.norm(normal)
    e2 = np.cross(normal, e1)
    
    def to_local(P):
        v = P - p0
        return np.array([np.dot(v, e1), np.dot(v, e2)])
    
    A = to_local(p0)
    B = to_local(p1)
    C = to_local(p2)
    
    # Bounding box in local coordinates.
    min_x = np.floor(min(A[0], B[0], C[0]))
    max_x = np.ceil(max(A[0], B[0], C[0]))
    min_y = np.floor(min(A[1], B[1], C[1]))
    max_y = np.ceil(max(A[1], B[1], C[1]))
    
    def point_in_triangle(P, A, B, C):
        # Using barycentric coordinates
        v0 = C - A
        v1 = B - A
        v2 = P - A
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    blocks = set()
    # Iterate over the bounding box in local coordinates.
    for i in range(int(min_x), int(max_x) + 1):
        for j in range(int(min_y), int(max_y) + 1):
            center_local = np.array([i + 0.5, j + 0.5])
            if point_in_triangle(center_local, A, B, C):
                # Convert the local coordinate back to world coordinates.
                P_world = p0 + (i + 0.5) * e1 + (j + 0.5) * e2
                block = tuple(np.round(P_world).astype(int))
                blocks.add(block)
    return blocks

def generate_block_data(vertices, triangles):
    """
    From the continuous geometry, produce sets of block coordinates for the framework (edges)
    and the panels (filled triangles). Framework blocks are removed from panel blocks.
    
    Returns
    -------
    framework_blocks : set of tuples
    panel_blocks : set of tuples
    """
    framework_blocks = set()
    panel_blocks = set()
    for tri in triangles:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        # Add line segments to the framework.
        framework_blocks.update(line_to_blocks(p0, p1))
        framework_blocks.update(line_to_blocks(p1, p2))
        framework_blocks.update(line_to_blocks(p2, p0))
        # Fill the triangle for panels.
        panel_blocks.update(fill_triangle(p0, p1, p2))
    # Remove blocks that are already part of the framework.
    panel_blocks = panel_blocks - framework_blocks
    return framework_blocks, panel_blocks

# ---------------------------
# Visualization Functions
# ---------------------------
def visualize_dome(vertices, triangles):
    """
    Visualize the continuous geodesic dome using PyVista.
    
    Panels are displayed as translucent white surfaces,
    and framework edges are drawn in black.
    """
    n_faces = triangles.shape[0]
    # Each face is prefixed by the number of vertices (3) for PyVista.
    faces = np.hstack((np.full((n_faces, 1), 3), triangles)).astype(np.int_)
    mesh = pv.PolyData(vertices, faces)
    edges = mesh.extract_all_edges()
    
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(mesh, color="white", opacity=0.2, show_edges=False)
    plotter.add_mesh(edges, color="black", line_width=2)
    plotter.show()

def visualize_blocks(framework_blocks, panel_blocks):
    """
    Visualize Minecraft block positions as cubes.
    Framework blocks: drawn as gray cubes with black edges.
    Panel blocks: drawn as white cubes (50% transparent) with gray edges.
    """
    plotter = pv.Plotter(window_size=[800, 600])
    
    def cube_at(coord, face_color, edge_color, opacity=1.0):
        x, y, z = coord
        # Create a unit cube centered at (x, y, z)
        cube = pv.Cube(bounds=(x - 0.5, x + 0.5,
                                 y - 0.5, y + 0.5,
                                 z - 0.5, z + 0.5))
        plotter.add_mesh(cube, color=face_color, opacity=opacity,
                         show_edges=True, edge_color=edge_color)
    
    # Draw the framework cubes.
    for block in framework_blocks:
        cube_at(block, face_color="gray", edge_color="black", opacity=1.0)
    
    # Draw the panel cubes.
    for block in panel_blocks:
        cube_at(block, face_color="white", edge_color="gray", opacity=0.5)
    
    plotter.show()

# ---------------------------
# Main Program
# ---------------------------
if __name__ == '__main__':
    # Get dome parameters from shell input.
    try:
        diameter_input = input("Enter dome diameter (in world units, e.g., 20): ")
        frequency_input = input("Enter subdivision frequency (an integer, e.g., 3): ")
        diameter = float(diameter_input)
        frequency = int(frequency_input)
    except ValueError:
        print("Invalid input. Using default values: diameter=20, frequency=3")
        diameter = 20.0
        frequency = 3

    radius = diameter / 2.0

    # Generate the continuous geodesic sphere.
    vertices, triangles = create_geodesic_sphere(frequency, radius=radius)
    # For a dome, filter to the upper hemisphere (y >= 0).
    vertices, dome_triangles = filter_dome(vertices, triangles, upward_axis=1)

    print("Continuous dome geometry generated.")
    print(f"Total vertices: {vertices.shape[0]}, Total triangles: {dome_triangles.shape[0]}")

    # Visualize the continuous dome structure.
    visualize_dome(vertices, dome_triangles)

    # Convert the continuous geometry to Minecraft block coordinates.
    fw_blocks, panel_blocks = generate_block_data(vertices, dome_triangles)
    print("Minecraft block conversion complete.")
    print("Framework blocks:", len(fw_blocks))
    print("Panel blocks:", len(panel_blocks))

    # Visualize the block positions as cubes.
    visualize_blocks(fw_blocks, panel_blocks)
