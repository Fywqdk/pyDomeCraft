import numpy as np
import pyvista as pv

# ---------------------------
# STEP 1: Geometry creation
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
    # normalize to unit sphere
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
    Subdivide a triangle (v1, v2, v3) into smaller triangles.
    The subdivision is done using barycentric coordinates.
    
    Returns
    -------
    vertices: np.ndarray
        Array of vertices for this subdivided triangle.
    triangles: np.ndarray
        Array of triangle definitions (indices into vertices).
    """
    points = []
    for i in range(frequency + 1):
        for j in range(frequency + 1 - i):
            k = frequency - i - j
            # Linear interpolation via barycentrics.
            point = (i * v1 + j * v2 + k * v3) / frequency
            points.append(point)
    points = np.array(points)
    
    # Build the triangles from the grid indices
    triangles = []
    # The lambda translates from grid coordinates (i,j) to a 1D index in points[]
    index = lambda i, j: int( i * (frequency + 1) - (i*(i-1))//2 + j )
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
    Project every point to lie on a sphere of the given radius.
    """
    projected = []
    for p in points:
        p_norm = np.linalg.norm(p)
        projected.append(p / p_norm * radius)
    return np.array(projected)

def create_geodesic_sphere(frequency, radius=1.0):
    """
    Create a geodesic sphere by subdividing each face of an icosahedron.
    
    Returns
    -------
    vertices: np.ndarray
        All vertices.
    triangles: np.ndarray
        Connectivity as an array of triangles (using vertex indices).
    """
    base_vertices, base_faces = create_icosahedron()
    all_vertices = []
    all_triangles = []
    vertex_offset = 0
    for face in base_faces:
        v1, v2, v3 = base_vertices[face[0]], base_vertices[face[1]], base_vertices[face[2]]
        verts, tris = subdivide_triangle(v1, v2, v3, frequency)
        # Project the subdivided points onto the sphere
        verts = project_to_sphere(verts, radius)
        all_vertices.append(verts)
        all_triangles.append(tris + vertex_offset)
        vertex_offset += verts.shape[0]
    all_vertices = np.vstack(all_vertices)
    all_triangles = np.vstack(all_triangles)
    return all_vertices, all_triangles

def filter_dome(vertices, triangles, upward_axis=1):
    """
    Filter the triangles for a dome (upper hemisphere; here, we take y>=0).
    
    Parameters
    ----------
    upward_axis: int
        Which axis is considered “up.” (0 for x, 1 for y, 2 for z). Default is y.
        
    Returns
    -------
    vertices: np.ndarray, triangles: np.ndarray
        Only the triangles whose all vertices have coordinate>=0 in the upward_axis.
    """
    keep = []
    for t in triangles:
        if np.all(vertices[t, upward_axis] >= 0):
            keep.append(t)
    return vertices, np.array(keep)

# ---------------------------
# STEP 2: Convert geometry to Minecraft block coordinates
# ---------------------------
def line_to_blocks(p0, p1, steps_mult=2):
    """
    Convert a continuous line segment from p0 to p1 into a set of block coordinates.
    This simple method uses linear interpolation and rounds coordinates to the nearest integer.
    
    Parameters
    ----------
    p0, p1: array-like
        Endpoints of the line.
    steps_mult: int
        Multiplier for the number of interpolation steps based on the distance.
    
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
    Approximate filling a triangle (given its 3D vertices) with blocks.
    The method projects the triangle onto a local 2D coordinate system,
    then checks for each block in the bounding box whether its center falls inside the triangle.
    
    Returns
    -------
    set of (x, y, z) tuples that are inside the triangle.
    """
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    
    # Build a local coordinate system for the triangle's plane.
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
    
    # Bounding box in local 2D coordinates.
    min_x = np.floor(min(A[0], B[0], C[0]))
    max_x = np.ceil(max(A[0], B[0], C[0]))
    min_y = np.floor(min(A[1], B[1], C[1]))
    max_y = np.ceil(max(A[1], B[1], C[1]))
    
    def point_in_triangle(P, A, B, C):
        # Using barycentric coordinate method
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
    for i in range(int(min_x), int(max_x)+1):
        for j in range(int(min_y), int(max_y)+1):
            center_local = np.array([i + 0.5, j + 0.5])
            if point_in_triangle(center_local, A, B, C):
                # Convert the local coordinate back to world coordinate.
                P_world = p0 + (i + 0.5)*e1 + (j + 0.5)*e2
                block = tuple(np.round(P_world).astype(int))
                blocks.add(block)
    return blocks

def generate_block_data(vertices, triangles):
    """
    From the continuous geometry (vertices and triangles),
    produce sets of block coordinates for the framework (edges)
    and the panels (filled triangles).
    
    The framework blocks are removed from the panels.
    """
    framework_blocks = set()
    panel_blocks = set()
    for tri in triangles:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        # Framework: add line segments along each edge.
        framework_blocks.update(line_to_blocks(p0, p1))
        framework_blocks.update(line_to_blocks(p1, p2))
        framework_blocks.update(line_to_blocks(p2, p0))
        # Panels: fill the triangle
        panel_blocks.update(fill_triangle(p0, p1, p2))
    # Remove any blocks that are already part of the framework.
    panel_blocks = panel_blocks - framework_blocks
    return framework_blocks, panel_blocks

# ---------------------------
# STEP 3: Visualization with PyVista
# ---------------------------
def visualize_dome(vertices, triangles):
    """
    Visualize the geodesic dome construction:
      - The panels (filled translucent triangles)
      - The framework (edges drawn as thicker black lines)
    """
    # Create a mesh for the panels; note that each face is defined by 3 vertices.
    n_faces = triangles.shape[0]
    faces = np.hstack((np.full((n_faces, 1), 3), triangles)).astype(np.int_)
    mesh = pv.PolyData(vertices, faces)
    # Extract the edges (the framework)
    edges = mesh.extract_all_edges()
    
    plotter = pv.Plotter()
    # Panels: white with 80% transparency (0.2 opacity)
    plotter.add_mesh(mesh, color="white", opacity=0.2, show_edges=False)
    # Framework: black edges
    plotter.add_mesh(edges, color="black", line_width=2)
    plotter.show()

def visualize_blocks(framework_blocks, panel_blocks):
    """
    Visualize the Minecraft block positions as two separate point clouds.
    """
    plotter = pv.Plotter()
    
    # Convert sets to numpy arrays
    if framework_blocks:
        fw_pts = np.array(list(framework_blocks))
        plotter.add_points(fw_pts, color="black", point_size=10, render_points_as_spheres=True)
    if panel_blocks:
        p_pts = np.array(list(panel_blocks))
        plotter.add_points(p_pts, color="white", opacity=0.2, point_size=10, render_points_as_spheres=True)
    
    plotter.show()

# ---------------------------
# Main assembly and run
# ---------------------------
if __name__ == '__main__':
    # Parameters:
    diameter = 20.0  # diameter of the geodesic dome in world units
    radius = diameter / 2.0
    frequency = 3    # level of subdivision; higher -> more detailed dome
    
    # Generate continuous dome geometry (entire sphere initially)
    vertices, triangles = create_geodesic_sphere(frequency, radius=radius)
    # For a dome, take only triangles with all vertices in the upper half (y>=0)
    vertices, dome_triangles = filter_dome(vertices, triangles, upward_axis=1)
    
    # Visualize the continuous dome (framework + panels)
    visualize_dome(vertices, dome_triangles)
    
    # Convert continuous geometry to Minecraft block coordinates.
    fw_blocks, panel_blocks = generate_block_data(vertices, dome_triangles)
    print("Framework blocks:", len(fw_blocks))
    print("Panel blocks:", len(panel_blocks))
    
    # Visualize the block positions (as a rough approximation)
    visualize_blocks(fw_blocks, panel_blocks)
