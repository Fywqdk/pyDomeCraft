> I want to make a tool to calculate how to build a geodesic dome in minecraft. I plan to use python and pyvista for visualization. Can you give me a hand? Basically I want to be able to build, in minecraft, a geodesic dome with struts in one block type and the panels in another. I need the program to calculate a the framework for a geodesic dome (triangles only is fine) of a given diameter, and then convert that data to blocks of one type. The intra-framework panels should then be the second type of block. For simplicity lets visualize it with black blocks for the framework, and 80% translucent white blocks for the panels.

Below is an example of how you might start building such a tool. The idea is to first generate the continuous geodesic dome geometry—using an icosahedron as a base and subdividing its triangular faces—then “snap” those continuous coordinates to Minecraft’s integer grid. For visualization, we can use PyVista to render the dome with two colors (black for the framework/struts and a mostly transparent white for the panels). You can then later export or “print” the block coordinates for use in Minecraft (for example, by writing out a schematic file or using an in‐game builder).

Below is a self-contained Python script that shows one approach. Note that the voxelization (converting a continuous line or triangle into a set of blocks) is a non‑trivial problem. The code below uses a simple “round‑and‑linspace” approach for lines (the “framework”) and a basic barycentric fill for triangles (the panels). You may need to refine these methods depending on how faithful you need the dome to be and how Minecraft tolerates slight gaps or extraneous blocks.

---

```python
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
```

---

### How This Works

1. **Geometry Generation:**  
   We start with an icosahedron (a simple polyhedron) and subdivide each triangular face using barycentric interpolation. After subdivision, each point is projected onto the sphere—here, scaled to our desired dome radius. Finally, we filter out only those triangles whose vertices sit in the upper half (assuming the y‑axis is “up”) to form the dome. This method gives you a framework composed entirely of triangles .

2. **Mapping to Minecraft Blocks:**  
   - **Framework (struts):** For each edge of the triangle, we sample points along the line (using linear interpolation) and round them to the nearest integer coordinate. This yields a set of block positions for the struts (which you can then set to black in your visualization or as one block type in Minecraft).  
   - **Panels:** For each triangle face, we determine its 2D projection in a local coordinate system and “fill” the triangle by iterating through its bounding box and checking, via barycentric coordinates, which grid cells fall inside. Blocks already used in the framework are then removed from the panel set so that the struts remain distinct.  

3. **Visualization:**  
   Using PyVista, the continuous dome is visualized as a translucent white dome with black edges. Additionally, a second visualization shows the computed block positions (as point clouds) for framework and panels.  

### Next Steps and Extensions

- **Refinement:** The voxelization algorithms above are rather rudimentary. In a production tool you might want to use optimized voxelization (for example, using a true 3D Bresenham algorithm for lines or a more robust triangle fill routine).  
- **Exporting to Minecraft:** Once you have block coordinates, you might export them to a schematic file or use an API (such as [MCEdit](https://www.mcedit.net/) or other Minecraft modding libraries) to place the blocks in your game world.  
- **User Input and Adjustments:** You could add a GUI (for instance, using Tkinter or PyQt) to let the user input the dome’s diameter, subdivision frequency, and block types, and see an immediate visual preview via PyVista.

This blueprint should give you a solid foundation to build a more fully featured geodesic dome constructor for Minecraft. Enjoy developing your tool—and happy dome building!

: This is a standard method used in geodesic dome design and has been adapted in various computer‐aided architectural planning tools.

> Very nice start. Thanks. Can you change the visualize_blocks function to draw cubes instead? Lets change the colors for the framework blocks to a gray with black edges, and the panel blocks to have a gray edge, to easily see each block

Below is an updated version of the `visualize_blocks` function. In this version, each block coordinate is represented by a unit cube (with its center placed at the block's coordinate). Framework cubes are drawn with a gray face and black edges, while panel cubes are drawn with a white face (with partial opacity) and gray edges. For clarity, each cube is generated using PyVista's `Cube` object, and then added to the plotter.

```python
def visualize_blocks(framework_blocks, panel_blocks):
    """
    Visualize Minecraft block positions as cubes.
    Framework blocks: drawn as gray cubes with black edges.
    Panel blocks: drawn as white cubes (with partial opacity) with gray edges.
    """
    import pyvista as pv

    plotter = pv.Plotter(window_size=[800, 600])
    
    def cube_at(coord, face_color, edge_color, opacity=1.0):
        # Place a unit cube centered at the block coordinate.
        x, y, z = coord
        cube = pv.Cube(bounds=(x - 0.5, x + 0.5, 
                                 y - 0.5, y + 0.5, 
                                 z - 0.5, z + 0.5))
        plotter.add_mesh(cube, color=face_color, opacity=opacity, 
                         show_edges=True, edge_color=edge_color)
    
    # Add framework cubes: gray cube faces with black edges.
    for block in framework_blocks:
        cube_at(block, face_color="gray", edge_color="black", opacity=1.0)
    
    # Add panel cubes: white cube faces (50% transparent) with gray edges.
    for block in panel_blocks:
        cube_at(block, face_color="white", edge_color="gray", opacity=0.5)
    
    plotter.show()
```

### How It Works

1. **Cube Placement:**  
   Each block coordinate is assumed to be the center of a unit cube. The function `cube_at` calculates the proper cube bounds (from x-0.5 to x+0.5, etc.) and creates a cube using `pv.Cube`.

2. **Coloring and Edges:**  
   For the framework cubes, we set the face color to gray and the edge color to black to emphasize the structure. For the panel cubes, we set the face color to white with 50% opacity and add gray edges so each block is clearly outlined.

3. **Rendering:**  
   All cubes are added to the same PyVista plotter and then rendered. This gives you a 3D visualization that mirrors Minecraft’s blocky style while clearly differentiating between the dome’s struts (framework) and its panels.

This approach is ideal for testing and visualization. For a larger number of blocks, you might consider combining meshes or using more efficient batching methods—but for many geodesic dome projects, this method is a clear and visual starting point.

If you want to dive deeper into refining the block conversion or exploring more advanced voxelization, just let me know. Happy coding and dome building!

> Perfect. Now refactor the whole thing to allow input of frequency and diameter. Lets skip the GUI and just use shell input

Below is a complete, refactored Python script that accepts shell input for the dome's diameter and subdivision frequency. This version builds the dome framework, computes the Minecraft block positions, and visualizes the blocks as cubes—framework blocks with gray faces and black edges, and panel blocks with white faces (partially transparent) and gray edges.

```python
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
```

---

### Explanation

1. **Geometry Construction:**  
   - The script starts by generating an icosahedron and subdividing its faces based on the given frequency.  
   - Each triangle is then projected onto a sphere whose radius is half the input diameter.  
   - Only the upper hemisphere (assuming the y‑axis is up) is kept to form the dome.

2. **Block Conversion:**  
   - The continuous edges are sampled to create framework block positions.  
   - Each triangular face is "filled" using a simple barycentric technique to generate panel block positions (with framework blocks removed).

3. **Visualization:**  
   - The continuous dome geometry is visualized with PyVista showing the translucent panels and bold black framework edges.  
   - Next, the Minecraft blocks are displayed as cubes. Framework cubes are drawn in gray with black outlines, and panel cubes are white (50% transparent) with gray edges to clearly distinguish them.

4. **Shell Input:**  
   - At runtime, the program prompts the user for the dome’s diameter and subdivision frequency. If the input is invalid, default values are used.

This refactored script gives you a command-line tool to quickly generate, inspect, and visualize a geodesic dome design ready for conversion into Minecraft blocks. Enjoy coding and happy dome building! 

If you’re curious about refining voxelization or adding export options for Minecraft schematics, there are many avenues to explore next.

> All the data is stored as numpy arrays right? So it would be trivial to add a feature outputting each block layer as a 2d plan (seen from above). Please give it a try. If possible with a slider to move through the layers.

Below is an updated, complete version of the code. In this version, after generating the dome and converting it into block coordinates, we add a new function that uses Matplotlib to display a “plan view” of each horizontal layer (using the y‑coordinate). A slider at the bottom lets you jump from layer to layer: each layer is drawn as a 2D plot (x vs. z), with the framework blocks being a filled gray square with black edges and the panel blocks as white squares (50% transparent) with gray edges.

```python
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
    # Normalize to lie on the unit sphere.
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
    Subdivide a triangle (v1, v2, v3) into smaller ones using barycentric coordinates.
    
    Returns:
      points : np.ndarray of subdivided vertices,
      triangles : np.ndarray: indices for the smaller triangles.
    """
    points = []
    # Generate points in barycentric coordinates.
    for i in range(frequency + 1):
        for j in range(frequency + 1 - i):
            k = frequency - i - j
            point = (i * v1 + j * v2 + k * v3) / frequency
            points.append(point)
    points = np.array(points)
    
    # Generate connectivity using grid indices.
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
    Project each point to lie on a sphere with the given radius.
    """
    projected = []
    for p in points:
        p_norm = np.linalg.norm(p)
        projected.append(p / p_norm * radius)
    return np.array(projected)

def create_geodesic_sphere(frequency, radius=1.0):
    """
    Creates a geodesic sphere from an icosahedron by subdividing each face.
    
    Returns:
      vertices : np.ndarray
      triangles : np.ndarray (each triangle is defined by vertex indices)
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
    Filters triangles to keep only those that lie on the upper hemisphere.
    
    Parameters:
      upward_axis : int, usually 1 for y-axis.
    
    Returns:
      vertices (unchanged), and triangles that belong to the dome.
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
    Convert a line segment from p0 to p1 into a set of block coordinates by sampling and rounding.
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
    Approximate filling a triangle with blocks. The triangle is projected to a local 2D plane,
    then each grid cell in the bounding box is checked using barycentric coordinates.
    """
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    
    # Establish a local coordinate system on the triangle's plane.
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
    
    # Determine bounding box in local coordinates.
    min_x = np.floor(min(A[0], B[0], C[0]))
    max_x = np.ceil(max(A[0], B[0], C[0]))
    min_y = np.floor(min(A[1], B[1], C[1]))
    max_y = np.ceil(max(A[1], B[1], C[1]))
    
    def point_in_triangle(P, A, B, C):
        # Barycentric coordinate test.
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
    # Iterate over the bounding box and check each grid cell in the local system.
    for i in range(int(min_x), int(max_x) + 1):
        for j in range(int(min_y), int(max_y) + 1):
            center_local = np.array([i + 0.5, j + 0.5])
            if point_in_triangle(center_local, A, B, C):
                P_world = p0 + (i + 0.5)*e1 + (j + 0.5)*e2
                block = tuple(np.round(P_world).astype(int))
                blocks.add(block)
    return blocks

def generate_block_data(vertices, triangles):
    """
    From the continuous geometry generate sets of block positions for:
      - the framework (edges), and
      - the panels (filled triangles).
    Framework blocks are then removed from the panels.
    """
    framework_blocks = set()
    panel_blocks = set()
    for tri in triangles:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        framework_blocks.update(line_to_blocks(p0, p1))
        framework_blocks.update(line_to_blocks(p1, p2))
        framework_blocks.update(line_to_blocks(p2, p0))
        panel_blocks.update(fill_triangle(p0, p1, p2))
    panel_blocks = panel_blocks - framework_blocks
    return framework_blocks, panel_blocks

# ---------------------------
# Visualization Functions
# ---------------------------
def visualize_dome(vertices, triangles):
    """
    Visualize the continuous geodesic dome using PyVista.
    Panels are rendered as a translucent white surface and the framework as bold black edges.
    """
    n_faces = triangles.shape[0]
    faces = np.hstack((np.full((n_faces, 1), 3), triangles)).astype(np.int_)
    mesh = pv.PolyData(vertices, faces)
    edges = mesh.extract_all_edges()
    
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(mesh, color="white", opacity=0.2, show_edges=False)
    plotter.add_mesh(edges, color="black", line_width=2)
    plotter.show()

def visualize_blocks(framework_blocks, panel_blocks):
    """
    Visualize Minecraft block positions as 3D cubes using PyVista.
    Framework blocks are drawn as gray cubes with black edges.
    Panel blocks are drawn as white cubes (50% transparent) with gray edges.
    """
    plotter = pv.Plotter(window_size=[800, 600])
    
    def cube_at(coord, face_color, edge_color, opacity=1.0):
        x, y, z = coord
        cube = pv.Cube(bounds=(x - 0.5, x + 0.5,
                                 y - 0.5, y + 0.5,
                                 z - 0.5, z + 0.5))
        plotter.add_mesh(cube, color=face_color, opacity=opacity,
                         show_edges=True, edge_color=edge_color)
    
    for block in framework_blocks:
        cube_at(block, face_color="gray", edge_color="black", opacity=1.0)
    for block in panel_blocks:
        cube_at(block, face_color="white", edge_color="gray", opacity=0.5)
    
    plotter.show()

def visualize_layer_slider(framework_blocks, panel_blocks):
    """
    Display a 2D plan view (from above) for each block layer (based on y coordinate).

    A Matplotlib slider allows you to move through the layers. In each layer, a block is represented 
    by a square centered at its (x, z) coordinate:
      - Framework squares have a gray fill with a black edge.
      - Panel squares have a white fill (50% transparent) with a gray edge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.patches import Rectangle

    # Combine blocks to compute overall range of y-values.
    all_blocks = list(framework_blocks.union(panel_blocks))
    ys = [b[1] for b in all_blocks]
    min_y = int(min(ys))
    max_y = int(max(ys))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)
    
    def draw_layer(y_layer):
        ax.clear()
        ax.set_title(f"Layer {y_layer} (Plan View)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal')
        # Plot framework blocks for the current layer.
        for block in framework_blocks:
            if block[1] == y_layer:
                x, _, z = block
                rect = Rectangle((x-0.5, z-0.5), 1, 1, facecolor="gray", edgecolor="black")
                ax.add_patch(rect)
        # Plot panel blocks for the current layer.
        for block in panel_blocks:
            if block[1] == y_layer:
                x, _, z = block
                rect = Rectangle((x-0.5, z-0.5), 1, 1, facecolor="white", edgecolor="gray", alpha=0.5)
                ax.add_patch(rect)
        # Adjust plot limits based on blocks in this layer.
        xs = [b[0] for b in all_blocks if b[1] == y_layer]
        zs = [b[2] for b in all_blocks if b[1] == y_layer]
        if xs and zs:
            ax.set_xlim(min(xs)-1, max(xs)+1)
            ax.set_ylim(min(zs)-1, max(zs)+1)
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        fig.canvas.draw_idle()

    initial_layer = (min_y + max_y) // 2
    draw_layer(initial_layer)

    # Add slider at the bottom.
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Layer', min_y, max_y, valinit=initial_layer, valstep=1)

    def update(val):
        layer = int(slider.val)
        draw_layer(layer)
    slider.on_changed(update)
    plt.show()

# ---------------------------
# Main Program
# ---------------------------
if __name__ == '__main__':
    # Get dome parameters.
    try:
        diameter_input = input("Enter dome diameter (e.g., 20): ")
        frequency_input = input("Enter subdivision frequency (e.g., 3): ")
        diameter = float(diameter_input)
        frequency = int(frequency_input)
    except ValueError:
        print("Invalid input. Using default values: diameter=20, frequency=3")
        diameter = 20.0
        frequency = 3

    radius = diameter / 2.0

    # Generate and filter the geodesic dome (upper hemisphere).
    vertices, triangles = create_geodesic_sphere(frequency, radius=radius)
    vertices, dome_triangles = filter_dome(vertices, triangles, upward_axis=1)
    print(f"Continuous dome geometry generated: {vertices.shape[0]} vertices, {dome_triangles.shape[0]} triangles.")

    # Visualize the continuous dome.
    visualize_dome(vertices, dome_triangles)

    # Compute Minecraft block coordinates.
    fw_blocks, panel_blocks = generate_block_data(vertices, dome_triangles)
    print("Block conversion complete.")
    print(f"Framework blocks: {len(fw_blocks)}, Panel blocks: {len(panel_blocks)}")
    
    # Visualize as 3D cubes.
    visualize_blocks(fw_blocks, panel_blocks)
    
    # Finally, show a 2D plan view with a slider to select layers.
    visualize_layer_slider(fw_blocks, panel_blocks)
```

---

### How It Works

1. **Dome and Block Generation:**  
   The code first builds the geodesic dome (with a given diameter and subdivision frequency) using an icosahedron as the base and subdividing its faces. It then filters the dome to keep only the upper hemisphere (using y ≥ 0) and converts the continuous geometry into Minecraft-style block positions for both the framework (edges) and the panels (triangle fills).

2. **3D Visualizations:**  
   - The `visualize_dome` function illustrates the continuous dome using PyVista.
   - The `visualize_blocks` function draws each block as a cube: framework cubes are shown in gray with black edges, while panel cubes are white (at 50% opacity) with gray edges.

3. **2D Plan View with Layer Slider:**  
   The new `visualize_layer_slider` function uses Matplotlib to create a 2D overhead plan. It determines the range of vertical layers (by the y‑coordinate) and draws a plan view for the selected layer by plotting squares (each block appears as a square with a 1×1 size). A slider at the bottom lets you switch through layers interactively.

This extended tool should be a fun playground for both designing your dome and reviewing it layer by layer in a Minecraft-style plan view. Enjoy experimenting and happy coding!

> Great. If I wanted to keep fixed x and y axis in the matplotlib 2d plan view, what are the right parameters?

Below is an explanation and a brief code extract to help you fix the coordinate ranges. In the 2D plan view, our blocks are plotted using their x‐ and z‑coordinates (with y determining the layer). In the current implementation, the limits are computed dynamically for each layer:

```python
        xs = [b[0] for b in all_blocks if b[1] == y_layer]
        zs = [b[2] for b in all_blocks if b[1] == y_layer]
        if xs and zs:
            ax.set_xlim(min(xs)-1, max(xs)+1)
            ax.set_ylim(min(zs)-1, max(zs)+1)
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
```

To fix the x and y (i.e. x and z) axes so that the coordinate system remains constant as you move through layers, you should pre-calculate fixed limits based on *all* of your block data. For example:

```python
# Compute global x and z limits once
all_xs = [b[0] for b in all_blocks]
all_zs = [b[2] for b in all_blocks]
fixed_xlim = (min(all_xs) - 1, max(all_xs) + 1)
fixed_ylim = (min(all_zs) - 1, max(all_zs) + 1)
```

Then, in your `draw_layer` function, you simply set these fixed limits each time you update the layer:

```python
def draw_layer(y_layer):
    ax.clear()
    ax.set_title(f"Layer {y_layer} (Plan View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect('equal')  # Ensure squares are truly square.
    
    # Plot blocks as before...
    for block in framework_blocks:
        if block[1] == y_layer:
            x, _, z = block
            rect = Rectangle((x-0.5, z-0.5), 1, 1, facecolor="gray", edgecolor="black")
            ax.add_patch(rect)
    for block in panel_blocks:
        if block[1] == y_layer:
            x, _, z = block
            rect = Rectangle((x-0.5, z-0.5), 1, 1, facecolor="white", edgecolor="gray", alpha=0.5)
            ax.add_patch(rect)
    
    ax.set_xlim(fixed_xlim)  # Fixed x limits.
    ax.set_ylim(fixed_ylim)   # Fixed z limits.
    fig.canvas.draw_idle()
```

In summary, the "right parameters" are:
- **`ax.set_xlim(fixed_xlim)`** with `fixed_xlim` set as `(min(all_xs)-1, max(all_xs)+1)`
- **`ax.set_ylim(fixed_ylim)`** with `fixed_ylim` set as `(min(all_zs)-1, max(all_zs)+1)`
- Additionally, use **`ax.set_aspect('equal')`** to maintain an equal scale on both axes.

This way, as you slide through the layers, the x and y (plan view) axes remain fixed.

> Perfect. One last thing: I would like to have the background on the matplotlib chart be green squares for easier visual interpretation?

Below is one approach to add a background made of green squares. The idea is to precompute the global x/z limits (i.e. the entire grid that covers all blocks) and then—for every grid cell in that region—draw a square in a “grass‐like” green color. You then draw your blocks on top of this grid. For example, modify your `draw_layer` function (used in the slider callback) as follows:

```python
def visualize_layer_slider(framework_blocks, panel_blocks):
    """
    Display a 2D plan view for each block layer (based on y coordinate) where the background
    is filled with green squares. The blocks are then drawn on top:
      - Framework: gray squares with black edges.
      - Panel: white squares (50% transparent) with gray edges.
    
    A Matplotlib slider lets you move through the layers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.patches import Rectangle

    # Combine all block coordinates.
    all_blocks = list(framework_blocks.union(panel_blocks))
    ys = [b[1] for b in all_blocks]
    min_y = int(min(ys))
    max_y = int(max(ys))

    # Pre-calculate fixed global limits from all blocks:
    all_xs = [b[0] for b in all_blocks]
    all_zs = [b[2] for b in all_blocks]
    fixed_xlim = (min(all_xs) - 1, max(all_xs) + 1)
    fixed_ylim = (min(all_zs) - 1, max(all_zs) + 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    def draw_layer(y_layer):
        ax.clear()
        ax.set_title(f"Layer {y_layer} (Plan View)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect("equal")

        # --- Draw the background grid of green squares ---
        # Iterate over a grid that spans the entire fixed x and z ranges.
        for x in range(int(fixed_xlim[0]), int(fixed_xlim[1]) + 1):
            for z in range(int(fixed_ylim[0]), int(fixed_ylim[1]) + 1):
                # Draw each green square covering one block.
                # Adjust these colors as desired.
                bg = Rectangle((x - 0.5, z - 0.5), 1, 1,
                               facecolor="lightgreen", edgecolor="green", lw=0.5)
                ax.add_patch(bg)

        # --- Draw the framework blocks for the given layer ---
        for block in framework_blocks:
            if block[1] == y_layer:
                x, _, z = block
                rect_fw = Rectangle((x - 0.5, z - 0.5), 1, 1,
                                     facecolor="gray", edgecolor="black", lw=1.5)
                ax.add_patch(rect_fw)

        # --- Draw the panel blocks for the given layer ---
        for block in panel_blocks:
            if block[1] == y_layer:
                x, _, z = block
                rect_panel = Rectangle((x - 0.5, z - 0.5), 1, 1,
                                        facecolor="white", edgecolor="gray", alpha=0.5, lw=1)
                ax.add_patch(rect_panel)

        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
        fig.canvas.draw_idle()

    # Use the middle layer as the initial view.
    initial_layer = (min_y + max_y) // 2
    draw_layer(initial_layer)

    # Add slider for layers.
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Layer', min_y, max_y, valinit=initial_layer, valstep=1)

    def update(val):
        layer = int(slider.val)
        draw_layer(layer)
    slider.on_changed(update)
    plt.show()
```

### Explanation

1. **Fixed Global Limits:**  
   We compute `fixed_xlim` and `fixed_ylim` from all block coordinates so that the plan view always uses the same bounds regardless of the current layer.

2. **Green Background Grid:**  
   Before drawing the blocks, we loop over the x and z ranges (using integer steps based on the fixed limits) and add a `Rectangle` for each grid cell. In this case, each square is centered on integer coordinates with a width and height of 1, filled with a light green color and edged with a darker green.

3. **Layer Drawing:**  
   With the background in place, the blocks (framework and panels) are drawn on top using their own colors and edge settings.

This way, not only do you have a fixed grid for easy spatial reference, but the green squares mimic a tiled “ground” that can enhance the Minecraft-style visualization. Enjoy the extra visual flair in your plan view!
