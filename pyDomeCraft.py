import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

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
