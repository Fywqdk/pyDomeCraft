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
