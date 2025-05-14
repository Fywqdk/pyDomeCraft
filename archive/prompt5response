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
