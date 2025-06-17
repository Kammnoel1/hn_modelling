import matplotlib.pyplot as plt


def plot_energy_values(
    energy_dict: dict[str, dict[str, list[float]]],
    max_patterns: int,
    path: str,
    column: str = "double",
):
    mm_to_in = 0.03937
    width_mm = 85 if column == "single" else 180
    width_in = width_mm * mm_to_in
    height_in = width_in * 0.7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in, height_in))

    # Define different markers for each pattern
    markers = ["o", "s", "^"]
    x_range = range(1, max_patterns + 1)

    # Plot overlap=True condition
    overlap_true_energies = energy_dict['overlap_true']
    for i, (key, y_vals) in enumerate(overlap_true_energies.items()):
        marker = markers[i % len(markers)]
        ax1.plot(x_range, y_vals, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="none")
    ax1.set_xlabel("Pattern Set Index", fontsize=9)
    ax1.set_ylabel("Energy", fontsize=9)
    ax1.set_xticks(range(1, max_patterns + 1))
    ax1.set_ylim(-90, -40)
    ax1.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax1.grid(True, linewidth=0.5, alpha=0.4)

    # Plot overlap=False condition
    overlap_false_energies = energy_dict['overlap_false']
    for i, (key, y_vals) in enumerate(overlap_false_energies.items()):
        marker = markers[i % len(markers)]
        ax2.plot(x_range, y_vals, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="none")
    ax2.set_xlabel("Pattern Set Index", fontsize=9)
    ax2.set_ylabel("Energy", fontsize=9)
    ax2.set_xticks(range(1, max_patterns + 1))
    ax2.set_ylim(-90, -40)
    ax2.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax2.grid(True, linewidth=0.5, alpha=0.4)

    fig.tight_layout()

    raster_ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    dpi = 300 if path.lower().endswith(raster_ext) else None
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_stored_vs_attractor_energies(
    stored_energy_dict: dict[str, dict[str, list[float]]],
    attractor_energy_dict: dict[str, dict[str, list[float]]],
    max_patterns: int,
    path: str,
    column: str = "double",
):
    """
    Publication-ready scatter plot: Stored vs Attractor energies.
    
    Parameters
    ----------
    stored_energy_dict : dict
        Dictionary with overlap conditions as keys, each containing pattern energies.
    attractor_energy_dict : dict
        Dictionary with overlap conditions as keys, each containing attractor energies.
    max_patterns : int
        Maximum number of pattern sets.
    path : str
        Output file path (pdf preferred; tiff/jpg for raster).
    column : {'single', 'double'}
        Target figure width for journal specs.
    """
    # ------ figure geometry ---------
    mm_to_in = 0.03937
    width_mm = 85 if column == "single" else 180
    width_in = width_mm * mm_to_in
    height_in = width_in * 0.7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in, height_in))

    # Define different markers for each pattern type
    markers = ["o", "s", "^"]  # circle, square, triangle for x, y, z patterns
    pattern_keys = ['x', 'y', 'z']
    
    # Total number of individual patterns = 3 * max_patterns
    total_patterns = 3 * max_patterns

    # Plot overlap=True condition
    overlap_true_stored = stored_energy_dict['overlap_true']
    overlap_true_attractor = attractor_energy_dict['overlap_true']
    
    # Create x-axis positions for all individual patterns in correct order
    pattern_indices = []
    stored_energies_flat = []
    attractor_energies_flat = []
    pattern_types = []
    
    # Order: x1, y1, z1, x2, y2, z2, x3, y3, z3, ...
    for j in range(max_patterns):  # For each pattern set
        for i, key in enumerate(pattern_keys):  # For each pattern type (x, y, z)
            pattern_index = j * 3 + i + 1  # Pattern index from 1 to total_patterns
            pattern_indices.append(pattern_index)
            stored_energies_flat.append(overlap_true_stored[key][j])
            attractor_energies_flat.append(overlap_true_attractor[key][j])
            pattern_types.append(i)  # 0, 1, 2 for x, y, z
    
    # Plot stored patterns: hollow markers (circumference only)
    for i, (pattern_idx, stored_energy, pattern_type) in enumerate(zip(pattern_indices, stored_energies_flat, pattern_types)):
        marker = markers[pattern_type]
        ax1.plot(pattern_idx, stored_energy, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="none")
    
    # Plot attractor patterns: solid markers
    for i, (pattern_idx, attractor_energy, pattern_type) in enumerate(zip(pattern_indices, attractor_energies_flat, pattern_types)):
        marker = markers[pattern_type]
        ax1.plot(pattern_idx, attractor_energy, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="black")
    
    ax1.set_xlabel("Pattern Index", fontsize=9)
    ax1.set_ylabel("Energy", fontsize=9)
    ax1.set_xticks(range(1, total_patterns + 1))
    ax1.set_ylim(-90, -40)
    ax1.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax1.grid(True, linewidth=0.5, alpha=0.4)

    # Plot overlap=False condition
    overlap_false_stored = stored_energy_dict['overlap_false']
    overlap_false_attractor = attractor_energy_dict['overlap_false']
    
    # Reset for overlap=False condition
    pattern_indices = []
    stored_energies_flat = []
    attractor_energies_flat = []
    pattern_types = []
    
    # Same ordering: x1, y1, z1, x2, y2, z2, x3, y3, z3, ...
    for j in range(max_patterns):  # For each pattern set
        for i, key in enumerate(pattern_keys):  # For each pattern type (x, y, z)
            pattern_index = j * 3 + i + 1
            pattern_indices.append(pattern_index)
            stored_energies_flat.append(overlap_false_stored[key][j])
            attractor_energies_flat.append(overlap_false_attractor[key][j])
            pattern_types.append(i)
    
    # Plot stored patterns: hollow markers (circumference only)
    for i, (pattern_idx, stored_energy, pattern_type) in enumerate(zip(pattern_indices, stored_energies_flat, pattern_types)):
        marker = markers[pattern_type]
        ax2.plot(pattern_idx, stored_energy, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="none")
    
    # Plot attractor patterns: solid markers
    for i, (pattern_idx, attractor_energy, pattern_type) in enumerate(zip(pattern_indices, attractor_energies_flat, pattern_types)):
        marker = markers[pattern_type]
        ax2.plot(pattern_idx, attractor_energy, marker=marker, linestyle="none", markersize=6, 
                markeredgewidth=1.5, color="black", markerfacecolor="black")
    
    ax2.set_xlabel("Pattern Index", fontsize=9)
    ax2.set_ylabel("Energy", fontsize=9)
    ax2.set_xticks(range(1, total_patterns + 1))
    ax2.set_ylim(-90, -40)
    ax2.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax2.grid(True, linewidth=0.5, alpha=0.4)

    fig.tight_layout()

    # ------ save ---------
    raster_ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    dpi = 300 if path.lower().endswith(raster_ext) else None
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)