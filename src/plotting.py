import matplotlib.pyplot as plt
from evaluation import compute_stored_and_attractor


def plot_energy_values(
    energy_dict: dict[str, list[float]],
    max_patterns: int,
    path: str,
    column: str = "double",
):
    mm_to_in = 0.03937
    width_mm = 85 if column == "single" else 180
    width_in = width_mm * mm_to_in
    height_in = width_in * 0.7
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Define mapping from internal keys to display names & colors
    label_map = {
        "x": ("Verbal Encoding", "blue"),
        "y": ("Gesture Encoding", "orange"),
        "z": ("Gesture-Enhanced Encoding", "red"),
    }

    marker_kw = dict(marker="x", linestyle="none", markersize=6, markeredgewidth=1.5)

    x_range = range(1, max_patterns + 1)
    for key, y_vals in energy_dict.items():
        label, color = label_map.get(key, (key, "black"))
        ax.plot(x_range, y_vals, label=label, color=color, **marker_kw)

    ax.set_xticks(range(1, max_patterns + 1))
    ax.set_xlabel("Number of pattern sets $p$", fontsize=9)
    ax.set_ylabel("Energy", fontsize=9)
    ax.set_title("Energy across encoding conditions", fontsize=10, pad=6)
    ax.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    raster_ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    dpi = 300 if path.lower().endswith(raster_ext) else None
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_inscribed_vs_attractor_energies(
    z_patterns, x_patterns, y_patterns, net, noise_level, max_steps,
    path="plots/Figure_2a.pdf",
    x_max=6, y_min=-100, y_max=-35,
    column="double"  # 'single' or 'double'
):
    """
    Publication-ready scatter plot: Stored vs Attractor energies.
    
    Parameters
    ----------
    z_patterns, x_patterns, y_patterns : list of patterns
        Each list contains the patterns per set.
    net : network object
        Must have a process_pattern function defined.
    noise_level : float
        Noise level for input perturbation.
    max_steps : int
        Max steps for attractor dynamics.
    path : str
        Output file path (pdf preferred; tiff/jpg for raster).
    x_max, y_min, y_max : float
        Axis limits.
    column : {'single', 'double'}
        Target figure width for journal specs.
    """
    # ------ figure geometry ---------
    mm_to_in = 0.03937
    width_mm = 85 if column == "single" else 180
    width_in  = width_mm * mm_to_in
    height_in = width_in * 0.7
    fig, ax   = plt.subplots(figsize=(width_in, height_in))

    p = len(z_patterns)
    stored_energies    = {"Verbal": [], "Gesture": [], "Gesture-Enhanced": []}
    attractor_energies = {"Verbal": [], "Gesture": [], "Gesture-Enhanced": []}
    indices = {
        "Verbal": [3 * i + 1 for i in range(p)],
        "Gesture": [3 * i + 2 for i in range(p)],
        "Gesture-Enhanced": [3 * i + 3 for i in range(p)]
    }

    # ------ compute energies ---------
    for i in range(p):
        E_stored_x, E_attractor_x = compute_stored_and_attractor(x_patterns[i], net, noise_level, max_steps)
        E_stored_y, E_attractor_y = compute_stored_and_attractor(y_patterns[i], net, noise_level, max_steps)
        E_stored_z, E_attractor_z = compute_stored_and_attractor(z_patterns[i], net, noise_level, max_steps)

        stored_energies["Verbal"].append(E_stored_x)
        stored_energies["Gesture"].append(E_stored_y)
        stored_energies["Gesture-Enhanced"].append(E_stored_z)

        attractor_energies["Verbal"].append(E_attractor_x)
        attractor_energies["Gesture"].append(E_attractor_y)
        attractor_energies["Gesture-Enhanced"].append(E_attractor_z)

    # ------ plot styles ---------
    palette = {
        "Verbal": "blue",
        "Gesture": "orange",
        "Gesture-Enhanced": "red"
    }

    # Stored: solid circles
    for key, label in zip(["Verbal", "Gesture", "Gesture-Enhanced"], ["X", "Y", "Z"]):
        ax.plot(
            indices[key], stored_energies[key],
            linestyle="none", marker="o", markersize=5, markeredgewidth=0.8,
            color=palette[key], label=f"Stored – {label}"
        )

    # Attractor: crosses
    for key, label in zip(["Verbal", "Gesture", "Gesture-Enhanced"], ["X", "Y", "Z"]):
        ax.plot(
            indices[key], attractor_energies[key],
            linestyle="none", marker="x", markersize=6, markeredgewidth=1.2,
            color=palette[key], label=f"Attractor – {label}"
        )

    # ------ axis & labels ---------
    ax.set_xlabel("Pattern Index", fontsize=9)
    ax.set_ylabel("Energy", fontsize=9)
    ax.set_title("Stored vs. Attractor Energies", fontsize=10, pad=6)

    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax.grid(True, linewidth=0.5, alpha=0.4)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=False)

    fig.tight_layout()

    # ------ save ---------
    raster_ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    dpi = 300 if path.lower().endswith(raster_ext) else None
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)