import numpy as np
from network import HopfieldNetwork, calculate_energy
from patterns import generate_xyz, add_noise

def test_recall(patterns, net, noise, steps):
    for p in patterns:
        noisy = add_noise(p, noise)
        rec, _ = net.recall(noisy, p, steps)
        if not np.array_equal(rec, p):
            return False
    return True

def compute_stored_and_attractor(pattern, net, noise_level, max_steps):
    """
    Compute the stored energy and attractor energy for a given pattern.
    """
    E_stored = calculate_energy(pattern, net.weights)
    noisy_pattern = add_noise(pattern, noise_level)
    recalled_pattern, _ = net.recall(noisy_pattern, pattern, max_steps)
    E_attractor = calculate_energy(recalled_pattern, net.weights)
    return E_stored, E_attractor

def compute_energy_vs_load(N, max_p, overlap):
    """
    Determine the memory capacity of a Hopfield network.
    For each pattern set, compute the success rate, energy values, and overlaps.
    (No trials loop; each value is computed from a single trial.)
    """
    energies = {"x":[],"y":[],"z":[]}

    for p in range(1, max_p + 1):
        x_pats, y_pats, z_pats = generate_xyz(p, N, overlap)
        all_pats = np.vstack(x_pats + y_pats + z_pats)
        net = HopfieldNetwork(size=N)
        net.train(all_pats)
        # Compute average energy values for each pattern type
        energies["x"].append(np.mean([calculate_energy(x, net.weights) for x in x_pats]))
        energies["y"].append(np.mean([calculate_energy(y, net.weights) for y in y_pats]))
        energies["z"].append(np.mean([calculate_energy(z, net.weights) for z in z_pats]))

    return energies