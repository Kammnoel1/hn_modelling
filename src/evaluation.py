import numpy as np
from network import calculate_energy
from patterns import generate_xyz, add_noise

def test_recall(patterns, net, noise, steps):
    for p in patterns:
        noisy = add_noise(p, noise)
        rec, _ = net.recall(noisy, p, steps)
        if not np.array_equal(rec, p):
            return False
    return True

def compute_attractor_energy(x_pats, y_pats, z_pats, net, noise_level, max_steps):
    """
    Compute the attractor energy for given pattern sets.
    """
    energies = {
        "x": [],
        "y": [],
        "z": []
    }
    
    for key, patterns in [("x", x_pats), ("y", y_pats), ("z", z_pats)]:
        for pattern in patterns:
            noisy_pattern = add_noise(pattern, noise_level)
            recalled_pattern, _ = net.recall(noisy_pattern, pattern, max_steps)
            E_attractor = calculate_energy(recalled_pattern, net.weights)
            energies[key].append(E_attractor)
    
    return energies

def compute_stored_energy(x_pats, y_pats, z_pats, net):
    """
    Determine the memory capacity of a Hopfield network.
    For each pattern set, compute the success rate, energy values, and overlaps.
    (No trials loop; each value is computed from a single trial.)
    """
    energies = {
        "x": [calculate_energy(x, net.weights) for x in x_pats],
        "y": [calculate_energy(y, net.weights) for y in y_pats],
        "z": [calculate_energy(z, net.weights) for z in z_pats]
    }
    
    return energies