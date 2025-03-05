import numpy as np 
import os 
import matplotlib.pyplot as plt
from network import HopfieldNetwork

def add_noise_to_pattern(pattern, noise_level=0.05):
    """Flip a fraction of bits in the pattern to add noise."""
    noisy_pattern = pattern.copy()
    num_flips = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1 
    return noisy_pattern

def initialize_binary_array(size):
    """Generate a random binary pattern with values (+1, -1)."""
    return np.random.choice([1, -1], size=size, p=[0.5, 0.5])

def generate_patterns(p, network_size): 
    """Generate x, y, and z patterns with the specified transformation."""
    x_patterns = [initialize_binary_array(network_size) for _ in range(p)]
    y_patterns = [initialize_binary_array(network_size) for _ in range(p)]
    z_patterns = [np.sign(x + y + (x * y)) for x, y in zip(x_patterns, y_patterns)]
    # z_patterns = [initialize_binary_array(network_size) for _ in range(p)]
    return x_patterns, y_patterns, z_patterns

def calculate_energy(pattern, weights):
    """Compute the Hopfield network energy for a given pattern."""
    return -0.5 * np.dot(pattern.T, np.dot(weights, pattern))

def test_recall(patterns, net, noise_level, max_steps): 
    """Test whether the Hopfield network successfully recalls stored patterns."""
    for original_pattern in patterns:
        noisy_pattern = add_noise_to_pattern(original_pattern, noise_level)
        recalled_pattern, _ = net.recall(noisy_pattern, original_pattern, max_steps)
        if not np.array_equal(recalled_pattern, original_pattern):
            return False  # Failure in recall
    return True  # All patterns recalled successfully

def compute_average_overlap(x_patterns, y_patterns, z_patterns):
    """Compute the average overlap between x, y, and z patterns."""
    overlap_xy = np.mean([np.sum(x == y) for x, y in zip(x_patterns, y_patterns)])
    overlap_xz = np.mean([np.sum(x == z) for x, z in zip(x_patterns, z_patterns)])
    overlap_yz = np.mean([np.sum(y == z) for y, z in zip(y_patterns, z_patterns)])
    return overlap_xy, overlap_xz, overlap_yz

def determine_memory_capacity(network_size, max_patterns, noise_level, max_steps):
    """
    Determine the memory capacity of a Hopfield network.
    For each pattern set, compute the success rate, energy values, and overlaps.
    (No trials loop; each value is computed from a single trial.)
    """
    success_rates = {}
    energy_dict = {"x": [], "y": [], "z": []}
    overlap_dict = {"xy": [], "xz": [], "yz": []}

    for p in range(1, max_patterns + 1):
        # Generate patterns and compute overlaps for this single trial
        x_patterns, y_patterns, z_patterns = generate_patterns(p, network_size)
        overlap_xy, overlap_xz, overlap_yz = compute_average_overlap(x_patterns, y_patterns, z_patterns)
        overlap_dict["xy"].append(overlap_xy)
        overlap_dict["xz"].append(overlap_xz)
        overlap_dict["yz"].append(overlap_yz)

        # Stack all patterns for training
        all_patterns = np.vstack(x_patterns + y_patterns + z_patterns)

        # Train the Hopfield network
        net = HopfieldNetwork(size=network_size)
        net.train(all_patterns)

        # Plot energies for inscribed vs. attractor patterns
        plot_inscribed_vs_attractor_energies(z_patterns, x_patterns, y_patterns, net, noise_level, max_steps)

        # Compute average energy values for each pattern type
        energy_dict["x"].append(np.mean([calculate_energy(x, net.weights) for x in x_patterns]))
        energy_dict["y"].append(np.mean([calculate_energy(y, net.weights) for y in y_patterns]))
        energy_dict["z"].append(np.mean([calculate_energy(z, net.weights) for z in z_patterns]))

        # Record success rate (1.0 for success, 0.0 for failure)
        success_rates[p] = 1.0 if test_recall(z_patterns, net, noise_level, max_steps) else 0.0

    return success_rates, energy_dict, overlap_dict

def plot_memory_capacity(success_rates, path="plots/memory_capacity.png"): 
    """Plot and save the memory capacity success rates."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(success_rates.keys(), success_rates.values(), marker='o')
    plt.xlabel('Number of Pattern Sets (p)')
    plt.ylabel('Success Rate')
    plt.title('Memory Capacity for Patterns')
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_energy_values(energy_dict, max_patterns, path="plots/energy_values.png"):
    """Plot and save the energy values for different pattern types as discrete crosses."""
    plt.figure(figsize=(10, 6))
    colors = {"x": "blue", "y": "orange", "z": "red"}
    for label, y_values in energy_dict.items():
        pattern_indices = list(range(1, max_patterns + 1))
        plt.scatter(pattern_indices, y_values, label=label, marker='x', color=colors.get(label, "black"))
    plt.xlabel("Number of Pattern Sets (p)")
    plt.ylabel("Energy Values")
    plt.title("Energy Values for Patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_overlaps(overlap_dict, max_patterns, path="plots/overlaps.png"): 
    """Plot and save the overlaps between patterns."""
    plt.figure(figsize=(10, 6))
    for label, y_values in overlap_dict.items():
        plt.plot(range(1, max_patterns + 1), y_values, label=label)
    plt.xlabel("Number of Pattern Sets (p)")
    plt.ylabel("Overlap")
    plt.title("Overlap between patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def process_pattern(pattern, net, noise_level, max_steps):
    """
    Compute the stored energy and attractor energy for a given pattern.
    """
    E_stored = calculate_energy(pattern, net.weights)
    noisy_pattern = add_noise_to_pattern(pattern, noise_level)
    recalled_pattern, _ = net.recall(noisy_pattern, pattern, max_steps)
    E_attractor = calculate_energy(recalled_pattern, net.weights)
    return E_stored, E_attractor

def plot_inscribed_vs_attractor_energies(z_patterns, x_patterns, y_patterns, net, noise_level, max_steps, 
                                           path="plots/energy_component_overlap_2.png", x_max=6, y_min=-80, y_max=-35):
    """
    For each pattern set, where each set consists of 3 patterns (X, Y, Z):
      - Compute the energy of the stored (inscribed) pattern.
      - Add noise and recall it, then compute the energy of the attractor.
      - Plot stored and attractor energies for all three pattern types on a global index.
      
    Global indexing:
      For each set i (starting from 0), assign:
         X-pattern: index = 3*i + 1
         Y-pattern: index = 3*i + 2
         Z-pattern: index = 3*i + 3
         
    The x-axis will always span from 1 to x_max (default=6) and the y-axis is fixed from y_min to y_max.
    """
    p = len(z_patterns)  # number of sets
    # Lists for energies
    stored_energies = {"x": [], "y": [], "z": []}
    attractor_energies = {"x": [], "y": [], "z": []}

    # Global indices for each pattern type
    indices = {"x": [3*i + 1 for i in range(p)],
               "y": [3*i + 2 for i in range(p)],
               "z": [3*i + 3 for i in range(p)]}

    # Compute energies for each pattern in each set
    for i in range(p):
        E_stored_x, E_attractor_x = process_pattern(x_patterns[i], net, noise_level, max_steps)
        E_stored_y, E_attractor_y = process_pattern(y_patterns[i], net, noise_level, max_steps)
        E_stored_z, E_attractor_z = process_pattern(z_patterns[i], net, noise_level, max_steps)

        stored_energies["x"].append(E_stored_x)
        stored_energies["y"].append(E_stored_y)
        stored_energies["z"].append(E_stored_z)

        attractor_energies["x"].append(E_attractor_x)
        attractor_energies["y"].append(E_attractor_y)
        attractor_energies["z"].append(E_attractor_z)

    plt.figure(figsize=(10, 6))
    # Plot stored energies (dots)
    plt.scatter(indices["x"], stored_energies["x"], color='blue', marker='o', label='Stored X-Patterns')
    plt.scatter(indices["y"], stored_energies["y"], color='orange', marker='o', label='Stored Y-Patterns')
    plt.scatter(indices["z"], stored_energies["z"], color='red', marker='o', label='Stored Z-Patterns')

    # Plot attractor energies (crosses)
    plt.scatter(indices["x"], attractor_energies["x"], color='blue', marker='x', s=100, label='Attractor X-Patterns')
    plt.scatter(indices["y"], attractor_energies["y"], color='orange', marker='x', s=100, label='Attractor Y-Patterns')
    plt.scatter(indices["z"], attractor_energies["z"], color='red', marker='x', s=100, label='Attractor Z-Patterns')

    plt.xlabel("Global Pattern Index")
    plt.ylabel("Energy")
    plt.title("Stored vs. Attractor Energies for Componenet Overlap Patterns")
    plt.legend()
    plt.grid(True)

    # Set consistent axes across all plots
    plt.xlim(0.5, x_max + 0.5)
    plt.ylim(y_min, y_max)
    
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
network_size = 100   # Number of neurons in the Hopfield network
max_patterns = 5     # Maximum number of pattern sets to test
noise_level = 0.2    # Noise level in patterns
max_steps = 10000    # Maximum update steps for recall

success_rates, energy_dict, overlap_dict = determine_memory_capacity(network_size, max_patterns, noise_level, max_steps)
plot_memory_capacity(success_rates)
plot_energy_values(energy_dict, max_patterns)
plot_overlaps(overlap_dict, max_patterns)