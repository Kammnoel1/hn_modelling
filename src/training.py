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
    p = len(x_patterns)
    overlap_xy = np.mean([np.sum(x == y) for x, y in zip(x_patterns, y_patterns)])
    overlap_xz = np.mean([np.sum(x == z) for x, z in zip(x_patterns, z_patterns)])
    overlap_yz = np.mean([np.sum(y == z) for y, z in zip(y_patterns, z_patterns)])
    return overlap_xy, overlap_xz, overlap_yz

def determine_memory_capacity(network_size, max_patterns, trials, noise_level, max_steps):
    """
    Determine the memory capacity of a Hopfield network and record success rates and energy values.
    """
    success_rates = {}
    energy_dict = {"x": [], "y": [], "z": []}
    overlap_dict = {"xy": [], "xz": [], "yz": []}

    for p in range(1, max_patterns + 1):
        success_count = 0
        total_energy_x = total_energy_y = total_energy_z = 0
        total_overlap_xy = total_overlap_xz = total_overlap_yz = 0

        for _ in range(trials):
            x_patterns, y_patterns, z_patterns = generate_patterns(p, network_size)
            overlap_xy, overlap_xz, overlap_yz = compute_average_overlap(x_patterns, y_patterns, z_patterns)

            total_overlap_xy += overlap_xy
            total_overlap_xz += overlap_xz
            total_overlap_yz += overlap_yz

            # Stack all patterns for training
            all_patterns = np.vstack(x_patterns + y_patterns + z_patterns)

            # Train the Hopfield network
            net = HopfieldNetwork(size=network_size)
            net.train(all_patterns)

            # Plot the energies of inscribed vs. attractor for z-patterns (just one example)
            plot_inscribed_vs_attractor_energies(z_patterns, x_patterns, y_patterns, net, noise_level, max_steps)


            # Accumulate energy values
            total_energy_x += sum(calculate_energy(x, net.weights) for x in x_patterns)
            total_energy_y += sum(calculate_energy(y, net.weights) for y in y_patterns)
            total_energy_z += sum(calculate_energy(z, net.weights) for z in z_patterns)

            if test_recall(z_patterns, net, noise_level, max_steps):
                success_count += 1

        # Compute averages
        success_rates[p] = success_count / trials
        num_patterns_total = 3 * p * trials
        energy_dict["x"].append(total_energy_x / num_patterns_total)
        energy_dict["y"].append(total_energy_y / num_patterns_total)
        energy_dict["z"].append(total_energy_z / num_patterns_total)
        overlap_dict["xy"].append(total_overlap_xy / trials)
        overlap_dict["xz"].append(total_overlap_xz / trials)
        overlap_dict["yz"].append(total_overlap_yz / trials)


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
    """Plot and save the energy values for different pattern types."""
    plt.figure(figsize=(10, 6))
    for label, y_values in energy_dict.items():
        plt.plot(range(1, max_patterns + 1), y_values, label=label)
    plt.xlabel("Number of Pattern Sets (p)")
    plt.ylabel("Energy Values")
    plt.title("Energy values for patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_overlaps(overlap_dict, max_patterns, path="plots/overlaps.png"): 
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

def plot_inscribed_vs_attractor_energies(z_patterns, x_patterns, y_patterns, net, noise_level, max_steps, path="plots/energy_per_pattern.png"):
    """
    For each pattern triplet (Z, X, Y):
      - Compute the energy of the stored (inscribed) pattern (blue dot for each type).
      - Add noise and recall it, then compute the energy of the attractor (red cross for each type).
      - Plot all energies on the same x-axis index with no lines connecting them.
    """
    # Initialize lists for stored and attractor energies for each pattern type
    energies_stored_z = []
    energies_attractor_z = []
    energies_stored_x = []
    energies_attractor_x = []
    energies_stored_y = []
    energies_attractor_y = []

    # Discrete x-axis indices for each pattern triplet
    pattern_indices = range(len(z_patterns))

    for i in pattern_indices:
        # Get each pattern from the triplet
        stored_z = z_patterns[i]
        stored_x = x_patterns[i]
        stored_y = y_patterns[i]

        # Compute energy of the inscribed (stored) patterns
        E_stored_z = calculate_energy(stored_z, net.weights)
        E_stored_x = calculate_energy(stored_x, net.weights)
        E_stored_y = calculate_energy(stored_y, net.weights)

        # Add noise and recall each pattern
        noisy_z = add_noise_to_pattern(stored_z, noise_level)
        recalled_z, _ = net.recall(noisy_z, stored_z, max_steps)
        noisy_x = add_noise_to_pattern(stored_x, noise_level)
        recalled_x, _ = net.recall(noisy_x, stored_x, max_steps)
        noisy_y = add_noise_to_pattern(stored_y, noise_level)
        recalled_y, _ = net.recall(noisy_y, stored_y, max_steps)

        # Compute energy of the attractors
        E_attractor_z = calculate_energy(recalled_z, net.weights)
        E_attractor_x = calculate_energy(recalled_x, net.weights)
        E_attractor_y = calculate_energy(recalled_y, net.weights)

        # Append the energies to the lists
        energies_stored_z.append(E_stored_z)
        energies_attractor_z.append(E_attractor_z)
        energies_stored_x.append(E_stored_x)
        energies_attractor_x.append(E_attractor_x)
        energies_stored_y.append(E_stored_y)
        energies_attractor_y.append(E_attractor_y)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot stored (inscribed) energies for each pattern type (blue dots)
    plt.scatter(pattern_indices, energies_stored_z, color='blue', marker='o', label='Stored Z-Patterns')
    plt.scatter(pattern_indices, energies_stored_x, color='cyan', marker='o', label='Stored X-Patterns')
    plt.scatter(pattern_indices, energies_stored_y, color='magenta', marker='o', label='Stored Y-Patterns')

    # Plot attractor energies for each pattern type (red crosses)
    plt.scatter(pattern_indices, energies_attractor_z, color='red', marker='x', label='Attractor Z-Patterns')
    plt.scatter(pattern_indices, energies_attractor_x, color='orange', marker='x', label='Attractor X-Patterns')
    plt.scatter(pattern_indices, energies_attractor_y, color='green', marker='x', label='Attractor Y-Patterns')

    plt.xlabel("Pattern Index")
    plt.ylabel("Energy")
    plt.title("Stored vs. Attractor Energies for Z-, X-, and Y-Patterns")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# Example usage
network_size = 100  # Number of neurons in the Hopfield network
max_patterns = 5     # Maximum number of pattern sets to test
trials = 1          # Number of trials per pattern set
noise_level = 0.2   # Noise level in patterns
max_steps = 10000   # Max update steps for recall

success_rates, energy_dict, overlap_dict = determine_memory_capacity(network_size, max_patterns, trials, noise_level, max_steps)
plot_memory_capacity(success_rates)
plot_energy_values(energy_dict, max_patterns)
plot_overlaps(overlap_dict, max_patterns)