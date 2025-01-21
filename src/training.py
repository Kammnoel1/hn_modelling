import numpy as np 
import os 
from network import HopfieldNetwork
import matplotlib.pyplot as plt


def add_noise_to_pattern(pattern, noise_level=0.05):
    """
    Add noise to a pattern by flipping a percentage of bits.
    :param pattern: The original binary pattern (+1, -1).
    :param noise_level: Fraction of bits to flip (0 to 1).
    :return: Noised pattern.
    """
    noisy_pattern = pattern.copy()
    num_flips = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1 
    return noisy_pattern

def create_plot(pattern, name): 
    # Reshape patterns to 12x12 for visualization
    pattern_reshaped = pattern.reshape(2, 12)
    # Create the plot
    plt.figure(figsize=(6, 2))
    plt.imshow(pattern_reshaped, cmap='gray', vmin=-1, vmax=1, aspect='equal')
    plt.title("Pattern")
    plt.axis('off')

    # Save the plot to a folder
    output_folder = "./plots"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, name if name.endswith(".png") else f"{name}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

def initialize_binary_array(size=12):
    """
    Initialize a NumPy array of given size with binary values (1, -1),
    each having a probability of 0.5.
    :param size: Size of the array (default: 12).
    :return: A NumPy array of shape (size,) with values 1 and -1.
    """
    return np.random.choice([1, -1], size=size, p=[0.5, 0.5])


def determine_memory_capacity(network_size, max_patterns, trials, noise_level, max_steps):
    """
    Determine the memory capacity of a Hopfield network.
    :param network_size: Number of neurons in the Hopfield network.
    :param max_patterns: Maximum number of pattern sets (p) to test.
    :param trials: Number of trials for each value of p.
    :param noise_level: Noise level to apply to patterns.
    :param max_steps: Maximum number of steps for recall convergence.
    :return: Dictionary mapping p to success rates.
    """
    success_rates = {}

    for p in range(0, max_patterns):
        success_count = 0

        for _ in range(trials):
            # Generate p sets of patterns
            x_patterns = [initialize_binary_array(network_size) for _ in range(p+1)]
            y_patterns = [initialize_binary_array(network_size) for _ in range(p+1)]
            z_patterns = [np.sign(x + y + 1 * (x * y)) for x, y in zip(x_patterns, y_patterns)]
            
            # vertically stack all patterns 
            all_patterns = np.vstack(x_patterns + y_patterns + z_patterns)  

            # Train the Hopfield network
            net = HopfieldNetwork(size=network_size)
            net.train(all_patterns)

            # Test recall for each pattern
            recall_success = True
            for original_pattern in z_patterns:
                noisy_pattern = add_noise_to_pattern(original_pattern, noise_level)
                recalled_pattern, _ = net.recall(noisy_pattern, original_pattern, max_steps)

                if not np.array_equal(recalled_pattern, original_pattern):
                    recall_success = False
                    break  

            if recall_success:
                success_count += 1

        success_rates[p] = success_count / trials

    return success_rates


# Example usage
network_size = 100  # Size of the Hopfield network
max_patterns = 14   # Maximum number of pattern sets to test
trials = 10         # Number of trials for each p
noise_level = 0.2   # Noise level in patterns
max_steps = 10000   # Maximum number of update steps of the network

success_rates = determine_memory_capacity(network_size, max_patterns, trials, noise_level, max_steps)

print(success_rates)




