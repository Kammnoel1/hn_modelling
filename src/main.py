import numpy as np
from evaluation import compute_energy_vs_load 
from plotting import plot_energy_values, plot_inscribed_vs_attractor_energies

SEED = 41
np.random.seed(SEED)

if __name__ == "__main__":
    N = 100                         # Number of neurons in the Hopfield network
    max_p = 5                       # Maximum number of pattern sets to test
    noise = 0.2                     # Noise level in patterns
    max_steps = 10000               # Maximum update steps for recall
    overlap = True                  # Whether to generate overlapping patterns
    path = "plots/Figure_1.pdf"     # Output path for the plot
    
    energies = compute_energy_vs_load(N, max_p, overlap)
    plot_energy_values(energies, max_p, path)
