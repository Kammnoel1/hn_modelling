import numpy as np
from evaluation import compute_stored_energy, compute_attractor_energy 
from plotting import plot_energy_values, plot_stored_vs_attractor_energies
from patterns import generate_xyz
from network import HopfieldNetwork 

SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":
    N = 100                                 # Number of neurons in the Hopfield network
    max_p = 5                               # Maximum number of pattern sets to test
    noise = 0.2                             # Noise level in patterns
    max_steps = 10000                       # Maximum update steps for recall
    path_memory = "figs/Figure_1.pdf"       # Output path for the memory plot
    path_retrieval = "figs/Figure_2.pdf"   # Output path for the retrieval plot
    
    # Define conditions to test
    conditions = [True, False]
    combined_stored_energies = {'overlap_true': {}, 'overlap_false': {}}
    combined_attractor_energies = {}
    
    # Loop through overlap conditions
    for overlap in conditions:
        # Generate patterns for current condition
        x_pats, y_pats, z_pats = generate_xyz(max_p, N, overlap=overlap)
        all_pats = np.vstack((x_pats, y_pats, z_pats))
        
        # Create and train network
        net = HopfieldNetwork(size=N)
        net.train(all_pats)
        
        # Compute energies
        stored_energies = compute_stored_energy(x_pats, y_pats, z_pats, net)
        attractor_energies = compute_attractor_energy(x_pats, y_pats, z_pats, net, noise, max_steps)
        
        # Store in combined dictionary
        condition_key = f'overlap_{str(overlap).lower()}'
        combined_stored_energies[condition_key] = stored_energies
        combined_attractor_energies[condition_key] = attractor_energies
    
    plot_energy_values(combined_stored_energies, max_p, path_memory)
    plot_stored_vs_attractor_energies(
        combined_stored_energies,
        combined_attractor_energies,
        max_p,
        path_retrieval,
        column="double"
    )

