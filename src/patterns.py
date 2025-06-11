import numpy as np 

def initialize_binary(size):
    """Generate a random binary pattern with values (+1, -1)."""
    return np.random.choice([1, -1], size=size, p=[0.5, 0.5])

def add_noise(pattern, noise_level):
    """Flip a fraction of bits in the pattern to add noise."""
    noisy = pattern.copy()
    num_flips = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)
    noisy[flip_indices] *= -1 
    return noisy

def generate_xyz(p: int, size: int, overlap: bool):
    x = [initialize_binary(size) for _ in range(p)]
    y = [initialize_binary(size) for _ in range(p)]
    if overlap: 
        z = [np.sign(x + y + (x * y)) for x, y in zip(x, y)]
    else: 
        z = [initialize_binary(size) for _ in range(p)]
    return x, y, z