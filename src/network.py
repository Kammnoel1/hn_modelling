import numpy as np 

class HopfieldNetwork: 
    def __init__(self, size):
        """
        Initiliaze Hopfield Network. 
        :param size: number of neurons. 
        """
        self.size = size 
        self.weights = np.zeros((size, size)) 
    
    def train(self, patterns):
        """
        Trains network according to the Hebbian Learning rule.
        :param patterns: array of binary patterns (+1, -1) with shape (num_patterns, pattern_size).
        """
        num_patterns, pattern_size = patterns.shape

        # Compute weights using matrix multiplication for all patterns
        self.weights = np.dot(patterns.T, patterns) / num_patterns

        # Zero out the diagonal to avoid self-connections
        np.fill_diagonal(self.weights, 0)


    def recall(self, pattern, stored_pattern, steps):
        """
        Recall a pattern from the network asynchronously and count updates.
        :param pattern: Noisy input binary pattern (+1, -1).
        :param stored_patterns: List of stored binary patterns (+1, -1).
        :param steps: Maximum number of update steps.
        :return: (output pattern after recall, number of updates).
        """
        output = pattern.copy()  # Initialize output with the noisy pattern
        for step in range(steps):
            # Choose a random neuron to update
            i = np.random.randint(0, self.size)
            activation = np.dot(self.weights[i, :], output)
            output[i] = 1 if activation >= 0 else -1

            # Check if the output matches any stored pattern
            if np.array_equal(output, stored_pattern):
                return output, step + 1

        # Return the result if max steps are reached
        return output, steps




