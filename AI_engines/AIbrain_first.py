import numpy as np
from numpy import random as np_random
import random
import copy
import string

# Number of inputs (raycasting distances)
N_INPUTS = 9
# Number of outputs [accelerate, brake, turn_left, turn_right]
N_ACTIONS = 4
# Hidden layer neurons
N_HIDDEN = 16


class AIbrain_first:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        
        # Car state (updated via passcardata)
        self.x = 0
        self.y = 0
        self.speed = 0

        self.init_param()

    def init_param(self):
        """
        Initialize model parameters with Xavier initialization.
        Network architecture: 9 inputs -> 16 hidden (ReLU) -> 4 outputs (sigmoid)
        """
        # Layer 1: input -> hidden (Xavier initialization)
        # W1 shape: (N_HIDDEN, N_INPUTS) = (16, 9)
        self.W1 = (np_random.rand(N_HIDDEN, N_INPUTS) - 0.5) * np.sqrt(2.0 / N_INPUTS)
        self.b1 = np.zeros(N_HIDDEN)
        
        # Layer 2: hidden -> output (Xavier initialization)
        # W2 shape: (N_ACTIONS, N_HIDDEN) = (4, 16)
        self.W2 = (np_random.rand(N_ACTIONS, N_HIDDEN) - 0.5) * np.sqrt(2.0 / N_HIDDEN)
        self.b2 = np.zeros(N_ACTIONS)
        
        self.NAME = "NewBrain"
        
        # Always call store() at the end!
        self.store()

    def decide(self, data):
        """
        Main decision function - forward pass through neural network.
        
        Args:
            data: List of 9 floats - distances to obstacles in tiles
                  Index 0: -90° (left), Index 4: 0° (forward), Index 8: +90° (right)
                  Low values = obstacle close, High values = clear path
        
        Returns:
            List of 4 values - actions to take
            Values > 0.5 will activate: [accelerate, brake, turn_left, turn_right]
        """
        # Convert input to numpy array
        x = np.asarray(data, dtype=float).ravel()
        
        # Handle input size mismatch
        n_expected = self.W1.shape[1]
        if x.size < n_expected:
            x = np.concatenate([x, np.zeros(n_expected - x.size)])
        elif x.size > n_expected:
            x = x[:n_expected]
        
        # Hidden layer with Leaky ReLU activation (prevents dead neurons)
        z1 = self.W1.dot(x) + self.b1
        hidden = np.where(z1 > 0, z1, 0.01 * z1)  # Leaky ReLU: max(0.01*x, x)
        
        # Output layer with Sigmoid (bounds output to 0-1 for threshold > 0.5)
        z2 = self.W2.dot(hidden) + self.b2
        # Clip to prevent overflow in exp
        output = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
        
        return output

    def mutate(self):
        """
        Apply Gaussian noise mutation to all weights and biases.
        Called during evolutionary training to create variations.
        """
        sigma = 0.15  # Mutation strength
        
        # Add Gaussian noise to all parameters
        self.W1 += np_random.randn(*self.W1.shape) * sigma
        self.b1 += np_random.randn(*self.b1.shape) * sigma
        self.W2 += np_random.randn(*self.W2.shape) * sigma
        self.b2 += np_random.randn(*self.b2.shape) * sigma
        
        # Update name to track mutation history
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        self.store()

    def store(self):
        """
        Store current parameters to self.parameters dict.
        Everything here will be saved to .npz files.
        """
        self.parameters = copy.deepcopy({
            "NAME": self.NAME,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        })

    def set_parameters(self, parameters):
        """
        Load parameters from dict or .npz file.
        """
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict
        
        # Load network weights
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.NAME = str(self.parameters["NAME"])

    def calculate_score(self, distance, time, no):
        """
        Calculate fitness score after a race.
        Rewards distance traveled and efficiency (distance/time).
        
        Args:
            distance: Total distance traveled
            time: Time taken
            no: Not used
        """
        # Avoid division by zero
        if time > 0:
            self.score = distance + (distance / time) * 0.5
        else:
            self.score = distance

    # ==================== DO NOT MODIFY BELOW ====================
    
    def passcardata(self, x, y, speed):
        """Receives car state each frame."""
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        """Returns the current score."""
        return self.score

    def get_parameters(self):
        """Returns a copy of stored parameters."""
        return copy.deepcopy(self.parameters)
