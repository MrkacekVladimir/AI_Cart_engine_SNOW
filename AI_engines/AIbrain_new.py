import numpy as np
from numpy import random as np_random
import random
import copy
import string

# Number of inputs (9 raycasting distances + 1 normalized speed)
N_INPUTS = 10
# Number of internal outputs (before conversion to 4 actions)
# Output 0: speed control (>0.5 = accelerate, <=0.5 = brake)
# Output 1: steering control (>0.5 = turn right, <=0.5 = turn left)
N_OUTPUTS = 2
# Hidden layer neurons
N_HIDDEN1 = 32  # First hidden layer
N_HIDDEN2 = 16  # Second hidden layer
# Max speed constant for normalization
MAX_SPEED = 500.0


class AIbrain_new:
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
        Network architecture: 10 inputs -> 32 hidden1 (ReLU) -> 16 hidden2 (ReLU) -> 2 outputs (sigmoid)
        
        Inputs: 9 ray distances + 1 normalized speed (0-1)
        Outputs:
        - Output 0: speed control (accelerate vs brake)
        - Output 1: steering control (turn left vs turn right)
        """
        # Generation counter for adaptive mutation
        self.generation = 0
        
        # Layer 1: input -> hidden1 (Xavier initialization)
        # W1 shape: (N_HIDDEN1, N_INPUTS) = (32, 10)
        self.W1 = (np_random.rand(N_HIDDEN1, N_INPUTS) - 0.5) * np.sqrt(2.0 / N_INPUTS)
        self.b1 = np.zeros(N_HIDDEN1)
        
        # Layer 2: hidden1 -> hidden2 (Xavier initialization)
        # W2 shape: (N_HIDDEN2, N_HIDDEN1) = (16, 32)
        self.W2 = (np_random.rand(N_HIDDEN2, N_HIDDEN1) - 0.5) * np.sqrt(2.0 / N_HIDDEN1)
        self.b2 = np.zeros(N_HIDDEN2)
        
        # Layer 3: hidden2 -> output (Xavier initialization)
        # W3 shape: (N_OUTPUTS, N_HIDDEN2) = (2, 16)
        self.W3 = (np_random.rand(N_OUTPUTS, N_HIDDEN2) - 0.5) * np.sqrt(2.0 / N_HIDDEN2)
        self.b3 = np.zeros(N_OUTPUTS)
        
        self.NAME = "DeepBrain"
        
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
        
        Control scheme (3-state for each axis):
            Speed (output 0):   < 0.33 = brake, 0.33-0.66 = cruise, > 0.66 = accelerate
            Steering (output 1): < 0.33 = left, 0.33-0.66 = straight, > 0.66 = right
        """
        # Convert ray input to numpy array
        rays = np.asarray(data, dtype=float).ravel()
        
        # Add normalized speed as 10th input (0-1 range)
        normalized_speed = self.speed / MAX_SPEED
        x = np.concatenate([rays, [normalized_speed]])
        
        # Handle input size mismatch (safety check)
        n_expected = self.W1.shape[1]
        if x.size < n_expected:
            x = np.concatenate([x, np.zeros(n_expected - x.size)])
        elif x.size > n_expected:
            x = x[:n_expected]
        
        # Hidden layer 1 with Leaky ReLU activation (prevents dead neurons)
        z1 = self.W1.dot(x) + self.b1
        hidden1 = np.where(z1 > 0, z1, 0.01 * z1)  # Leaky ReLU
        
        # Hidden layer 2 with Leaky ReLU activation
        z2 = self.W2.dot(hidden1) + self.b2
        hidden2 = np.where(z2 > 0, z2, 0.01 * z2)  # Leaky ReLU
        
        # Output layer with Sigmoid (bounds output to 0-1)
        # 2 outputs: [speed_control, steering_control]
        z3 = self.W3.dot(hidden2) + self.b3
        raw_output = 1.0 / (1.0 + np.exp(-np.clip(z3, -500, 500)))
        
        speed_ctrl = raw_output[0]
        steer_ctrl = raw_output[1]
        
        # 3-state speed control: brake / cruise / accelerate
        if speed_ctrl < 0.33:
            accelerate, brake = 0.0, 1.0  # Brake
        elif speed_ctrl > 0.66:
            accelerate, brake = 1.0, 0.0  # Accelerate
        else:
            accelerate, brake = 0.0, 0.0  # Cruise (coast)
        
        # 3-state steering: left / straight / right
        if steer_ctrl < 0.33:
            turn_left, turn_right = 1.0, 0.0  # Turn left
        elif steer_ctrl > 0.66:
            turn_left, turn_right = 0.0, 1.0  # Turn right
        else:
            turn_left, turn_right = 0.0, 0.0  # Go straight
        
        return [accelerate, brake, turn_left, turn_right]

    def mutate(self, generation=None):
        """
        Apply Gaussian noise mutation to all weights and biases.
        Called during evolutionary training to create variations.
        
        Uses adaptive mutation rate: starts high (0.3) for exploration,
        decays to low (0.05) for fine-tuning as generations progress.
        
        Args:
            generation: Current generation number (optional, uses internal counter if not provided)
        """
        # Update generation counter
        if generation is not None:
            self.generation = generation
        else:
            self.generation += 1
        
        # Adaptive mutation rate: decay from 0.3 to 0.05 over ~100 generations
        # Formula: sigma = sigma_max * decay^generation + sigma_min
        sigma_max = 0.25
        sigma_min = 0.05
        decay = 0.97  # ~50% reduction every 23 generations
        sigma = sigma_max * (decay ** self.generation) + sigma_min
        
        # Add Gaussian noise to all parameters
        self.W1 += np_random.randn(*self.W1.shape) * sigma
        self.b1 += np_random.randn(*self.b1.shape) * sigma
        self.W2 += np_random.randn(*self.W2.shape) * sigma
        self.b2 += np_random.randn(*self.b2.shape) * sigma
        self.W3 += np_random.randn(*self.W3.shape) * sigma
        self.b3 += np_random.randn(*self.b3.shape) * sigma
        
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
            "W3": self.W3,
            "b3": self.b3,
            "generation": self.generation,
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
        self.W3 = np.array(self.parameters["W3"], dtype=float)
        self.b3 = np.array(self.parameters["b3"], dtype=float)
        self.NAME = str(self.parameters["NAME"])
        
        # Load generation counter (default to 0 for backwards compatibility)
        self.generation = int(self.parameters.get("generation", 0))

    def calculate_score(self, distance, time, no):
        """
        Calculate fitness score after a race.
        
        Components:
        1. Base score: distance traveled (PRIMARY - dominant factor)
        2. Efficiency bonus: small reward for covering distance quickly
        3. Survival bonus: rewards staying alive longer (capped)
        
        Args:
            distance: Total distance traveled
            time: Time taken (in frames or seconds)
            no: Not used
        """
        # Base score: distance traveled (PRIMARY objective)
        base_score = distance
        
        # Efficiency bonus: small reward for speed (reduced from 10.0 to 2.0)
        # This is now a tiebreaker, not a major factor
        if time > 0:
            efficiency_bonus = (distance / time) * 2.0
        else:
            efficiency_bonus = 0
        
        # Survival bonus: reward for staying alive (capped at 50)
        # Reduced cap so it doesn't overshadow distance
        survival_bonus = min(time * 0.3, 50)
        
        # Total score
        self.score = base_score + efficiency_bonus + survival_bonus

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
