import random
import string
import copy
import numpy as np

"""
Simple rule-based AI brain using only if-statements.
No neural network, no training - just hardcoded logic based on sensor data.

Ray indices:
  0: -90° (perpendicular left)
  1: -45° (diagonal left)
  2: -20° (slight left)
  3:  -5° (almost forward, tiny left)
  4:   0° (straight ahead)
  5:  +5° (almost forward, tiny right)
  6: +20° (slight right)
  7: +45° (diagonal right)
  8: +90° (perpendicular right)

Output: [accelerate, brake, left, right] - values > 0.5 activate the action
"""


class AIbrain_dumb:
    def __init__(self):
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.x = 0
        self.y = 0
        self.speed = 0
        self.init_param()

    def init_param(self):
        """Initialize - just set a name, no trainable parameters."""
        self.NAME = "Dumb_" + ''.join(random.choices(self.chars, k=5))
        self.store()

    def decide(self, data):
        """
        Simple if-statement based decision making.
        
        Returns [accelerate, brake, left, right] where > 0.5 activates action.
        """
        rays = np.asarray(data, dtype=float)
        
        # Handle uninitialized rays (all zeros on first frame)
        if np.sum(rays) < 0.01:
            return [1.0, 0.0, 0.0, 0.0]  # Just accelerate if no data
        
        # Key ray readings
        forward = rays[4]           # Straight ahead (0°)
        left_side = (rays[0] + rays[1]) / 2    # Average of -90° and -45°
        right_side = (rays[7] + rays[8]) / 2   # Average of +45° and +90°
        
        # Initialize outputs (0 = off, 1 = on)
        accelerate = 0.0
        brake = 0.0
        turn_left = 0.0
        turn_right = 0.0
        
        # === SPEED CONTROL ===
        if forward < 1.5:
            # Wall ahead - brake!
            brake = 1.0
        else:
            # Clear enough - accelerate!
            accelerate = 1.0
        
        # === STEERING (defensive - turn early) ===
        if forward < 3.0:
            # Wall coming up - turn toward more open side
            if left_side > right_side:
                turn_left = 1.0
            else:
                turn_right = 1.0
        
        # Don't brake to a complete stop if we need to turn
        if self.speed < 30 and (turn_left > 0.5 or turn_right > 0.5):
            brake = 0.0
            accelerate = 1.0
        
        return [accelerate, brake, turn_left, turn_right]

    def store(self):
        """Store parameters (just the name for this simple AI)."""
        self.parameters = {'NAME': self.NAME}

    def mutate(self):
        """No-op - rule-based AI doesn't mutate."""
        self.NAME += "_MUT"
        self.store()

    def calculate_score(self, distance, time, no):
        """Calculate fitness score."""
        self.score = distance / time + no

    def passcardata(self, x, y, speed):
        """Receive current car state from the simulation."""
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        """Load parameters (just the name)."""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = copy.deepcopy(parameters)
        
        self.NAME = str(self.parameters.get('NAME', 'Dumb_loaded'))
