from numpy import random as np_random
import random
import numpy as np
import copy
import string
import logging

# Configure logging (only if not already configured)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

# Configure logger for this module
logger = logging.getLogger(__name__)

# Set to True to enable decision logging (every N frames)
DEBUG_LOGGING = True
LOG_EVERY_N_FRAMES = 30  # Log every N frames to avoid spam

# Ray indices: [-90, -45, -20, -5, 0, 5, 20, 45, 90]
# Forward rays: indices 2,3,4,5,6 (-20, -5, 0, 5, 20)
# Left rays: indices 0,1 (-90, -45)
# Right rays: indices 7,8 (45, 90)

FORWARD_RAY_INDICES = [2, 3, 4, 5, 6]  # -20, -5, 0, 5, 20 degrees
LEFT_RAY_INDICES = [0, 1]              # -90, -45 degrees
RIGHT_RAY_INDICES = [7, 8]             # 45, 90 degrees

# Network sizes
SPEED_INPUT_SIZE = 6   # 5 forward rays + 1 normalized speed
SPEED_HIDDEN_SIZE = 16
SPEED_OUTPUT_SIZE = 2  # accelerate, brake

STEER_INPUT_SIZE = 6   # 4 side rays + 2 speed outputs
STEER_HIDDEN_SIZE = 16
STEER_OUTPUT_SIZE = 2  # left, right

MAX_SPEED_NORM = 500.0  # For normalizing speed input


class AIbrain_logistic:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        self.frame_count = 0  # For logging frequency control
        self.x = 0
        self.y = 0
        self.speed = 0

        self.init_param()

    def init_param(self):
        """Initialize two-stage network parameters."""
        
        # Stage 1: Speed Control Network
        # Input: 5 forward rays + normalized speed = 6 inputs
        # Output: 2 (accelerate, brake)
        self.W1_speed = (np_random.rand(SPEED_HIDDEN_SIZE, SPEED_INPUT_SIZE) - 0.5) / SPEED_INPUT_SIZE
        self.b1_speed = (np_random.rand(SPEED_HIDDEN_SIZE) - 0.5)
        self.W2_speed = (np_random.rand(SPEED_OUTPUT_SIZE, SPEED_HIDDEN_SIZE) - 0.5) / SPEED_HIDDEN_SIZE
        self.b2_speed = (np_random.rand(SPEED_OUTPUT_SIZE) - 0.5)
        
        # Stage 2: Steering Network
        # Input: 2 left rays + 2 right rays + 2 speed outputs = 6 inputs
        # Output: 2 (left, right)
        self.W1_steer = (np_random.rand(STEER_HIDDEN_SIZE, STEER_INPUT_SIZE) - 0.5) / STEER_INPUT_SIZE
        self.b1_steer = (np_random.rand(STEER_HIDDEN_SIZE) - 0.5)
        self.W2_steer = (np_random.rand(STEER_OUTPUT_SIZE, STEER_HIDDEN_SIZE) - 0.5) / STEER_HIDDEN_SIZE
        self.b2_steer = (np_random.rand(STEER_OUTPUT_SIZE) - 0.5)
        
        self.NAME = "Logistic_" + ''.join(random.choices(self.chars, k=5))
        self.store()

    def sigmoid(self, x):
        """Sigmoid activation for smooth 0-1 outputs."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def decide(self, data):
        """
        Two-stage decision process:
        Stage 1: Decide acceleration/braking based on forward space and current speed
        Stage 2: Decide steering based on side clearance and speed decision
        """
        rays = np.asarray(data, dtype=float)
        
        # Handle uninitialized rays (all zeros on first frame)
        # If rays are all zero, assume open road to get the car moving
        if np.sum(rays) < 0.01:
            rays = np.full(9, 10.0)  # Assume 10 tiles of clearance everywhere
        
        # === STAGE 1: Speed Control ===
        # Extract forward rays and normalize
        forward_rays = rays[FORWARD_RAY_INDICES]
        
        # Normalize speed (0-1 range)
        norm_speed = self.speed / MAX_SPEED_NORM
        
        # Build speed network input
        speed_input = np.concatenate([forward_rays, [norm_speed]])
        
        # Forward pass through speed network
        h_speed = np.maximum(0, self.W1_speed.dot(speed_input) + self.b1_speed)  # ReLU
        z_speed = self.W2_speed.dot(h_speed) + self.b2_speed
        
        # Apply sigmoid for probability-like outputs
        speed_output = self.sigmoid(z_speed)
        
        # Speed modulation: adjust acceleration based on forward clearance
        min_forward_dist = np.min(forward_rays)
        
        # Also check side clearance - if we can turn, we should keep moving!
        left_rays = rays[LEFT_RAY_INDICES]
        right_rays = rays[RIGHT_RAY_INDICES]
        max_side_clearance = max(np.max(left_rays), np.max(right_rays))
        
        # Base rule: Consider both forward space AND escape routes (side clearance)
        # Key insight: Car needs speed to turn! If there's space to turn into, accelerate.
        
        if min_forward_dist < 0.5:
            # CRITICAL: About to hit wall
            if max_side_clearance > 1.0:
                # There's an escape route - accelerate slowly to enable turning
                base_accel = 0.6
                base_brake = 0.0
            else:
                # Nowhere to go - brake
                base_accel = 0.0
                base_brake = 1.0
        elif min_forward_dist < 1.5:
            # Tight forward space - but check if we can turn
            if max_side_clearance > 1.5:
                # Good escape route - accelerate to turn
                base_accel = 0.7
                base_brake = 0.0
            else:
                # Limited options - slow acceleration to maneuver
                base_accel = 0.55
                base_brake = 0.0
        elif min_forward_dist < 3.0:
            # Moderate forward space - accelerate
            space_factor = (min_forward_dist - 1.5) / 1.5  # 0 to 1
            base_accel = 0.6 + space_factor * 0.2  # 0.6 to 0.8
            base_brake = 0.0
        elif min_forward_dist < 5.0:
            # Good space - accelerate but watch speed
            speed_limit_factor = 1.0 - norm_speed * 0.3
            base_accel = 0.8 * speed_limit_factor
            base_brake = 0.0
        else:
            # Open road - full acceleration, modulate by current speed
            speed_limit_factor = 1.0 - norm_speed * 0.5
            base_accel = 0.9 * speed_limit_factor
            base_brake = 0.0
        
        # Combine base rule with network output (network can fine-tune)
        network_accel_shift = (speed_output[0] - 0.5) * 0.3  # -0.15 to +0.15
        network_brake_shift = (speed_output[1] - 0.5) * 0.3
        
        accel_decision = np.clip(base_accel + network_accel_shift, 0.0, 1.0)
        brake_decision = np.clip(base_brake + network_brake_shift, 0.0, 1.0)
        
        # Don't brake if we need to turn (car can't turn at speed 0!)
        if self.speed < 50 and max_side_clearance > 1.0:
            brake_decision = min(brake_decision, 0.3)
            accel_decision = max(accel_decision, 0.6)
        
        # === STAGE 2: Steering Control ===
        # (left_rays and right_rays already extracted in Stage 1)
        
        # Build steering network input: side rays + speed decision outputs
        steer_input = np.concatenate([left_rays, right_rays, [accel_decision, brake_decision]])
        
        # Forward pass through steering network
        h_steer = np.maximum(0, self.W1_steer.dot(steer_input) + self.b1_steer)  # ReLU
        z_steer = self.W2_steer.dot(h_steer) + self.b2_steer
        
        # Apply sigmoid
        steer_output = self.sigmoid(z_steer)
        
        # Steering: use network output as base, add bias toward more open side
        left_clearance = np.mean(left_rays)
        right_clearance = np.mean(right_rays)
        
        # Determine which way has more space
        if min_forward_dist < 1.5:
            # MUST turn - strong override toward open side
            if left_clearance > right_clearance + 0.3:
                # Turn LEFT - clear space on left
                left_decision = 0.9
                right_decision = 0.1
            elif right_clearance > left_clearance + 0.3:
                # Turn RIGHT - clear space on right
                left_decision = 0.1
                right_decision = 0.9
            else:
                # Similar clearance - let network decide but boost the winner
                if steer_output[0] > steer_output[1]:
                    left_decision = 0.7
                    right_decision = 0.2
                else:
                    left_decision = 0.2
                    right_decision = 0.7
        elif min_forward_dist < 3.0:
            # Should turn - moderate bias toward open side
            clearance_diff = left_clearance - right_clearance
            bias = np.clip(clearance_diff * 0.3, -0.4, 0.4)
            left_decision = np.clip(steer_output[0] + bias, 0.0, 1.0)
            right_decision = np.clip(steer_output[1] - bias, 0.0, 1.0)
        else:
            # Open road - network decides with small bias
            clearance_diff = left_clearance - right_clearance
            bias = np.clip(clearance_diff * 0.1, -0.2, 0.2)
            left_decision = np.clip(steer_output[0] + bias, 0.0, 1.0)
            right_decision = np.clip(steer_output[1] - bias, 0.0, 1.0)
        
        # Prevent simultaneous LEFT+RIGHT (pick the stronger one)
        if left_decision > 0.5 and right_decision > 0.5:
            if left_decision > right_decision:
                right_decision = 0.3
            else:
                left_decision = 0.3
        
        # Build final decision vector
        decision = np.array([accel_decision, brake_decision, left_decision, right_decision])
        
        # === STRUCTURED LOGGING ===
        self.frame_count += 1
        if DEBUG_LOGGING and self.frame_count % LOG_EVERY_N_FRAMES == 0:
            # Determine which actions are active (> 0.5 threshold)
            actions = []
            if accel_decision > 0.5:
                actions.append("ACCEL")
            if brake_decision > 0.5:
                actions.append("BRAKE")
            if left_decision > 0.5:
                actions.append("LEFT")
            if right_decision > 0.5:
                actions.append("RIGHT")
            if not actions:
                actions.append("COAST")
            
            logger.info(
                f"[{self.NAME}] Frame {self.frame_count} | "
                f"Decision: [{accel_decision:.2f}, {brake_decision:.2f}, {left_decision:.2f}, {right_decision:.2f}] | "
                f"Actions: {'+'.join(actions)} | "
                f"Speed: {self.speed:.0f} | "
                f"FwdMin: {min_forward_dist:.1f} | "
                f"L/R: {left_clearance:.1f}/{right_clearance:.1f}"
            )
        
        return decision

    def store(self):
        """Store all network parameters for saving/loading."""
        self.parameters = copy.deepcopy({
            'W1_speed': self.W1_speed,
            'b1_speed': self.b1_speed,
            'W2_speed': self.W2_speed,
            'b2_speed': self.b2_speed,
            'W1_steer': self.W1_steer,
            'b1_steer': self.b1_steer,
            'W2_steer': self.W2_steer,
            'b2_steer': self.b2_steer,
            'NAME': self.NAME,
        })

    def mutate(self):
        """Mutate all network parameters for evolutionary training."""
        mutation_strength = 0.2
        
        # Mutate speed network
        self.W1_speed += (np_random.rand(*self.W1_speed.shape) - 0.5) * mutation_strength
        self.b1_speed += (np_random.rand(*self.b1_speed.shape) - 0.5) * mutation_strength
        self.W2_speed += (np_random.rand(*self.W2_speed.shape) - 0.5) * mutation_strength
        self.b2_speed += (np_random.rand(*self.b2_speed.shape) - 0.5) * mutation_strength
        
        # Mutate steering network
        self.W1_steer += (np_random.rand(*self.W1_steer.shape) - 0.5) * mutation_strength
        self.b1_steer += (np_random.rand(*self.b1_steer.shape) - 0.5) * mutation_strength
        self.W2_steer += (np_random.rand(*self.W2_steer.shape) - 0.5) * mutation_strength
        self.b2_steer += (np_random.rand(*self.b2_steer.shape) - 0.5) * mutation_strength
        
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        self.store()

    def calculate_score(self, distance, time, no):
        """Calculate fitness score for evolutionary selection."""
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
        """Load network parameters from saved data."""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = copy.deepcopy(parameters)

        # Load speed network
        self.W1_speed = np.array(self.parameters['W1_speed'], dtype=float)
        self.b1_speed = np.array(self.parameters['b1_speed'], dtype=float)
        self.W2_speed = np.array(self.parameters['W2_speed'], dtype=float)
        self.b2_speed = np.array(self.parameters['b2_speed'], dtype=float)
        
        # Load steering network
        self.W1_steer = np.array(self.parameters['W1_steer'], dtype=float)
        self.b1_steer = np.array(self.parameters['b1_steer'], dtype=float)
        self.W2_steer = np.array(self.parameters['W2_steer'], dtype=float)
        self.b2_steer = np.array(self.parameters['b2_steer'], dtype=float)
        
        self.NAME = str(self.parameters['NAME'])
