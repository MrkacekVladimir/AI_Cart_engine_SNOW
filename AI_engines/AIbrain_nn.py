from numpy import random as np_random
import random
import numpy as np
import copy
import string

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]
N_HIDDEN = 16  # Hidden layer neurons

# vždy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_nn:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # pro potreby náhdných znaků
        self.decider = 0
        self.x = 0 # sem se ulozí souradnice x, max HEIGHT*1.3
        self.y = 0 # sem se ulozí souradnice y, max HEIGHT (800)
        self.speed = 0 # sem se ukládá souradnice, max MAXSPEED ( 500)
        
        # Speed tracking for better scoring
        self.speed_sum = 0.0
        self.speed_samples = 0
        self.max_speed_reached = 0.0

        self.init_param()

    def init_param(self):
        # zde si vytvoríme promnenne co potrebujeme pro nas model
        # parametry modely vzdy inicializovat v této metode
        # Layer 1: input -> hidden (Xavier-like initialization)
        self.W1 = (np_random.rand(N_HIDDEN, N_INPUTS) - 0.5) * np.sqrt(2.0 / N_INPUTS)
        self.b1 = np.zeros(N_HIDDEN)
        # Layer 2: hidden -> output
        self.W2 = (np_random.rand(N_ACTIONS, N_HIDDEN) - 0.5) * np.sqrt(2.0 / N_HIDDEN)
        self.b2 = np.zeros(N_ACTIONS)

        self.NAME = "SAFR_nn"

        # vždy uložit!
        self.store()

    def decide(self, data):
        self.decider += 1

        x = np.asarray(data, dtype=float).ravel()

        n_w = self.W1.shape[1]
        if x.size < n_w:
            x = np.concatenate([x, np.zeros(n_w - x.size)])
        elif x.size > n_w:
            x = x[:n_w]

        # Hidden layer with Leaky ReLU activation (prevents dead neurons)
        h = self.W1.dot(x) + self.b1
        hidden = np.where(h > 0, h, 0.01 * h)
        # Output layer with Sigmoid (bounds output to 0-1 for threshold > 0.5)
        z = self.W2.dot(hidden) + self.b2
        output = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

        return output

    def mutate(self):
        """
        Mutace: všechny váhy i biasy se malé náhodně posunou.
        Tím se skutečně mění lineární kombinace pro každou akci.
        """
        # náhodné perturbace ~ [-0.075, 0.075]
        delta_W1 = (np_random.rand(*self.W1.shape) - 0.5) * 0.15
        delta_b1 = (np_random.rand(*self.b1.shape) - 0.5) * 0.15
        delta_W2 = (np_random.rand(*self.W2.shape) - 0.5) * 0.15
        delta_b2 = (np_random.rand(*self.b2.shape) - 0.5) * 0.15

        self.W1 = self.W1 + delta_W1
        self.b1 = self.b1 + delta_b1
        self.W2 = self.W2 + delta_W2
        self.b2 = self.b2 + delta_b2

        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))

        self.store()

    def store(self):
        # vše, co se má ukládat do .npz
        self.parameters = copy.deepcopy({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "NAME": self.NAME,
        })

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Zde nastavit co chceme ukládat:
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.NAME = str(self.parameters["NAME"])


    def calculate_score(self, distance, time, finish_order):
        # Calculate average speed from tracked data
        avg_speed = self.speed_sum / max(self.speed_samples, 1)
        
        # Weighted scoring formula (no finish_order used)
        # - distance * 2: reward covering more ground
        # - avg_speed * 0.05: reward consistent movement
        # - max_speed * 0.02: bonus for achieving high speeds
        # - time * 0.1: small survival bonus
        self.score = (distance * 2.0) + (avg_speed * 0.05) + (self.max_speed_reached * 0.02) + (time * 0.1)

    ##################### do těchto funkcí není potřeba zasahovat:
    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        
        # Track speed statistics
        self.speed_sum += speed
        self.speed_samples += 1
        if speed > self.max_speed_reached:
            self.max_speed_reached = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)
