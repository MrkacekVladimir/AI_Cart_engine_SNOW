from numpy import random as np_random
import random
import numpy as np
import copy
import string

class AIbrain_TeamName:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0

        self.init_param()

    def init_param(self):
        self.hidden_size = 64  # tunable
        
        # Layer 1: 9 inputs -> 16 hidden
        self.W1 = (np.random.rand(self.hidden_size, 9) - 0.5) / 9
        self.b1 = (np.random.rand(self.hidden_size) - 0.5)
        
        # Layer 2: 16 hidden -> 4 outputs
        self.W2 = (np.random.rand(4, self.hidden_size) - 0.5) / self.hidden_size
        self.b2 = (np.random.rand(4) - 0.5)
        
        self.NAME ="Vlad_"+''.join(random.choices(self.chars, k=5))
        self.store()

    def decide(self, data):
        x = np.asarray(data, dtype=float)
        
        # Hidden layer with ReLU
        h = np.maximum(0, self.W1.dot(x) + self.b1)
        
        # Output layer
        z = self.W2.dot(h) + self.b2
        return z

    # def decide(self, data):
    #     self.decider += 1
    #     if np.round(self.decider) % 2 == 1:
    #         return np.round(self.w1)
    #     else:
    #         return np.round(self.w2)

    def store(self):
        self.parameters = copy.deepcopy({
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            "NAME": self.NAME,
        })


    def mutate(self):
        mutation_strength = 0.2  # Smaller for more params
        
        # Mutate all 4 parameter arrays
        self.W1 += (np.random.rand(*self.W1.shape) - 0.5) * mutation_strength
        self.b1 += (np.random.rand(*self.b1.shape) - 0.5) * mutation_strength
        self.W2 += (np.random.rand(*self.W2.shape) - 0.5) * mutation_strength
        self.b2 += (np.random.rand(*self.b2.shape) - 0.5) * mutation_strength
        
        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))
        self.store()

    def calculate_score(self, distance, time, no):
        self.score = distance/time + no

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = copy.deepcopy(parameters)

        self.w1 = self.parameters['w1']
        self.w2 = self.parameters['w2']
        self.NAME = self.parameters['NAME']
