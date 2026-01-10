from numpy import random as np_random
import random
import numpy as np
import copy
import string
import constants
"""
class AIbrain_zupicz:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0
        self.generation_count = 0
        # Inicializace parametrů
        self.init_param()

    def init_param(self):
        #Inicializace sítě a náhodných vah.
        # 1. Zjistíme počet senzorů
        try:
            n_sensors = len(constants.RAYCAST_ANGLES)
        except AttributeError:
            n_sensors = 5 # Fallback
            
        # 2. Nastavíme vstupy sítě
        # Vstupy = Senzory + 1 (Rychlost auta)
        self.n_inputs = n_sensors + 1 
        self.n_hidden = 12 # Zvýšili jsme počet neuronů, protože máme víc vstupů
        self.n_outputs = 2 # [Steering, Throttle]

        # 3. Inicializace vah (He initialization pro ReLU)
        self.w1 = np.random.randn(self.n_inputs, self.n_hidden) * np.sqrt(2.0/self.n_inputs)
        self.b1 = np.zeros(self.n_hidden)

        self.w2 = np.random.randn(self.n_hidden, self.n_outputs) * np.sqrt(2.0/self.n_hidden)
        
        # 4. Randomized Bias Hack (Tvá připomínka)
        # Základ [0.0, 0.5] + náhodný šum.
        # Zajistí, že se auta rozjedou, ale každé má trochu jinou "osobnost".
        base_bias = np.array([0.0, 0.5])
        noise = np.random.randn(self.n_outputs) * 0.2
        self.b2 = base_bias + noise

        # Telemetrie a Skóre
        self.score = 0
        self.fitness = 0
        self.car_speed = 0
        self.car_x = 0
        self.car_y = 0
        self.generation_count = 0

    def relu(self, x):
        #ReLU aktivace: Všechno záporné změní na nulu. Lineární pro kladné.
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        #Sigmoid: 0 až 1.
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def decide(self, sensor_data):
        # --- 1. PŘÍPRAVA VSTUPŮ ---
        raw_sensors = np.array(sensor_data)
        
        # A) Normalizace senzorů (Dlaždice)
        # Zjistíme max dosvit (obvykle 5 nebo 10 dlaždic)
        ray_len = getattr(constants, 'RAYCAST_LENGTH', 5) 
        norm_sensors = raw_sensors / ray_len
        
        # B) Normalizace rychlosti
        # Rychlost musíme dostat cca mezi 0 a 1. 
        # Max rychlost bývá kolem 20-30 jednotek. Vydělíme to 30.
        norm_speed = self.car_speed / 30.0
        
        # C) Spojení do jednoho vektoru
        # Vstupní vektor = [Senzor1, Senzor2, ..., Senzor9, Rychlost]
        x = np.append(norm_sensors, norm_speed)

        # Ošetření změny počtu senzorů (pro jistotu)
        if x.shape[0] != self.n_inputs:
            # Pokud se změní mapa a počet paprsků, musíme resetovat mozek
            # (Poznámka: Toto zahodí natrénovaný mozek, ale zabrání pádu)
            self.init_param()
            # Musíme znovu vytvořit x se správnou délkou pro novou inicializaci, 
            # ale v tomto framu to prostě necháme proběhnout s chybou nebo vrátíme default
            return [False, False, False, False]

        # --- 2. PRŮCHOD SÍTÍ (Inference) ---
        
        # Skrytá vrstva (ReLU)
        z1 = np.dot(x, self.w1) + self.b1
        h1 = self.relu(z1)

        # Výstupní vrstva (Sigmoid)
        z2 = np.dot(h1, self.w2) + self.b2
        y = self.sigmoid(z2) 

        # --- 3. PŘEVOD NA KLÁVESY ---
        actions = [False, False, False, False] # UP, DOWN, LEFT, RIGHT
        steering, throttle = y[0], y[1]

        # Řízení (Steering) - jemnější deadzone
        if steering < 0.4: actions[2] = True  # Left
        elif steering > 0.6: actions[3] = True # Right
        
        # Plyn (Throttle)
        if throttle > 0.5: actions[0] = True   # Gas
        elif throttle < 0.3: actions[1] = True # Brake
        
        return actions

    # --- FUNKCE PRO KOMPATIBILITU S ENGINEM ---

    def passcardata(self, x, y, speed):
        #Zde získáváme rychlost pro příští rozhodnutí.
        self.car_x = x
        self.car_y = y
        self.car_speed = speed

    def calculate_score(self, distance, time, no):
        #Fitness funkce s důrazem na rychlost v pozdějších fázích.
        if time > 0:
            avg_speed = distance / time
        else:
            avg_speed = 0
            
        # Zjistíme generaci z našeho interního počítadla (bezpečnější než parametr 'no')
        gen = self.generation_count
        
        start_speed_gen = 5
        max_speed_gen = 40 # Posunuto, ať mají víc času na učení ovládání
        
        if gen < start_speed_gen:
            speed_imp = 0.0
        elif gen >= max_speed_gen:
            speed_imp = 1.0
        else:
            speed_imp = (gen - start_speed_gen) / (max_speed_gen - start_speed_gen)

        # Skóre
        self.score = distance + (avg_speed * 50 * speed_imp)
        self.fitness = self.score

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        self.generation_count += 1
        
        def mutate_arr(arr):
            mask = np.random.rand(*arr.shape) < mutation_rate
            noise = np.random.randn(*arr.shape) * mutation_strength
            arr[mask] += noise[mask]
        
        mutate_arr(self.w1)
        mutate_arr(self.b1)
        mutate_arr(self.w2)
        mutate_arr(self.b2)

    def get_parameters(self):
        return np.concatenate([self.w1.flatten(), self.b1.flatten(), self.w2.flatten(), self.b2.flatten()])

    def set_parameters(self, parameters):
        idx1 = self.w1.size
        idx2 = idx1 + self.b1.size
        idx3 = idx2 + self.w2.size
        self.w1 = parameters[:idx1].reshape(self.w1.shape)
        self.b1 = parameters[idx1:idx2].reshape(self.b1.shape)
        self.w2 = parameters[idx2:idx3].reshape(self.w2.shape)
        self.b2 = parameters[idx3:].reshape(self.b2.shape)

    def save(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, gen=self.generation_count)
        print(f"Saved to {filename}")

    def load(self, filename):
        try:
            data = np.load(filename)
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            # Zkusíme načíst i generaci, pokud tam je
            if 'gen' in data:
                self.generation_count = data['gen']
            print(f"Loaded from {filename}")
        except Exception as e:
            print(f"Error loading: {e}")
"""

class AIbrain_zupicz:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0
        self.init_param()

    def init_param(self):
        try:
            n_sensors = len(constants.RAYCAST_ANGLES)
        except AttributeError:
            n_sensors = 5
            
        self.n_inputs = n_sensors + 1 
        self.n_hidden = 16 
        self.n_outputs = 3 # [Steering, Gas, Brake]

        # He Initialization
        self.w1 = np.random.randn(self.n_inputs, self.n_hidden) * np.sqrt(2.0/self.n_inputs)
        self.b1 = np.zeros(self.n_hidden)
        self.w2 = np.random.randn(self.n_hidden, self.n_outputs) * np.sqrt(2.0/self.n_hidden)
        
        self.NAME ="Zupicz_"+''.join(random.choices(self.chars, k=5))
        self.store()

        # --- BIAS ---
        # Nastavíme jen jemnou preferenci pro jízdu vpřed, ale žádné extrémy.
        # Steering (0): 0.0
        # Gas (1): 0.2 (Jemné popostrčení, zbytek se musí naučit)
        # Brake (2): -0.5 (Aby nebrzdilo bez důvodu)
        self.b2 = np.array([0.0, 0.2, -0.5]) 
        self.b2 += np.random.randn(3) * 0.2

        self.score = 0
        self.fitness = 0
        self.car_speed = 0
        self.car_x = 0
        self.car_y = 0
        self.generation_count = 0

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def decide(self, sensor_data):
        # --- 1. VSTUPY ---
        raw_sensors = np.array(sensor_data)
        ray_len = getattr(constants, 'RAYCAST_LENGTH', 5) 
        
        # Normalizace, Clamping a Inverze
        # 1.0 = Zeď je bezprostředně u nás (PANIKA)
        # 0.0 = Zeď je daleko
        norm_sensors = np.clip(raw_sensors / ray_len, 0.0, 1.0)
        inputs = 1.0 - norm_sensors 
        
        # Rychlost (0.0 až 1.0)
        norm_speed = np.clip(self.car_speed / 40.0, 0.0, 1.0)
        x = np.append(inputs, norm_speed)

        if x.shape[0] != self.n_inputs:
            self.init_param()
            return [False, False, False, False]

        # --- 2. PRŮCHOD SÍTÍ ---
        z1 = np.dot(x, self.w1) + self.b1
        h1 = self.relu(z1)
        z2 = np.dot(h1, self.w2) + self.b2
        y = self.sigmoid(z2) 

        # --- 3. AKCE (Čistá logika) ---
        actions = [False, False, False, False] # UP, DOWN, LEFT, RIGHT
        
        out_steer = y[0]
        out_gas   = y[1]
        out_brake = y[2]

        # A) Zatáčení
        # 0.0 - 0.45: Left
        # 0.45 - 0.55: Rovně
        # 0.55 - 1.0: Right
        if out_steer < 0.45: actions[2] = True
        elif out_steer > 0.55: actions[3] = True
        
        # B) Plyn
        if out_gas > 0.5:
            actions[0] = True
            
        # C) Brzda
        if out_brake > 0.5:
            actions[1] = True
            actions[0] = False # Fyzika: Brzda má přednost před plynem
        
        # Žádné další "if speed < X then force gas".
        # Síť se musí sama naučit, že když stojí, nedostává body.
        
        return actions
    
    def passcardata(self, x, y, speed):
        self.car_x = x
        self.car_y = y
        self.car_speed = speed

    def calculate_score(self, distance, time, no):
        # Opět použijeme naše bezpečné počítadlo generací
        if time > 0:
            avg_speed = distance / time
        else:
            avg_speed = 0
            
        gen = self.generation_count
        
        start_speed_gen = 5
        max_speed_gen = 40
        
        if gen < start_speed_gen:
            speed_imp = 0.0
        elif gen >= max_speed_gen:
            speed_imp = 1.0
        else:
            speed_imp = (gen - start_speed_gen) / (max_speed_gen - start_speed_gen)

        self.score = distance + (avg_speed * 50 * speed_imp)
        self.fitness = self.score

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        self.generation_count += 1
        
        def mutate_arr(arr):
            mask = np.random.rand(*arr.shape) < mutation_rate
            noise = np.random.randn(*arr.shape) * mutation_strength
            arr[mask] += noise[mask]
        
        mutate_arr(self.w1)
        mutate_arr(self.b1)
        mutate_arr(self.w2)
        mutate_arr(self.b2)

    def get_parameters(self):
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            # Můžeme zkusit přibalit i generaci, engine to při uložení vezme
            "gen": self.generation_count 
        }

    def set_parameters(self, parameters):
        # Musíme ošetřit, že 'parameters' je slovník (což při načítání bude)
        self.w1 = parameters["w1"]
        self.b1 = parameters["b1"]
        self.w2 = parameters["w2"]
        self.b2 = parameters["b2"]
        
        # Pokud je v datech i číslo generace, načteme ho
        if "gen" in parameters:
            self.generation_count = int(parameters["gen"])

    def store(self):
        self.parameters = copy.deepcopy({
            'w1': self.w1,
            'w2': self.w2,
            "NAME": self.NAME,
        })

    def load(self, filename):
        try:
            data = np.load(filename)
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            if 'gen' in data:
                self.generation_count = data['gen']
            print(f"Loaded from {filename}")
        except Exception as e:
            print(f"Error loading: {e}")