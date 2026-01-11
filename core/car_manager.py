import os
import time

import pygame
from my_sprites.AI_car import AI_car, HeadlessCar
from constants import WHITE, SPEED, MAX_SPEED, TILESIZE, TURN_SPEED, BREAK_SPEED, FRICTION_SPEED, PATH_SAVES
import copy
from pathlib import Path
import numpy as np

SPAWN_Y_JITTER_PX = 12


class HeadlessCarManager:
    """
    Car manager for headless training mode.
    No sprites, no rendering - just pure simulation.
    """
    
    def __init__(self, pocet_aut, pocet_generaci, max_ticks, cars_to_next, save_as, load_from):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_ticks = max_ticks  # Frame count per generation
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from
        
        self.cur_epoch = 0
        self.tick_count = 0  # Current tick in generation
        self.running = False
        self.pustene = False
        
        self.car_list = []
        self.best_cars_list = []
        self.defaultbrain = None
        self.brain_list = []
    
    def add_defaultbrain(self, brain):
        self.defaultbrain = brain
        self.reset_brains()
    
    def reset_brains(self):
        self.brain_list = [self.defaultbrain() for _ in range(self.pocet_aut)]
    
    def start(self):
        """Start training from scratch."""
        self.car_list = []
        self.cur_epoch = 0
        self.tick_count = 0
        
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)
        
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = HeadlessCar(base_x, y, 10, 20, self.brain_list[i], 180 + np.random.randint(-45, +45))
            self.car_list.append(c)
        
        self.running = True
        self.pustene = True
    
    def load(self, file="userbrain.npz"):
        """Load parameters and start training."""
        self.running = False
        params = np.load(Path(PATH_SAVES + file))
        self.reset_brains()
        
        self.car_list = []
        self.cur_epoch = 0
        self.tick_count = 0
        
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)
        
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = HeadlessCar(base_x, y, 10, 20, self.brain_list[i], 180 + np.random.randint(-45, +45))
            c.brain.set_parameters(params)
            if i > 0:
                c.brain.mutate()
            self.car_list.append(c)
        
        self.running = True
        self.pustene = True
    
    def setup_next_epoch(self):
        """Setup the next generation."""
        self.score_cars()
        self.best_cars_list = self.best_cars_list[:self.cars_to_next]
        
        self.car_list = []
        self.reset_brains()
        self.tick_count = 0
        self.cur_epoch += 1
        
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)
        
        # Keep best cars
        for i in range(self.cars_to_next):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = HeadlessCar(base_x, y, 10, 20,
                           copy.deepcopy(self.best_cars_list[i].brain),
                           180 + np.random.randint(-45, +45))
            self.car_list.append(c)
        
        # Fill with mutated versions
        for j in range(self.pocet_aut - self.cars_to_next):
            i = j % self.cars_to_next
            c = HeadlessCar(TILESIZE * 4 + int(TILESIZE / 2),
                           TILESIZE * 8 + int(TILESIZE / 2),
                           10, 20,
                           copy.deepcopy(self.best_cars_list[i].brain))
            c.brain.mutate()
            self.car_list.append(c)
        
        self.running = True
    
    def score_cars(self):
        self.best_cars_list = sorted(self.car_list, key=lambda obj: obj.brain.score, reverse=True)
    
    def update_step(self, dt, blocks):
        """
        Single update step for all cars (headless - no sprites).
        Returns True if epoch is still running, False if epoch ended.
        """
        if not self.running:
            return False
        
        for c in self.car_list:
            if c.running:
                c.update(dt, blocks)
                # Check collision using mask
                if c.check_collision(blocks):
                    c.running = False
        
        self.tick_count += 1
        
        # Check for epoch end (tick-based)
        if self.tick_count >= self.max_ticks:
            self.running = False
            return False
        
        return True
    
    def autosave(self):
        self.save(self.save_as)
    
    def save(self, file):
        self.score_cars()
        np.savez(Path(PATH_SAVES + file), **self.best_cars_list[0].brain.get_parameters())
    
    def get_best_score(self):
        """Get the current best score."""
        if self.car_list:
            return max(c.brain.score for c in self.car_list)
        return 0.0
    
    def get_avg_score(self):
        """Get average score of all cars."""
        if self.car_list:
            return sum(c.brain.score for c in self.car_list) / len(self.car_list)
        return 0.0
    
    def get_running_count(self):
        """Get count of cars still running."""
        return sum(1 for c in self.car_list if c.running)
    
    def run_training(self, blocks, dt=1/60, callback=None):
        """
        Run complete training loop (headless).
        
        Args:
            blocks: Blocks object with collision mask
            dt: Fixed timestep (default 1/60s for ~60fps equivalent)
            callback: Optional callback(epoch, best_score, avg_score, ticks, time_elapsed)
        """
        start_time = time.time()
        
        for epoch in range(self.pocet_generaci):
            epoch_start = time.time()
            
            # Run one epoch
            while self.running:
                self.update_step(dt, blocks)
            
            epoch_time = time.time() - epoch_start
            
            # Report progress
            if callback:
                callback(
                    epoch + 1,
                    self.get_best_score(),
                    self.get_avg_score(),
                    self.tick_count,
                    epoch_time
                )
            
            # Setup next epoch if not done
            if epoch < self.pocet_generaci - 1:
                self.setup_next_epoch()
        
        # Final save
        self.autosave()
        self.pustene = False
        
        total_time = time.time() - start_time
        return total_time

class Car_manager():
    def __init__(self, pocet_aut, pocet_generaci, max_time, cars_to_next,save_as, load_from):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from

        self.epoch = 0
        self.cur_epoch = 0
        self.total_time = 0
        self.running = False # tykas se zda bezi nejaká epocha
        self.pustene = False # týká se zda bezí celkvoe trening

        self.sprite_list = list() # registrujeme do vsech - abych je pak byl schopen prebrat
        self.sprite_running = pygame.sprite.Group() # praucjeme jen s running ve kole
        self.best_cars_list = list() # bezst brain z minulé epochy

    def setup(self, pocet_aut, pocet_generaci, max_time, cars_to_next, save_as, load_from):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from

    def setup_next_epoch(self):
        self.score_cars()# oskoruji auta
        self.best_cars_list = [c for c in self.best_cars_list[0:self.cars_to_next]]# ulozim si strnaou nejlepsích n

        self.sprite_list = list()# vyresetuji cekový list
        for c in self.sprite_running:# vymazu vse
            c.kill()
        self.reset_brains()
        self.total_time = 0
        self.cur_epoch += 1

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        for i in range(self.cars_to_next):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain),  180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # doplneneí zmutovanou generací prvních n aut:
        for j in range(self.pocet_aut-self.cars_to_next):
            i = j % self.cars_to_next
            c = AI_car(TILESIZE * 4 + int(TILESIZE / 2), TILESIZE * 8 + int(TILESIZE / 2), 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain))

            c.brain.mutate()
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        self.running = True

    def score_cars(self):
        self.best_cars_list = list()
        self.best_cars_list = sorted(self.sprite_list, key=lambda obj: obj.brain.score, reverse=True)

    def add_defaultbrain(self, brain):
        self.defaultbrain = brain
        self.reset_brains()

    # start - od začátku vše, zresetuje co jde a nastaví auta znova
    def start(self):
        # zapnu start a vymazu vsechny informace pokud nekde jsou:
        if len(self.sprite_list) > 0:
            self.sprite_list = list()  # registrujeme do vsech - abych je pak byl schopen prebrat
        if len(self.sprite_running) > 0:
            for c in self.sprite_running:
                c.kill()
            self.sprite_running = pygame.sprite.Group()  # praucjeme jen s running ve kole

        self.cur_epoch = 0
        self.total_time = 0

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20, self.brain_list[i], 180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # a zapnu tréning
        self.running = True
        self.pustene = True

    def get_sprite_group(self):
        return self.sprite_running

    def reset_brains(self):
        self.brain_list = [self.defaultbrain() for _ in range(self.pocet_aut)]

    def draw(self, screen):
        self.sprite_running.draw(screen)

    def stop(self):
        self.running = False

    def autosave(self):
        self.save(self.save_as)

    def save(self, file):
        self.score_cars()
        print(f"probiha save souboru {file}")
        print(os.getcwd())
        print(self.best_cars_list[0].brain.get_parameters())# nejlepsí je na prvním íste :)
        np.savez(Path(PATH_SAVES+file), **self.best_cars_list[0].brain.get_parameters())

    def load(self, file = "userbrain.npz"):
        self.running = False
        print(f"probiha load souboru {file}")
        params =  np.load(Path(PATH_SAVES+file))
        print({key: params[key] for key in params.files})
        self.reset_brains()

        if len(self.sprite_list) > 0:
            self.sprite_list = list()  # registrujeme do vsech - abych je pak byl schopen prebrat
        if len(self.sprite_running) > 0:
            for c in self.sprite_running:
                c.kill()
            self.sprite_running = pygame.sprite.Group()  # praucjeme jen s running ve kole

        self.cur_epoch = 0
        self.total_time = 0
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20, self.brain_list[i],  180+np.random.randint(-45,+45))
            c.brain.set_parameters(params)
            if i>0:
                c.brain.mutate()
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # a zapnu tréning
        self.running = True
        self.pustene = True

    def update(self,  dt, keys, blocks):
        if self.running:
            for c in self.sprite_running:
                c.update(dt, keys, blocks)
                # nárazy do zdí
                hit = pygame.sprite.spritecollideany(c, blocks.sprites)
                if hit is not None:
                    c.running = False

            self.total_time += dt


        # enco jako if total time> max time - new epoch
        # ted musím doresit epochy, kolize a save a load
        if self.total_time > self.max_time  and self.pustene:
            self.running = False # timer pro epochu

            if self.cur_epoch < self.pocet_generaci:
                print(f"-----epocha {self.cur_epoch}------------------------")
                #for sprite in self.sprite_list:
                #    print(sprite.brain.parameters)
                #print(f"----------------------------------------------------")
                self.setup_next_epoch()
            else:
                self.pustene = False
                self.autosave()


        #print(f"total time: {self.total_time}")
        #print(f"setting, pocet aut: {self.pocet_aut}, pocet generaci: {self.pocet_generaci}, epoch: {self.epoch}")
        #print(f"vnitrni data: len sprite list: {len(self.sprite_list)}, len sprite running: {len(self.sprite_running)}")