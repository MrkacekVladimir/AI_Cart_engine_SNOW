import os

import pygame
from my_sprites.AI_car import AI_car
from constants import WHITE, SPEED, MAX_SPEED, TILESIZE, TURN_SPEED, BREAK_SPEED, FRICTION_SPEED,PATH_SAVES
import copy
from pathlib import Path
import numpy as np

SPAWN_Y_JITTER_PX = 12

class Car_manager():
    def __init__(self, pocet_aut, pocet_generaci, max_time, cars_to_next, save_as, load_from, checkpoint_every=0, cars_to_next_decay=1.0):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.cars_to_next_initial = cars_to_next  # Store initial value
        self.cars_to_next_decay = cars_to_next_decay  # Decay factor per epoch (1.0 = no decay, 0.95 = 5% reduction)
        self.cars_to_next_accumulator = 0.0  # Accumulator for fractional decay
        self.save_as = save_as
        self.load_from = load_from
        self.checkpoint_every = checkpoint_every  # Save checkpoint every N epochs (0 = disabled)
        
        self.best_score_ever = float('-inf')  # Track best score across all epochs
        
        # Create checkpoint folder based on save filename
        self.checkpoint_folder = self._get_checkpoint_folder(save_as)

        self.epoch = 0
        self.cur_epoch = 0
        self.total_time = 0
        self.running = False # tykas se zda bezi nejaká epocha
        self.pustene = False # týká se zda bezí celkvoe trening

        self.sprite_list = list() # registrujeme do vsech - abych je pak byl schopen prebrat
        self.sprite_running = pygame.sprite.Group() # praucjeme jen s running ve kole
        self.best_cars_list = list() # bezst brain z minulé epochy
    
    def _get_checkpoint_folder(self, filename):
        """Create and return the checkpoint folder path based on filename"""
        base_name = filename.replace('.npz', '')
        folder_path = Path(PATH_SAVES) / base_name
        # Create folder if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def setup(self, pocet_aut, pocet_generaci, max_time, cars_to_next, save_as, load_from, checkpoint_every=0, cars_to_next_decay=1.0):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.cars_to_next_initial = cars_to_next
        self.cars_to_next_decay = cars_to_next_decay
        self.cars_to_next_accumulator = 0.0  # Reset accumulator on setup
        self.save_as = save_as
        self.load_from = load_from
        self.checkpoint_every = checkpoint_every
        
        # Update checkpoint folder when setup is called
        self.checkpoint_folder = self._get_checkpoint_folder(save_as)

    def setup_next_epoch(self):
        self.score_cars()# oskoruji auta
        self.log_epoch_stats()  # Log statistics after scoring
        self.checkpoint()  # Save checkpoint if needed
        self.checkpoint_best_score()  # Save checkpoint on new best score
        
        # Apply decay to cars_to_next (elite count) with accumulation
        if self.cars_to_next_decay < 1.0:
            # Calculate the decay amount (what we lose)
            decay_amount = self.cars_to_next * (1.0 - self.cars_to_next_decay)
            self.cars_to_next_accumulator += decay_amount
            
            # Only decrease when accumulator >= 1 (one full car)
            if self.cars_to_next_accumulator >= 1.0:
                cars_to_remove = int(self.cars_to_next_accumulator)
                new_cars_to_next = self.cars_to_next - cars_to_remove
                # Ensure at least 10 cars go to next generation (minimum for diversity)
                new_cars_to_next = max(10, new_cars_to_next)
                
                if new_cars_to_next != self.cars_to_next:
                    print(f"  Elite count decay: {self.cars_to_next} → {new_cars_to_next} (accumulated {self.cars_to_next_accumulator:.2f})")
                    # Subtract the actual amount removed from accumulator
                    actual_removed = self.cars_to_next - new_cars_to_next
                    self.cars_to_next_accumulator -= actual_removed
                    self.cars_to_next = new_cars_to_next
        
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
    
    def log_epoch_stats(self):
        """Logs statistics about scores for the current epoch"""
        if len(self.sprite_list) == 0:
            return
        
        scores = [car.brain.score for car in self.sprite_list]
        min_score = np.min(scores)
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        
        # Get mutation rate and average speed from best car
        if len(self.best_cars_list) > 0 and hasattr(self.best_cars_list[0].brain, 'mutation_rate'):
            mutation_rate = self.best_cars_list[0].brain.mutation_rate
            mutation_std = self.best_cars_list[0].brain.mutation_std
            
            # Get average speed from best car
            avg_speed = getattr(self.best_cars_list[0].brain, 'average_speed', 0.0)
            
            mutation_info = f", MutRate={mutation_rate:.4f}, MutStd={mutation_std:.4f}, AvgSpeed={avg_speed:.1f}"
        else:
            mutation_info = ""
        
        # Add elite count info
        elite_info = f", Elite={self.cars_to_next}"
        
        print(f"Epoch {self.cur_epoch}: Min={min_score:.2f}, Mean={mean_score:.2f}, Max={max_score:.2f}{mutation_info}{elite_info}")
        
        # Print top 5 cars
        if len(self.best_cars_list) > 0:
            top_n = min(5, len(self.best_cars_list))
            print(f"  Top {top_n} cars: ", end="")
            top_scores = [f"{self.best_cars_list[i].brain.score:.2f}" for i in range(top_n)]
            print(", ".join(top_scores))

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
    
    def checkpoint(self):
        """Save a checkpoint with epoch number in filename"""
        if self.checkpoint_every <= 0:
            return
        
        # Only checkpoint at specified intervals
        if self.cur_epoch % self.checkpoint_every == 0 and self.cur_epoch > 0:
            # Create checkpoint filename in the checkpoint folder
            checkpoint_file = f"epoch{self.cur_epoch}.npz"
            checkpoint_path = self.checkpoint_folder / checkpoint_file
            
            self.score_cars()
            print(f"💾 Checkpoint: Saving to {checkpoint_path}")
            np.savez(checkpoint_path, **self.best_cars_list[0].brain.get_parameters())
    
    def checkpoint_best_score(self):
        """Save a checkpoint when a new best score is achieved"""
        if len(self.best_cars_list) == 0:
            return
        
        current_best_score = self.best_cars_list[0].brain.score
        
        # Check if this is a new best score
        if current_best_score > self.best_score_ever:
            old_best = self.best_score_ever
            self.best_score_ever = current_best_score
            
            # Create checkpoint filename with best score in the checkpoint folder
            checkpoint_file = f"best_score{current_best_score:.2f}.npz"
            checkpoint_path = self.checkpoint_folder / checkpoint_file
            
            print(f"🏆 New best score: {current_best_score:.2f} (previous: {(old_best if old_best > float('-inf') else float('-inf')):.2f}")
            print(f"💾 Saving best model to {checkpoint_path}")
            np.savez(checkpoint_path, **self.best_cars_list[0].brain.get_parameters())

    def save(self, file):
        self.score_cars()
        self.log_epoch_stats()  # Log final statistics
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
            
            # Check if all cars have stopped (early stopping)
            all_cars_stopped = all(not c.running for c in self.sprite_list)
            if all_cars_stopped and self.pustene:
                print(f"  ⚡ Early stop: All cars stopped at {self.total_time:.2f}s (max={self.max_time}s)")
                self.running = False


        # enco jako if total time> max time - new epoch
        # ted musím doresit epochy, kolize a save a load
        if (self.total_time > self.max_time or not self.running) and self.pustene:
            self.running = False # timer pro epochu

            if self.cur_epoch < self.pocet_generaci:
                print(f"-----epocha {self.cur_epoch}------------------------")
                #for sprite in self.sprite_list:
                #    print(sprite.brain.parameters)
                #print(f"----------------------------------------------------")
                self.setup_next_epoch()
            else:
                self.pustene = False
                print(f"-----Final Results (Epoch {self.cur_epoch})--------")
                self.autosave()


        #print(f"total time: {self.total_time}")
        #print(f"setting, pocet aut: {self.pocet_aut}, pocet generaci: {self.pocet_generaci}, epoch: {self.epoch}")
        #print(f"vnitrni data: len sprite list: {len(self.sprite_list)}, len sprite running: {len(self.sprite_running)}")