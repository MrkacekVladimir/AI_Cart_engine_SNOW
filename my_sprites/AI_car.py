import math
import pygame
from constants import WHITE, SPEED, MAX_SPEED, TILESIZE, TURN_SPEED, BREAK_SPEED, FRICTION_SPEED, RAYCAST_ANGLES
from core.TextureAtlas import TextureAtlas


class HeadlessCar:
    """
    Lightweight car for headless training mode.
    No sprites, no images - just physics and raycasting.
    """
    
    def __init__(self, x, y, width, height, brain, angle=180.0):
        # Position as vector
        self.pos = pygame.math.Vector2(x, y)
        self.width = width
        self.height = height
        
        # Physics
        self.speed = 0.0
        self.angle = angle
        self.max_speed = MAX_SPEED
        
        # Control parameters
        self.acceleration = SPEED
        self.brake = BREAK_SPEED
        self.friction = FRICTION_SPEED
        self.turn_speed = TURN_SPEED
        
        # Raycasting
        self.ray_angles = RAYCAST_ANGLES
        self.ray_distances: list[float] = [0.0 for _ in self.ray_angles]
        
        self.brain = brain
        
        # Logging for brain
        self.logs_distance = 0
        self.logs_dt = 0
        self.no = 100_000
        
        # Running state
        self.running = True
    
    def update(self, dt, blocks):
        """Update car physics and raycasting (no rendering)."""
        if not self.running:
            return
        
        self.logs_dt += dt
        prev_pos = self.pos.copy()
        
        # Brain decision
        self.brain.passcardata(self.pos[0], self.pos[1], self.speed)
        decision = self.brain.decide(self.ray_distances)
        
        # Accelerate
        if decision[0] > 0.5:
            self.speed += self.acceleration * dt
        
        # Brake
        if decision[1] > 0.5:
            self.speed -= self.brake * dt
        
        if self.speed < 0:
            self.speed = 0
        
        # Friction
        if not decision[0] > 0.5 and not decision[1] > 0.5:
            if self.speed > 0:
                self.speed -= self.friction * dt
                if self.speed < 0:
                    self.speed = 0
        
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        
        # Turning
        if self.speed > 0:
            if decision[2] > 0.5:  # left
                self.angle += self.turn_speed * dt
            if decision[3] > 0.5:  # right
                self.angle -= self.turn_speed * dt
        
        # Movement
        direction = pygame.math.Vector2(1, 0).rotate(-self.angle)
        self.pos += direction * self.speed * dt
        
        # Raycasting
        self.update_rays(blocks)
        
        # Distance tracking
        frame_dist_px = self.pos.distance_to(prev_pos)
        frame_dist_tiles = frame_dist_px / TILESIZE
        self.logs_distance += frame_dist_tiles
        
        self.brain.calculate_score(self.logs_distance, self.logs_dt, self.no)
    
    def update_rays(self, blocks, step=4.0):
        """
        Calculate distance to nearest block for each ray angle.
        
        Args:
            blocks: Blocks object with collision mask
            step: Ray step size in pixels (larger = faster but less accurate)
                  Default 4.0 gives ~2x speedup with minimal accuracy loss
        """
        if blocks.mask is None:
            return
        
        mask = blocks.mask
        w, h = blocks.image.get_size()
        max_dist_px = math.hypot(w, h)
        
        results = []
        
        for rel_angle in self.ray_angles:
            world_angle = self.angle + rel_angle
            direction = pygame.math.Vector2(1, 0).rotate(-world_angle)
            pos = self.pos.copy()
            
            dist = 0.0
            hit = False
            
            while dist < max_dist_px:
                pos += direction * step
                dist += step
                
                ix = int(pos.x)
                iy = int(pos.y)
                
                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    break
                
                if mask.get_at((ix, iy)):
                    hit = True
                    break
            
            if hit:
                dist_tiles = dist / TILESIZE
            else:
                dist_tiles = max_dist_px / TILESIZE
            
            results.append(dist_tiles)
        
        self.ray_distances = results
    
    def check_collision(self, blocks):
        """Check if car position collides with wall mask."""
        if blocks.mask is None:
            return False
        
        ix = int(self.pos.x)
        iy = int(self.pos.y)
        w, h = blocks.image.get_size()
        
        # Out of bounds = collision
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return True
        
        return blocks.mask.get_at((ix, iy))


class AI_car(pygame.sprite.Sprite):
    atlas: TextureAtlas | None = None

    @classmethod
    def set_atlas(cls, atlas: TextureAtlas) -> None:
        cls.atlas = atlas

    def __init__(self, x, y, width, height, brain, angle = 180.0):
        super().__init__()

        if AI_car.atlas is None:
            raise RuntimeError("car.atlas není nastaven, zavolej car.set_atlas(...)")

        # pozice jako vektor (float), pracujeme s centrem auta
        self.pos = pygame.math.Vector2(x, y)

        # fyzika auta
        self.speed = 0.0
        self.angle = angle          # 0 = doprava
        self.max_speed = MAX_SPEED

        # parametry řízení
        self.acceleration = SPEED
        self.brake = BREAK_SPEED
        self.friction = FRICTION_SPEED
        self.turn_speed = TURN_SPEED

        # základní obrázek z atlasu (nahoru)
        base_image = AI_car.atlas.get("car_black_1.png")

        if (width, height) != base_image.get_size():
            self.original_image = pygame.transform.smoothscale(base_image, (width, height))
        else:
            self.original_image = base_image

        # první orientace
        self.image = pygame.transform.rotate(self.original_image, self.angle - 90)
        self.rect = self.image.get_rect(center=self.pos)

        # raycasting – relativní úhly vůči směru auta
        self.ray_angles = RAYCAST_ANGLES
        self.ray_distances: list[float] = [0.0 for _ in self.ray_angles]

        self.brain = brain

        # logs pro brain pak:
        self.logs_distance = 0
        self.logs_dt = 0
        self.no = 100_000

        # zda závodí ( tedy nenarazilo do zdi)
        self.running = True

    def get_rel_pos(self):
        return (
            self.pos.x / (TILESIZE * 10),
            self.pos.y / (TILESIZE * 10)
        )

    def update(self, dt, keys, blocks):
        if not self.running:
            return None

        self.logs_dt += dt
        prev_pos = self.pos.copy()

        # řízení rychlosti
        self.brain.passcardata(self.pos[0], self.pos[1], self.speed)
        decision = self.brain.decide(self.ray_distances)

        if decision[0]>0.5:# up
            self.speed += self.acceleration * dt

        if decision[1]>0.5:# down
            self.speed -= self.brake * dt

        if self.speed < 0:
            self.speed = 0

        if not decision[0]>0.5 and not decision[1]>0.5:
            if self.speed > 0:
                self.speed -= self.friction * dt
                if self.speed < 0:
                    self.speed = 0

        if self.speed > self.max_speed:
            self.speed = self.max_speed

        if self.speed > 0:
            if decision[2]>0.5:#left
                self.angle += self.turn_speed * dt
            if decision[3]>0.5:#right
                self.angle -= self.turn_speed * dt

        # pohyb
        direction = pygame.math.Vector2(1, 0).rotate(-self.angle)
        self.pos += direction * self.speed * dt

        # obrázek
        self.image = pygame.transform.rotate(self.original_image, self.angle - 90)
        self.rect = self.image.get_rect(center=self.pos)

        # raycasting
        self.update_rays(blocks)

        # vyhodntím uejtou vzdálenost:
        frame_dist_px = self.pos.distance_to(prev_pos)  # v pixelech
        frame_dist_tiles = frame_dist_px / TILESIZE  # v dlaždicích
        self.logs_distance += frame_dist_tiles # kvuli porovnatelnosti.

        # dočasně: vytisknout vzdálenosti v dlaždicích
        # (bacha, bude to spamovat konzoli, ale na test je to dobré)
        # print([round(d, 2) for d in self.ray_distances])
        self.brain.calculate_score(self.logs_distance, self.logs_dt, self.no)


    def update_rays(self, blocks):
        """Spočítá vzdálenost k nejbližšímu bloku pro každou ray_angle."""
        if blocks.mask is None:
            return

        mask = blocks.mask
        w, h = blocks.image.get_size()
        max_dist_px = math.hypot(w, h)  # maximální vzdálenost (diagonála mapy)

        results = []

        for rel_angle in self.ray_angles:
            world_angle = self.angle + rel_angle

            # stejná konvence jako pro pohyb
            direction = pygame.math.Vector2(1, 0).rotate(-world_angle)
            pos = self.pos.copy()

            step = 1.0  # px krok, tenký blok = 1 px, takže krok nesmí být větší
            dist = 0.0
            hit = False

            while dist < max_dist_px:
                pos += direction * step
                dist += step

                ix = int(pos.x)
                iy = int(pos.y)

                # mimo mapu
                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    break

                if mask.get_at((ix, iy)):
                    hit = True
                    break

            if hit:
                dist_tiles = dist / TILESIZE
            else:
                dist_tiles = max_dist_px / TILESIZE  # „nic nenašel“

            results.append(dist_tiles)

        self.ray_distances = results
