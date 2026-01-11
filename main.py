import pygame, sys, os
import argparse
import math
import numpy as np
from scenes.MenuScene import MenuScene
from scenes.SceneManager import SceneManager
from scenes.MapEditorScene import MapEditor
from scenes.PlayGameScene import PlayGame
from scenes.TrainingScene import Training
from core.TextureAtlas import TextureAtlas
from core.CsvTilemap import CsvTileMap
from scenes.DuelScene import DuelScene

# Parse command-line arguments
parser = argparse.ArgumentParser(description='AI Cart Racing Engine')
parser.add_argument('--headless', action='store_true', 
                    help='Run in headless mode (no rendering, faster training)')
parser.add_argument('--map', type=str, default='DefaultRace',
                    help='Map name to use for training (e.g., DefaultRace, DefaultReset, or custom map name)')
parser.add_argument('--pocet_aut', type=int, default=100,
                    help='Number of cars (individuals) per generation')
parser.add_argument('--pocet_generaci', type=int, default=10,
                    help='Number of generations to simulate')
parser.add_argument('--max_time', type=int, default=5,
                    help='Maximum time per epoch in seconds')
parser.add_argument('--cars_to_next', type=int, default=20,
                    help='Number of best cars to keep as elite for next generation')
parser.add_argument('--save_as', type=str, default='userbrain.npz',
                    help='Filename to save the best brain (saved to UserData/SAVES/)')
parser.add_argument('--load_from', type=str, default=None,
                    help='Optional: Load existing brain from file (in UserData/SAVES/) to continue training')
parser.add_argument('--checkpoint_every', type=int, default=0,
                    help='Save checkpoint every N epochs (0 = disabled, e.g., 10 = save every 10 epochs)')
parser.add_argument('--cars_to_next_decay', type=float, default=1.0,
                    help='Decay factor for elite count per epoch (1.0 = no decay, 0.95 = 5%% reduction per epoch)')
parser.add_argument('--speed', type=float, default=100.0,
                    help='Time acceleration multiplier for headless mode (default: 100.0, e.g., 100.0 = 100x faster, 1000.0 = 1000x faster)')
args = parser.parse_args()

# Problematickej mixer resim pres SDL, ted vypnuto
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
# Set headless video driver if headless mode is enabled
if args.headless:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    print("Running in headless mode - rendering disabled for faster training")

## Start pygame + start modulů!
pygame.init()

######################################################################
# VS studio na windwos:
# tvorba virtual env: python3 -m venv .venv  
# aktivace linux: source .venv/bin/activate
# ctrl+shift+P - vybrat interpeter, ten nainstalovej - venv
# ve visual studio code F1 - pyhton select interpreter
# ############################################################
############################################################
############################################################
############################################################
### Students part:

#from AI_engines.AIbrain_TeamName import AIbrain_TeamName as trainbrain
from AI_engines.AIbrain_linear_SNOW import AIbrain_linear_SNOW as trainbrain



#### Students part end here.
############################################################
############################################################


# Konstatny
from constants import WIDTH, HEIGHT, FPS, tile_path_png, tile_path_xml, tile_path_default_setting_csv, TILESIZE
from constants import tilecicle, EMPTY_TILE, car_path_png, car_path_xml


# Nastaveni okna aj.
# In headless mode, we still create a surface but won't render it
screen = pygame.display.set_mode((WIDTH, HEIGHT))
if not args.headless:
    pygame.display.set_caption("AI engines training")

# Grafika - preload dat
atlas = TextureAtlas(tile_path_png, tile_path_xml, scale_to=(TILESIZE, TILESIZE), scale_mode="smooth")
atlas.load()
atlas.convert_all()
tmap = CsvTileMap(
    atlas,
    tile_path_default_setting_csv,
    tile_w=TILESIZE,
    tile_h=TILESIZE,
    base_tile=EMPTY_TILE,   #  základní výplň mapy
    empty_symbol="."            #  POZOR! prázdné buňky v CSV znamenají „jen base“
)
tmap.prerender()

# vehicle:

vehicles_atlas = TextureAtlas(
    image_path=car_path_png,
    xml_path=car_path_xml,
    add_alias_without_ext=True,      # pak můžete psát jméno bez .png
    scale_to=(64, 128),              # nebo None pokud nechcete škálovat
    scale_mode="smooth",
)

vehicles_atlas.load()
vehicles_atlas.convert_all()


# hodiny - FPS CLOCK / heart rate
clock = pygame.time.Clock()

# Kolecke spritů
my_sprites = pygame.sprite.Group()

# start herní smyčky:
running = True

# Scene manager - rídí běh hry. to jaká "scéna" polozka menu je právě načtená
# scene manager a provtní setting  - registruji jak manager do scény tak scénu do managera
# SCENE manager je shell pro scény
scene_manager = SceneManager(screen)
scene_manager.set_cur_tmap(tmap) # pred pridáním scéne, jde do scén pak!
scene_manager.set_atlas_tmap(atlas)
scene_manager.set_default_tmap_name(tile_path_default_setting_csv)
scene_manager.set_TILESIZE(TILESIZE)
scene_manager.set_vehicle_atlas(vehicles_atlas)

# registrace polozek menu, vzdy  registruji do scene manager a manager naopak do scény
scene_manager.add_menu(MenuScene(scene_manager))
scene_manager.add_mapeditor(MapEditor(scene_manager, tilecicle))
scene_manager.add_playgame(PlayGame(scene_manager))
scene_manager.add_training(Training(scene_manager))
scene_manager.add_brain(trainbrain)

scene_manager.add_duel(DuelScene(scene_manager))

# Determine map path
if args.map in ("DefaultRace", "DefaultReset"):
    map_path = f"DefaultSettings/{args.map}.csv"
else:
    map_path = f"UserData/{args.map}.csv"

# Switch to training scene
scene_manager.set_training(map_path)

# In headless mode, automatically start training with provided parameters
if args.headless:

    
    # Set training parameters
    training_scene = scene_manager.training_scene
    training_scene.input_data = {
        "pocet_aut": args.pocet_aut,
        "pocet_generaci": args.pocet_generaci,
        "max_time": args.max_time,
        "cars_to_next": args.cars_to_next,
        "save_as": args.save_as,
        "load_from": args.load_from if args.load_from else "userbrain.npz",
        "checkpoint_every": args.checkpoint_every,
        "cars_to_next_decay": args.cars_to_next_decay
    }
    
    # Update car manager with new parameters
    training_scene.cars_manager.setup(**training_scene.input_data)
    training_scene.cars_manager.add_defaultbrain(scene_manager.get_brain())
    
    # If load_from is specified, load existing brain
    if args.load_from:
        print(f"Loading existing brain from {args.load_from}")
        training_scene.cars_manager.load(args.load_from)
    else:
        # Start fresh training
        print(f"Starting training with parameters:")
        print(f"  Map: {args.map}")
        print(f"  Cars per generation: {args.pocet_aut}")
        print(f"  Generations: {args.pocet_generaci}")
        print(f"  Max time per epoch: {args.max_time}s (simulation time)")
        print(f"  Elite cars: {args.cars_to_next}")
        print(f"  Save as: {args.save_as}")
        print(f"  Speed multiplier: {args.speed}x (simulation time advances {args.speed}x faster than real time)")
        training_scene.start()



# cyklus udrzujici okno v chodu
while running:
    # FPS kontrola / jeslti bezi dle rychlosti!
    # In headless mode, don't limit FPS - run as fast as possible
    if args.headless:
        # In headless mode, use accelerated time to simulate much faster
        # Use multiple steps with stable dt to maintain accuracy while accelerating
        # Allow larger dt_per_step for very high speeds to reduce computational overhead
        if args.speed >= 1000:
            max_dt_per_step = 1.0 / 10.0  # Larger dt for very high speeds (10 FPS equivalent)
        elif args.speed >= 100:
            max_dt_per_step = 1.0 / 20.0  # Medium dt for high speeds (20 FPS equivalent)
        else:
            max_dt_per_step = 1.0 / 30.0  # Smaller dt for lower speeds (30 FPS equivalent)
        
        target_dt_per_frame = (1.0 / 60.0) * args.speed  # Total time to advance this frame
        steps_needed = max(1, math.ceil(target_dt_per_frame / max_dt_per_step))  # Number of steps needed
        dt_per_step = target_dt_per_frame / steps_needed  # Actual dt per step
        
        clock.tick(0)  # tick(0) = no FPS limit
        
        # Process events once per frame (not every step) for efficiency
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif (not scene_manager.is_active()):
                running = False
                break
        
        if not running:
            break
        
        # Run multiple steps per frame to advance simulation time faster
        keys = pygame.key.get_pressed()  # Get keys once per frame
        for _ in range(steps_needed):
            # Update with stable time step
            scene_manager.update(dt_per_step, keys)
            
            # Check if training is complete and exit
            training_scene = scene_manager.training_scene
            if hasattr(training_scene, 'cars_manager') and training_scene.cars_manager.pustene == False:
                print("Training completed! Exiting...")
                running = False
                break
            
            if not running:
                break
    else:
        # Normal mode with real-time simulation
        dt = clock.tick(FPS)/1000.0
        
        # Eventy a zavření okna:
        for event in pygame.event.get():
            # print(event) - pokud potrebujete info co se zmacklo.
            if event.type == pygame.QUIT:
                running = False
                break
            elif (not scene_manager.is_active()):
                running = False
                break
            else:
                # eventy prebrat jen kdyz orapvdu nezavírám, predtím musí být vzdy jinak break!!!
                scene_manager.event(event) # herní eventy handluje scene_manager který je predává dle potřeby.

        if not running:
            break

        # Update
        keys = pygame.key.get_pressed()
        scene_manager.update(dt, keys)

        # Render
        scene_manager.draw()
        pygame.display.flip()

print("Vypinani")
# POZOR! pokud chceme ukozit a pustit proces quitu ta NESMIME na linuxu zavolat event ci draw a nebo flip!!
# mením tím pak i šablonu do buduoucna - jank bude vyset process
# novinka, na linuxu obcas blokuje zavrení okna kdyz neukončím okn
# a další problém je kdyz vykrelsím do screnu ci udleám update proto dávám break
try:
    pygame.display.quit()
except Exception:
    pass

try:
    pygame.mixer.fadeout(200)
    pygame.time.delay(210)
    pygame.mixer.stop()
    pygame.mixer.quit()
except Exception:
    pass

pygame.quit()
sys.exit(0)
