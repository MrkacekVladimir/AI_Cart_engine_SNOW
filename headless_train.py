#!/usr/bin/env python3
"""
Headless Training Mode for AI Racing Cars

Run AI training without GUI rendering for dramatically faster generation cycles.
Uses tick-based generations for maximum speed.

Usage:
    python headless_train.py --map UserData/my_map.csv --cars 50 --generations 100 --ticks 1200 --best 10 --save my_save.npz

"""
import argparse
import importlib
import os
import sys
import time

# Set dummy video driver BEFORE importing pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import numpy as np

from constants import TILESIZE, tilesides
from core.headless_loader import HeadlessTileMap
from core.car_manager import HeadlessCarManager
from my_sprites.block import Blocks


def load_brain_class(brain_name: str):
    """
    Dynamically load a brain class from AI_engines module.
    
    Args:
        brain_name: Name like 'AIbrain_best_sofar' or 'best_sofar'
    
    Returns:
        Brain class
    """
    # Normalize name
    if not brain_name.startswith("AIbrain_"):
        brain_name = f"AIbrain_{brain_name}"
    
    module_name = f"AI_engines.{brain_name}"
    
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Error: Brain module '{module_name}' not found.")
        print("Available brains:")
        for f in os.listdir("AI_engines"):
            if f.startswith("AIbrain_") and f.endswith(".py"):
                print(f"  - {f[:-3]}")
        sys.exit(1)
    
    # Find the brain class in the module
    # Convention: class name is similar to module name but may vary
    brain_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and attr_name.startswith("AIbrain"):
            brain_class = attr
            break
    
    if brain_class is None:
        print(f"Error: No AIbrain class found in module '{module_name}'")
        sys.exit(1)
    
    return brain_class


def training_callback(epoch, best_score, avg_score, ticks, epoch_time):
    """Callback for printing training progress."""
    print(f"[Gen {epoch:4d}] Best: {best_score:7.1f} | Avg: {avg_score:7.1f} | Ticks: {ticks:5d} | Time: {epoch_time:.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Headless AI Car Training - Fast generation without GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--map", "-m",
        type=str,
        default="DefaultSettings/DefaultReset.csv",
        help="Path to CSV map file"
    )
    
    parser.add_argument(
        "--cars", "-c",
        type=int,
        default=50,
        help="Number of cars per generation"
    )
    
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=100,
        help="Number of generations to train"
    )
    
    parser.add_argument(
        "--ticks", "-t",
        type=int,
        default=1200,
        help="Max ticks (frames) per generation (1200 ticks ≈ 20s at 60fps)"
    )
    
    parser.add_argument(
        "--best", "-b",
        type=int,
        default=10,
        help="Number of best cars to keep for next generation"
    )
    
    parser.add_argument(
        "--brain",
        type=str,
        default="AIbrain_best_sofar",
        help="Brain module to use (e.g., 'AIbrain_best_sofar' or 'best_sofar')"
    )
    
    parser.add_argument(
        "--save", "-s",
        type=str,
        default="headless_output.npz",
        help="Output filename for trained model (in UserData/SAVES/)"
    )
    
    parser.add_argument(
        "--load", "-l",
        type=str,
        default=None,
        help="Load existing model to continue training (in UserData/SAVES/)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-generation output"
    )
    
    args = parser.parse_args()
    
    # Initialize pygame in headless mode
    pygame.init()
    
    # Create a small dummy display (required for some pygame functions)
    pygame.display.set_mode((1, 1))
    
    print("=" * 60)
    print("HEADLESS TRAINING MODE")
    print("=" * 60)
    print(f"Map:         {args.map}")
    print(f"Cars:        {args.cars}")
    print(f"Generations: {args.generations}")
    print(f"Ticks/Gen:   {args.ticks}")
    print(f"Best kept:   {args.best}")
    print(f"Brain:       {args.brain}")
    print(f"Save to:     UserData/SAVES/{args.save}")
    if args.load:
        print(f"Load from:   UserData/SAVES/{args.load}")
    print("=" * 60)
    print()
    
    # Load map
    print("Loading map...", end=" ", flush=True)
    tilemap = HeadlessTileMap(args.map, TILESIZE)
    tilemap.load()
    print(f"OK ({tilemap.width_tiles}x{tilemap.height_tiles} tiles)")
    
    # Create collision blocks
    print("Building collision mask...", end=" ", flush=True)
    blocks = Blocks(TILESIZE, 255, tilemap.grid, tilesides)
    blocks.construct_mask_only()
    print("OK")
    
    # Load brain class
    print(f"Loading brain '{args.brain}'...", end=" ", flush=True)
    brain_class = load_brain_class(args.brain)
    print(f"OK ({brain_class.__name__})")
    
    # Create car manager
    manager = HeadlessCarManager(
        pocet_aut=args.cars,
        pocet_generaci=args.generations,
        max_ticks=args.ticks,
        cars_to_next=args.best,
        save_as=args.save,
        load_from=args.load or args.save
    )
    manager.add_defaultbrain(brain_class)
    
    # Start training
    print()
    print("Starting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    if args.load:
        manager.load(args.load)
    else:
        manager.start()
    
    # Custom training loop with progress output
    # dt=1/60 is just a physics multiplier, NOT a frame limiter
    # Training runs at full CPU speed (no throttling)
    callback = None if args.quiet else training_callback
    
    total_time = manager.run_training(blocks, dt=1/60, callback=callback)
    
    print("-" * 60)
    print()
    print(f"Training complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time/generation: {total_time/args.generations:.2f}s")
    print(f"Saved to: UserData/SAVES/{args.save}")
    
    # Final score
    manager.score_cars()
    if manager.best_cars_list:
        print(f"Best final score: {manager.best_cars_list[0].brain.score:.1f}")
    
    pygame.quit()


if __name__ == "__main__":
    main()
