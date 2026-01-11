# core/headless_loader.py
"""
Lightweight loader for headless training mode.
Loads CSV tilemaps and creates collision masks without requiring texture atlases.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import csv


class HeadlessTileMap:
    """
    Minimal tilemap loader that only reads the CSV grid.
    No textures, no rendering - just the grid data for collision detection.
    """
    
    def __init__(self, csv_path: str | Path, tile_size: int):
        self.csv_path = Path(csv_path)
        self.grid: List[List[str]] = []
        self.tile_size = tile_size
        self.width_tiles = 0
        self.height_tiles = 0
        
    def load(self) -> None:
        """Load the CSV grid."""
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            grid = []
            for row in reader:
                if not row:
                    continue
                items = [cell.strip() for cell in row]
                grid.append(items)
        
        if not grid:
            raise ValueError(f"CSV '{self.csv_path}' is empty.")
        
        # Validate consistent row lengths
        w0 = len(grid[0])
        for i, r in enumerate(grid):
            if len(r) != w0:
                raise ValueError(f"Row {i} has different length ({len(r)}) than first row ({w0}).")
        
        self.grid = grid
        self.height_tiles = len(grid)
        self.width_tiles = len(grid[0]) if grid else 0
    
    @property
    def width_px(self) -> int:
        return self.width_tiles * self.tile_size
    
    @property
    def height_px(self) -> int:
        return self.height_tiles * self.tile_size
