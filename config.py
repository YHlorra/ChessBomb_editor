"""
Configuration module for Chess Bomb Editor
Contains all constants, settings, and resource paths
"""

import sys
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for development and PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return str(base_path / relative_path)

# Resource paths
FONT_PATH = get_resource_path("assets/font")
PIECES_PATH = get_resource_path("assets/peices")
ICON_PATH = get_resource_path("assets/icon")
SKULL_PATH = get_resource_path("assets/skull")

# Skull types
WHITE_SKULL = 1
GRAY_SKULL = 2
BOSS_SKULL = 3

# Chess piece types
PAWN = 'P'
KNIGHT = 'N'
BISHOP = 'B'
ROOK = 'R'
QUEEN = 'Q'
KING = 'K'

# Piece names in Chinese
PIECE_NAMES = {
    PAWN: "兵",
    KNIGHT: "马",
    BISHOP: "象",
    ROOK: "车",
    QUEEN: "皇后",
    KING: "王"
}

# Skull colors
SKULL_COLORS = {
    WHITE_SKULL: (255, 255, 255),
    GRAY_SKULL: (150, 150, 150),
    BOSS_SKULL: (100, 0, 0)
}

# Enhanced UI Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
LIGHT_BROWN = (222, 184, 135)
DARK_BROWN = (139, 69, 19)
LIGHT_GREEN = (144, 238, 144)
DARK_GREEN = (0, 128, 0)
LIGHT_RED = (255, 182, 193)
DARK_RED = (139, 0, 0)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)
HIGHLIGHT_COLOR = (255, 255, 0, 128)
HOVER_COLOR = (255, 165, 0)

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BOARD_SIZE = 360
CELL_SIZE = BOARD_SIZE // 8

# UI Layout
CONTROL_PANEL_X = 390
CONTROL_PANEL_Y = 70
CONTROL_PANEL_WIDTH = WINDOW_WIDTH - CONTROL_PANEL_X - 30
CONTROL_PANEL_HEIGHT = 450

INFO_PANEL_X = 30
INFO_PANEL_Y = 520
INFO_PANEL_WIDTH = WINDOW_WIDTH - 60
INFO_PANEL_HEIGHT = 180

# Enhanced UI Elements
BORDER_RADIUS = 12
PANEL_BORDER_WIDTH = 3
BUTTON_BORDER_WIDTH = 2
CELL_BORDER_WIDTH = 2
HOVER_SCALE = 1.05
ANIMATION_SPEED = 200  # milliseconds

# Solver settings
DEFAULT_BEAM_WIDTH = 15
DEFAULT_MAX_DEPTH = 20

# Font settings
DEFAULT_FONT_SIZE = 24
TITLE_FONT_SIZE = 30