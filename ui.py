"""
UI module for Chess Bomb Editor
Contains Pygame rendering and event handling
"""

import pygame
import os
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from config import (
    FONT_PATH, PIECES_PATH, SKULL_PATH,
    WHITE_SKULL, GRAY_SKULL, BOSS_SKULL,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    PIECE_NAMES, SKULL_COLORS,
    BLACK, WHITE, GRAY, LIGHT_GRAY, DARK_GRAY, LIGHT_BROWN, DARK_BROWN,
    LIGHT_GREEN, DARK_GREEN, LIGHT_RED, DARK_RED, LIGHT_BLUE, DARK_BLUE,
    HIGHLIGHT_COLOR, HOVER_COLOR,
    WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE, CELL_SIZE,
    CONTROL_PANEL_X, CONTROL_PANEL_Y, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT,
    INFO_PANEL_X, INFO_PANEL_Y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT,
    DEFAULT_FONT_SIZE, TITLE_FONT_SIZE,
    BORDER_RADIUS, PANEL_BORDER_WIDTH, BUTTON_BORDER_WIDTH, CELL_BORDER_WIDTH
)
from board import ATTACK_PATTERNS
from solver import solve_with_alns, validate_board_and_pieces, format_solution


class BoardEditor:
    """Main board editor class using Pygame"""
    
    def __init__(self):
        # Initialize tkinter root window
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

        # Initialize pygame
        pygame.init()
        self.solution_ready = False
        self.solution = None
        self.solving = False
        self.solution_message = ""

        # Set up window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess Bomb编辑器")

        # Load fonts
        try:
            self.font = pygame.font.Font(f"{FONT_PATH}/simsun.ttc", DEFAULT_FONT_SIZE)
            self.title_font = pygame.font.Font(f"{FONT_PATH}/simsun.ttc", TITLE_FONT_SIZE)
        except:
            self.font = pygame.font.Font(None, DEFAULT_FONT_SIZE)
            self.title_font = pygame.font.Font(None, TITLE_FONT_SIZE)

        # Attack preview
        self.show_attack_preview = False

        # Board data
        self.board_data = np.zeros((8, 8), dtype=int)
        self.current_skull_type = WHITE_SKULL
        self.available_pieces = {
            PAWN: 0, KNIGHT: 0,
            BISHOP: 0, ROOK: 0,
            QUEEN: 0, KING: 0
        }

        # UI state
        self.info_title = "解决方案"
        self.info_messages = []
        
        # Enhanced UI state
        self.hover_pos = None
        self.selected_piece = None
        self.animation_states = {}
        self.last_click_time = 0
        self.double_click_threshold = 300  # milliseconds

        # Load assets
        self._load_assets()

    def _load_assets(self):
        """Load skull and piece images"""
        # Load skull images
        self.skull_images = {}
        try:
            skull_paths = {
                WHITE_SKULL: f"{SKULL_PATH}/white_skull.png",
                GRAY_SKULL: f"{SKULL_PATH}/gray_skull.png",
                BOSS_SKULL: f"{SKULL_PATH}/boss_skull.png"
            }

            for skull_type, path in skull_paths.items():
                if os.path.exists(path):
                    img = pygame.image.load(path)
                    self.skull_images[skull_type] = pygame.transform.scale(
                        img, (CELL_SIZE - 12, CELL_SIZE - 12)
                    )
        except Exception:
            pass

        # Load piece images
        self.piece_images = {}
        try:
            piece_paths = {
                QUEEN: f"{PIECES_PATH}/wQ.svg",
                ROOK: f"{PIECES_PATH}/wR.svg",
                BISHOP: f"{PIECES_PATH}/wB.svg",
                KNIGHT: f"{PIECES_PATH}/wN.svg",
                KING: f"{PIECES_PATH}/wK.svg",
                PAWN: f"{PIECES_PATH}/wP.svg"
            }

            for piece_type, path in piece_paths.items():
                if os.path.exists(path):
                    img = pygame.image.load(path)
                    self.piece_images[piece_type] = pygame.transform.scale(img, (24, 24))
        except Exception:
            pass

    def draw_board(self):
        """Draw the chess board with enhanced visual design"""
        # Draw board background with shadow
        shadow_rect = pygame.Rect(32, 14, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=BORDER_RADIUS)
        
        board_rect = pygame.Rect(28, 10, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, BLACK, board_rect, CELL_BORDER_WIDTH, border_radius=BORDER_RADIUS)

        for row in range(8):
            # Enhanced row numbers with background
            row_bg = pygame.Rect(10, 10 + row * CELL_SIZE, 16, CELL_SIZE)
            pygame.draw.rect(self.screen, LIGHT_GRAY, row_bg, border_radius=4)
            pygame.draw.rect(self.screen, DARK_GRAY, row_bg, 1, border_radius=4)
            
            row_text = self.font.render(str(8 - row), True, BLACK)
            text_rect = row_text.get_rect(center=row_bg.center)
            self.screen.blit(row_text, text_rect)

            for col in range(8):
                # Enhanced column letters with background
                if row == 7:
                    col_bg = pygame.Rect(30 + col * CELL_SIZE, BOARD_SIZE + 16, CELL_SIZE, 16)
                    pygame.draw.rect(self.screen, LIGHT_GRAY, col_bg, border_radius=4)
                    pygame.draw.rect(self.screen, DARK_GRAY, col_bg, 1, border_radius=4)
                    
                    col_text = self.font.render(chr(97 + col), True, BLACK)
                    text_rect = col_text.get_rect(center=col_bg.center)
                    self.screen.blit(col_text, text_rect)

                rect = pygame.Rect(30 + col * CELL_SIZE,
                                   10 + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)

                # Enhanced checkerboard pattern with gradients
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.screen, LIGHT_BROWN, rect)
                    # Add subtle gradient effect
                    gradient_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height // 3)
                    lighter_brown = (min(255, LIGHT_BROWN[0] + 20), 
                                   min(255, LIGHT_BROWN[1] + 20), 
                                   min(255, LIGHT_BROWN[2] + 20))
                    pygame.draw.rect(self.screen, lighter_brown, gradient_rect)
                else:
                    pygame.draw.rect(self.screen, DARK_BROWN, rect)
                    # Add subtle gradient effect
                    gradient_rect = pygame.Rect(rect.x, rect.y + rect.height * 2 // 3, 
                                              rect.width, rect.height // 3)
                    lighter_dark = (min(255, DARK_BROWN[0] + 15), 
                                  min(255, DARK_BROWN[1] + 15), 
                                  min(255, DARK_BROWN[2] + 15))
                    pygame.draw.rect(self.screen, lighter_dark, gradient_rect)

                # Add hover effect
                if self.hover_pos:
                    mouse_x, mouse_y = self.hover_pos
                    if rect.collidepoint(mouse_x, mouse_y):
                        hover_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                        hover_surface.fill(HOVER_COLOR + (80,))
                        self.screen.blit(hover_surface, rect)

                # Draw skulls with enhanced visuals
                skull_type = self.board_data[row, col]
                if skull_type > 0:
                    # Add skull glow effect
                    glow_rect = rect.inflate(-4, -4)
                    glow_color = (*SKULL_COLORS[skull_type], 50)
                    glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=6)
                    self.screen.blit(glow_surface, glow_rect)
                    
                    if skull_type in self.skull_images:
                        img = self.skull_images[skull_type]
                        self.screen.blit(img, (rect.x + 5, rect.y + 5))
                    else:
                        # Enhanced skull drawing with gradient
                        pygame.draw.circle(self.screen,
                                         SKULL_COLORS[skull_type],
                                         (rect.centerx, rect.centery),
                                         CELL_SIZE // 3)
                        # Add highlight
                        highlight_pos = (rect.centerx - CELL_SIZE // 8, rect.centery - CELL_SIZE // 8)
                        pygame.draw.circle(self.screen, WHITE, highlight_pos, CELL_SIZE // 12)

                    # 移除骷髅生命值显示

    def draw_skull_selector(self):
        """Draw enhanced skull type selector"""
        # Draw panel with shadow and rounded corners
        shadow_rect = pygame.Rect(BOARD_SIZE + 64, 14, 320, 120)
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=BORDER_RADIUS)
        
        skull_area = pygame.Rect(BOARD_SIZE + 60, 10, 320, 120)
        pygame.draw.rect(self.screen, LIGHT_GRAY, skull_area, 0, border_radius=BORDER_RADIUS)
        pygame.draw.rect(self.screen, DARK_GRAY, skull_area, PANEL_BORDER_WIDTH, border_radius=BORDER_RADIUS)

        # Enhanced title
        selector_title = self.title_font.render("骷髅类型", True, BLACK)
        title_rect = selector_title.get_rect(center=(skull_area.centerx, skull_area.y + 25))
        self.screen.blit(selector_title, title_rect)

        skull_options = [
            ("白色骷髅", WHITE_SKULL, LIGHT_GRAY),
            ("灰色骷髅", GRAY_SKULL, GRAY),
            ("首领骷髅", BOSS_SKULL, DARK_GRAY)
        ]

        for i, (name, sk_type, bg_color) in enumerate(skull_options):
            option_rect = pygame.Rect(skull_area.x + 20, skull_area.y + 45 + i * 22, 280, 20)

            # Enhanced selection highlight
            if self.current_skull_type == sk_type:
                # Draw selection background
                selection_surface = pygame.Surface((option_rect.width, option_rect.height), pygame.SRCALPHA)
                selection_surface.fill(LIGHT_BLUE + (100,))
                self.screen.blit(selection_surface, option_rect)
                pygame.draw.rect(self.screen, DARK_BLUE, option_rect, 2, border_radius=6)
            else:
                # Add hover effect
                if self.hover_pos and option_rect.collidepoint(self.hover_pos):
                    hover_surface = pygame.Surface((option_rect.width, option_rect.height), pygame.SRCALPHA)
                    hover_surface.fill(HOVER_COLOR + (50,))
                    self.screen.blit(hover_surface, option_rect)
                pygame.draw.rect(self.screen, GRAY, option_rect, 1, border_radius=4)

            # Enhanced skull preview with glow
            skull_preview_rect = pygame.Rect(option_rect.x + 5, option_rect.y + 2, 16, 16)
            if sk_type in self.skull_images:
                img = pygame.transform.scale(self.skull_images[sk_type], (16, 16))
                self.screen.blit(img, skull_preview_rect)
            else:
                # Draw skull with gradient and highlight
                pygame.draw.circle(self.screen, SKULL_COLORS[sk_type],
                                 (skull_preview_rect.centerx, skull_preview_rect.centery), 8)
                highlight_pos = (skull_preview_rect.centerx - 2, skull_preview_rect.centery - 2)
                pygame.draw.circle(self.screen, WHITE, highlight_pos, 2)

            # Enhanced text
            text = self.font.render(name, True, BLACK)
            self.screen.blit(text, (option_rect.x + 28, option_rect.y + 2))
            
            # 移除HP指示器

    def draw_piece_editor(self):
        """Draw enhanced piece count editor"""
        # Draw panel with shadow and rounded corners
        shadow_rect = pygame.Rect(BOARD_SIZE + 64, 144, 320, 280)
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=BORDER_RADIUS)
        
        pieces_area = pygame.Rect(BOARD_SIZE + 60, 140, 320, 280)
        pygame.draw.rect(self.screen, LIGHT_GRAY, pieces_area, 0, border_radius=BORDER_RADIUS)
        pygame.draw.rect(self.screen, DARK_GRAY, pieces_area, PANEL_BORDER_WIDTH, border_radius=BORDER_RADIUS)

        # Enhanced title
        pieces_title = self.title_font.render("可用棋子", True, BLACK)
        title_rect = pieces_title.get_rect(center=(pieces_area.centerx, pieces_area.y + 25))
        self.screen.blit(pieces_title, title_rect)

        piece_info = [
            ("皇后", QUEEN, 9),
            ("战车", ROOK, 5),
            ("主教", BISHOP, 3),
            ("骑士", KNIGHT, 3),
            ("国王", KING, 4),
            ("士兵", PAWN, 1)
        ]

        if not hasattr(self, 'button_rects'):
            self.button_rects = {}

        for i, (name, piece_type, value) in enumerate(piece_info):
            row_rect = pygame.Rect(pieces_area.x + 15, pieces_area.y + 50 + i * 38, 290, 32)
            
            # Enhanced row background with gradient
            pygame.draw.rect(self.screen, WHITE, row_rect, 0, border_radius=6)
            # Add subtle gradient
            gradient_rect = pygame.Rect(row_rect.x, row_rect.y, row_rect.width, row_rect.height // 3)
            lighter_white = (min(255, 240), min(255, 245), min(255, 250))
            pygame.draw.rect(self.screen, lighter_white, gradient_rect, border_radius=6)
            
            # Add hover effect
            if self.hover_pos and row_rect.collidepoint(self.hover_pos):
                hover_surface = pygame.Surface((row_rect.width, row_rect.height), pygame.SRCALPHA)
                hover_surface.fill(HOVER_COLOR + (30,))
                self.screen.blit(hover_surface, row_rect)
            
            pygame.draw.rect(self.screen, GRAY, row_rect, 1, border_radius=6)

            # Enhanced piece image with glow
            piece_rect = pygame.Rect(row_rect.x + 8, row_rect.y + 4, 24, 24)
            if piece_type in self.piece_images:
                img = self.piece_images[piece_type]
                self.screen.blit(img, piece_rect)
            else:
                # Draw piece symbol
                symbol = PIECE_NAMES.get(piece_type, name[0])
                symbol_text = self.font.render(symbol, True, BLACK)
                symbol_rect = symbol_text.get_rect(center=piece_rect.center)
                self.screen.blit(symbol_text, symbol_rect)

            # Enhanced piece name and value
            name_text = self.font.render(name, True, BLACK)
            self.screen.blit(name_text, (row_rect.x + 40, row_rect.y + 2))

            # Enhanced count display with background
            count_bg = pygame.Rect(row_rect.x + 150, row_rect.y + 4, 40, 24)
            pygame.draw.rect(self.screen, LIGHT_BLUE, count_bg, border_radius=4)
            pygame.draw.rect(self.screen, DARK_BLUE, count_bg, 1, border_radius=4)
            
            count_text = self.font.render(str(self.available_pieces[piece_type]), True, BLACK)
            count_rect = count_text.get_rect(center=count_bg.center)
            self.screen.blit(count_text, count_rect)

            # Enhanced decrease button
            minus_rect = pygame.Rect(row_rect.x + 200, row_rect.y + 4, 28, 24)
            button_color = LIGHT_RED if self.available_pieces[piece_type] > 0 else GRAY
            pygame.draw.rect(self.screen, button_color, minus_rect, 0, border_radius=6)
            pygame.draw.rect(self.screen, DARK_RED, minus_rect, BUTTON_BORDER_WIDTH, border_radius=6)
            
            minus_text = self.font.render("−", True, WHITE)  # Using proper minus sign
            minus_rect_center = minus_text.get_rect(center=minus_rect.center)
            self.screen.blit(minus_text, minus_rect_center)

            # Enhanced increase button
            plus_rect = pygame.Rect(row_rect.x + 235, row_rect.y + 4, 28, 24)
            pygame.draw.rect(self.screen, LIGHT_GREEN, plus_rect, 0, border_radius=6)
            pygame.draw.rect(self.screen, DARK_GREEN, plus_rect, BUTTON_BORDER_WIDTH, border_radius=6)
            
            plus_text = self.font.render("+", True, WHITE)
            plus_rect_center = plus_text.get_rect(center=plus_rect.center)
            self.screen.blit(plus_text, plus_rect_center)

            # Store button positions for click detection
            self.button_rects[f"minus_{piece_type}"] = minus_rect
            self.button_rects[f"plus_{piece_type}"] = plus_rect
            
            # Add hover effect for buttons
            if self.hover_pos:
                if minus_rect.collidepoint(self.hover_pos):
                    hover_surface = pygame.Surface((minus_rect.width, minus_rect.height), pygame.SRCALPHA)
                    hover_surface.fill(WHITE + (30,))
                    self.screen.blit(hover_surface, minus_rect)
                elif plus_rect.collidepoint(self.hover_pos):
                    hover_surface = pygame.Surface((plus_rect.width, plus_rect.height), pygame.SRCALPHA)
                    hover_surface.fill(WHITE + (30,))
                    self.screen.blit(hover_surface, plus_rect)

    def draw_action_buttons(self):
        """Draw enhanced action buttons"""
        buttons_y = 470
        button_width = 100
        button_height = 40
        button_spacing = 20
        total_width = button_width * 2 + button_spacing
        buttons_x = (WINDOW_WIDTH - total_width) // 2

        # Enhanced clear button
        clear_button = pygame.Rect(buttons_x, buttons_y, button_width, button_height)
        
        # Draw shadow
        shadow_rect = clear_button.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=8)
        
        # Draw button with gradient effect
        pygame.draw.rect(self.screen, LIGHT_RED, clear_button, 0, border_radius=8)
        pygame.draw.rect(self.screen, DARK_RED, clear_button, BUTTON_BORDER_WIDTH, border_radius=8)
        
        # Add hover effect
        if self.hover_pos and clear_button.collidepoint(self.hover_pos):
            hover_surface = pygame.Surface((clear_button.width, clear_button.height), pygame.SRCALPHA)
            hover_surface.fill(WHITE + (40,))
            self.screen.blit(hover_surface, clear_button)
        
        clear_text = self.font.render("清空棋盘", True, WHITE)
        clear_text_rect = clear_text.get_rect(center=clear_button.center)
        self.screen.blit(clear_text, clear_text_rect)

        # Enhanced solve button
        solve_button = pygame.Rect(buttons_x + button_width + button_spacing, buttons_y, button_width, button_height)
        
        # Draw shadow
        shadow_rect = solve_button.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=8)
        
        # Button color based on state
        if self.solving:
            button_color = GRAY
            border_color = DARK_GRAY
        else:
            button_color = LIGHT_GREEN
            border_color = DARK_GREEN
            
        pygame.draw.rect(self.screen, button_color, solve_button, 0, border_radius=8)
        pygame.draw.rect(self.screen, border_color, solve_button, BUTTON_BORDER_WIDTH, border_radius=8)
        
        # Add hover effect
        if self.hover_pos and solve_button.collidepoint(self.hover_pos) and not self.solving:
            hover_surface = pygame.Surface((solve_button.width, solve_button.height), pygame.SRCALPHA)
            hover_surface.fill(WHITE + (40,))
            self.screen.blit(hover_surface, solve_button)
        
        solve_text = self.font.render("开始求解", True, WHITE)
        solve_text_rect = solve_text.get_rect(center=solve_button.center)
        self.screen.blit(solve_text, solve_text_rect)

        # Store button positions for click detection
        self.clear_button_rect = clear_button
        self.solve_button_rect = solve_button

    def draw_info_panel(self):
        """Draw enhanced information panel"""
        # Draw panel with shadow and rounded corners
        shadow_rect = pygame.Rect(INFO_PANEL_X + 4, INFO_PANEL_Y + 4,
                                 INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, shadow_rect, border_radius=BORDER_RADIUS)
        
        panel_rect = pygame.Rect(INFO_PANEL_X, INFO_PANEL_Y,
                                 INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 0, border_radius=BORDER_RADIUS)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, PANEL_BORDER_WIDTH, border_radius=BORDER_RADIUS)

        # Enhanced title with gradient background
        title_bg = pygame.Rect(INFO_PANEL_X + 10, INFO_PANEL_Y + 10, INFO_PANEL_WIDTH - 20, 35)
        pygame.draw.rect(self.screen, LIGHT_BLUE, title_bg, 0, border_radius=6)
        pygame.draw.rect(self.screen, DARK_BLUE, title_bg, 1, border_radius=6)
        
        title_text = self.title_font.render(self.info_title, True, WHITE)
        title_rect = title_text.get_rect(center=title_bg.center)
        self.screen.blit(title_text, title_rect)

        # Enhanced status display
        status_y = INFO_PANEL_Y + 55
        if self.solving:
            # Animated status text
            import time
            dots = "." * ((int(time.time() * 2) % 4))
            status_text = self.font.render(f"ALNS算法求解中{dots}", True, DARK_BLUE)
            status_bg_color = LIGHT_BLUE
        elif self.solution:
            status_text = self.font.render("✓ 已找到解决方案", True, DARK_GREEN)
            status_bg_color = LIGHT_GREEN
        elif self.solution_message:
            status_text = self.font.render(f"✗ {self.solution_message}", True, DARK_RED)
            status_bg_color = LIGHT_RED
        else:
            status_text = self.font.render("等待开始求解", True, BLACK)
            status_bg_color = LIGHT_GRAY

        # Status background
        status_bg = pygame.Rect(INFO_PANEL_X + 10, status_y, INFO_PANEL_WIDTH - 20, 25)
        pygame.draw.rect(self.screen, status_bg_color, status_bg, 0, border_radius=4)
        pygame.draw.rect(self.screen, DARK_GRAY, status_bg, 1, border_radius=4)
        
        status_rect = status_text.get_rect(center=status_bg.center)
        self.screen.blit(status_text, status_rect)

        # Enhanced separator
        pygame.draw.line(self.screen, DARK_GRAY,
                         (INFO_PANEL_X + 10, INFO_PANEL_Y + 90),
                         (INFO_PANEL_X + INFO_PANEL_WIDTH - 10, INFO_PANEL_Y + 90), 2)

        # Display solution steps with enhanced formatting
        if self.info_messages:
            y_offset = 100
            x_offset = 0
            col_width = INFO_PANEL_WIDTH // 2 - 20
            step_number = 1

            for i, message in enumerate(self.info_messages[:20]):  # Limit to 20 steps
                if i == 10:
                    y_offset = 100
                    x_offset = col_width + 20

                # Step background with alternating colors
                step_bg = pygame.Rect(INFO_PANEL_X + 10 + x_offset, y_offset - 2, 
                                    col_width - 5, 22)
                bg_color = (LIGHT_GRAY if i % 2 == 0 else WHITE)
                pygame.draw.rect(self.screen, bg_color, step_bg, border_radius=3)
                pygame.draw.rect(self.screen, GRAY, step_bg, 1, border_radius=3)

                # Step number
                step_text = self.font.render(f"{step_number}.", True, DARK_BLUE)
                self.screen.blit(step_text, (INFO_PANEL_X + 15 + x_offset, y_offset))
                
                # Step content
                text = self.font.render(message, True, BLACK)
                self.screen.blit(text, (INFO_PANEL_X + 35 + x_offset, y_offset))
                
                y_offset += 25
                step_number += 1

    def draw_ui(self):
        """Draw the complete user interface"""
        self.screen.fill(WHITE)
        self.draw_board()
        self.draw_skull_selector()
        self.draw_piece_editor()
        self.draw_action_buttons()

        # Instructions
        instruction1 = self.font.render("左键点击棋盘放置骷髅", True, BLACK)
        instruction2 = self.font.render("右键点击棋盘清除骷髅", True, BLACK)
        self.screen.blit(instruction1, (30, BOARD_SIZE + 40))
        self.screen.blit(instruction2, (30, BOARD_SIZE + 70))

        self.draw_info_panel()

        if self.solving:
            solving_text = self.font.render("正在使用ALNS算法计算中...", True, RED)
            self.screen.blit(solving_text, (WINDOW_WIDTH // 2 - solving_text.get_width() // 2,
                                           BOARD_SIZE + 110))

    def handle_mouse_click(self, pos, is_right_click=False):
        """Handle mouse clicks"""
        x, y = pos

        # Check board clicks
        if 30 <= x < 30 + BOARD_SIZE and 10 <= y < 10 + BOARD_SIZE:
            col = (x - 30) // CELL_SIZE
            row = (y - 10) // CELL_SIZE

            if is_right_click:
                self.board_data[row, col] = 0
            else:
                self.board_data[row, col] = self.current_skull_type
            return

        # Check skull selector
        skull_area = pygame.Rect(BOARD_SIZE + 60, 10, 320, 120)
        if skull_area.collidepoint(x, y):
            for i, sk_type in enumerate([WHITE_SKULL, GRAY_SKULL, BOSS_SKULL]):
                option_rect = pygame.Rect(skull_area.x + 20, skull_area.y + 45 + i * 25, 280, 20)
                if option_rect.collidepoint(x, y):
                    self.current_skull_type = sk_type
                    return

        # Check piece editor buttons
        if hasattr(self, 'button_rects'):
            for piece_type in [QUEEN, ROOK, BISHOP, KNIGHT, KING, PAWN]:
                minus_key = f"minus_{piece_type}"
                plus_key = f"plus_{piece_type}"
                
                if minus_key in self.button_rects and self.button_rects[minus_key].collidepoint(x, y):
                    if self.available_pieces[piece_type] > 0:
                        self.available_pieces[piece_type] -= 1
                    return
                    
                if plus_key in self.button_rects and self.button_rects[plus_key].collidepoint(x, y):
                    self.available_pieces[piece_type] += 1
                    return

        # Check action buttons
        if hasattr(self, 'clear_button_rect') and self.clear_button_rect.collidepoint(x, y):
            self.board_data = np.zeros((8, 8), dtype=int)
            return

        if hasattr(self, 'solve_button_rect') and self.solve_button_rect.collidepoint(x, y) and not self.solving:
            self.start_solving()

    def start_solving(self):
        """Start solving the puzzle in a separate thread"""
        self.solving = True
        self.solution = None
        self.solution_message = ""
        self.solution_ready = False

        def solve_thread():
            try:
                board = self.board_data.copy()
                available_pieces = self.available_pieces.copy()

                is_valid, message = validate_board_and_pieces(board, available_pieces)
                if not is_valid:
                    self.solution_message = message
                    self.solving = False
                    return

                from board import ChessState
                initial_state = ChessState(board, available_pieces)
                solution = solve_with_alns(initial_state, max_iterations=1000, time_limit=30)

                if solution:
                    try:
                        formatted_solution = format_solution(solution)
                        # 计算棋子使用统计
                        piece_counts = {}
                        for step in formatted_solution:
                            piece = step['piece']
                            piece_counts[piece] = piece_counts.get(piece, 0) + 1
                        
                        self.solution = solution
                        self.solution_ready = True
                        self.formatted_solution = formatted_solution
                        # 使用增强的描述信息
                        self.info_messages = [step['description'] for step in formatted_solution[:10]]
                    except Exception as format_error:
                        self.solution_message = f"格式化解决方案错误: {str(format_error)}"
                else:
                    self.solution_message = "未找到解决方案"
                    
            except Exception as e:
                self.solution_message = f"求解错误: {str(e)}"
            finally:
                self.solving = False

        thread = threading.Thread(target=solve_thread)
        thread.daemon = True
        thread.start()

    def show_solution_window(self):
        """Show solution window"""
        if self.solution:
            # 创建可以独立操作的解决方案窗口
            solution_window = SolutionWindow(self.tk_root, self.solution)
            self.tk_root.update()

    def run(self):
        """Run the main editor loop"""
        clock = pygame.time.Clock()
        running = True

        while running:
            self.screen.fill(WHITE)
            self.draw_ui()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return None

                # Track mouse position for hover effects
                if event.type == pygame.MOUSEMOTION:
                    self.hover_pos = event.pos

                if not self.solving:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        current_time = pygame.time.get_ticks()
                        
                        # Check for double click
                        if event.button == 1:
                            if current_time - self.last_click_time < self.double_click_threshold:
                                # Double click detected - clear the cell
                                self.handle_mouse_click(event.pos, True)
                            else:
                                # Single click
                                self.handle_mouse_click(event.pos)
                            self.last_click_time = current_time
                        elif event.button == 3:
                            self.handle_mouse_click(event.pos, True)

            if self.solution_ready:
                self.show_solution_window()
                self.solution_ready = False

            pygame.display.flip()
            clock.tick(60)  # Increased frame rate for smoother animations

        pygame.quit()
        return None


class SolutionWindow:
    """独立的解决方案显示窗口，可以独立于主界面进行关闭和最小化"""
    
    def __init__(self, parent, solution):
        # 创建窗口
        self.window = tk.Toplevel(parent)
        self.window.title("Chess Bomb - 解决方案详情")
        self.window.geometry("800x600")
        
        # 确保窗口可以独立操作
        self.window.transient(None)  # 不绑定到父窗口
        
        # 确保最小化和关闭按钮正常工作
        self.window.attributes('-topmost', False)  # 不总是显示在顶部
        
        # 绑定关闭事件处理
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 确保不阻塞主窗口
        self.window.grab_release()
        
        self.solution = solution
        self.current_step = 0
        self.formatted_solution = format_solution(solution)

        # Create main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header with summary
        self._create_header(main_frame)
        
        # Create navigation controls
        self._create_navigation(main_frame)
        
        # Create content area with tabs
        self._create_content_area(main_frame)
        
        # Create footer with export options
        self._create_footer(main_frame)

        # Insert solution content
        self._display_solution()

    def _create_header(self, parent):
        """Create header section with solution summary"""
        header_frame = ttk.LabelFrame(parent, text="解决方案摘要", padding="10")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Calculate statistics
        total_moves = len(self.formatted_solution)
        piece_counts = {}
        for step in self.formatted_solution:
            piece = step['piece']
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        # Display summary
        summary_text = f"总步数: {total_moves}  |  "
        summary_text += "  |  ".join([f"{piece}: {count}" for piece, count in piece_counts.items()])
        
        ttk.Label(header_frame, text=summary_text, font=('Arial', 11, 'bold')).pack()
    def _create_navigation(self, parent):
        """Create minimal navigation controls"""
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Only keep basic info label
        ttk.Label(nav_frame, text="解决方案详情", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

    def _create_content_area(self, parent):
        """Create content area with only detailed steps"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for tabs (keep the structure but only one tab)
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Only keep detailed steps tab
        self._create_steps_tab()

    def _create_steps_tab(self):
        """Create detailed steps tab"""
        steps_frame = ttk.Frame(self.notebook)
        self.notebook.add(steps_frame, text="解决方案")
        
        # Create text widget with scrollbar
        text_container = ttk.Frame(steps_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(text_container, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_container, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for better formatting
        self.output_text.tag_configure('header', font=('Arial', 12, 'bold'), foreground='#2c3e50')
        self.output_text.tag_configure('step', font=('Arial', 10, 'bold'), foreground='#2980b9')
        self.output_text.tag_configure('position', font=('Consolas', 10, 'bold'), foreground='#27ae60')
        self.output_text.tag_configure('piece', font=('Arial', 10), foreground='#8e44ad')
        
        # Only keep mousewheel event for scrolling
        self.output_text.bind("<MouseWheel>", self._on_mousewheel)



    def _create_footer(self, parent):
        """Create minimal footer with only close button"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X)
        
        # Only keep close button centered
        ttk.Button(footer_frame, text="关闭", command=self.on_close, width=15).pack(side=tk.RIGHT, padx=10, pady=5)

    def _display_solution(self):
        """Display the complete solution with improved readability"""
        self.output_text.delete(1.0, tk.END)
        
        # 添加更多的文本标签样式
        self.output_text.tag_configure('success', font=('Arial', 10), foreground='#27ae60')
        self.output_text.tag_configure('info', font=('Arial', 10), foreground='#3498db')
        self.output_text.tag_configure('description', font=('Arial', 10), foreground='#2c3e50')
        self.output_text.tag_configure('symbol', font=('Arial', 12))
        
        # 插入标题和摘要
        self.output_text.insert(tk.END, "🏆 Chess Bomb 解决方案 🏆\n\n", 'header')
        self.output_text.insert(tk.END, f"📝 使用 ALNS 算法生成的最优解\n", 'info')
        
        # 解决方案统计信息
        total_moves = len(self.formatted_solution)
        piece_counts = {}
        for step in self.formatted_solution:
            piece = step['piece']
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        self.output_text.insert(tk.END, f"🔢 总步数: {total_moves} 步\n", 'info')
        
        # 棋子使用统计
        self.output_text.insert(tk.END, f"\n🧩 棋子使用统计:\n", 'info')
        for piece, count in sorted(piece_counts.items()):
            piece_symbol = next((s for s, p in {'👑':'皇后', '🚗':'战车', '🛐':'主教', '🐎':'骑士', '⚔️':'士兵'}.items() if p == piece), '')
            self.output_text.insert(tk.END, f"  {piece_symbol} {piece}: {count} 个\n", 'description')
        
        # 分隔线
        self.output_text.insert(tk.END, "\n" + "="*60 + "\n\n", 'info')
        
        # 详细步骤列表（表格形式）
        self.output_text.insert(tk.END, f"📋 解决方案步骤:\n\n", 'header')
        
        # 表头
        self.output_text.insert(tk.END, f"{'步骤':^8}  {'棋子':^12}  {'位置':^8}  {'移动说明':^25}\n", 'info')
        self.output_text.insert(tk.END, f"{'-'*8}  {'-'*12}  {'-'*8}  {'-'*25}\n", 'info')
        
        # 步骤内容
        for i, step in enumerate(self.formatted_solution):
            step_line = f"{step['step']:^8}  {step['symbol']} {step['piece'][:5]:<8}  {step['position']:^8}  {step['description']:^25}\n"
            
            if i > 0 and (i + 1) % 10 == 0 and i < len(self.formatted_solution) - 1:
                self.output_text.insert(tk.END, step_line, 'description')
                self.output_text.insert(tk.END, f"{'-'*8}  {'-'*12}  {'-'*8}  {'-'*25}\n", 'info')
            else:
                self.output_text.insert(tk.END, step_line, 'description')
        
        self.output_text.config(state='disabled')



    def _update_step_display(self):
        """Simplified update method - no step navigation needed"""
        pass





    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.output_text.yview_scroll(-1 * (event.delta // 120), "units")

    def on_close(self):
        """关闭窗口并释放资源"""
        try:
            self.window.destroy()
        except:
            # 忽略已经被销毁或其他异常情况
            pass