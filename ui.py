"""
UI module for Chess Bomb Editor
Contains Pygame rendering and event handling
"""

import pygame
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
            self.title_font = pygame.font.Font(f"{FONT_PATH}/simsun.ttc", TITLE_FONT_SIZE, bold=True)
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
        except Exception as e:
            print(f"加载骷髅图像时出错: {e}")

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
        except Exception as e:
            print(f"加载棋子图像时出错: {e}")

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

                    # Enhanced skull health display
                    hp_bg = pygame.Rect(rect.centerx - 12, rect.centery - 12, 24, 20)
                    pygame.draw.rect(self.screen, WHITE, hp_bg, border_radius=4)
                    pygame.draw.rect(self.screen, BLACK, hp_bg, 1, border_radius=4)
                    
                    hp_text = self.font.render(str(skull_type), True, BLACK)
                    text_rect = hp_text.get_rect(center=hp_bg.center)
                    self.screen.blit(hp_text, text_rect)

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
            ("白色骷髅 (1HP)", WHITE_SKULL, LIGHT_GRAY),
            ("灰色骷髅 (2HP)", GRAY_SKULL, GRAY),
            ("首领骷髅 (3HP)", BOSS_SKULL, DARK_GRAY)
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
            
            # Add HP indicator
            hp_text = self.font.render(f"HP: {sk_type}", True, DARK_GRAY)
            self.screen.blit(hp_text, (option_rect.x + 220, option_rect.y + 2))

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
                img = pygame.piece_images[piece_type]
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
            
            value_text = self.font.render(f"值:{value}", True, DARK_GRAY)
            self.screen.blit(value_text, (row_rect.x + 40, row_rect.y + 16))

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
                    print("\n最终解决方案:")
                    formatted_solution = format_solution(solution)
                    for step in formatted_solution:
                        print(f"步骤 {step['step']}: 在 {step['position']} 放置 {step['piece']}")
                    
                    self.solution = solution
                    self.solution_ready = True
                    self.info_messages = [step['notation'] for step in formatted_solution[:10]]
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
            SolutionWindow(self.tk_root, self.solution)
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
    """Enhanced solution display window using Tkinter"""
    
    def __init__(self, parent, solution):
        self.window = tk.Toplevel(parent)
        self.window.title("解决方案详情")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
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

        # Center window on parent
        self.window.transient(parent)
        self.window.grab_set()

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
        
        # Quality indicator
        if total_moves <= 10:
            quality = "优秀 ⭐⭐⭐"
            color = "green"
        elif total_moves <= 15:
            quality = "良好 ⭐⭐"
            color = "orange"
        else:
            quality = "一般 ⭐"
            color = "red"
            
        ttk.Label(header_frame, text=f"解决方案质量: {quality}", 
                 font=('Arial', 10), foreground=color).pack(pady=(5, 0))

    def _create_navigation(self, parent):
        """Create navigation controls"""
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Step counter
        self.step_label = ttk.Label(nav_frame, text="", font=('Arial', 10))
        self.step_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation buttons
        ttk.Button(nav_frame, text="◀ 上一步", command=self._prev_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="下一步 ▶", command=self._next_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="自动播放", command=self._auto_play).pack(side=tk.LEFT, padx=2)
        
        # Speed control
        ttk.Label(nav_frame, text="播放速度:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.StringVar(value="1.0")
        speed_combo = ttk.Combobox(nav_frame, textvariable=self.speed_var, 
                                   values=["0.5", "1.0", "1.5", "2.0"], width=5)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.set("1.0")
        
        self._update_step_display()

    def _create_content_area(self, parent):
        """Create content area with tabs"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Detailed steps
        self._create_steps_tab()
        
        # Tab 2: Chess board visualization
        self._create_board_tab()
        
        # Tab 3: Statistics
        self._create_stats_tab()

    def _create_steps_tab(self):
        """Create detailed steps tab"""
        steps_frame = ttk.Frame(self.notebook)
        self.notebook.add(steps_frame, text="详细步骤")
        
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
        self.output_text.tag_configure('current', background='#fff3cd', border=1, relief='solid')
        
        # Bind mouse events for step highlighting
        self.output_text.bind("<Button-1>", self._on_step_click)
        self.output_text.bind("<MouseWheel>", self._on_mousewheel)

    def _create_board_tab(self):
        """Create chess board visualization tab"""
        board_frame = ttk.Frame(self.notebook)
        self.notebook.add(board_frame, text="棋盘视图")
        
        # Create canvas for board display
        canvas_frame = ttk.Frame(board_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.board_canvas = tk.Canvas(canvas_frame, bg='white', width=400, height=400)
        self.board_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create step info panel
        info_frame = ttk.LabelFrame(board_frame, text="当前步骤信息", padding="10")
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.step_info_text = tk.Text(info_frame, width=30, height=15, font=('Arial', 10))
        info_scrollbar = ttk.Scrollbar(info_frame, command=self.step_info_text.yview)
        self.step_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.step_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Draw initial board
        self._draw_board()

    def _create_stats_tab(self):
        """Create statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="统计分析")
        
        # Create statistics display
        stats_container = ttk.Frame(stats_frame, padding="20")
        stats_container.pack(fill=tk.BOTH, expand=True)
        
        # Piece usage statistics
        piece_frame = ttk.LabelFrame(stats_container, text="棋子使用统计", padding="10")
        piece_frame.pack(fill=tk.X, pady=(0, 10))
        
        piece_counts = {}
        for step in self.formatted_solution:
            piece = step['piece']
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        for piece, count in sorted(piece_counts.items()):
            ttk.Label(piece_frame, text=f"{piece}: {count} 个", 
                     font=('Arial', 10)).pack(anchor=tk.W)
        
        # Position analysis
        position_frame = ttk.LabelFrame(stats_container, text="位置分布", padding="10")
        position_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_counts = {}
        rank_counts = {}
        for step in self.formatted_solution:
            pos = step['position']
            file_counts[pos[0]] = file_counts.get(pos[0], 0) + 1
            rank_counts[pos[1:]] = rank_counts.get(pos[1:], 0) + 1
        
        ttk.Label(position_frame, text="文件分布: " + ", ".join([f"{file}:{count}" for file, count in sorted(file_counts.items())]),
                 font=('Arial', 10)).pack(anchor=tk.W)
        ttk.Label(position_frame, text="横排分布: " + ", ".join([f"{rank}:{count}" for rank, count in sorted(rank_counts.items())]),
                 font=('Arial', 10)).pack(anchor=tk.W)

    def _create_footer(self, parent):
        """Create footer with export options"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X)
        
        ttk.Button(footer_frame, text="导出为文本", command=self._export_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(footer_frame, text="导出为JSON", command=self._export_json).pack(side=tk.LEFT, padx=2)
        ttk.Button(footer_frame, text="复制到剪贴板", command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(footer_frame, text="关闭", command=self.on_close).pack(side=tk.RIGHT, padx=2)

    def _display_solution(self):
        """Display the complete solution"""
        self.output_text.delete(1.0, tk.END)
        
        # Insert header
        self.output_text.insert(tk.END, "🏆 Chess Bomb 解决方案 🏆\n\n", 'header')
        self.output_text.insert(tk.END, f"使用 ALNS 算法生成的最优解\n", 'header')
        self.output_text.insert(tk.END, f"总步数: {len(self.formatted_solution)} 步\n\n", 'header')
        
        # Insert steps with formatting
        for i, step in enumerate(self.formatted_solution):
            step_text = f"步骤 {step['step']:2d}: "
            self.output_text.insert(tk.END, step_text, 'step')
            
            pos_text = f"{step['position']}"
            self.output_text.insert(tk.END, "位置 ", '')
            self.output_text.insert(tk.END, pos_text, 'position')
            
            piece_text = f" 放置 {step['piece']}\n"
            self.output_text.insert(tk.END, piece_text, 'piece')
            
            # Store step position for highlighting
            start_idx = self.output_text.index(f"{i + 4}.0")
            end_idx = self.output_text.index(f"{i + 4}.end")
            self.step_positions = getattr(self, 'step_positions', {})
            self.step_positions[i] = (start_idx, end_idx)
        
        self.output_text.config(state='disabled')

    def _draw_board(self):
        """Draw chess board with current step highlighted"""
        self.board_canvas.delete("all")
        
        board_size = 360
        cell_size = board_size // 8
        
        # Draw board
        for row in range(8):
            for col in range(8):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Checkerboard pattern
                color = '#f0d9b5' if (row + col) % 2 == 0 else '#b58863'
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
                
                # Add coordinates
                if col == 0:
                    self.board_canvas.create_text(x1 - 10, y1 + cell_size//2, 
                                                text=str(8-row), font=('Arial', 8))
                if row == 7:
                    self.board_canvas.create_text(x1 + cell_size//2, y2 + 10, 
                                                text=chr(97 + col), font=('Arial', 8))
        
        # Draw pieces up to current step
        for i in range(min(self.current_step, len(self.formatted_solution))):
            step = self.formatted_solution[i]
            pos = step['position']
            col = ord(pos[0]) - 97
            row = 8 - int(pos[1:])
            
            x = col * cell_size + cell_size // 2
            y = row * cell_size + cell_size // 2
            
            # Draw piece
            self.board_canvas.create_oval(x-15, y-15, x+15, y+15, 
                                        fill='#4a90e2', outline='#2c3e50', width=2)
            self.board_canvas.create_text(x, y, text=step['piece'][0], 
                                        font=('Arial', 12, 'bold'), fill='white')

    def _update_step_display(self):
        """Update step display and board"""
        if not self.formatted_solution:
            return
            
        self.step_label.config(text=f"步骤: {self.current_step + 1} / {len(self.formatted_solution)}")
        
        # Update step info
        if hasattr(self, 'step_info_text'):
            self.step_info_text.delete(1.0, tk.END)
            if self.current_step < len(self.formatted_solution):
                step = self.formatted_solution[self.current_step]
                info = f"步骤 {step['step']}\n"
                info += f"棋子: {step['piece']}\n"
                info += f"位置: {step['position']}\n"
                info += f"记号: {step['notation']}\n\n"
                
                # Add move description
                info += "移动说明:\n"
                info += f"在棋盘位置 {step['position']} 放置 {step['piece']}。\n"
                
                self.step_info_text.insert(1.0, info)
        
        # Highlight current step in text
        if hasattr(self, 'output_text') and hasattr(self, 'step_positions'):
            self.output_text.tag_remove('current', 1.0, tk.END)
            if self.current_step in self.step_positions:
                start, end = self.step_positions[self.current_step]
                self.output_text.tag_add('current', start, end)
                self.output_text.see(start)
        
        # Update board
        if hasattr(self, 'board_canvas'):
            self._draw_board()

    def _prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_step_display()

    def _next_step(self):
        """Go to next step"""
        if self.current_step < len(self.formatted_solution) - 1:
            self.current_step += 1
            self._update_step_display()

    def _auto_play(self):
        """Auto-play solution"""
        if hasattr(self, 'auto_playing') and self.auto_playing:
            self.auto_playing = False
            return
            
        self.auto_playing = True
        self._play_next()

    def _play_next(self):
        """Play next step in auto-play"""
        if not hasattr(self, 'auto_playing') or not self.auto_playing:
            return
            
        if self.current_step < len(self.formatted_solution) - 1:
            self.current_step += 1
            self._update_step_display()
            speed = float(self.speed_var.get())
            delay = int(1000 / speed)  # Convert speed to delay
            self.window.after(delay, self._play_next)
        else:
            self.auto_playing = False

    def _on_step_click(self, event):
        """Handle step click to jump to that step"""
        index = self.output_text.index(f"@{event.x},{event.y}")
        line = int(index.split('.')[0])
        
        # Calculate step number (account for header lines)
        step_num = line - 4
        if 0 <= step_num < len(self.formatted_solution):
            self.current_step = step_num
            self._update_step_display()

    def _export_text(self):
        """Export solution as text file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Chess Bomb 解决方案\n")
                f.write("=" * 50 + "\n\n")
                for step in self.formatted_solution:
                    f.write(f"步骤{step['step']}: {step['notation']}\n")
            messagebox.showinfo("导出成功", f"解决方案已导出到: {filename}")

    def _export_json(self):
        """Export solution as JSON file"""
        import json
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.formatted_solution, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("导出成功", f"解决方案已导出到: {filename}")

    def _copy_to_clipboard(self):
        """Copy solution to clipboard"""
        solution_text = "\n".join([f"步骤{step['step']}: {step['notation']}" 
                                  for step in self.formatted_solution])
        self.window.clipboard_clear()
        self.window.clipboard_append(solution_text)
        messagebox.showinfo("复制成功", "解决方案已复制到剪贴板")

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.output_text.yview_scroll(-1 * (event.delta // 120), "units")

    def on_close(self):
        """Handle window close"""
        self.window.destroy()