"""
Board module for Chess Bomb Editor
Contains board state management and game logic using NumPy
"""

import numpy as np
from config import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE_SKULL, GRAY_SKULL, BOSS_SKULL
)


def precalculate_attack_patterns():
    """Pre-calculate attack patterns for all chess pieces on all board positions"""
    patterns = {}
    for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
        patterns[piece_type] = {}
        for x in range(8):
            for y in range(8):
                affected = set()

                # 兵的攻击模式：十字形两格范围
                if piece_type == PAWN:
                    directions = [
                        (0, 1), (0, 2),  # 下方
                        (0, -1), (0, -2),  # 上方
                        (1, 0), (2, 0),  # 右侧
                        (-1, 0), (-2, 0)  # 左侧
                    ]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            affected.add((nx, ny))

                # 马的日字型攻击
                elif piece_type == KNIGHT:
                    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                             (1, -2), (1, 2), (2, -1), (2, 1)]
                    for dx, dy in moves:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            affected.add((nx, ny))

                # 象的对角线无限攻击
                elif piece_type == BISHOP:
                    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nx, ny = x, y
                        while 0 <= (nx := nx + dx) < 8 and 0 <= (ny := ny + dy) < 8:
                            affected.add((nx, ny))

                # 车的十字线无限攻击
                elif piece_type == ROOK:
                    for dx in [-1, 1]:
                        nx, ny = x, y
                        while 0 <= (nx := nx + dx) < 8:
                            affected.add((nx, y))
                    for dy in [-1, 1]:
                        nx, ny = x, y
                        while 0 <= (ny := ny + dy) < 8:
                            affected.add((x, ny))

                # 皇后攻击（车+象）
                elif piece_type == QUEEN:
                    # Rook moves
                    for dx in [-1, 1]:
                        nx = x
                        while 0 <= (nx := nx + dx) < 8:
                            affected.add((nx, y))
                    for dy in [-1, 1]:
                        ny = y
                        while 0 <= (ny := ny + dy) < 8:
                            affected.add((x, ny))
                    # Bishop moves
                    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nx, ny = x, y
                        while 0 <= (nx := nx + dx) < 8 and 0 <= (ny := ny + dy) < 8:
                            affected.add((nx, ny))

                # 王的周围8格攻击
                elif piece_type == KING:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == dy == 0: 
                                continue
                            if 0 <= (nx := x + dx) < 8 and 0 <= (ny := y + dy) < 8:
                                affected.add((nx, ny))
                
                patterns[piece_type][(x, y)] = affected

    return patterns


# Global cache of attack patterns
ATTACK_PATTERNS = precalculate_attack_patterns()


class ChessState:
    """Represents the current state of the chess board puzzle"""
    
    def __init__(self, board, available_pieces=None):
        self.board = board  # 棋盘状态
        self.bombs_used = []  # 已使用棋子的列表
        if available_pieces is None:
            self.available_pieces = {
                PAWN: 0,
                KNIGHT: 0,
                BISHOP: 0,
                ROOK: 0,
                QUEEN: 0,
                KING: 0
            }
        else:
            self.available_pieces = available_pieces.copy()

    def remaining_health(self):
        """Calculate total remaining health of all skulls"""
        return np.sum(np.maximum(self.board, 0))

    def is_solved(self):
        """Check if all skulls have been destroyed"""
        return np.all(self.board <= 0)

    def copy(self):
        """Create a deep copy of the current state"""
        new_state = ChessState(np.copy(self.board), self.available_pieces.copy())
        new_state.bombs_used = self.bombs_used.copy()
        return new_state

    def get_affected_cells(self, piece_type, x, y):
        """获取特定棋子在位置(x, y)能攻击到的所有位置"""
        return ATTACK_PATTERNS[piece_type][(x, y)]

    def place_piece(self, piece_type, x, y):
        """Place a chess piece at the specified position and apply damage"""
        if self.board[x][y] != 0 or self.available_pieces[piece_type] <= 0:
            return None

        new_state = self.copy()
        new_state.board[x][y] = -1  # 标记为已放置棋子
        new_state.available_pieces[piece_type] -= 1
        new_state.bombs_used.append((piece_type, x, y))

        # 获取受影响的单元格
        affected_cells = self.get_affected_cells(piece_type, x, y)

        # 应用伤害
        for i, j in affected_cells:
            if new_state.board[i][j] > 0:
                new_state.board[i][j] -= 1

        return new_state

    def calculate_piece_efficiency(self, piece_type, x, y):
        """Calculate how much damage a piece would deal at position (x, y)"""
        if self.board[x][y] != 0 or self.available_pieces[piece_type] <= 0:
            return -1

        damage = 0
        affected_cells = self.get_affected_cells(piece_type, x, y)

        # 计算伤害
        for i, j in affected_cells:
            if self.board[i][j] > 0:
                damage += 1

        return damage

    def get_valid_moves(self):
        """Get all valid moves for the current state"""
        moves = []
        for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
            if self.available_pieces[piece_type] <= 0:
                continue
            for x in range(8):
                for y in range(8):
                    if self.board[x][y] == 0:
                        moves.append((piece_type, x, y))
        return moves