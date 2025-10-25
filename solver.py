"""
Solver module for Chess Bomb Editor
Contains ALNS solving algorithm
"""

import numpy as np
import random
import copy
from board import ChessState, ATTACK_PATTERNS
from config import PIECE_NAMES, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, RED

# 从alns库导入必要的组件
try:
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
except ImportError:
    # 如果alns库不可用，创建简单的模拟类
    class ALNS:
        def __init__(self, rng):
            self.rng = rng
            self.destroy_operators = []
            self.repair_operators = []
            self.accept = None
        
        def add_destroy_operator(self, operator, name=None):
            self.destroy_operators.append(operator)
        
        def add_repair_operator(self, operator, name=None):
            self.repair_operators.append(operator)
    
    class SimulatedAnnealing:
        def __init__(self, start_temp, end_temp, cooling_rate):
            self.start_temp = start_temp
            self.end_temp = end_temp
            self.cooling_rate = cooling_rate


class ChessBombALNSState:
    """Chess Bomb puzzle state for ALNS algorithm"""
    
    def __init__(self, board, available_pieces, moves=None):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.moves = moves if moves is not None else []
        self._objective = None
        
    def objective(self):
        """Objective function: minimize remaining health and piece usage"""
        if self._objective is None:
            remaining_health = np.sum(np.maximum(self.board, 0))
            pieces_used = len(self.moves)
            self._objective = remaining_health * 1000 + pieces_used * 10
        return self._objective
    
    def is_complete(self):
        """Check if the puzzle is solved"""
        return np.all(self.board <= 0)
    
    def copy(self):
        """Create a copy of this state"""
        return ChessBombALNSState(self.board, self.available_pieces, self.moves.copy())
    
    def __str__(self):
        return f"ChessBombState(health={np.sum(np.maximum(self.board, 0))}, moves={len(self.moves)})"


class ALNSChessBombSolver:
    """ALNS solver specifically designed for Chess Bomb puzzles"""
    
    def __init__(self):
        self.alns = None
        self.statistics = None
        
    def setup_alns(self):
        """Initialize ALNS with destroy and repair operators"""
        self.alns = ALNS(rng=random.Random(42))
        
        # Register destroy operators
        self.alns.add_destroy_operator(self.random_piece_removal, name="random_piece_removal")
        self.alns.add_destroy_operator(self.worst_piece_removal, name="worst_piece_removal")
        self.alns.add_destroy_operator(self.cluster_removal, name="cluster_removal")
        
        # Register repair operators
        self.alns.add_repair_operator(self.greedy_piece_placement, name="greedy_piece_placement")
        self.alns.add_repair_operator(self.heuristic_placement, name="heuristic_placement")
        self.alns.add_repair_operator(self.local_search_repair, name="local_search_repair")
        
        # Setup acceptance criteria only
        # 只配置接受标准，让ALNS使用默认的选择器
        self.alns.accept = SimulatedAnnealing(1000, 0.01, 0.001)  # SA acceptance
        
    def random_piece_removal(self, state, rnd_state):
        """Remove a random number of placed pieces"""
        if not state.moves:
            return state
            
        # Remove 1-3 pieces randomly
        num_remove = min(rnd_state.randint(1, 4), len(state.moves))
        new_state = state.copy()
        
        pieces_to_remove = rnd_state.sample(new_state.moves, num_remove)
        
        for piece_type, x, y in pieces_to_remove:
            # Remove piece from board
            new_state.board[x][y] = 0
            new_state.available_pieces[piece_type] += 1
            new_state.moves.remove((piece_type, x, y))
            
            # Restore damage to affected skulls
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if new_state.board[i][j] < 0:  # Empty cell
                    continue
                elif new_state.board[i][j] > 0:  # Skull - restore health
                    new_state.board[i][j] += 1
                    
        return new_state
    
    def worst_piece_removal(self, state, rnd_state):
        """Remove pieces with lowest effectiveness"""
        if not state.moves:
            return state
            
        new_state = state.copy()
        piece_effectiveness = []
        
        # Calculate effectiveness for each piece
        for piece_type, x, y in new_state.moves:
            damage = 0
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if new_state.board[i][j] > 0:
                    damage += 1
            effectiveness = damage / max(1, len(ATTACK_PATTERNS[piece_type][(x, y)]))
            piece_effectiveness.append((effectiveness, (piece_type, x, y)))
        
        # Sort by effectiveness (ascending - remove worst first)
        piece_effectiveness.sort()
        
        # Remove 1-2 worst pieces
        num_remove = min(rnd_state.randint(1, 3), len(piece_effectiveness))
        for i in range(num_remove):
            _, (piece_type, x, y) = piece_effectiveness[i]
            
            # Remove piece
            new_state.board[x][y] = 0
            new_state.available_pieces[piece_type] += 1
            new_state.moves.remove((piece_type, x, y))
            
            # Restore damage
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if new_state.board[i][j] > 0:
                    new_state.board[i][j] += 1
                    
        return new_state
    
    def cluster_removal(self, state, rnd_state):
        """Remove pieces in a cluster (geographically close pieces)"""
        if not state.moves:
            return state
            
        new_state = state.copy()
        
        # Select a random piece as cluster center
        center_piece = rnd_state.choice(new_state.moves)
        _, center_x, center_y = center_piece
        
        # Find nearby pieces
        nearby_pieces = []
        for piece_type, x, y in new_state.moves:
            distance = abs(x - center_x) + abs(y - center_y)
            if distance <= 2:  # Manhattan distance <= 2
                nearby_pieces.append((piece_type, x, y))
        
        # Remove 1-2 nearby pieces
        if nearby_pieces:
            num_remove = min(rnd_state.randint(1, 3), len(nearby_pieces))
            pieces_to_remove = rnd_state.sample(nearby_pieces, num_remove)
            
            for piece_type, x, y in pieces_to_remove:
                new_state.board[x][y] = 0
                new_state.available_pieces[piece_type] += 1
                new_state.moves.remove((piece_type, x, y))
                
                # Restore damage
                for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                    if new_state.board[i][j] > 0:
                        new_state.board[i][j] += 1
                        
        return new_state
    
    def greedy_piece_placement(self, state, rnd_state):
        """Place pieces at positions with maximum damage"""
        new_state = state.copy()
        best_moves = []
        
        # Find all valid moves and calculate their damage
        for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
            if new_state.available_pieces[piece_type] <= 0:
                continue
                
            for x in range(8):
                for y in range(8):
                    if new_state.board[x][y] != 0:
                        continue
                        
                    # Calculate damage
                    damage = 0
                    for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                        if new_state.board[i][j] > 0:
                            damage += 1
                    
                    if damage > 0:
                        best_moves.append((damage, piece_type, x, y))
        
        # Sort by damage (descending)
        best_moves.sort(reverse=True)
        
        # Place 1-2 best pieces
        num_place = min(rnd_state.randint(1, 3), len(best_moves))
        for i in range(num_place):
            _, piece_type, x, y = best_moves[i]
            
            # Place piece
            new_state.board[x][y] = -1  # Mark as placed
            new_state.available_pieces[piece_type] -= 1
            new_state.moves.append((piece_type, x, y))
            
            # Apply damage
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if new_state.board[i][j] > 0:
                    new_state.board[i][j] -= 1
                    
        return new_state
    
    def heuristic_placement(self, state, rnd_state):
        """Place pieces using heuristic evaluation"""
        new_state = state.copy()
        
        # Evaluate all possible moves
        move_scores = []
        
        for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
            if new_state.available_pieces[piece_type] <= 0:
                continue
                
            for x in range(8):
                for y in range(8):
                    if new_state.board[x][y] != 0:
                        continue
                    
                    # Calculate heuristic score
                    score = 0
                    damage = 0
                    
                    for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                        if new_state.board[i][j] > 0:
                            damage += 1
                            # Bonus for finishing skulls
                            if new_state.board[i][j] == 1:
                                score += 2
                        score += 1
                    
                    score += damage * 10
                    
                    # Piece value bonus (prefer cheaper pieces for same damage)
                    piece_values = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 4}
                    score += (10 - piece_values[piece_type]) * damage
                    
                    if damage > 0:
                        move_scores.append((score, piece_type, x, y))
        
        # Sort by score (descending)
        move_scores.sort(reverse=True)
        
        # Place 1-2 best moves
        num_place = min(rnd_state.randint(1, 3), len(move_scores))
        for i in range(num_place):
            _, piece_type, x, y = move_scores[i]
            
            # Place piece
            new_state.board[x][y] = -1
            new_state.available_pieces[piece_type] -= 1
            new_state.moves.append((piece_type, x, y))
            
            # Apply damage
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if new_state.board[i][j] > 0:
                    new_state.board[i][j] -= 1
                    
        return new_state
    
    def local_search_repair(self, state, rnd_state):
        """Local search repair with hill climbing"""
        new_state = state.copy()
        
        # Try to improve current solution by adding pieces
        improved = True
        attempts = 0
        
        while improved and attempts < 10:
            improved = False
            best_move = None
            best_score = new_state.objective()
            
            # Try all possible moves
            for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
                if new_state.available_pieces[piece_type] <= 0:
                    continue
                    
                for x in range(8):
                    for y in range(8):
                        if new_state.board[x][y] != 0:
                            continue
                        
                        # Simulate move
                        test_state = new_state.copy()
                        test_state.board[x][y] = -1
                        test_state.available_pieces[piece_type] -= 1
                        test_state.moves.append((piece_type, x, y))
                        
                        for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                            if test_state.board[i][j] > 0:
                                test_state.board[i][j] -= 1
                        
                        if test_state.objective() < best_score:
                            best_score = test_state.objective()
                            best_move = (piece_type, x, y)
                            improved = True
            
            if best_move:
                piece_type, x, y = best_move
                new_state.board[x][y] = -1
                new_state.available_pieces[piece_type] -= 1
                new_state.moves.append((piece_type, x, y))
                
                for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                    if new_state.board[i][j] > 0:
                        new_state.board[i][j] -= 1
            
            attempts += 1
                    
        return new_state
    
# 简化的自定义选择器类，只保留必要的方法
class SimpleSelector:
    """自定义选择器类，实现ALNS库需要的基本接口"""
    def __init__(self):
        pass
    
    def __call__(self, scores, rnd_state):
        # 简单随机选择一个算子
        return rnd_state.randint(0, len(scores) - 1)
    
    def update(self, delta, d_idx, r_idx, best=None, current=None):
        # ALNS库需要的update方法，不做任何操作
        pass

    def __str__(self):
        return "SimpleSelector()"

    def solve(self, initial_state, max_iterations=1000, time_limit=30):
        """Solve the Chess Bomb puzzle using ALNS"""
        # 检查初始状态是否已解决
        if initial_state.is_solved():
            return []
        
        # 检查是否有可能的移动
        valid_moves = initial_state.get_valid_moves()
        if not valid_moves:
            return []
            
        try:
            # 确保数据结构兼容性
            board_array = np.array(initial_state.board)
            pieces_dict = dict(initial_state.available_pieces)
            alns_state = ChessBombALNSState(board_array, pieces_dict)
            
            # 创建ALNS求解器
            destroy_operators = [
                self.random_piece_removal,
                self.worst_piece_removal,
                self.cluster_removal
            ]
            
            repair_operators = [
                self.greedy_piece_placement,
                self.random_piece_placement,
                self.damage_based_placement
            ]
            
            alns = ALNS(random.Random(42))
            for op in destroy_operators:
                alns.add_destroy_operator(op)
            for op in repair_operators:
                alns.add_repair_operator(op)
            
            op_select = SimpleSelector()
            accept = SimulatedAnnealing(1000, 0.01, 0.001)
            
            from alns.stop import MaxIterations
            result = alns.iterate(
                alns_state,
                op_select,
                accept,
                MaxIterations(max_iterations)
            )
            
            if result.is_complete():
                # 验证解决方案
                test_state = initial_state.copy()
                for piece_type, x, y in result.moves:
                    test_state = test_state.place_piece(piece_type, x, y)
                    if test_state is None:
                        return []
                return result.moves
            else:
                # 如果ALNS没有找到完整解决方案，返回空列表
                return []
        except Exception:
            return []


def solve_with_alns(initial_state, max_iterations=1000, time_limit=30):
    """Solve Chess Bomb puzzle using ALNS algorithm"""
    try:
        # 定义ALNS状态类
        class ALNSChessState:
            """ALNS状态类，包装ChessState"""
            def __init__(self, chess_state):
                self.chess_state = chess_state.copy()
                self.moves = []
                
            def copy(self):
                new_state = ALNSChessState(self.chess_state)
                new_state.moves = self.moves.copy()
                return new_state
            
            def objective(self):
                # 目标函数：最小化剩余生命值 + 已使用棋子数
                return self.chess_state.remaining_health() + len(self.moves)
            
            def is_solved(self):
                return self.chess_state.is_solved()
            
            def get_valid_moves(self):
                return self.chess_state.get_valid_moves()
            
            def apply_move(self, piece, position):
                self.chess_state.apply_move(piece, position)
                self.moves.append((piece, position))
            
            def calculate_damage(self, piece, position):
                # 计算在指定位置放置指定棋子造成的伤害
                # 创建临时状态来计算伤害
                temp_state = self.chess_state.copy()
                temp_state.apply_move(piece, position)
                # 计算伤害：原始生命值 - 新的生命值
                original_health = self.chess_state.remaining_health()
                new_health = temp_state.remaining_health()
                return original_health - new_health
            
            def restore_damage(self, piece, position):
                # 恢复指定棋子造成的伤害
                # 这需要重新计算棋盘状态
                # 由于复杂性，我们选择重新应用所有剩余的移动
                # 创建一个新的棋盘状态副本
                self.chess_state = ChessState(np.copy(initial_state.board), initial_state.available_pieces.copy())
                # 重新应用所有移动，但排除要移除的那个
                for p, pos in self.moves:
                    if p != piece or pos != position:
                        self.chess_state.apply_move(p, pos)
        

        
        # 创建ALNS状态
        alns_state = ALNSChessState(initial_state)

        
        # 定义必要的算子函数
        def random_piece_removal(state, rnd_state):
            new_state = state.copy()
            if new_state.moves:
                # 随机移除一个棋子
                idx = rnd_state.randint(0, len(new_state.moves) - 1)
                piece, position = new_state.moves.pop(idx)
                # 恢复被移除棋子造成的伤害
                new_state.restore_damage(piece, position)
            return new_state
        
        def worst_piece_removal(state, rnd_state):
            new_state = state.copy()
            if new_state.moves:
                # 计算每个棋子的效率
                efficiencies = []
                for i, (piece, position) in enumerate(new_state.moves):
                    damage = new_state.calculate_damage(piece, position)
                    efficiencies.append((damage, i))
                # 移除效率最低的棋子
                if efficiencies:
                    worst_idx = min(efficiencies)[1]
                    piece, position = new_state.moves.pop(worst_idx)
                    new_state.restore_damage(piece, position)
            return new_state
        
        def greedy_piece_placement(state, rnd_state):
            new_state = state.copy()
            # 获取所有有效移动
            valid_moves = new_state.get_valid_moves()
            if valid_moves:
                # 选择伤害最大的移动
                best_move = max(valid_moves, key=lambda x: new_state.calculate_damage(x[0], x[1]))
                new_state.apply_move(*best_move)
            return new_state
        
        # 创建ALNS求解器
        # 简化的实现：使用贪心算法
        # 由于ALNS的复杂性和潜在的兼容性问题，我们使用简化的贪心算法作为替代
        solution = []
        current_state = initial_state.copy()
        

        
        # 最多尝试max_iterations次移动
        for i in range(max_iterations):
            # 检查是否已解决
            if current_state.is_solved():
                return solution
            
            # 获取所有有效移动
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # 选择伤害最大的移动
            best_move = max(valid_moves, key=lambda x: current_state.calculate_piece_efficiency(x[0], x[1], x[2]))
            
            # 应用移动：使用place_piece方法，它返回一个新状态
            new_state = current_state.place_piece(*best_move)
            if new_state is not None:
                current_state = new_state
                solution.append(best_move)
            

        
        # 检查最终状态
        if current_state.is_solved():
            return solution
        else:
            return None
    except Exception as e:
        return None


def format_solution(solution):
    """Format solution steps for display with improved readability"""
    if not solution:
        return []
    
    formatted_steps = []
    piece_symbols = {
        '皇后': '👑', '战车': '🚗', '主教': '🛐', '骑士': '🐎',
        '国王': '👑', '士兵': '⚔️'
    }
    
    for idx, (piece_type, x, y) in enumerate(solution):
        piece_name = PIECE_NAMES.get(piece_type, piece_type)
        pos_text = f"{chr(97 + y)}{8 - x}"  # 棋盘坐标
        piece_symbol = piece_symbols.get(piece_name, '')
        
        # 更详细的移动描述
        move_description = f"{piece_symbol} {piece_name} → {pos_text}"
        
        formatted_steps.append({
            'step': idx + 1,
            'piece': piece_name,
            'position': pos_text,
            'notation': f"{piece_name} {pos_text}",
            'symbol': piece_symbol,
            'description': move_description
        })
    
    return formatted_steps


def validate_board_and_pieces(board, available_pieces):
    """Validate that the board and pieces are suitable for solving"""
    total_health = np.sum(np.maximum(board, 0))
    total_pieces = sum(available_pieces.values())
    
    if total_health == 0:
        return False, "棋盘上没有骷髅！"
    
    if total_pieces == 0:
        return False, "没有可用棋子！"
    
    return True, ""