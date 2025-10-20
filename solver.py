"""
Solver module for Chess Bomb Editor
Contains solving algorithms including beam search and ALNS
"""

import numpy as np
import random
from alns import State, ALNS, Statistics
from alns.accept import HillClimbing, SimulatedAnnealing, RecordToRecordTravel
from alns.select import RouletteWheel, SimpleRandom
from board import ChessState, ATTACK_PATTERNS
from config import PIECE_NAMES, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING


def heuristic(state):
    """Heuristic function for evaluating board states"""
    remaining_health = state.remaining_health()
    pieces_used = len(state.bombs_used)
    return -remaining_health * 1000 - pieces_used


def beam_search(initial_state, beam_width=15, max_depth=20):
    """Beam search algorithm for solving the puzzle"""
    beam = [{
        'state': initial_state,
        'score': heuristic(initial_state),
        'moves': []
    }]

    for _ in range(max_depth):
        candidates = []
        for candidate in beam:
            for move in candidate['state'].get_valid_moves():
                new_state = candidate['state'].place_piece(*move)
                if new_state:
                    new_score = heuristic(new_state)
                    candidates.append({
                        'state': new_state,
                        'score': new_score,
                        'moves': candidate['moves'] + [move]
                    })
        
        if not candidates:
            break
            
        candidates.sort(key=lambda x: (-x['score'], len(x['moves'])))
        beam = candidates[:beam_width]

        if beam and beam[0]['state'].is_solved():
            return beam[0]['moves']

    return None


class ChessBombALNSState(State):
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
        self.alns = ALNS(default_rng_state=42)
        
        # Register destroy operators
        self.alns.add_destroy_operator(self.random_piece_removal, name="random_piece_removal")
        self.alns.add_destroy_operator(self.worst_piece_removal, name="worst_piece_removal")
        self.alns.add_destroy_operator(self.cluster_removal, name="cluster_removal")
        
        # Register repair operators
        self.alns.add_repair_operator(self.greedy_piece_placement, name="greedy_piece_placement")
        self.alns.add_repair_operator(self.heuristic_placement, name="heuristic_placement")
        self.alns.add_repair_operator(self.local_search_repair, name="local_search_repair")
        
        # Setup selection and acceptance criteria
        self.alns.select = RouletteWheel([0.5, 0.3, 0.2], 0.8)  # Operator weights
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
    
    def solve(self, initial_state, max_iterations=1000, time_limit=30):
        """Solve the Chess Bomb puzzle using ALNS"""
        self.setup_alns()
        self.statistics = Statistics()
        
        # Create initial state
        alns_state = ChessBombALNSState(
            initial_state.board, 
            initial_state.available_pieces
        )
        
        # If initial state is already solved, return empty solution
        if alns_state.is_complete():
            return []
        
        # Run ALNS
        try:
            result = self.alns.iterate(
                alns_state,
                max_iterations,
                time_limit=time_limit,
                statistics=self.statistics
            )
            
            if result.is_complete():
                return result.moves
            else:
                # Try beam search as fallback
                return beam_search(initial_state, beam_width=15, max_depth=20)
                
        except Exception as e:
            print(f"ALNS failed: {e}, falling back to beam search")
            return beam_search(initial_state, beam_width=15, max_depth=20)


def solve_with_alns(initial_state, max_iterations=1000, time_limit=30):
    """Convenience function to solve using ALNS"""
    solver = ALNSChessBombSolver()
    return solver.solve(initial_state, max_iterations, time_limit)


def format_solution(solution):
    """Format solution steps for display"""
    if not solution:
        return []
    
    formatted_steps = []
    for idx, (piece_type, x, y) in enumerate(solution):
        piece_name = PIECE_NAMES.get(piece_type, piece_type)
        pos_text = f"{chr(97 + y)}{8 - x}"  # 棋盘坐标
        formatted_steps.append({
            'step': idx + 1,
            'piece': piece_name,
            'position': pos_text,
            'notation': f"{piece_name} {pos_text}"
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