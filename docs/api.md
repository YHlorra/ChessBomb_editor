# Chess Bomb Editor API Documentation

## Overview

This document describes the public APIs and interfaces provided by the Chess Bomb Editor modules. These APIs are intended for developers who want to extend or integrate with the Chess Bomb Editor.

## Module APIs

### config.py

#### Constants

```python
# Game constants
WHITE_SKULL = 1
GRAY_SKULL = 2
BOSS_SKULL = 3

PAWN = 'P'
KNIGHT = 'N'
BISHOP = 'B'
ROOK = 'R'
QUEEN = 'Q'
KING = 'K'

# UI Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# ... (see config.py for complete list)

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BOARD_SIZE = 360
CELL_SIZE = BOARD_SIZE // 8
```

#### Functions

```python
def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource file.
    
    Args:
        relative_path: Relative path from module directory
        
    Returns:
        Absolute path to resource
        
    Example:
        font_path = get_resource_path("assets/font/simsun.ttc")
    """
```

### board.py

#### Classes

##### ChessState

```python
class ChessState:
    def __init__(self, board: np.ndarray, available_pieces: dict = None):
        """
        Initialize chess board state.
        
        Args:
            board: 8x8 numpy array with skull health values
            available_pieces: Dictionary mapping piece types to counts
            
        Example:
            board = np.zeros((8, 8), dtype=int)
            pieces = {QUEEN: 1, ROOK: 2}
            state = ChessState(board, pieces)
        """
    
    def remaining_health(self) -> int:
        """Calculate total remaining skull health."""
        
    def is_solved(self) -> bool:
        """Check if all skulls are destroyed."""
        
    def copy(self) -> 'ChessState':
        """Create a deep copy of this state."""
        
    def get_affected_cells(self, piece_type: str, x: int, y: int) -> set:
        """
        Get cells affected by a piece at given position.
        
        Args:
            piece_type: Type of chess piece
            x, y: Board coordinates (0-7)
            
        Returns:
            Set of (x, y) tuples for affected cells
        """
        
    def place_piece(self, piece_type: str, x: int, y: int) -> Optional['ChessState']:
        """
        Place a piece and apply damage.
        
        Args:
            piece_type: Type of chess piece to place
            x, y: Board coordinates
            
        Returns:
            New ChessState if placement valid, None otherwise
        """
        
    def calculate_piece_efficiency(self, piece_type: str, x: int, y: int) -> int:
        """
        Calculate damage efficiency for piece placement.
        
        Args:
            piece_type: Type of chess piece
            x, y: Board coordinates
            
        Returns:
            Number of skulls damaged, or -1 if invalid placement
        """
        
    def get_valid_moves(self) -> list:
        """
        Get all valid piece placements.
        
        Returns:
            List of (piece_type, x, y) tuples
        """
```

#### Functions

```python
def precalculate_attack_patterns() -> dict:
    """
    Pre-calculate attack patterns for all pieces and positions.
    
    Returns:
        Dictionary: {piece_type: {(x, y): set of affected cells}}
    """

# Global cache
ATTACK_PATTERNS = precalculate_attack_patterns()
```

### solver.py

#### ALNS Classes

##### ChessBombALNSState

```python
class ChessBombALNSState(State):
    def __init__(self, board: np.ndarray, available_pieces: dict, moves: list = None):
        """
        ALNS state representation.
        
        Args:
            board: 8x8 numpy array
            available_pieces: Available piece counts
            moves: List of moves made so far
        """
    
    def objective(self) -> int:
        """Objective function value (lower is better)."""
        
    def is_complete(self) -> bool:
        """Check if puzzle is solved."""
        
    def copy(self) -> 'ChessBombALNSState':
        """Create state copy."""
```

##### ALNSChessBombSolver

```python
class ALNSChessBombSolver:
    def __init__(self):
        """Initialize ALNS solver with default configuration."""
    
    def solve(self, initial_state: ChessState, max_iterations: int = 1000, 
              time_limit: int = 30) -> list:
        """
        Solve puzzle using ALNS algorithm.
        
        Args:
            initial_state: Initial ChessState
            max_iterations: Maximum ALNS iterations
            time_limit: Time limit in seconds
            
        Returns:
            List of (piece_type, x, y) moves, or None if no solution
        """
    
    def get_statistics(self) -> Statistics:
        """Get ALNS performance statistics."""
```

#### Solver Functions

```python
def solve_with_alns(initial_state: ChessState, max_iterations: int = 1000, 
                   time_limit: int = 30) -> list:
    """
    Convenience function for ALNS solving.
    
    Args:
        initial_state: Initial ChessState
        max_iterations: Maximum iterations
        time_limit: Time limit in seconds
        
    Returns:
        List of moves or None
    """

def beam_search(initial_state: ChessState, beam_width: int = 15, 
                max_depth: int = 20) -> list:
    """
    Solve puzzle using beam search algorithm.
    
    Args:
        initial_state: Initial ChessState
        beam_width: Number of concurrent paths
        max_depth: Maximum solution length
        
    Returns:
        List of moves or None
    """

def format_solution(solution: list) -> list:
    """
    Format solution into user-friendly steps.
    
    Args:
        solution: List of (piece_type, x, y) moves
        
    Returns:
        List of dictionaries with step information:
        [{'step': 1, 'piece': '皇后', 'position': 'a1', 'notation': '皇后 a1'}, ...]
    """

def validate_board_and_pieces(board: np.ndarray, available_pieces: dict) -> tuple:
    """
    Validate board and piece configuration.
    
    Args:
        board: 8x8 numpy array
        available_pieces: Available piece counts
        
    Returns:
        (is_valid: bool, message: str)
    """

def heuristic(state: ChessState) -> int:
    """
    Heuristic function for board evaluation.
    
    Args:
        state: ChessState to evaluate
        
    Returns:
        Heuristic score (higher is better)
    """
```

### ui.py

#### Main Classes

##### BoardEditor

```python
class BoardEditor:
    def __init__(self):
        """Initialize the chess bomb editor interface."""
    
    def run(self) -> Optional[int]:
        """
        Run the main application loop.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
    
    def start_solving(self):
        """Start solving current board configuration."""
    
    def show_solution_window(self):
        """Display solution in separate window."""
```

##### SolutionWindow

```python
class SolutionWindow:
    def __init__(self, parent: tk.Tk, solution: list):
        """
        Create enhanced solution display window.
        
        Args:
            parent: Parent Tkinter window
            solution: List of (piece_type, x, y) moves
        """
    
    def _export_text(self):
        """Export solution as text file."""
    
    def _export_json(self):
        """Export solution as JSON file."""
    
    def _copy_to_clipboard(self):
        """Copy solution to clipboard."""
```

## Usage Examples

### Basic Puzzle Setup and Solving

```python
import numpy as np
from board import ChessState
from solver import solve_with_alns, format_solution
from config import QUEEN, ROOK, PAWN, WHITE_SKULL

# Create board with skulls
board = np.zeros((8, 8), dtype=int)
board[0][0] = WHITE_SKULL  # Skull at a1
board[0][1] = WHITE_SKULL  # Skull at a2

# Define available pieces
available_pieces = {
    QUEEN: 1,
    ROOK: 1,
    PAWN: 2
}

# Create initial state
initial_state = ChessState(board, available_pieces)

# Solve puzzle
solution = solve_with_alns(initial_state, max_iterations=500, time_limit=10)

if solution:
    formatted = format_solution(solution)
    for step in formatted:
        print(f"Step {step['step']}: Place {step['piece']} at {step['position']}")
else:
    print("No solution found")
```

### Custom ALNS Configuration

```python
from solver import ALNSChessBombSolver
from alns.accept import RecordToRecordTravel
from alns.select import SimpleRandom

# Create solver with custom configuration
solver = ALNSChessBombSolver()

# Customize acceptance criteria
solver.alns.accept = RecordToRecordTravel(0.05, 0.01, 0.2)

# Customize operator selection
solver.alns.select = SimpleRandom([0.4, 0.3, 0.2, 0.1])

# Solve with custom parameters
solution = solver.solve(initial_state, max_iterations=2000, time_limit=60)
```

### Board State Manipulation

```python
from board import ChessState
from config import KING, WHITE_SKULL, GRAY_SKULL

# Create initial board
board = np.zeros((8, 8), dtype=int)
board[3][3] = WHITE_SKULL   # d4
board[3][4] = GRAY_SKULL    # e4

available_pieces = {KING: 2}
state = ChessState(board, available_pieces)

# Test piece placement efficiency
efficiency = state.calculate_piece_efficiency(KING, 4, 3)  # Place at d5
print(f"King at d5 efficiency: {efficiency}")

# Get all valid moves
valid_moves = state.get_valid_moves()
print(f"Total valid moves: {len(valid_moves)}")

# Place a piece
new_state = state.place_piece(KING, 4, 3)
if new_state:
    print("Piece placed successfully")
    print(f"Remaining health: {new_state.remaining_health()}")
    print(f"Is solved: {new_state.is_solved()}")
```

### Custom Destroy/Repair Operators

```python
from alns import State
import random

class CustomChessBombState(ChessBombALNSState):
    """Custom state with additional operators"""
    
    def custom_destroy_heuristic(self, rnd_state):
        """Custom destroy operator using heuristic selection"""
        if not self.moves:
            return self
            
        # Select pieces with lowest efficiency
        efficiencies = []
        for piece_type, x, y in self.moves:
            damage = 0
            for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
                if self.board[i][j] > 0:
                    damage += 1
            efficiency = damage / max(1, len(ATTACK_PATTERNS[piece_type][(x, y)]))
            efficiencies.append((efficiency, (piece_type, x, y)))
        
        efficiencies.sort()  # Sort by efficiency (ascending)
        
        # Remove least efficient piece
        new_state = self.copy()
        piece_type, x, y = efficiencies[0][1]
        
        # Remove piece and restore damage
        new_state.board[x][y] = 0
        new_state.available_pieces[piece_type] += 1
        new_state.moves.remove((piece_type, x, y))
        
        for i, j in ATTACK_PATTERNS[piece_type][(x, y)]:
            if new_state.board[i][j] > 0:
                new_state.board[i][j] += 1
                
        return new_state

# Register custom operator
solver = ALNSChessBombSolver()
solver.alns.add_destroy_operator(
    CustomChessBombState.custom_destroy_heuristic,
    name="heuristic_destroy"
)
```

### Integration with Custom UI

```python
import tkinter as tk
from ui import BoardEditor
from solver import solve_with_alns, format_solution

class CustomApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main tkinter window
        
        # Create custom editor
        self.editor = CustomBoardEditor()
        
    def run(self):
        """Run custom application"""
        return self.editor.run()

class CustomBoardEditor(BoardEditor):
    """Custom board editor with additional features"""
    
    def __init__(self):
        super().__init__()
        self.custom_features = True
        
    def start_solving(self):
        """Override solving with custom logic"""
        # Use custom solver configuration
        from solver import ALNSChessBombSolver
        
        if self.solving:
            return
            
        self.solving = True
        
        def custom_solve_thread():
            try:
                # Custom solving logic
                board = self.board_data.copy()
                available_pieces = self.available_pieces.copy()
                
                from board import ChessState
                initial_state = ChessState(board, available_pieces)
                
                # Use custom solver
                solver = ALNSChessBombSolver()
                solution = solver.solve(initial_state, max_iterations=1500, time_limit=45)
                
                if solution:
                    self.solution = solution
                    self.solution_ready = True
                    self.info_messages = [step['notation'] for step in format_solution(solution)[:10]]
                else:
                    self.solution_message = "Custom solver found no solution"
                    
            except Exception as e:
                self.solution_message = f"Custom solver error: {e}"
            finally:
                self.solving = False
                
        thread = threading.Thread(target=custom_solve_thread)
        thread.daemon = True
        thread.start()

# Run custom application
if __name__ == "__main__":
    app = CustomApplication()
    app.run()
```

## Error Handling

### Common Exceptions

```python
# Solver exceptions
try:
    solution = solve_with_alns(initial_state)
except ValueError as e:
    print(f"Invalid board configuration: {e}")
except RuntimeError as e:
    print(f"Solver runtime error: {e}")
except MemoryError:
    print("Insufficient memory for solving")

# Board state exceptions
try:
    new_state = state.place_piece(QUEEN, 0, 0)
except IndexError:
    print("Invalid board coordinates")
except ValueError as e:
    print(f"Invalid piece placement: {e}")

# UI exceptions
try:
    editor = BoardEditor()
    result = editor.run()
except pygame.error as e:
    print(f"Pygame initialization failed: {e}")
except Exception as e:
    print(f"UI error: {e}")
```

### Validation Functions

```python
from solver import validate_board_and_pieces

# Validate before solving
is_valid, message = validate_board_and_pieces(board, available_pieces)
if not is_valid:
    print(f"Configuration invalid: {message}")
    return

# Validate piece placement
if state.calculate_piece_efficiency(piece_type, x, y) < 0:
    print("Invalid piece placement")
    return
```

## Performance Considerations

### Memory Management

```python
# Use efficient state copying
state_copy = state.copy()  # Uses numpy array copying

# Clean up solver statistics
solver = ALNSChessBombSolver()
result = solver.solve(initial_state)
stats = solver.get_statistics()
# solver object will be garbage collected

# Batch operations for efficiency
def solve_multiple_boards(boards_config):
    """Solve multiple board configurations efficiently"""
    results = []
    solver = ALNSChessBombSolver()  # Reuse solver instance
    
    for board, pieces in boards_config:
        state = ChessState(board, pieces)
        solution = solver.solve(state, max_iterations=500)
        results.append(solution)
    
    return results
```

### Threading Considerations

```python
import threading
from solver import solve_with_alns

class ThreadedSolver:
    def __init__(self):
        self.solver_threads = {}
        self.results = {}
    
    def solve_async(self, board_id, initial_state):
        """Solve board asynchronously"""
        if board_id in self.solver_threads:
            return False  # Already solving
        
        def solve_thread():
            try:
                solution = solve_with_alns(initial_state)
                self.results[board_id] = solution
            finally:
                if board_id in self.solver_threads:
                    del self.solver_threads[board_id]
        
        thread = threading.Thread(target=solve_thread)
        thread.daemon = True
        thread.start()
        self.solver_threads[board_id] = thread
        
        return True
    
    def get_result(self, board_id):
        """Get solving result"""
        return self.results.get(board_id)
    
    def is_solving(self, board_id):
        """Check if board is being solved"""
        return board_id in self.solver_threads
```

## Testing Support

### Test Utilities

```python
def create_test_board(skull_positions, skull_types):
    """Create test board with specified skulls"""
    board = np.zeros((8, 8), dtype=int)
    for (x, y), skull_type in zip(skull_positions, skull_types):
        board[x][y] = skull_type
    return board

def create_test_pieces(piece_counts):
    """Create test piece configuration"""
    return piece_counts.copy()

def verify_solution(board, available_pieces, solution):
    """Verify that solution is valid and solves the puzzle"""
    state = ChessState(board, available_pieces.copy())
    
    for piece_type, x, y in solution:
        new_state = state.place_piece(piece_type, x, y)
        if new_state is None:
            return False, f"Invalid move: {piece_type} at ({x},{y})"
        state = new_state
    
    return state.is_solved(), "Solution valid" if state.is_solved() else "Puzzle not solved"

# Example test
def test_simple_puzzle():
    board = create_test_board([(0,0), (1,1)], [WHITE_SKULL, WHITE_SKULL])
    pieces = create_test_pieces({QUEEN: 1, ROOK: 1})
    
    solution = solve_with_alns(ChessState(board, pieces))
    
    if solution:
        is_valid, message = verify_solution(board, pieces, solution)
        print(f"Solution valid: {is_valid}, Message: {message}")
        print(f"Solution length: {len(solution)}")
    else:
        print("No solution found")
```