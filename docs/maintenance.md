# Chess Bomb Editor - Maintenance and Development Guide

## Overview

This guide provides comprehensive information for maintaining, extending, and troubleshooting the Chess Bomb Editor codebase. It covers development workflows, coding standards, testing procedures, and deployment practices.

## Development Environment Setup

### Prerequisites
- Python 3.8+
- Git
- Development IDE (VS Code, PyCharm, or similar)
- Virtual environment tool

### Development Setup
```bash
# 1. Clone the repository
git clone https://github.com/YHlorra/ChessBomb_editor.git
cd ChessBomb_editor

# 2. Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# or
dev_env\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 4. Install pre-commit hooks
pre-commit install
```

### Development Dependencies
```txt
# requirements-dev.txt
pytest>=6.0.0
pytest-cov>=2.10.0
black>=21.0.0
flake8>=3.8.0
mypy>=0.800
pre-commit>=2.15.0
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
```

## Code Standards

### Python Style Guide

#### 1. Code Formatting
- Use **Black** for code formatting
- Line length: 88 characters
- Use double quotes for strings
- Use f-strings for string formatting

```bash
# Format code
black *.py

# Check formatting
black --check *.py
```

#### 2. Linting
- Use **Flake8** for linting
- Follow PEP 8 guidelines
- Import order: standard library, third-party, local

```bash
# Lint code
flake8 *.py

# Configuration in .flake8
[flake8]
max-line-length = 88
ignore = E203, W503
```

#### 3. Type Hints
- Use **MyPy** for type checking
- Add type hints to all public functions
- Use type hints for complex data structures

```bash
# Type checking
mypy *.py

# Configuration in mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
```

### Naming Conventions

```python
# Constants: UPPER_SNAKE_CASE
WINDOW_WIDTH = 800
WHITE_SKULL = 1

# Classes: PascalCase
class ChessState:
    class ALNSChessBombSolver:

# Functions and variables: snake_case
def calculate_piece_efficiency():
    piece_type = "queen"
    remaining_health = 0

# Private methods: prefix with underscore
def _load_assets(self):
    def _validate_move(self, move):
```

### Documentation Standards

#### Docstring Format (Google Style)
```python
def solve_with_alns(initial_state: ChessState, 
                    max_iterations: int = 1000, 
                    time_limit: int = 30) -> list:
    """Solve Chess Bomb puzzle using ALNS algorithm.
    
    Args:
        initial_state: Initial chess board state
        max_iterations: Maximum ALNS iterations to perform
        time_limit: Maximum solving time in seconds
        
    Returns:
        List of moves as (piece_type, x, y) tuples, or None if no solution
        
    Raises:
        ValueError: If initial state is invalid
        RuntimeError: If solver encounters internal error
        
    Example:
        >>> board = np.zeros((8, 8), dtype=int)
        >>> pieces = {QUEEN: 1}
        >>> state = ChessState(board, pieces)
        >>> solution = solve_with_alns(state, time_limit=10)
        >>> print(len(solution))
        3
    """
```

#### Module Documentation
```python
"""
Chess Bomb Editor - Module Description

This module provides [functionality] for [purpose].

Key components:
- Class1: Description
- Function1: Description
- Constant1: Description

Example usage:
    [code example]
"""
```

## Testing

### Test Structure
```
tests/
├── unit/
│   ├── test_board.py
│   ├── test_solver.py
│   ├── test_ui.py
│   └── test_config.py
├── integration/
│   ├── test_solver_integration.py
│   └── test_ui_integration.py
├── fixtures/
│   ├── sample_boards.py
│   └── test_data.json
└── conftest.py
```

### Writing Tests

#### Unit Tests
```python
# tests/unit/test_board.py
import pytest
import numpy as np
from board import ChessState
from config import QUEEN, WHITE_SKULL

class TestChessState:
    def test_initialization(self):
        """Test ChessState initialization."""
        board = np.zeros((8, 8), dtype=int)
        pieces = {QUEEN: 1}
        state = ChessState(board, pieces)
        
        assert state.remaining_health() == 0
        assert not state.is_solved()
        assert len(state.get_valid_moves()) == 64
    
    def test_piece_placement(self):
        """Test piece placement mechanics."""
        board = np.zeros((8, 8), dtype=int)
        board[0][0] = WHITE_SKULL
        pieces = {QUEEN: 1}
        state = ChessState(board, pieces)
        
        new_state = state.place_piece(QUEEN, 1, 1)
        assert new_state is not None
        assert new_state.remaining_health() == 0
        assert new_state.is_solved()
    
    @pytest.mark.parametrize("piece_type,expected_damage", [
        (QUEEN, 9),  # Queen can damage 9 cells from center
        (ROOK, 7),  # Rook can damage 7 cells from center
        (PAWN, 4),  # Pawn can damage 4 cells from center
    ])
    def test_piece_damage(self, piece_type, expected_damage):
        """Test piece damage calculation."""
        # Implementation here
        pass
```

#### Integration Tests
```python
# tests/integration/test_solver_integration.py
import pytest
from board import ChessState
from solver import solve_with_alns, format_solution

class TestSolverIntegration:
    def test_simple_puzzle_solving(self):
        """Test complete puzzle solving workflow."""
        board = np.zeros((8, 8), dtype=int)
        board[0][0] = 1  # White skull at a1
        pieces = {'Q': 1}
        
        state = ChessState(board, pieces)
        solution = solve_with_alns(state, time_limit=5)
        
        assert solution is not None
        assert len(solution) <= 2
        
        formatted = format_solution(solution)
        assert len(formatted) == len(solution)
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_board.py

# Run with specific markers
pytest -m "unit"  # Only unit tests
pytest -m "slow"  # Only slow tests

# Generate coverage report
pytest --cov=. --cov-report=term-missing
```

### Test Fixtures
```python
# tests/conftest.py
import pytest
import numpy as np
from board import ChessState
from config import QUEEN, ROOK, PAWN, WHITE_SKULL, GRAY_SKULL

@pytest.fixture
def empty_board():
    """Create empty 8x8 board."""
    return np.zeros((8, 8), dtype=int)

@pytest.fixture
def simple_state():
    """Create simple chess state with one skull."""
    board = np.zeros((8, 8), dtype=int)
    board[0][0] = WHITE_SKULL
    pieces = {QUEEN: 1}
    return ChessState(board, pieces)

@pytest.fixture
def complex_state():
    """Create complex chess state with multiple skulls."""
    board = np.zeros((8, 8), dtype=int)
    board[0][0] = WHITE_SKULL
    board[1][1] = GRAY_SKULL
    board[2][2] = WHITE_SKULL
    pieces = {QUEEN: 1, ROOK: 1, PAWN: 2}
    return ChessState(board, pieces)
```

## Performance Monitoring

### Profiling
```python
# Profile solver performance
import cProfile
import pstats
from solver import solve_with_alns
from board import ChessState
import numpy as np

def profile_solver():
    board = np.zeros((8, 8), dtype=int)
    # ... setup board
    state = ChessState(board, pieces)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    solution = solve_with_alns(state)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

if __name__ == "__main__":
    profile_solver()
```

### Memory Monitoring
```python
import tracemalloc
import gc

def monitor_memory_usage():
    """Monitor memory usage during solving."""
    tracemalloc.start()
    
    # Run solver
    solution = solve_with_alns(state)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
    gc.collect()  # Force garbage collection
```

### Performance Benchmarks
```python
# benchmarks/solver_benchmarks.py
import time
from statistics import mean, stdev
from solver import solve_with_alns

def benchmark_solver(test_cases, iterations=10):
    """Benchmark solver performance."""
    results = []
    
    for case in test_cases:
        case_times = []
        case_solutions = []
        
        for _ in range(iterations):
            start_time = time.time()
            solution = solve_with_alns(case['state'], time_limit=5)
            end_time = time.time()
            
            case_times.append(end_time - start_time)
            case_solutions.append(len(solution) if solution else None)
        
        results.append({
            'case': case['name'],
            'avg_time': mean(case_times),
            'std_time': stdev(case_times),
            'avg_solution_length': mean([s for s in case_solutions if s is not None]),
            'success_rate': sum(1 for s in case_solutions if s is not None) / len(case_solutions)
        })
    
    return results
```

## Debugging

### Logging Setup
```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Usage in main modules
import logging
logger = logging.getLogger(__name__)

def some_function():
    logger.info("Starting function")
    try:
        # Function logic
        logger.debug("Function completed successfully")
    except Exception as e:
        logger.error(f"Function failed: {e}")
        raise
```

### Debug Mode
```python
# config.py
DEBUG_MODE = os.getenv('CHESSBOMB_DEBUG', 'false').lower() == 'true'

if DEBUG_MODE:
    setup_logging(logging.DEBUG, 'chessbomb_debug.log')
else:
    setup_logging(logging.INFO)
```

### Common Debugging Scenarios

#### 1. Solver Not Finding Solutions
```python
def debug_solver_issue(state):
    """Debug solver issues."""
    print(f"Initial state health: {state.remaining_health()}")
    print(f"Available pieces: {state.available_pieces}")
    print(f"Valid moves count: {len(state.get_valid_moves())}")
    
    # Try beam search as comparison
    from solver import beam_search
    beam_solution = beam_search(state)
    print(f"Beam search solution: {beam_solution}")
    
    # Try ALNS with debug output
    alns_solution = solve_with_alns(state, time_limit=5)
    print(f"ALNS solution: {alns_solution}")
```

#### 2. UI Rendering Issues
```python
def debug_ui_rendering():
    """Debug UI rendering problems."""
    import pygame
    
    # Check pygame initialization
    print(f"Pygame version: {pygame.version.ver}")
    print(f"Display driver: {pygame.display.get_driver()}")
    print(f"Display info: {pygame.display.Info()}")
    
    # Test basic rendering
    screen = pygame.display.set_mode((100, 100))
    screen.fill((255, 0, 0))
    pygame.display.flip()
    
    # Check resource loading
    print(f"Font path exists: {os.path.exists(FONT_PATH)}")
    print(f"Images loaded: {len(skull_images)}")
```

## Release Management

### Version Control Workflow

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-alns-operators

# Make changes and commit
git add .
git commit -m "feat: add new ALNS destroy operators"

# Push to feature branch
git push origin feature/new-alns-operators
```

#### 2. Release Preparation
```bash
# Update version numbers
# Update __init__.py if using package version
# Update CHANGELOG.md

# Create release branch
git checkout -b release/v2.1.0

# Final testing
pytest
black --check .
flake8 .

# Tag release
git tag -a v2.1.0 -m "Release version 2.1.0"
git push origin v2.1.0
```

#### 3. Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Performance benchmarks run
- [ ] Security scan completed
- [ ] Dependencies updated

### Automated Testing

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Test with pytest
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

## Security Considerations

### Input Validation
```python
def validate_board_input(board_data):
    """Validate board input for security."""
    if not isinstance(board_data, (list, np.ndarray)):
        raise ValueError("Board must be array-like")
    
    if len(board_data) != 8:
        raise ValueError("Board must be 8x8")
    
    for row in board_data:
        if len(row) != 8:
            raise ValueError("Board must be 8x8")
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 3:
                raise ValueError("Cell values must be 0-3")
    
    return True
```

### File Operations
```python
import os
from pathlib import Path

def safe_load_asset(filename, asset_dir):
    """Safely load asset file."""
    # Resolve path to prevent directory traversal
    asset_path = Path(asset_dir) / filename
    
    # Ensure path is within asset directory
    try:
        asset_path.resolve().relative_to(Path(asset_dir).resolve())
    except ValueError:
        raise ValueError("Invalid asset path")
    
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset not found: {filename}")
    
    return asset_path
```

## Community Contribution

### Contributing Guidelines

#### 1. Before Contributing
- Read this documentation thoroughly
- Set up development environment
- Run existing tests
- Check existing issues for similar requests

#### 2. Making Changes
- Create feature branch from `main`
- Write tests for new functionality
- Update documentation
- Ensure all tests pass

#### 3. Submitting Changes
- Create pull request with descriptive title
- Include description of changes
- Reference any related issues
- Wait for code review

### Code Review Process
1. **Automated Checks**: CI pipeline runs tests and linting
2. **Manual Review**: Maintainer reviews code quality and functionality
3. **Testing**: Reviewer tests changes manually
4. **Approval**: Changes merged after approval

## Troubleshooting Guide

### Common Development Issues

#### 1. Import Errors
```python
# Problem: Circular imports
# Solution: Reorganize imports, use local imports

def solve_puzzle():
    from solver import solve_with_alns  # Local import
    return solve_with_alns(state)
```

#### 2. Test Failures
```bash
# Problem: Tests failing due to missing test data
# Solution: Check test fixtures and paths

pytest -v tests/unit/test_board.py::TestChessState::test_piece_placement
```

#### 3. Performance Regression
```python
# Problem: New code causing performance issues
# Solution: Profile before and after changes

import time

def time_function(func):
    start = time.time()
    result = func()
    end = time.time()
    print(f"{func.__name__}: {end - start:.3f}s")
    return result
```

### Getting Help

1. **Documentation**: Read relevant docs in `docs/` directory
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Ask questions in GitHub Discussions
4. **Debug Mode**: Enable debug logging for detailed output

## Future Development

### Roadmap Planning

#### Short Term (Next 3 months)
- [ ] Add more ALNS operators
- [ ] Improve UI animations
- [ ] Add puzzle import/export
- [ ] Performance optimizations

#### Medium Term (3-6 months)
- [ ] Multi-language support
- [ ] Custom puzzle themes
- [ ] Advanced solver analytics
- [ ] Mobile-friendly interface

#### Long Term (6+ months)
- [ ] Web-based version
- [ ] Multiplayer support
- [ ] AI-powered puzzle generation
- [ ] Integration with chess platforms

### Architecture Evolution

#### 1. Plugin System
```python
# Future: Plugin architecture for solvers
class SolverPlugin:
    def solve(self, state, **kwargs):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

solver_registry.register("custom_solver", CustomSolverPlugin())
```

#### 2. Configuration System
```python
# Future: External configuration
import yaml

class Config:
    def __init__(self, config_file="config.yaml"):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
```

This maintenance guide provides a comprehensive foundation for the continued development and maintenance of the Chess Bomb Editor. Regular updates and community contributions will help ensure the project remains robust and feature-rich.