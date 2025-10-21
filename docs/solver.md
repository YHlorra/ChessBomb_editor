# Chess Bomb Solver Documentation

## Overview

The Chess Bomb Editor implements two main solving algorithms: the Adaptive Large Neighborhood Search (ALNS) algorithm as the primary solver, with beam search as a fallback. This document details the implementation, configuration, and optimization of these solvers.

## ALNS Solver Implementation

### Algorithm Overview

Adaptive Large Neighborhood Search (ALNS) is a metaheuristic optimization algorithm that iteratively improves solutions by destroying parts of the current solution and repairing them with different operators. The ALNS implementation is specifically tailored for the Chess Bomb puzzle domain.

### Key Components

#### 1. ChessBombALNSState
```python
class ChessBombALNSState(State):
    def __init__(self, board, available_pieces, moves=None):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.moves = moves if moves is not None else []
        self._objective = None
```

**Features**:
- Implements the ALNS `State` interface
- Lazy evaluation of objective function
- Efficient state copying with NumPy arrays
- Complete solution detection

#### 2. ALNSChessBombSolver
```python
class ALNSChessBombSolver:
    def __init__(self):
        self.alns = None
        self.statistics = None
```

**Features**:
- Configurable destroy and repair operators
- Adaptive operator selection
- Multiple acceptance criteria
- Performance statistics tracking

### Destroy Operators

Destroy operators remove pieces from the current solution to create room for improvement:

#### 1. Random Piece Removal
```python
def random_piece_removal(self, state, rnd_state):
    num_remove = min(rnd_state.randint(1, 4), len(state.moves))
    # Remove random pieces and restore damage
```

- **Purpose**: Diversification by random removal
- **Removal Count**: 1-3 pieces randomly selected
- **Damage Restoration**: Fully restores skull health for removed pieces

#### 2. Worst Piece Removal
```python
def worst_piece_removal(self, state, rnd_state):
    # Calculate effectiveness = damage / attack_pattern_size
    # Remove pieces with lowest effectiveness
```

- **Purpose**: Remove poorly performing pieces
- **Effectiveness Metric**: Damage dealt per attacked cell
- **Selection**: Removes 1-2 worst pieces

#### 3. Cluster Removal
```python
def cluster_removal(self, state, rnd_state):
    # Select random piece as center
    # Remove geographically nearby pieces (Manhattan distance <= 2)
```

- **Purpose**: Local neighborhood exploration
- **Cluster Definition**: Manhattan distance ≤ 2 from center
- **Geographic Focus**: Promotes spatial reorganization

### Repair Operators

Repair operators add pieces to improve the current solution:

#### 1. Greedy Piece Placement
```python
def greedy_piece_placement(self, state, rnd_state):
    # Calculate damage for all possible moves
    # Place pieces with highest damage
```

- **Purpose**: Rapid improvement through maximum damage
- **Selection Criteria**: Highest damage dealing moves
- **Placement Count**: 1-2 pieces per iteration

#### 2. Heuristic Placement
```python
def heuristic_placement(self, state, rnd_state):
    score = damage * 10
    score += (10 - piece_value) * damage
    # Bonus for finishing skulls (HP = 1)
```

- **Purpose**: Balance damage and piece efficiency
- **Scoring Formula**: 
  - Base score: `damage * 10`
  - Piece efficiency bonus: `(10 - piece_value) * damage`
  - Finishing bonus: `+2` for killing blows
- **Piece Values**: Pawn(1), Knight/Bishop(3), King(4), Rook(5), Queen(9)

#### 3. Local Search Repair
```python
def local_search_repair(self, state, rnd_state):
    # Hill climbing with immediate improvement
    # Try all possible moves, keep best improvement
```

- **Purpose**: Fine-tuning through local optimization
- **Algorithm**: Hill climbing with steepest ascent
- **Convergence**: Stops when no improvement found or 10 attempts

### ALNS Configuration

#### Operator Selection
```python
self.alns.select = RouletteWheel([0.5, 0.3, 0.2], 0.8)
```
- **Initial Weights**: Reflect operator perceived effectiveness
- **Decay Rate**: 0.8 (gradual weight adjustment)
- **Adaptation**: Weights updated based on operator performance

#### Acceptance Criteria
```python
self.alns.accept = SimulatedAnnealing(1000, 0.01, 0.001)
```
- **Initial Temperature**: 1000 (high initial acceptance)
- **Final Temperature**: 0.01 (low final acceptance)
- **Cooling Rate**: 0.001 (gradual temperature reduction)

#### Termination Conditions
- **Time Limit**: 30 seconds (configurable)
- **Iteration Limit**: 1000 iterations (configurable)
- **Solution Quality**: Stops when optimal solution found

## Beam Search Solver

### Algorithm Overview

Beam search is a complete search algorithm that maintains a fixed-size set of promising partial solutions (the "beam") and expands them iteratively.

### Implementation Details

```python
def beam_search(initial_state, beam_width=15, max_depth=20):
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
        
        candidates.sort(key=lambda x: (-x['score'], len(x['moves'])))
        beam = candidates[:beam_width]
        
        if beam and beam[0]['state'].is_solved():
            return beam[0]['moves']
    
    return None
```

### Heuristic Function
```python
def heuristic(state):
    remaining_health = state.remaining_health()
    pieces_used = len(state.bombs_used)
    return -remaining_health * 1000 - pieces_used
```

- **Primary Objective**: Minimize remaining skull health
- **Secondary Objective**: Minimize number of pieces used
- **Weighting**: Health heavily weighted (1000x) over piece count

### Configuration Parameters
- **Beam Width**: 15 (number of concurrent paths)
- **Max Depth**: 20 (maximum solution length)
- **Heuristic**: Linear combination of health and piece efficiency

## Solution Integration

### Solver Selection Strategy
```python
def solve_with_alns(initial_state, max_iterations=1000, time_limit=30):
    try:
        result = alns_solver.solve(initial_state, max_iterations, time_limit)
        if result.is_complete():
            return result.moves
        else:
            return beam_search(initial_state, beam_width=15, max_depth=20)
    except Exception as e:
        print(f"ALNS failed: {e}, falling back to beam search")
        return beam_search(initial_state, beam_width=15, max_depth=20)
```

### Fallback Mechanism
1. **Primary**: ALNS solver with 30-second time limit
2. **Fallback**: Beam search if ALNS fails or doesn't find complete solution
3. **Final**: Return None if no solution found

## Performance Optimization

### 1. Attack Pattern Caching
- Pre-calculate all attack patterns at module initialization
- Store in global `ATTACK_PATTERNS` dictionary
- O(1) lookup during solution evaluation

### 2. Efficient State Management
- Use NumPy arrays for board representation
- Lazy evaluation of objective functions
- Efficient copy operations with shared data where possible

### 3. Early Termination
- Stop immediately when optimal solution found
- Timeout protection for long-running searches
- Memory usage monitoring

## Parameter Tuning Guide

### ALNS Parameters

#### Time Limit
- **Default**: 30 seconds
- **Adjustment**: 
  - Increase for complex puzzles (up to 120 seconds)
  - Decrease for real-time requirements (5-10 seconds)
- **Impact**: Solution quality vs. response time trade-off

#### Iteration Count
- **Default**: 1000 iterations
- **Adjustment**: 
  - Increase for difficult instances (2000-5000)
  - Decrease for rapid solving (500-800)
- **Impact**: Solution thoroughness vs. speed

#### Operator Weights
- **Default**: RouletteWheel([0.5, 0.3, 0.2], 0.8)
- **Tuning Strategy**:
  - Monitor operator performance statistics
  - Adjust weights based on success rates
  - Consider problem-specific characteristics

### Beam Search Parameters

#### Beam Width
- **Default**: 15
- **Range**: 5-50
- **Trade-off**: 
  - Smaller: Faster but less thorough
  - Larger: Better solutions but slower and more memory

#### Max Depth
- **Default**: 20
- **Range**: 10-50
- **Consideration**: Maximum expected solution length

## Benchmarking and Evaluation

### Performance Metrics
1. **Solution Quality**: Number of moves and piece efficiency
2. **Success Rate**: Percentage of puzzles solved optimally
3. **Time to Solution**: Algorithm execution time
4. **Memory Usage**: Peak memory consumption

### Test Scenarios
1. **Simple Puzzles**: 1-5 skulls, few pieces
2. **Medium Puzzles**: 5-15 skulls, moderate piece availability
3. **Complex Puzzles**: 15+ skulls, constrained piece sets
4. **Edge Cases**: No solution, impossible configurations

### Evaluation Framework
```python
def benchmark_solver(solver_func, test_cases):
    results = []
    for case in test_cases:
        start_time = time.time()
        solution = solver_func(case.initial_state)
        end_time = time.time()
        
        results.append({
            'case': case.name,
            'solution_length': len(solution) if solution else 0,
            'execution_time': end_time - start_time,
            'optimal': len(solution) == case.optimal_length,
            'success': solution is not None
        })
    return results
```

## Troubleshooting

### Common Issues

#### 1. ALNS Convergence Problems
- **Symptoms**: Solver stuck in local optima
- **Solutions**: 
  - Increase initial temperature
  - Adjust operator weights
  - Increase iteration count

#### 2. Memory Issues
- **Symptoms**: Out of memory errors on large puzzles
- **Solutions**:
  - Reduce beam width
  - Limit solution depth
  - Implement state garbage collection

#### 3. Slow Performance
- **Symptoms**: Excessive solving time
- **Solutions**:
  - Optimize attack pattern lookups
  - Reduce problem complexity
  - Tune algorithm parameters

### Debug Mode
Enable debug output for solver analysis:
```python
# In solver configuration
solver.debug_mode = True
solver.verbose_output = True
```

This provides detailed operator performance, iteration statistics, and convergence analysis.