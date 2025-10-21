# Chess Bomb Editor - Architecture Documentation

## Overview

The Chess Bomb Editor has been refactored from a monolithic architecture into a modular, maintainable system. This document describes the new architecture, module responsibilities, and design patterns used.

## Module Structure

### Core Modules

```
chess-bomb-editor/
├── main.py              # Application entry point
├── config.py            # Configuration constants and settings
├── board.py             # Game logic and board state management
├── solver.py            # Solving algorithms (ALNS and beam search)
├── ui.py                # User interface components
├── requirements.txt     # Python dependencies
└── docs/               # Documentation directory
```

### Module Dependencies

```
main.py
└── ui.py
    ├── config.py
    ├── board.py
    └── solver.py
        ├── board.py
        └── config.py
```

## Module Responsibilities

### main.py
- **Purpose**: Application entry point and orchestration
- **Responsibilities**:
  - Initialize the application
  - Handle top-level error management
  - Launch the main UI
- **Key Functions**:
  - `main()`: Main application entry point

### config.py
- **Purpose**: Centralized configuration and constants
- **Responsibilities**:
  - Define game constants (piece types, skull types)
  - Store UI colors and dimensions
  - Manage resource paths
  - Configure solver parameters
- **Key Components**:
  - Game constants (PAWN, KNIGHT, BISHOP, etc.)
  - UI color schemes and layouts
  - Resource path management
  - Solver configuration parameters

### board.py
- **Purpose**: Game logic and board state management
- **Responsibilities**:
  - Manage chess board state
  - Handle piece placement and damage calculations
  - Pre-calculate attack patterns
  - Validate moves and game state
- **Key Classes**:
  - `ChessState`: Represents current game state
  - Attack pattern calculation and caching
- **Key Functions**:
  - `precalculate_attack_patterns()`: Cache attack patterns for performance
  - Move validation and piece efficiency calculations

### solver.py
- **Purpose**: Puzzle solving algorithms
- **Responsibilities**:
  - Implement ALNS (Adaptive Large Neighborhood Search) solver
  - Provide beam search fallback
  - Format and validate solutions
- **Key Classes**:
  - `ChessBombALNSState`: ALNS state representation
  - `ALNSChessBombSolver`: Main ALNS solver implementation
  - `SolutionWindow`: Enhanced solution display
- **Key Algorithms**:
  - ALNS with ChessBomb-specific operators
  - Beam search algorithm as fallback
  - Multiple destroy and repair operators

### ui.py
- **Purpose**: User interface components and interaction
- **Responsibilities**:
  - Main game interface using Pygame
  - Solution display window using Tkinter
  - Handle user input and events
  - Visual feedback and animations
- **Key Classes**:
  - `BoardEditor`: Main Pygame interface
  - `SolutionWindow`: Enhanced solution display with tabs
- **Features**:
  - Enhanced visual design with gradients and shadows
  - Hover effects and animations
  - Multi-tab solution viewer
  - Export functionality

## Design Patterns

### 1. Modular Architecture
- **Pattern**: Separation of Concerns
- **Implementation**: Each module has a single, well-defined responsibility
- **Benefits**: Improved maintainability, testability, and reusability

### 2. Strategy Pattern
- **Pattern**: Multiple solving algorithms
- **Implementation**: ALNS solver with beam search fallback
- **Benefits**: Algorithm flexibility and improved success rates

### 3. Observer Pattern
- **Pattern**: UI updates based on solver state changes
- **Implementation**: Solution display updates when solver completes
- **Benefits**: Decoupled UI and solver logic

### 4. Factory Pattern
- **Pattern**: Solution formatting
- **Implementation**: `format_solution()` function creates standardized solution format
- **Benefits**: Consistent solution representation across UI components

## Data Flow

```
User Input → UI Module → Board State → Solver → Solution → UI Display
     ↑                                                            ↓
     └─────────────── Solution Navigation and Export ←─────────────┘
```

1. **Input Handling**: UI module captures user interactions
2. **State Management**: Board module maintains game state
3. **Solving**: Solver module processes the state and generates solutions
4. **Display**: UI module presents solutions with enhanced visualization
5. **Navigation**: Users can navigate, export, and analyze solutions

## Performance Optimizations

### 1. Attack Pattern Caching
- Pre-calculate all possible attack patterns at startup
- Cache results in `ATTACK_PATTERNS` dictionary
- Benefit: O(1) lookup time during solving

### 2. Modular Loading
- Load resources (images, fonts) only when needed
- Graceful fallback when resources are missing
- Benefit: Faster startup and better error handling

### 3. Efficient State Copying
- Use NumPy arrays for board state
- Implement efficient copy operations in `ChessState.copy()`
- Benefit: Fast state manipulation during solving

## Error Handling

### 1. Resource Loading
- Graceful degradation when assets are missing
- Fallback rendering for missing images
- Console logging for debugging

### 2. Solver Error Handling
- ALNS solver falls back to beam search on failure
- Comprehensive exception handling in solving threads
- User-friendly error messages

### 3. UI Error Handling
- Try-catch blocks around file operations
- Input validation for user interactions
- Window management error handling

## Testing Considerations

### Unit Testing
- Test each module in isolation
- Mock external dependencies (files, GUI)
- Test solver algorithms with known inputs/outputs

### Integration Testing
- Test module interactions
- Test complete user workflows
- Test resource loading and error handling

### Performance Testing
- Benchmark solver performance
- Test UI responsiveness during solving
- Memory usage monitoring

## Future Extensibility

### 1. New Solvers
- Plugin architecture for new algorithms
- Standardized solver interface
- Performance comparison framework

### 2. UI Themes
- Configurable color schemes
- Different board styles
- Accessibility options

### 3. Puzzle Variants
- Different board sizes
- New piece types
- Alternative skull configurations

## Maintenance Guidelines

### 1. Code Style
- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Comprehensive docstrings for all public functions

### 2. Version Control
- Semantic versioning
- Descriptive commit messages
- Feature branches for new development

### 3. Documentation
- Keep this architecture document updated
- Document new features and changes
- Maintain inline code comments

### 4. Dependencies
- Pin dependency versions in requirements.txt
- Regular security updates
- Compatibility testing with new versions