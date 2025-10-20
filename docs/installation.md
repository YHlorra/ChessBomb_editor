# Chess Bomb Editor - Installation and Setup Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 512 MB RAM
- **Storage**: 100 MB available space
- **Graphics**: Any graphics card capable of running Pygame

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **Memory**: 2 GB RAM
- **Storage**: 500 MB available space
- **Graphics**: Dedicated graphics card for better performance

## Installation Steps

### Option 1: Download and Run (Recommended for Users)

1. **Download the Application**
   ```bash
   # Clone the repository
   git clone https://github.com/YHlorra/ChessBomb_editor.git
   cd ChessBomb_editor
   ```

2. **Install Dependencies**
   ```bash
   # Create virtual environment (recommended)
   python -m venv chessbomb_env
   
   # Activate virtual environment
   # Windows:
   chessbomb_env\Scripts\activate
   # macOS/Linux:
   source chessbomb_env/bin/activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

### Option 2: Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/your-username/ChessBomb_editor.git
   cd ChessBomb_editor
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv dev_env
   
   # Activate virtual environment
   # Windows:
   dev_env\Scripts\activate
   # macOS/Linux:
   source dev_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (optional)
   pip install pytest black flake8 mypy
   ```

3. **Verify Installation**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Check code style
   flake8 *.py
   black --check *.py
   
   # Type checking
   mypy *.py
   ```

## Dependencies

### Core Dependencies
```
numpy>=1.20.0          # Numerical computing and board state management
pygame>=2.0.0,<2.3.0   # Game interface and graphics
alns>=2.0.0            # Adaptive Large Neighborhood Search solver
```

### Development Dependencies (Optional)
```
pytest>=6.0.0          # Testing framework
black>=21.0.0           # Code formatting
flake8>=3.8.0           # Linting
mypy>=0.800             # Type checking
```

## Asset Files

The application requires the following asset files in the `assets/` directory:

```
assets/
├── font/
│   └── simsun.ttc       # Chinese font for UI text
├── peices/
│   ├── wQ.svg           # White Queen
│   ├── wR.svg           # White Rook
│   ├── wB.svg           # White Bishop
│   ├── wN.svg           # White Knight
│   ├── wK.svg           # White King
│   └── wP.svg           # White Pawn
└── skull/
    ├── white_skull.png  # White skull (1 HP)
    ├── gray_skull.png   # Gray skull (2 HP)
    └── boss_skull.png   # Boss skull (3 HP)
```

**Note**: The application will run without these assets, but will use fallback graphics and fonts.

## Platform-Specific Setup

### Windows

1. **Install Python**
   - Download from [python.org](https://python.org)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version`

2. **Install Visual C++ Redistributable**
   - Required for some Python packages
   - Download from Microsoft's website

3. **Run the Application**
   ```cmd
   python main.py
   ```

### macOS

1. **Install Python**
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.10
   
   # Or download from python.org
   ```

2. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

3. **Run the Application**
   ```bash
   python3 main.py
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and System Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev
   sudo apt install python3-dev python3-numpy
   ```

2. **Run the Application**
   ```bash
   python3 main.py
   ```

## Configuration

### Environment Variables (Optional)

```bash
# Set custom font path
export CHESSBOMB_FONT_PATH="/path/to/custom/font"

# Set custom assets path
export CHESSBOMB_ASSETS_PATH="/path/to/assets"

# Enable debug mode
export CHESSBOMB_DEBUG=1
```

### Application Configuration

The application can be configured through the `config.py` file:

```python
# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Solver settings
DEFAULT_BEAM_WIDTH = 15
DEFAULT_MAX_DEPTH = 20

# UI settings
DEFAULT_FONT_SIZE = 24
TITLE_FONT_SIZE = 30
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'pygame'`

**Solution**:
```bash
# Install missing dependencies
pip install pygame numpy alns

# If using virtual environment, ensure it's activated
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 2. Font Loading Issues

**Problem**: Chinese text not displaying correctly

**Solution**:
1. Ensure `simsun.ttc` is in `assets/font/` directory
2. Install Chinese fonts on your system
3. Set environment variable for font path

#### 3. Pygame Display Issues

**Problem**: Application window doesn't open

**Solution**:
```bash
# Linux: Set display environment
export DISPLAY=:0

# Test pygame installation
python -c "import pygame; pygame.init(); print('Pygame working')"

# Update graphics drivers
sudo apt update
sudo apt install mesa-utils
```

#### 4. ALNS Solver Issues

**Problem**: Solver crashes or takes too long

**Solution**:
1. Reduce solver time limit:
   ```python
   solution = solve_with_alns(state, time_limit=10)  # 10 seconds
   ```
2. Use beam search fallback:
   ```python
   from solver import beam_search
   solution = beam_search(state)
   ```

#### 5. Memory Issues

**Problem**: Application runs out of memory

**Solution**:
1. Reduce beam width in beam search
2. Limit ALNS iterations
3. Close other applications to free memory

### Platform-Specific Issues

#### Windows

**Problem**: `dll not found` errors

**Solution**:
- Install Microsoft Visual C++ Redistributable
- Use 64-bit Python with 64-bit packages

#### macOS

**Problem**: Permission denied errors

**Solution**:
```bash
# Give Python permission to access screen
sudo tccutil reset All
# Or run from Terminal with security preferences
```

#### Linux

**Problem**: Audio not working

**Solution**:
```bash
# Install audio libraries
sudo apt install libsdl2-mixer-dev

# Test audio system
python3 -c "import pygame; pygame.mixer.init()"
```

## Performance Optimization

### Hardware Recommendations

1. **CPU**: Multi-core processor for faster solving
2. **RAM**: 4GB+ for complex puzzles
3. **Graphics**: Any modern GPU for smooth UI
4. **Storage**: SSD for faster asset loading

### Software Optimization

1. **Use Virtual Environment**: Isolates dependencies and improves performance
2. **Close Background Applications**: Frees up memory and CPU
3. **Update Python**: Use latest Python version for better performance
4. **Configure Solver Settings**: Adjust time limits based on puzzle complexity

## First Run

When you first run the application:

1. **Interface Overview**
   - Left panel: Chess board (8x8 grid)
   - Right panel: Skull selector, piece counter, and action buttons
   - Bottom panel: Solution display and status

2. **Basic Usage**
   - Left-click board cells to place skulls
   - Right-click to remove skulls
   - Select skull type from the selector
   - Adjust available pieces using +/- buttons
   - Click "开始求解" to start solving

3. **Solution Display**
   - Solution appears in the bottom panel
   - Detailed solution window opens automatically
   - Navigate steps using arrow buttons
   - Export solutions using the export options

## Getting Help

### Documentation

- **Architecture Guide**: `docs/architecture.md`
- **Solver Documentation**: `docs/solver.md`
- **API Reference**: `docs/api.md`
- **Maintenance Guide**: `docs/maintenance.md`

### Community Support

- **GitHub Issues**: Report bugs and request features
- **Wiki**: Additional documentation and tutorials
- **Discussions**: Community forum for questions

### Debug Mode

Enable debug mode for detailed logging:

```python
# In config.py
DEBUG_MODE = True

# Or via environment variable
export CHESSBOMB_DEBUG=1
python main.py
```

Debug output includes:
- Solver iteration statistics
- Performance metrics
- Error details
- Asset loading status

## Updates and Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run tests to verify
python -m pytest
```

### Backup and Restore

```bash
# Backup configuration and assets
cp -r assets/ assets_backup/
cp config.py config_backup.py

# Restore if needed
cp -r assets_backup/ assets/
cp config_backup.py config.py
```

### Uninstallation

```bash
# If using virtual environment
deactivate
rm -rf chessbomb_env/

# Remove application directory
rm -rf ChessBomb_editor/

# Note: This does not affect system Python installation
```

## Next Steps

After successful installation:

1. **Read the Documentation**: Browse `docs/` directory for detailed guides
2. **Try the Tutorials**: Check the wiki for step-by-step tutorials
3. **Experiment**: Create simple puzzles to understand the interface
4. **Join the Community**: Participate in discussions and share your creations
5. **Contribute**: Consider contributing to the project development