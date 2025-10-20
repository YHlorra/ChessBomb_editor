# Frontend Guideline Document

This document explains how the frontend of the ChessBomb Editor is built, why it’s organized the way it is, and how to keep it consistent and easy to maintain. It doesn’t assume a deep technical background, so anyone on the team can understand it.

## 1. Frontend Architecture

The ChessBomb Editor’s user interface is split into two parts:

- **Pygame Editor Window**
  - Shows the chessboard grid, skulls, piece counters, and buttons.
  - Handles all user input (clicks, drags, keyboard) in real time.
- **Tkinter Solution Window**
  - Pops up after solving to list each piece placement step.
  - Offers a simple, scrollable text view separate from the main grid.

Underneath those windows, the app follows a loose Model-View-Controller pattern:

- **Model (`ChessState`)**: Keeps all board data (skull health, piece counts, placed pieces).
- **View (`BoardEditor` and `SolutionWindow`)**: Draws the interface and updates it on screen.
- **Controller (`BoardEditor` event handlers & helper functions)**: Responds to clicks, runs the solver, and updates the model.

This separation:

- Makes it easy to add or swap out parts (for example, replacing Tkinter with another UI toolkit).
- Keeps code in small, focused sections so it’s easier to find and fix bugs.
- Improves performance by precomputing heavy data (attack patterns) and using a separate thread for solving.

## 2. Design Principles

1. **Usability**  
   - Clear grid layout and intuitive click/drag controls for placing or removing skulls.  
   - On-screen counters and buttons are grouped logically (piece selection, solve controls).  
2. **Accessibility**  
   - Keyboard shortcuts (where possible) to switch modes or cancel the solver.  
   - High-contrast colors for skulls, pieces, and board squares.  
   - Tooltips or labels on buttons to explain their function.  
3. **Responsiveness**  
   - The solver runs in a background thread to prevent freezing.  
   - A progress indicator and “Stop Solving” button keep users informed and in control.  
4. **Consistency**  
   - All buttons share the same shape, padding, and hover behavior.  
   - Similar visual treatment for interactive elements (buttons, counters, board cells).  

## 3. Styling and Theming

### Styling Approach

- We rely on **image assets** (PNG files) for chess pieces and skulls, loaded from an `assets/` folder.  
- UI elements (buttons, panels) are drawn with Pygame’s rectangle and text functions.  
- Configuration variables centrally define colors, font sizes, and padding so style changes affect the whole app.

### Visual Style

- **Modern Flat** with subtle shadows and clear shapes.  
- Light, neutral background on the board and darker accents for controls.  

### Color Palette

- Primary Board Light: `#FFFFFF`  
- Primary Board Dark: `#E0E0E0`  
- Accent (buttons, highlights): `#2E8B57` (sea green)  
- Warning/Errors (no solution): `#D32F2F` (red)  
- Text Primary: `#212121` (near black)  

### Fonts

- Load a clear **sans-serif** font (e.g., `OpenSans-Regular.ttf`) from assets.  
- Fallback: system default sans-serif.  
- Use consistent sizes: board labels (14px), button text (16px), solution list (12px).

## 4. Component Structure

- **`main.py`**: Entry point. Loads resources, initializes attack patterns, and starts the editor.
- **`ChessState`**: Encapsulates the board grid, skull health, piece counts, and placed moves.
- **`valid_moves`, `apply_move`, `heuristic`, `beam_search`**: Pure functions handling solver logic.
- **`BoardEditor`**: Pygame-based class. Manages drawing, input, and threading for solving.
- **`SolutionWindow`**: Tkinter-based class. Displays the final solution steps.

Folder layout:

assets/         # images and fonts
main.py         # application code

Reusing drawing logic and helper functions keeps each class focused and makes it easy to add features or refactor without side effects.

## 5. State Management

- **Single Source of Truth**: All game data lives in one `ChessState` object.
- **Immutable Copies for Solver**: When exploring moves, the solver clones `ChessState` so the editor’s data stays unchanged.
- **Shared State Updates**: After solving, the main thread reads the result and updates the UI to show steps or an error message.

This keeps the editor responsive and prevents accidental overlaps between user edits and solver calculations.

## 6. Routing and Navigation

Although this is a desktop app, it has three main UI states:

1. **Edit Mode**: Default. Users place/remove skulls and pick pieces.
2. **Solving Mode**: Solver runs. A progress bar and “Stop” button are visible.
3. **Solution View**: Tkinter window shows step-by-step placements.

Transition logic lives in `BoardEditor`:

- Clicking “Solve” switches to Solving Mode and starts the thread.  
- Solver success/failure triggers a switch to Solution View or back to Edit Mode with an error message.  

## 7. Performance Optimization

- **Precomputed Attack Patterns**: On startup, we build a lookup table for each piece type and board position. This removes repeated calculations during the beam search.
- **NumPy Grid**: The board is stored as a NumPy array for fast updates and health calculations.
- **Threaded Solver**: Runs the beam search in a background thread to keep the UI loop running at full speed.
- **Limited Beam Width & Depth**: Controls search size, preventing runaway computation.
- **Asset Caching**: Images and fonts load once and stay in memory.

Together, these strategies ensure the editor stays snappy, even on puzzles that take longer to solve.

## 8. Testing and Quality Assurance

- **Unit Tests** (pytest):  
  - `ChessState` methods (`is_solved`, `remaining_health`, efficiency calculations).  
  - Solver helper functions (`valid_moves`, `apply_move`, `heuristic`).
- **Integration Tests**:  
  - Simulate a full solve on small boards to ensure the editor and solver coordinate correctly.
- **End-to-End Tests**:  
  - Use a GUI automation tool (e.g., PyAutoGUI) to click through basic workflows: placing skulls, solving, and viewing results.
- **Linting and Style Checks**:  
  - flake8 or pylint to enforce consistent Python style (PEP 8).  
  - Automated checks in CI to catch errors early.

## 9. Conclusion and Overall Frontend Summary

This guideline shows how the ChessBomb Editor’s frontend is built for clarity, speed, and easy updates. By splitting the UI into clear components, following usability principles, and optimizing heavy tasks, we ensure both developers and end users have a smooth experience. Unique aspects—like Pygame for rich board interaction combined with a simple Tkinter solution window and threaded solving—set this app apart and make it a flexible foundation for future growth.