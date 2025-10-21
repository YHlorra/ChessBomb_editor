# Project Requirements Document (PRD)

## 1. Project Overview

ChessBomb Editor is a desktop application that provides an intuitive, visual sandbox for designing and solving custom ChessBomb puzzles. In the editor, users place “skulls” (targets with health values) on a chessboard-like grid, then specify how many of each standard chess piece (King, Queen, Rook, Bishop, Knight, Pawn) they have available. Once the puzzle is set up, the built-in solver uses a beam search algorithm to find a sequence of piece placements that eliminate all skulls, and it displays those moves step by step in a separate window.

This tool is being built to streamline both the creative and analytical aspects of puzzle design for game designers and enthusiasts. The key objectives for version one are:
- A responsive, easy-to-use Pygame-based board editor for skull placement and piece count configuration.
- An efficient beam search solver that runs in the background without freezing the UI.
- A clear solution display via a lightweight Tkinter window.

Success will be measured by:
1. Editor responsiveness (drawing and input lag ≤ 100 ms).  
2. Solver finding valid solutions within 10 seconds on average puzzles.  
3. No UI hangs or crashes during typical use.

## 2. In-Scope vs. Out-of-Scope

In-Scope (Version 1):
- Interactive grid editor in Pygame for placing/removing skulls of different health types.
- On-screen controls to set available piece counts (King, Queen, Rook, Bishop, Knight, Pawn).
- Precomputation of attack patterns for all pieces at every board position.
- Beam search solver with configurable beam width and max depth.
- Background threading of the solver to keep UI responsive.
- Solution visualization in a separate Tkinter window with step-by-step text output.
- Robust asset loading that works both in source mode and when packaged with PyInstaller.

Out-of-Scope (Later Phases):
- Undo/redo history for editor actions.
- Save/load puzzle configurations to/from disk.
- Exporting board images or solution logs as files.
- Alternative search algorithms (A*, genetic, etc.).
- Mobile or web app versions.
- User accounts, networking, or cloud storage.

## 3. User Flow

When a user launches ChessBomb Editor, they see a blank chessboard grid in a Pygame window, alongside controls for choosing skull type (by health level) and entering the number of each chess piece they wish to use. The user clicks on grid cells to place or remove skulls, toggles skull health types via keyboard or UI buttons, and adjusts numeric inputs for piece counts. All changes update the board view and piece-count display in real time.

Once the configuration feels right, the user clicks the “Solve” button. The solver starts in a background thread; the UI shows a progress indicator (e.g., a spinning icon or status text). When the solver finishes or reaches its depth limit, a new Tkinter window pops up listing each placement step (e.g., “Place Queen at D5 eliminates skull at E6”). The user reads through the steps, closes the solution window, and can then modify the puzzle or exit the app.

## 4. Core Features

- **Board Editor (Pygame):** Click/drag to place or remove skulls; keyboard shortcuts for changing skull types.
- **Piece Count Controls:** Numeric inputs or up/down buttons to set how many of each chess piece are available.
- **Precomputed Attack Patterns:** Global lookup table generated at startup for fast attack checks.
- **Beam Search Solver:** Heuristic-driven search that balances remaining skull health and piece usage, with configurable beam width and maximum depth.
- **Asynchronous Processing:** Solver runs in a separate thread to keep the main event loop responsive.
- **Solution Window (Tkinter):** Displays the ordered list of moves that solve the puzzle.
- **Asset Loader:** `get_resource_path()` handles images and fonts whether running from source or a PyInstaller bundle.

## 5. Tech Stack & Tools

- **Language:** Python 3.8+  
- **Frontend/UI:** Pygame (main editor), Tkinter (solution window)  
- **Data & Logic:** NumPy for board representations and fast array operations  
- **Search Algorithm:** Custom beam search implementation  
- **Packaging:** PyInstaller for building standalone executables  
- **Threading:** Python `threading` module to offload the solver  
- **IDE/Plugins (optional):** VS Code, PyCharm; linting via flake8, black auto-formatter

_No external AI models or cloud services are required._

## 6. Non-Functional Requirements

- **Performance:**  
  • Editor redraw rate ≥ 30 fps.  
  • Solver should find solutions for typical puzzles (<50 skulls) within 10 seconds.  
- **Responsiveness:** UI must never freeze during solving; status updates every 0.5 seconds.  
- **Usability:** Clear labels for all controls; consistent mouse/keyboard shortcuts; no modal dialogs that block the main thread.  
- **Reliability:** Graceful handling if no solution is found (display a friendly “No solution within depth limit” message).  
- **Portability:** Runs on Windows, macOS, Linux with minimal setup.  
- **Security & Privacy:** No network calls; user data remains local.

## 7. Constraints & Assumptions

- Python environment includes Pygame, NumPy, and Tkinter.  
- Typical user has a mouse and keyboard; no touch‐screen support required.  
- Beam search performance depends on beam width and depth—solutions may not exist within given limits.  
- All assets (images, fonts) are shipped locally; no dynamic downloads.  
- The board size is fixed (8×8 grid).  
- No external dependencies on databases or web APIs.

## 8. Known Issues & Potential Pitfalls

- **Search Depth & Timeouts:** Too shallow a max depth may miss solutions; too deep may exceed acceptable solve times.  → Mitigation: expose both beam width and depth as editable settings with sensible defaults.
- **Thread Safety:** Pygame is not inherently thread-safe.  → Mitigation: Only update shared state via thread-safe flags and schedule UI updates on the main thread.
- **Tkinter & Pygame Coexistence:** Two GUI frameworks in one app can conflict in event loops.  → Mitigation: Keep Tkinter window simple (read-only text) and run it only after the main loop pauses or solves.
- **Large Precomputation Overhead:** Calculating attack patterns for every square and piece type could add a 1–2 second startup delay.  → Mitigation: Show a “Loading resources…” splash screen or progress bar.
- **Asset Path Failures:** If `sys._MEIPASS` lookup fails, images will not load.  → Mitigation: Fall back to relative paths and display placeholder graphics upon failure.

---

This PRD fully defines the first-release scope, user experience, core modules, and constraints for the ChessBomb Editor, ensuring clear guidance for all subsequent technical and development documents.