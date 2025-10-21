# Backend Structure Document for ChessBomb Editor

## 1. Backend Architecture

This section describes how the core logic of the ChessBomb Editor is organized, the main design patterns in use, and how the setup supports performance and future growth.

• Python-based application organized around three main parts:
  • **Model (ChessState):** Holds the board state (an 8×8 grid), remaining piece counts, and the list of placed pieces.  
  • **View/Controller (BoardEditor):** Manages the Pygame window—drawing the board, handling user clicks, and coordinating solver execution.  
  • **Auxiliary View (SolutionWindow):** A simple Tkinter window that displays the solver’s step-by-step results.

• **Design Patterns and Practices:**
  • **Loose MVC:** ChessState = Model; BoardEditor’s draw methods = View; BoardEditor’s event handlers = Controller.  
  • **Precomputed Lookup (Singleton-style):** At startup, all piece attack patterns are calculated once and stored in a global dictionary for fast lookup during solving.  
  • **Asynchronous Processing:** The beam search runs in a background thread to keep the Pygame UI responsive.

### Scalability, Maintainability, Performance

• **Scalability:** The modular separation of model, solver, and UI code lets you swap or extend each part (for example, add new piece types or a different search algorithm) without large rewrites.  
• **Maintainability:** Clear class boundaries, descriptive function and variable names, and centralized configuration constants make the code easier to read and modify.  
• **Performance:** Using NumPy for board representations and precomputing attack patterns minimizes repeated calculations during search. Threading keeps the interface smooth even on complex puzzles.

## 2. Database Management

The ChessBomb Editor runs entirely in memory and does not use an external database. Instead:

• **In-Memory Structures:**  
  • The board is a NumPy 2D array where each cell value represents skull health or emptiness.  
  • Piece counts are kept in a simple Python dictionary (e.g., {"Queen": 2, "Knight": 3}).

• **Persistence (Planned):**  
  • No disk-based save/load is currently implemented.  
  • A future enhancement could serialize the board state and piece counts to JSON or a lightweight local file.

## 3. Database Schema

_Not applicable_: There is no SQL or NoSQL database in this project. All data lives in Python objects during runtime. Should a file-based save feature be added, the schema might look like:

• **Example JSON Structure (for future use):**
  • `board`: List of 8 lists, each with 8 integers (skull health or 0).  
  • `pieces`: Key-value pairs mapping piece names to remaining counts.  
  • `settings`: Optional properties (beam width, max depth).

## 4. API Design and Endpoints

There is no network API. Instead, the codebase exposes internal functions and methods that act like an API for the solver and editor:

• **get_resource_path(relative_path):** Returns the correct file path for assets whether running as a bundled executable or from source.  
• **precalculate_attack_patterns():** Builds a global lookup of attack moves for every piece type at every board position.  
• **valid_moves(state):** Generates all legal piece placements from a given ChessState.  
• **apply_move(state, move):** Creates a new ChessState by placing one piece and updating the board and piece counts.  
• **heuristic(state):** Scores a ChessState based on remaining skull health and pieces used. Guides the beam search.  
• **beam_search(initial_state, beam_width, max_depth):** Orchestrates the solver, returning a solution sequence or failure.

These functions form a clear interface between the UI and the solver logic.

## 5. Hosting Solutions

This is a desktop application—no cloud or on-premises servers are involved. Key points:

• **Local Execution:** Runs on the user’s machine under a Python interpreter.  
• **Standalone Bundling:** Supports packaging with PyInstaller, using `sys._MEIPASS` to locate assets in the bundled app.  

**Benefits:**

• No ongoing hosting costs.  
• Zero network latency for local use.  
• Easy distribution as a single executable file.

## 6. Infrastructure Components

Although there is no traditional server infrastructure, several components work together under the hood:

• **Threading:** Uses Python’s `threading` module to run the solver in parallel with the main Pygame loop, preventing UI freezes.  
• **Asset Loader:** The `get_resource_path` function abstracts resource lookup so images and fonts load correctly in both development and bundled modes.  
• **Global Lookup Table:** The `ATTACK_PATTERNS` dictionary serves as a CPU-side cache of precomputed moves, drastically speeding up repeated calculations.  

Together, these parts ensure a snappy user experience with responsive graphics and fast puzzle solving.

## 7. Security Measures

As a local, offline application, security concerns are minimal. However, a few best practices are in place:

• **Safe Resource Handling:** Asset loading is wrapped in try/except blocks to catch missing or corrupted files and provide user-friendly error messages.  
• **Thread Safety:** Although Python’s GIL limits true parallelism, careful use of threads avoids race conditions by not sharing mutable state between the UI thread and solver thread.  
• **Input Validation (UI):** Mouse and keyboard inputs are checked to ensure only valid board positions and piece counts are accepted.

## 8. Monitoring and Maintenance

Currently, basic strategies ensure the application runs smoothly and can be maintained over time:

• **Logging (Recommended):** While minimal print statements exist, integrating Python’s `logging` module would track solver progress, errors, and performance metrics.  
• **Exception Handling:** Key operations (asset loading, state transitions) are guarded with error handling to prevent crashes.  
• **Code Organization:** A single main file simplifies navigation, but splitting into modules (e.g., `model.py`, `solver.py`, `ui.py`) is recommended as the project grows.  
• **Automated Testing (Future):** Unit tests for ChessState methods, valid move generation, and heuristic scoring would catch regressions early.

## 9. Conclusion and Overall Backend Summary

The ChessBomb Editor’s backend is a clean, in-memory Python architecture that pairs a model (board and piece data) with a Pygame/Tkinter user interface and a beam search solver. It emphasizes performance through NumPy and precomputed attack patterns, and prioritizes responsiveness via threading. While no external database or server hosting is required, the design is modular enough to add features like persistent puzzles, logging, and more advanced search algorithms. This straightforward, maintainable setup lets puzzle designers focus on creating challenges, confident that the underlying engine is both fast and reliable.