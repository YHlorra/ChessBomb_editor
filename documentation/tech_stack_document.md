# Tech Stack Document for ChessBomb Editor

This document explains in simple terms the technologies used to build the ChessBomb puzzle editor and solver. You don’t need a technical background to understand why each tool was chosen and how they all work together.

## 1. Frontend Technologies

These are the tools that handle everything you see and interact with.

• **Pygame**  
  - Provides the main graphical window where you place skulls and pieces.  
  - Manages drawing the board grid, images, buttons, and captures mouse and keyboard events.  
  - Chosen because it’s simple to use for custom 2D visuals and real-time interaction.

• **Tkinter**  
  - Opens a secondary, lightweight window to display the solver’s steps.  
  - Shows text instructions (e.g., “Place Queen at D4”) without disturbing the main screen.  
  - Ideal for quick pop-up dialogs and text displays without adding complexity to the main interface.

## 2. Backend Technologies

These components run the logic behind the scenes, keeping track of the board and solving the puzzle.

• **Python**  
  - The core language powering all parts of the application.  
  - Known for readability and a large ecosystem of libraries.

• **NumPy**  
  - Represents the board as a fast, 2D array.  
  - Makes it quick to compute which skulls would be hit by a chess piece.

• **Beam Search Algorithm**  
  - A smart search method that looks ahead through possible piece placements.  
  - Keeps only a limited number of the most promising paths (the “beam”), speeding up the solution process.  
  - Balances between finding a good solution quickly and exploring too many options.

• **Resource Management (`get_resource_path`)**  
  - Ensures images and fonts load correctly whether you run the code directly or from a packaged executable.  
  - Detects if the application is bundled (e.g., with PyInstaller) and finds assets accordingly.

• **Threading**  
  - Runs the beam search in a background thread so the main window never freezes.  
  - Allows users to cancel or monitor progress without interruption.

## 3. Infrastructure and Deployment

How the application is built, stored, and released.

• **PyInstaller**  
  - Packs the Python code, assets, and the Python interpreter into a single executable file.  
  - Lets end users run the app on Windows/macOS/Linux without installing Python or extra libraries.

• **Git (Version Control)**  
  - Tracks all code changes over time in a central repository (e.g., GitHub).  
  - Enables multiple developers to work together and roll back to earlier versions if needed.

• **CI/CD Pipelines (Optional)**  
  - Automated processes (e.g., GitHub Actions) can run tests and build new executables whenever code is updated.  
  - Ensures that each release is consistent and that errors are caught early.

## 4. Third-Party Integrations

Services or libraries added from outside to extend functionality.

• **PyInstaller** (Packaging Library)  
  - Already covered under deployment, but worth noting here as a key external tool.

*(No external payment gateways, analytics services, or cloud APIs are used in this version.)*

## 5. Security and Performance Considerations

Steps taken to keep the application reliable, safe, and fast.

### Security

• **Bundled Executable**  
  - Packaging with PyInstaller reduces the risk of missing or tampered files.  
  - Keeps all assets and code together in a protected bundle.

• **Input Validation**  
  - Although basic, the code checks user inputs for piece counts to prevent negative values or invalid text.

### Performance

• **Precomputed Attack Patterns**  
  - At startup, all possible moves for each piece type on each square are calculated once and stored.  
  - During solving, the app simply looks up the results instead of recalculating, greatly speeding up search.

• **NumPy Arrays**  
  - Handling the board as a numeric grid is much faster than using standard Python lists.

• **Multithreading**  
  - Running the solver separately from the user interface ensures smooth interaction even on complex puzzles.

## 6. Conclusion and Overall Tech Stack Summary

ChessBomb Editor combines simple yet powerful tools to deliver a responsive, user-friendly puzzle editor and solver:

• **Python & NumPy** for clear logic and fast board calculations.  
• **Pygame** for an interactive, drag-and-drop style editor.  
• **Tkinter** for clean solution display without cluttering the main interface.  
• **Beam Search** for efficient puzzle solving guided by a smart heuristic.  
• **PyInstaller & Git** for reliable releases and safe collaboration.

These choices strike a balance between performance, maintainability, and ease of use—ensuring that both puzzle creators and solvers enjoy a seamless experience with ChessBomb Editor.