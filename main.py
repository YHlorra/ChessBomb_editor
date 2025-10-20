"""
Chess Bomb Editor - Main Entry Point
This is the main entry point for the Chess Bomb Editor application.
The application has been refactored into a modular architecture for better maintainability.
"""

import sys
from ui import BoardEditor


def main():
    """Main entry point for the Chess Bomb Editor application"""
    try:
        # Create and run the board editor
        editor = BoardEditor()
        result = editor.run()
        return result
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())