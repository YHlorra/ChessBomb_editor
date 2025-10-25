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
        # 使用更友好的方式处理错误
        return 1


if __name__ == "__main__":
    sys.exit(main())