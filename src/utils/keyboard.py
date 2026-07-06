import os
import sys

class KeyboardController:
    @staticmethod
    def check_key(target_key: str) -> bool:
        """
        Check if a specific key was pressed without blocking.
        Supports 'c' (for cancel), '\\x1b' (for ESC), etc.
        """
        target_key = target_key.lower()
        if os.name == 'nt':
            import msvcrt
            if msvcrt.kbhit():
                try:
                    ch = msvcrt.getch()
                    # Some special keys return multiple bytes, safe decode
                    key = ch.decode('utf-8', errors='ignore').lower()
                    if key == target_key:
                        return True
                except:
                    pass
        else:
            # Basic fallback for Unix (requires tty, termios).
            # For simplicity, if not available, just return False.
            try:
                import select
                # Check if stdin has data
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == target_key:
                        return True
            except:
                pass
        return False
