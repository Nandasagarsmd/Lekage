import os
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, log_dir="output", filename="log.txt"):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            os.makedirs(log_dir, exist_ok=True)
            cls._instance.path = os.path.join(log_dir, filename)
            with open(cls._instance.path, "w") as f:
                f.write(f"===== Simulation Log Started: {datetime.now()} =====\n\n")
            cls._instance._indent = 0
        return cls._instance

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def write(self, msg, level=0):
        """
        Write a message with timestamp and indentation.
        Example:
            logger.write("Hopping rate = ...", level=1)
        """
        prefix = "    " * (self._indent + level)
        line = f"[{self._timestamp()}] {prefix}{msg}"
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def section(self, title):
        line = "=" * 80
        with open(self.path, "a") as f:
            f.write(f"\n{line}\n>>> {title}\n{line}\n")

    def push(self):
        """Increase indentation level."""
        self._indent += 1

    def pop(self):
        """Decrease indentation level (never below 0)."""
        self._indent = max(0, self._indent - 1)
