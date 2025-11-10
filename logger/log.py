import datetime
import os

class Logger:
    """
    Simple timestamped file logger for simulation output.
    Creates directories automatically and supports clean sections.
    """

    def __init__(self, filepath="output/log.txt"):
        # ensure folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.file = open(filepath, "w", encoding="utf-8")

        # initial header
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"=== Simulation Log Started @ {start_time} ===\n")
        self.file.write("=" * 80 + "\n")

    def write(self, text: str):
        """Write one line with timestamp."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.file.write(f"[{timestamp}] {text}\n")
        self.file.flush()

    def section(self, title: str):
        """Write a visually separated section header."""
        self.file.write("\n" + "=" * 80 + "\n")
        self.file.write(f"### {title}\n")
        self.file.write("=" * 80 + "\n")

    def close(self):
        """Close the file properly."""
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"\n=== Simulation Ended @ {end_time} ===\n")
        self.file.close()
