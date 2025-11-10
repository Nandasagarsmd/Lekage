import os
import datetime
import pandas as pd

class Logger:
    """
    Excel-based multi-sheet structured logger.
    Each sheet corresponds to a simulation section (Setup, Loop, RateManager, Summary).
    """

    def __init__(self, filepath="output/log.xlsx"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.data = {"Setup": [], "Loop": [], "RateManager": [], "Summary": []}
        self.current_sheet = "Setup"

    def section(self, title: str):
        """Switch active sheet context based on section name."""
        if "RateManager" in title or "Transition" in title:
            self.current_sheet = "RateManager"
        elif "Loop" in title:
            self.current_sheet = "Loop"
        elif "End" in title or "Summary" in title:
            self.current_sheet = "Summary"
        else:
            self.current_sheet = "Setup"

    def write(self, **kwargs):
        """Append structured record (key-value pairs)."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        record = {"Timestamp": timestamp, **kwargs}
        self.data[self.current_sheet].append(record)

    def close(self):
        """Save all accumulated records to an Excel workbook."""
        with pd.ExcelWriter(self.filepath, engine="openpyxl") as writer:
            for sheet, rows in self.data.items():
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_excel(writer, index=False, sheet_name=sheet)
        print(f"[INFO] Logs saved to {self.filepath}")
