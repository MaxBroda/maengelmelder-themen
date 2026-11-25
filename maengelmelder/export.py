from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

# Excel erkennt UTF-8 in einer CSV nur am BOM, sonst werden Umlaute und ß falsch angezeigt.
CSV_ENCODING = "utf-8-sig"


def write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding=CSV_ENCODING)


class ResultWriter:
    """Schreibt jede Ergebnistabelle als CSV und als gleichnamiges Blatt in eine gemeinsame Excel-Datei.

    Mehrere Skripte schreiben in dieselbe Arbeitsmappe; ein vorhandenes Blatt gleichen Namens wird ersetzt.
    """

    MAX_COLUMN_WIDTH = 100

    def __init__(self, directory: Path, workbook: str = "results.xlsx") -> None:
        self._directory = directory
        self._workbook = directory / workbook

    def write(self, frame: pd.DataFrame, name: str, index: bool = False) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        write_csv(frame, self._directory / f"{name}.csv", index=index)
        self._write_sheet(frame, name, index)

    def _write_sheet(self, frame: pd.DataFrame, name: str, index: bool) -> None:
        append = self._workbook.exists()
        with pd.ExcelWriter(
            self._workbook,
            engine="openpyxl",
            mode="a" if append else "w",
            if_sheet_exists="replace" if append else None,
        ) as writer:
            frame.to_excel(writer, sheet_name=name, index=index)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[get_column_letter(column[0].column)].width = min(width + 2, self.MAX_COLUMN_WIDTH)
