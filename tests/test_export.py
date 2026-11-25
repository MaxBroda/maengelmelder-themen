import pandas as pd

from maengelmelder.export import ResultWriter


def test_csv_starts_with_bom_so_excel_reads_utf8(tmp_path):
    ResultWriter(tmp_path).write(pd.DataFrame({"Top-Wörter": ["straße"]}), "topics")

    raw = (tmp_path / "topics.csv").read_bytes()

    assert raw.startswith(b"\xef\xbb\xbf")
    assert pd.read_csv(tmp_path / "topics.csv").columns.tolist() == ["Top-Wörter"]


def test_tables_share_one_workbook_and_rewrite_replaces_sheet(tmp_path):
    writer = ResultWriter(tmp_path)
    writer.write(pd.DataFrame({"a": [1]}), "first")
    writer.write(pd.DataFrame({"b": [2]}), "second")
    writer.write(pd.DataFrame({"a": [3]}), "first")

    sheets = pd.read_excel(tmp_path / "results.xlsx", sheet_name=None)

    assert set(sheets) == {"first", "second"}
    assert sheets["first"]["a"].tolist() == [3]
