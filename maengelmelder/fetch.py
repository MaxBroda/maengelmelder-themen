"""Lädt alle Meldungen des Mängelmelders Dresden und speichert sie als CSV.

    python -m maengelmelder.fetch [--start 2022-02-01] [--out data/raw/requests.csv]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from maengelmelder.open311 import Open311Client, TimeWindow

BERLIN = ZoneInfo("Europe/Berlin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2022-02-01", help="Startdatum (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, default=Path("data/raw/requests.csv"))
    args = parser.parse_args()

    period = TimeWindow(
        start=datetime.fromisoformat(args.start).replace(tzinfo=BERLIN),
        end=datetime.now(BERLIN),
    )
    client = Open311Client(requests.Session())
    frame = pd.DataFrame(client.requests(period))
    frame["requested_datetime"] = pd.to_datetime(frame["requested_datetime"], utc=True)
    frame = frame.sort_values("requested_datetime").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"{len(frame)} Meldungen ({period.start:%Y-%m-%d} bis {period.end:%Y-%m-%d}) -> {args.out}")


if __name__ == "__main__":
    main()
