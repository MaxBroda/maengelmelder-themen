from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator, Protocol

DRESDEN_ENDPOINT = (
    "https://buergerbeteiligung.sachsen.de/api/rest/public/open311/v2/beteiligung/1024377"
)


class Open311Error(Exception):
    pass


class HttpSession(Protocol):
    def get(self, url: str, params: dict[str, str], timeout: float) -> Any: ...


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("TimeWindow braucht zeitzonenbehaftete Zeitpunkte")
        if self.start >= self.end:
            raise ValueError(f"start {self.start} liegt nicht vor end {self.end}")

    def split(self) -> tuple[TimeWindow, TimeWindow]:
        middle = self.start + (self.end - self.start) / 2
        return TimeWindow(self.start, middle), TimeWindow(middle, self.end)

    def months(self) -> Iterator[TimeWindow]:
        cursor = self.start
        while cursor < self.end:
            next_month = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
            upper = min(next_month, self.end)
            yield TimeWindow(cursor, upper)
            cursor = upper

    def as_params(self) -> dict[str, str]:
        return {
            "start_date": self.start.isoformat(timespec="seconds"),
            "end_date": self.end.isoformat(timespec="seconds"),
        }


class Open311Client:
    # Die API liefert stillschweigend höchstens so viele Meldungen pro Anfrage
    # und ignoriert Paging-Parameter. Ein volles Fenster wird daher geteilt.
    PAGE_LIMIT = 1000
    MIN_WINDOW = timedelta(minutes=1)

    def __init__(
        self,
        session: HttpSession,
        endpoint: str = DRESDEN_ENDPOINT,
        timeout: float = 60.0,
        pause: float = 0.2,
    ) -> None:
        self._session = session
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._pause = pause

    def services(self) -> list[dict[str, Any]]:
        return self._get("services.json", {})

    def requests(self, period: TimeWindow) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for month in period.months():
            for record in self._requests_in(month):
                collected[record["service_request_id"]] = record
        return list(collected.values())

    def _requests_in(self, window: TimeWindow) -> list[dict[str, Any]]:
        records = self._get("requests.json", window.as_params())
        if len(records) < self.PAGE_LIMIT:
            return records
        if window.end - window.start <= self.MIN_WINDOW:
            raise Open311Error(f"Fenster {window} ist voll und nicht weiter teilbar")
        first, second = window.split()
        return self._requests_in(first) + self._requests_in(second)

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        url = f"{self._endpoint}/{path}"
        response = self._session.get(url, params=params, timeout=self._timeout)
        if response.status_code != 200:
            raise Open311Error(f"{url} {params} -> HTTP {response.status_code}")
        time.sleep(self._pause)
        payload = response.json()
        if not isinstance(payload, list):
            raise Open311Error(f"{url} {params} -> unerwartete Antwort: {payload!r:.200}")
        return payload
