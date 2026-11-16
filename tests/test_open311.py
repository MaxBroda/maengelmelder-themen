from datetime import datetime, timedelta, timezone

import pytest

from maengelmelder.open311 import Open311Client, TimeWindow


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeSession:
    """Simuliert die API: liefert alle Meldungen im Fenster, aber höchstens `limit`."""

    def __init__(self, timestamps, limit):
        self._timestamps = timestamps
        self._limit = limit
        self.calls = 0

    def get(self, url, params, timeout):
        self.calls += 1
        start = datetime.fromisoformat(params["start_date"])
        end = datetime.fromisoformat(params["end_date"])
        hits = [
            {"service_request_id": str(i), "requested_datetime": ts.isoformat()}
            for i, ts in enumerate(self._timestamps)
            if start <= ts < end
        ]
        return FakeResponse(hits[: self._limit])


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_months_cover_period_without_gaps():
    windows = list(TimeWindow(utc(2022, 1, 15), utc(2022, 4, 10)).months())

    assert [w.start for w in windows] == [utc(2022, 1, 15), utc(2022, 2, 1), utc(2022, 3, 1), utc(2022, 4, 1)]
    assert windows[-1].end == utc(2022, 4, 10)


def test_window_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        TimeWindow(datetime(2022, 1, 1), datetime(2022, 2, 1))


def test_full_window_is_split_until_nothing_is_lost(monkeypatch):
    monkeypatch.setattr(Open311Client, "PAGE_LIMIT", 10)
    timestamps = [utc(2022, 3, 1) + timedelta(hours=i) for i in range(35)]
    session = FakeSession(timestamps, limit=10)

    records = Open311Client(session, pause=0).requests(TimeWindow(utc(2022, 3, 1), utc(2022, 3, 31)))

    assert len(records) == 35
    assert session.calls > 1
