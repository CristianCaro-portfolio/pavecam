from datetime import date

import httpx
import pytest

from pavecam.iem import IEMClient

LISTING = """<a href="IDOT-001-00_202601051459.jpg">x</a>
<a href="IDOT-001-00_202601051503.jpg">x</a><a href="IDOT-001-00_202601051459.jpg">dup</a>"""


def _client(tmp_path, handler):
    c = IEMClient("https://example.test", tmp_path)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_frame_listing_is_parsed_deduplicated_and_cached(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, text=LISTING)

    with _client(tmp_path, handler) as c:
        t1 = c.frame_times("IDOT-001-00", date(2026, 1, 5))
        t2 = c.frame_times("IDOT-001-00", date(2026, 1, 5))
    assert [t.strftime("%H%M") for t in t1] == ["1459", "1503"]
    assert t1 == t2 and len(calls) == 1


def test_missing_day_returns_empty(tmp_path):
    with _client(tmp_path, lambda req: httpx.Response(404)) as c:
        assert c.frame_times("IDOT-001-00", date(2026, 1, 5)) == []


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    import tenacity

    monkeypatch.setattr(tenacity.nap, "sleep", lambda s: None)
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, content=b"ok")

    with _client(tmp_path, handler) as c:
        assert c.download("https://example.test/x.jpg") == b"ok"
    assert n["i"] == 3


def test_client_errors_are_not_retried(tmp_path):
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(400)

    with _client(tmp_path, handler) as c, pytest.raises(httpx.HTTPStatusError):
        c.download("https://example.test/x.jpg")
    assert n["i"] == 1
