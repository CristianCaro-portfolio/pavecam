from datetime import UTC, date, datetime, timedelta

import polars as pl

from pavecam import dataset as ds

UTC = UTC


def _rwis(rows):
    return pl.DataFrame(
        rows,
        schema={"station": pl.Utf8, "obtime": pl.Datetime("us", "UTC"), "tmpf": pl.Float64, "dwpf": pl.Float64,
                "tfs0": pl.Float64, "tfs0_text": pl.Utf8, "tfs1": pl.Float64, "tfs1_text": pl.Utf8},
        orient="row",
    )


def test_rwis_labels_maps_codes_and_keeps_raw_code():
    t = datetime(2026, 1, 5, 18, tzinfo=UTC)
    out = ds.rwis_labels(_rwis([("A", t, 20.0, 18.0, 25.0, "Ice Warning", None, None),
                                ("A", t, 40.0, 30.0, 41.0, "Other", 40.0, "Dry")]))
    assert out["label"].to_list() == ["frozen", "dry"]
    assert out["sensor_code"].to_list() == ["Ice Warning", "Dry"]


def test_site_quality_rejects_stuck_and_sparse_sensors():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(100):
        t = t0 + timedelta(minutes=5 * i)
        rows.append(("GOOD", t, "frozen" if i % 4 == 0 else "dry"))
        rows.append(("STUCK", t, "dry"))
        rows.append(("SPARSE", t, "wet" if i < 30 else None))
    labels = pl.DataFrame(rows, schema={"station": pl.Utf8, "obtime": pl.Datetime("us", "UTC"), "label": pl.Utf8},
                          orient="row")
    q = {x.station: x for x in ds.site_quality(labels)}
    assert q["GOOD"].accepted
    assert not q["STUCK"].accepted and "stuck" in q["STUCK"].reason
    assert not q["SPARSE"].accepted and "usable" in q["SPARSE"].reason


def test_select_days_covers_all_kinds_without_overlap():
    rows = []
    for d in range(1, 29):
        day = date(2026, 1, d)
        kind = "frozen" if d <= 6 else "wet" if d <= 12 else "dry"
        for s in "ABCD":
            rows.append((s, datetime(day.year, day.month, day.day, 18, tzinfo=UTC), kind))
    labels = pl.DataFrame(rows, schema={"station": pl.Utf8, "obtime": pl.Datetime("us", "UTC"), "label": pl.Utf8},
                          orient="row")
    days = ds.select_days(labels, n_frozen=3, n_wet=3, n_dry=4, seed=1)
    counts = dict(days.group_by("kind").len().iter_rows())
    assert counts == {"frozen": 3, "wet": 3, "dry": 4}
    assert days["day"].n_unique() == days.height


def test_asof_joins_respect_tolerance():
    slot = datetime(2026, 1, 5, 18, tzinfo=UTC)
    cands = ds.candidate_slots([slot.date()], ("18:00",), ["A", "B"])
    frames = pl.DataFrame({"station": ["A", "B"],
                           "frame_time": [slot + timedelta(minutes=3), slot + timedelta(minutes=25)]},
                          schema={"station": pl.Utf8, "frame_time": pl.Datetime("us", "UTC")})
    out = ds.attach_frames(cands, frames, tolerance_min=10).sort("station")
    assert out["frame_time"][0] is not None
    assert out["frame_time"][1] is None


def test_weather_is_backward_only():
    slot = datetime(2026, 1, 5, 18, tzinfo=UTC)
    cands = ds.candidate_slots([slot.date()], ("18:00",), ["A"])
    asos = pl.DataFrame({"asos_station": ["X", "X"], "asos_time": [slot - timedelta(minutes=40), slot + timedelta(minutes=5)],
                         "air_f": [20.0, 50.0]},
                        schema={"asos_station": pl.Utf8, "asos_time": pl.Datetime("us", "UTC"), "air_f": pl.Float64})
    out = ds.attach_weather(cands, asos, {"A": "X"}, tolerance_min=90)
    assert out["air_f"][0] == 20.0


def test_wx_flags_split_rain_from_freezing_rain():
    t = datetime(2026, 1, 5, 18, tzinfo=UTC)
    raw = pl.DataFrame({
        "station": ["X"] * 4, "valid": [t + timedelta(hours=i) for i in range(4)],
        "tmpf": [30.0, 31.0, 36.0, 20.0], "dwpf": [28.0, 29.0, 33.0, 15.0], "relh": [90.0] * 4,
        "sknt": [10.0] * 4, "vsby": [2.0] * 4, "p01i": [0.01, 0.02, 0.1, None],
        "wxcodes": ["-FZRA", "-SN BR", "-RA", None],
    }, schema_overrides={"valid": pl.Datetime("us", "UTC")})
    f = ds.asos_features(raw)
    assert f["wx_freezing"].to_list() == [1, 0, 0, 0]
    assert f["wx_rain"].to_list() == [0, 0, 1, 0]
    assert f["wx_snow"].to_list() == [0, 1, 0, 0]
    assert f["wx_fog"].to_list() == [0, 1, 0, 0]
    h = ds.add_weather_history(f)
    last = h.row(-1, named=True)
    assert last["snow_12h"] == 1 and last["freezing_12h"] == 1
    assert last["min_air_12h"] == 20.0
    assert abs(last["precip_12h"] - 0.13) < 1e-9
